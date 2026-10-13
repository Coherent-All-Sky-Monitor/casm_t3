"""Per-node worker: turn fresh beam dumps into candidate plots and alerts.

Runs on each backend node. T2 drops a JSON trigger card in the local spool
directory when it requests a dump from one of this node's casm_cand_dump
daemons; this daemon picks the card up, waits for the .dada file(s) to
finish landing, extracts the detection beam, renders the candidate figure
twice (hella's own RFI flags applied, and the raw beam), writes the
schema-2 event folder (casm_io docs/t3_events.md) and ships the small
artifacts (PNGs, summary JSON, manifest) to the archive directory on corr1.
Bulk dump data never leaves the node.

hella's flags come from the per-gulp records it writes (``--mask-records``,
casm_io docs/hella_masks.md); they are looked up by absolute sample index,
which the dump header's OBS_OFFSET gives. Without usable records the main
figure is the raw beam labelled "hella RFI flags unavailable (reason)".

Processed cards are renamed to .done (or .failed) in place, which doubles
as the restart bookkeeping.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import socket
import subprocess
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from casm_io.candidates import hella_mask, t3_event
from casm_t2 import beams as t2_beams
from casm_t2 import logsetup

from casm_t3 import dump_reader, event_archive, fil_reader, plotting

logger = logging.getLogger("t3.dump_plotter")

LOCAL_HOSTNAME = socket.gethostname().split(".")[0]


def find_dump_files(dump_dir: Path, not_before: float,
                    not_after: float | None = None) -> list[Path]:
    """List .dada files in dump_dir whose mtime falls in the given window.

    Triggers are rate-limited to >=120 s apart per stream, so a window of a
    minute or so around a card's mtime isolates that card's dump from its
    neighbours in the shared stream directory.
    """
    hits = [p for p in dump_dir.glob("*.dada")
            if p.stat().st_mtime >= not_before
            and (not_after is None or p.stat().st_mtime <= not_after)]
    return sorted(hits, key=lambda p: p.name)


def wait_for_dump(dump_dir: Path, after_mtime: float, timeout_s: float,
                  settle_s: float = 3.0) -> list[Path]:
    """Wait for new .dada files in dump_dir and for their sizes to settle."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        cands = [p for p in dump_dir.glob("*.dada") if p.stat().st_mtime >= after_mtime]
        if cands:
            sizes = {p: p.stat().st_size for p in cands}
            time.sleep(settle_s)
            again = [p for p in dump_dir.glob("*.dada") if p.stat().st_mtime >= after_mtime]
            if {p: p.stat().st_size for p in again} == sizes and again:
                return sorted(again, key=lambda p: p.name)
        time.sleep(2.0)
    return []


def ship_artifacts(files: list[Path], candname: str, archive_host: str, archive_dir: str,
                   final=()) -> None:
    """Copy small result files to the archive tree on the archive host.

    ``final`` (the manifest) is copied after every file in ``files``, so a
    reader that waits for it never sees a half-shipped event.
    """
    dest = f"{archive_dir}/{candname}"
    if archive_host == LOCAL_HOSTNAME:
        Path(dest).mkdir(parents=True, exist_ok=True)
        for f in [*files, *final]:
            shutil.copy2(f, dest)
    else:
        subprocess.run(["ssh", archive_host, f"mkdir -p {dest}"], check=True, timeout=30)
        for group in (files, final):
            if group:
                subprocess.run(["scp", "-q", *map(str, group), f"{archive_host}:{dest}/"],
                               check=True, timeout=120)
    logger.info("archived %s -> %s:%s", candname, archive_host, dest)


def render_card(card: dict, dump_files: list[Path], out_png: Path,
                layout: str | None = None,
                fil_path: Path | None = None,
                subbands: int | None = None,
                clean: str | None = None, scrunches=None,
                sumthreshold: dict | None = None) -> tuple[Path, dict]:
    """Read the detection beam from dump_files and render the candidate figure.

    Pure transform — no dump triggers, no artifact shipping, no alerting —
    so the offline t3-replot CLI can share it safely with the daemon.
    Returns the PNG path and the result record (card + provenance) for the
    caller to persist or discard.

    With fil_path set, the detection beam is also written there as a
    single-beam float32 filterbank; a failure to write it is logged and
    never blocks the figure.
    """
    header, data = dump_reader.read_beams(dump_files, [card["local_beam"]])

    fil_written = None
    if fil_path is not None:
        try:
            fil_written = event_archive.write_event_fil(data[0], header, card, fil_path)
        except (OSError, ValueError) as exc:
            logger.exception("could not write %s: %s", fil_path, exc)

    event_utc = datetime.fromisoformat(card["event_utc"])
    t_rel = (event_utc - header.t0).total_seconds()
    if not 0 <= t_rel <= data.shape[2] * header.tsamp_s:
        logger.warning("event time %.2fs falls outside dump of %.2fs — timing offset?",
                       t_rel, data.shape[2] * header.tsamp_s)

    png = plotting.make_candidate_figure(
        data[0], header.freqs_mhz, header.tsamp_s, t_rel, card, out_png, layout=layout,
        subbands=subbands, clean=clean, scrunches=scrunches, sumthreshold=sumthreshold)

    result = dict(card)
    result.update(host=LOCAL_HOSTNAME, dump_files=[str(f) for f in dump_files],
                  n_samples=int(data.shape[2]), plot=str(png))
    if fil_written is not None:
        result["fil"] = str(fil_written)
    return png, result


def render_card_from_fil(card: dict, fil_path: Path, out_png: Path,
                         layout: str | None = None,
                         subbands: int | None = None,
                         clean: str | None = None, scrunches=None,
                sumthreshold: dict | None = None) -> tuple[Path, dict]:
    """Render the figure from the archived single-beam .fil instead of a dump.

    The .dada is deleted minutes after a trigger and the per-event .fil is
    kept, so this is how an archived event is re-plotted (t3-replot). A pure
    transform like render_card: no dumps, no shipping, no alerting.
    """
    data, header = fil_reader.read_fil(fil_path)
    event_utc = datetime.fromisoformat(card["event_utc"])
    t_rel = (event_utc - header.t0).total_seconds()
    span_s = data.shape[1] * header.tsamp_s
    if not 0 <= t_rel <= span_s:
        logger.warning("event time %.2fs falls outside %s of %.2fs; timing offset?",
                       t_rel, Path(fil_path).name, span_s)
    png = plotting.make_candidate_figure(data, header.freqs_mhz, header.tsamp_s, t_rel,
                                         card, out_png, layout=layout, subbands=subbands,
                                         clean=clean, scrunches=scrunches,
                                         sumthreshold=sumthreshold)
    result = dict(card)
    result.update(host=LOCAL_HOSTNAME, fil=str(fil_path), n_samples=int(data.shape[1]),
                  plot=str(png))
    if clean:
        result["clean"] = clean
        if scrunches:
            result["scrunches"] = [list(st) for st in scrunches]
        if sumthreshold:
            result["sumthreshold"] = dict(sumthreshold)
    return png, result


# --------------------------------------------------------------------------
# hella's flags for one dump
# --------------------------------------------------------------------------

def lookup_mask(mask_records: str, header, card: dict,
                nsamp: int) -> tuple[hella_mask.MaskWindow | None, str | None]:
    """hella's flags over the dump's samples for the detection beam.

    Returns (window, reason). ``reason`` is a short text when the window is
    None (records off, dump not aligned, lookup failed) or covers no sample.
    Never raises: a mask is an addition to the figure, not a precondition.
    """
    if not mask_records:
        return None, "mask records off"
    if not header.contiguous:
        logger.warning("no mask: %s", header.contiguity_note)
        return None, "dump files not contiguous"
    if header.sample0 is None:
        return None, "dump has no OBS_OFFSET"
    utc_start = header.raw.get("UTC_START")
    if not utc_start:
        return None, "dump header has no UTC_START"
    try:
        freqs = header.freqs_mhz
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            win = hella_mask.window_mask(
                mask_records, utc_start, int(card["beam"]), header.sample0, int(nsamp),
                fch1=float(freqs[0]), foff=float(freqs[1] - freqs[0]))
        for w in caught:
            logger.warning("hella mask: %s", w.message)
    except Exception as exc:  # noqa: BLE001 - never blocks the figure
        logger.warning("mask lookup in %s failed: %s: %s", mask_records,
                       type(exc).__name__, exc)
        if isinstance(exc, FileNotFoundError):
            return None, "mask record dir missing"
        return None, f"mask lookup failed: {type(exc).__name__}"
    if not win.coverage.any():
        return win, "no hella record covers the dump"
    return win, None


@dataclass
class EventRender:
    """What render_event produced, for the summary and the delete decision."""
    header: dump_reader.DumpHeader
    nsamp: int
    png: Path                       # the Slack figure: masked, or raw labelled unavailable
    raw_png: Path | None            # None when the raw twin failed
    fil: Path | None
    mask_npz: Path | None
    win: hella_mask.MaskWindow | None   # the window behind the masked figure, else None
    mask_reason: str | None         # why there is no mask (None when the masked figure was drawn)
    masked: bool
    sub_incoh: bool | None
    sub_incoh_source: str


def _remove(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("could not remove %s: %s", path, exc)


def render_event(card: dict, dump_files: list[Path], event_dir: Path,
                 mask_records: str = "", layout: str | None = None,
                 fil_path: Path | None = None, subbands: int | None = None) -> EventRender:
    """Read the detection beam and write the figure pair, the .fil and the mask.

    ``<event>.png`` has hella's flags applied when records cover the dump
    (``<event>.mask.npz`` is written beside it), else it is the raw beam
    labelled "hella RFI flags unavailable (reason)", also when the masked
    render itself fails. ``<event>_raw.png`` is always the raw beam, labelled
    with the IB subtraction state read from the dump header; a failure there is
    logged and leaves ``raw_png`` None. A failed .fil or mask.npz write is
    logged and never blocks a figure. The main figure's own failure raises.
    """
    candname = card["candname"]
    event_dir = Path(event_dir)
    png = event_dir / f"{candname}.png"
    raw_png = event_dir / f"{candname}_raw.png"
    mask_path = event_dir / t3_event.product_names(candname)["mask"]

    header, data = dump_reader.read_beams(dump_files, [card["local_beam"]])
    nsamp = int(data.shape[2])
    fil_written = None
    if fil_path is not None:
        try:
            fil_written = event_archive.write_event_fil(data[0], header, card, fil_path)
        except (OSError, ValueError) as exc:
            logger.exception("could not write %s: %s", fil_path, exc)

    event_utc = datetime.fromisoformat(card["event_utc"])
    t_rel = (event_utc - header.t0).total_seconds()
    if not 0 <= t_rel <= nsamp * header.tsamp_s:
        logger.warning("event time %.2fs falls outside dump of %.2fs; timing offset?",
                       t_rel, nsamp * header.tsamp_s)

    # products of an earlier pass over this card must not outlive a failure now
    _remove(raw_png)
    _remove(mask_path)

    def draw(out: Path, **kw) -> Path:
        return plotting.make_candidate_figure(
            data[0], header.freqs_mhz, header.tsamp_s, t_rel, card, out, layout=layout,
            subbands=subbands, **kw)

    win, reason = lookup_mask(mask_records, header, card, nsamp)
    masked, mask_npz = win is not None and reason is None, None
    if masked:
        try:
            mask_npz = hella_mask.save_event_mask(mask_path, win)
        except (OSError, ValueError) as exc:
            logger.exception("could not write %s: %s", mask_path, exc)
        try:
            draw(png, mask=win.combined, coverage=win.coverage)
        except Exception as exc:  # noqa: BLE001 - fall back to the raw figure
            logger.exception("masked render of %s failed: %s", candname, exc)
            masked, win, reason = False, None, f"mask render failed: {type(exc).__name__}"
            if mask_npz is not None:
                _remove(mask_npz)
                mask_npz = None
    if not masked:
        draw(png, mask_label=event_archive.unavailable_label(reason))

    sub_incoh, sub_source = event_archive.sub_incoh_from_header(header.raw)
    raw_written = None
    try:
        raw_written = draw(raw_png, mask_label=event_archive.raw_label(sub_incoh))
    except Exception as exc:  # noqa: BLE001 - the main figure is already written
        logger.exception("raw render of %s failed: %s", candname, exc)
    return EventRender(header=header, nsamp=nsamp, png=png, raw_png=raw_written,
                       fil=fil_written, mask_npz=mask_npz, win=win if masked else None,
                       mask_reason=reason, masked=masked, sub_incoh=sub_incoh,
                       sub_incoh_source=sub_source)


def _rmdir_if_empty(path: Path) -> None:
    """Remove a directory only when it holds nothing (rmdir refuses otherwise)."""
    try:
        path.rmdir()
    except OSError:
        pass


def _mirror(files: list[Path], dest_dir: Path) -> None:
    """Copy files into dest_dir, fail-soft: a full staging disk never blocks shipping."""
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            if Path(f).parent.resolve() != dest_dir.resolve():
                shutil.copy2(f, dest_dir / Path(f).name)
    except OSError as exc:
        logger.exception("could not copy %s to %s: %s", [Path(f).name for f in files],
                         dest_dir, exc)


def process_card(card_path: Path, args: argparse.Namespace) -> None:
    card = json.loads(card_path.read_text())
    candname = card["candname"]
    dump_dir = Path(card["dump_dir"])
    logger.info("processing %s (beam %d, snr %.1f)", candname, card["beam"], card["snr"])

    files = wait_for_dump(dump_dir, after_mtime=card_path.stat().st_mtime - 10,
                          timeout_s=args.dump_timeout)
    if not files:
        raise RuntimeError(f"no dump appeared in {dump_dir} within {args.dump_timeout}s")

    plots_dir = Path(args.plots_dir)
    events_root = Path(args.events_root) if args.events_root else None
    event_dir = fil_path = None
    if events_root is not None:
        try:
            (events_root / candname).mkdir(parents=True, exist_ok=True)
            event_dir = events_root / candname
            fil_path = event_dir / f"{candname}.fil"
        except OSError as exc:
            logger.exception("could not create event dir under %s: %s", events_root, exc)
    # The bundle (figures, fil, mask, CSVs, summary, manifest) is built in the
    # event dir; with the archive off it is built flat in the plots dir.
    bundle_dir = event_dir or plots_dir
    bundle_dir.mkdir(parents=True, exist_ok=True)
    names = t3_event.product_names(candname)

    try:
        r = render_event(card, files, bundle_dir, mask_records=getattr(args, "mask_records", ""),
                         fil_path=fil_path)
    except Exception:
        if event_dir is not None:
            _rmdir_if_empty(event_dir)       # a card that failed before any product
        raise

    # Plot-then-delete keeps the dump budget disk-neutral. Known-source dumps
    # are kept for folding (the janitor ages them out). The dump goes only when
    # every product that makes the event reproducible was written: the fil (when
    # the archive is on), both figures and, for a masked figure, its mask.npz
    # here, the summary and the manifest below.
    delete_after = (args.delete_dumps
                    and not str(card.get("trigger_reason", "")).startswith("known_source"))
    missing = [name for name, ok in (("fil", events_root is None or r.fil is not None),
                                     ("raw figure", r.raw_png is not None),
                                     ("mask.npz of the masked figure",
                                      not r.masked or r.mask_npz is not None)) if not ok]
    will_delete = delete_after and not missing

    try:
        t3_event.write_sidecars(bundle_dir, candname, card)
    except Exception as exc:  # noqa: BLE001 - the CSVs duplicate the card, which is kept
        logger.exception("could not write the CSVs for %s: %s", candname, exc)
    # None for a product that does not exist (no mask, failed raw render, no fil)
    products = {role: (name if (bundle_dir / name).is_file() else None)
                for role, name in names.items()}
    products["json"], products["manifest"] = names["json"], names["manifest"]

    sub_incoh, sub_source = r.sub_incoh, r.sub_incoh_source
    provenance = {
        "obs_utc_start": r.header.raw.get("UTC_START"),
        "t2d_commit": card.get("t2d_commit"),
        "t3_commit": getattr(args, "t3_commit", None) or "unknown",
        "sub_incoh": sub_incoh, "sub_incoh_source": sub_source,
        "utc_clock": event_archive.UTC_CLOCK,
        **event_archive.hella_provenance(r.win),
    }
    if r.masked:
        mask_block = t3_event.hella_mask_block(r.win, file=r.mask_npz.name if r.mask_npz else None)
        if r.mask_npz is None:
            mask_block["reason"] = "mask.npz could not be written"
    else:
        mask_block = t3_event.hella_mask_block(None, reason=r.mask_reason)
    summary = t3_event.summary_from_card(
        dict(card, data_available=not will_delete), products=products,
        dump=event_archive.dump_span(r.header, r.nsamp, LOCAL_HOSTNAME),
        provenance=provenance, hella_mask=mask_block, kind="candidate")
    # Summary, then the manifest last: t3-collect posts only once it exists.
    summary_json = t3_event.write_summary(bundle_dir, candname, summary)
    manifest = t3_event.write_manifest(bundle_dir, candname)

    shipped = [bundle_dir / names["png"], summary_json]
    if r.raw_png is not None:
        shipped.insert(1, r.raw_png)
    if bundle_dir != plots_dir:
        _mirror([*shipped, manifest], plots_dir)
    ship_artifacts(shipped, candname, args.archive_host, args.archive_dir, final=[manifest])
    # Slack posting moved to t3-collect on corr1 (2026-08-13): corr2 has no
    # internet, so the single poster watches the corr1 archive instead.

    if delete_after and missing:
        logger.warning("keeping the dump of %s: %s not written", candname, ", ".join(missing))
    if will_delete:
        for f in files:
            try:
                size_gb = f.stat().st_size / 1e9
                f.unlink()
                logger.info("deleted dump %s (%.1f GB) after plotting", f.name, size_gb)
            except OSError as exc:
                logger.warning("could not delete %s: %s", f, exc)


def _mark(card_path: Path, suffix: str) -> None:
    """Rename a spool card to its .done/.failed marker.

    Tolerates the card having been removed externally mid-flight (operator
    archiving the spool): one missing card must not stop the polling loop.
    """
    try:
        card_path.rename(card_path.with_suffix(suffix))
    except FileNotFoundError:
        logger.warning("card %s vanished before it could be marked %s",
                       card_path.name, suffix)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot and alert on fresh beam dumps")
    p.add_argument("--spool", default=t2_beams.T2_SPOOL_DIR)
    p.add_argument("--plots-dir", default="/mnt/nvme4/data/casm/t3_plots")
    p.add_argument("--archive-host", default="casm-corr1")
    p.add_argument("--archive-dir", default="/mnt/nvme5/casm_pipeline/candidates")
    p.add_argument("--events-root", default="/mnt/nvme3/T3/EVENTS",
                   help="per-event archive of the Slack PNG, result JSON and "
                        "single-beam .fil; --events-root '' disables it")
    p.add_argument("--mask-records", default="",
                   help="hella's per-gulp flag records (casm_io docs/hella_masks.md); with "
                        "it, the Slack figure has hella's flags applied and <event>.mask.npz "
                        "is stored. '' (default) turns the lookup off")
    p.add_argument("--dump-timeout", type=float, default=180.0)
    p.add_argument("--poll", type=float, default=2.0)
    p.add_argument("--log-file", default=None,
                   help="also log to this file (journald always gets a copy)")
    p.add_argument("--delete-dumps", action="store_true",
                   help="delete .dada files after a successful plot (known-source dumps kept)")
    args = p.parse_args()

    logsetup.setup(args.log_file)
    args.t3_commit = event_archive.source_version()
    spool = Path(args.spool)
    spool.mkdir(parents=True, exist_ok=True)
    Path(args.plots_dir).mkdir(parents=True, exist_ok=True)
    logger.info("watching %s on %s (casm_t3 %s, mask records %s)", spool, LOCAL_HOSTNAME,
                args.t3_commit, args.mask_records or "off")

    while True:
        for card_path in sorted(spool.glob("*.json")):
            try:
                process_card(card_path, args)
            except Exception as exc:  # noqa: BLE001 - one bad card must not stop the loop
                logger.exception("failed on %s: %s", card_path.name, exc)
                _mark(card_path, ".json.failed")
            else:
                _mark(card_path, ".json.done")
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
