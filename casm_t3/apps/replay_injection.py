"""Replay an injection into its beam dump and re-render the T3 candidate figure.

Injections are merged into the assembled stream (d2) that hella searches,
while the dump daemons read d0, upstream of the merge — so the pulse that
fired the trigger is by construction absent from every dump (casm-wiki
``dump-stream-content.md``). This tool puts it back: it reads the injection's
row from the T2 ledger and its matched cluster, reads the detection beam out
of the dump the operator supplies, adds the *same* synthetic pulse the
injection bot generated (``casm_t3.injection``) at the arrival time hella
reported, and renders the ordinary v2 candidate figure from the result.

Nothing here writes to the live system: the sqlite is opened read-only, no
FIFO is touched, no dump is deleted.

    t3-replay-injection --inject-id 661 --dump /path/to/dumpdir --out /tmp/inj661.png
    t3-replay-injection --inject-id 661 --dump ... --no-pulse --out /tmp/inj661_raw.png
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from casm_t2 import beams as t2_beams
from casm_t2 import timing, weights_registry

from casm_t3 import dump_reader, event_archive, injection, plotting, single_pulse

logger = logging.getLogger("t3.replay_injection")

DEFAULT_DB = "/mnt/nvme5/casm_pipeline/db/t2.sqlite"
DEFAULT_CANDS_DIR = "/mnt/nvme4/data/casm/hella_cands"
DEFAULT_CONTEXT_WINDOW_S = 4.0


# --------------------------------------------------------------- ledger

def _rows(conn: sqlite3.Connection, sql: str, args: tuple) -> list[dict]:
    cur = conn.execute(sql, args)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def read_injection(db_path: str, inject_id: int) -> tuple[dict, dict | None]:
    """The ledger row for an injection and its matched cluster row, if any.

    Plain SQL with ``SELECT *`` so a checkout whose schema predates some of
    the injection-bot columns still works: callers must use ``.get``.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        inj = _rows(conn, "SELECT * FROM injections WHERE id = ?", (inject_id,))
        if not inj:
            raise SystemExit(f"no injection id {inject_id} in {db_path}")
        inj = inj[0]
        cluster = None
        cid = inj.get("matched_cluster")
        if cid is not None:
            hits = _rows(conn, "SELECT * FROM clusters WHERE id = ?", (cid,))
            cluster = hits[0] if hits else None
    finally:
        conn.close()
    return inj, cluster


# ---------------------------------------------------------------- dumps

def _dump_span(path: Path) -> tuple[dump_reader.DumpHeader, float]:
    hdr = dump_reader.read_header(path)
    frame_bytes = hdr.nbeam * hdr.nchan * hdr.ninner * hdr.nbit // 8
    nframe = max(0, (path.stat().st_size - hdr.hdr_size)) // frame_bytes
    return hdr, nframe * hdr.ninner * hdr.tsamp_s


def select_dump_files(dump: Path, event_utc: datetime | None,
                      pad_s: float = 2.0) -> list[Path]:
    """The .dada file(s) to read: a file as given, or the ones in a directory
    whose time span brackets the event."""
    if dump.is_file():
        return [dump]
    files = sorted(dump.glob("*.dada"), key=lambda p: p.name)
    if not files:
        raise SystemExit(f"no .dada files in {dump}")
    if event_utc is None:
        return files
    hits = []
    for p in files:
        try:
            hdr, span_s = _dump_span(p)
        except (OSError, ValueError) as exc:
            logger.warning("skipping %s: %s", p.name, exc)
            continue
        if hdr.t0 - timedelta(seconds=pad_s) <= event_utc <= hdr.t0 + timedelta(seconds=span_s + pad_s):
            hits.append(p)
    if not hits:
        raise SystemExit(f"no .dada in {dump} covers {event_utc.isoformat()}")
    return hits


# -------------------------------------------------------------- context

def cands_members(cands_dir: Path, obs_utc_start: str, stream: int,
                  event_utc: datetime, window_s: float) -> list[list[float]]:
    """Raw T1 trials around the event from hella's cands file, as card members.

    Columns of the file are snr samp time_days width dm_idx dm beam (one
    header line). Missing file -> empty list: the T1 panel then just says so.
    """
    path = Path(cands_dir) / f"cands_{obs_utc_start}.dat.{stream}"
    if not path.exists():
        logger.warning("no cands file %s — T1 panel will be empty", path)
        return []
    try:
        utc_start = timing.parse_dada_utc(obs_utc_start)
        arr = np.loadtxt(path, skiprows=1, ndmin=2)
    except (OSError, ValueError) as exc:
        logger.warning("could not read %s: %s", path, exc)
        return []
    if arr.size == 0:
        return []
    t0_off = (utc_start - event_utc).total_seconds()
    dt = arr[:, 1] * timing.TSAMP_S + t0_off
    keep = np.abs(dt) <= window_s
    return [[round(float(d), 4), int(b), float(dm), float(snr), int(w)]
            for d, snr, w, dm, b in zip(dt[keep], arr[keep, 0], arr[keep, 3],
                                        arr[keep, 5], arr[keep, 6])]


# ----------------------------------------------------------------- card

def _width_index(sigma_ms: float, tsamp_s: float) -> int:
    """Boxcar index (ibox) closest to the injected pulse's FWHM."""
    n = injection.fwhm_ms(sigma_ms) / (1000.0 * tsamp_s)
    return int(max(0, min(9, round(np.log2(max(1.0, n))))))


def build_card(inj: dict, cluster: dict | None, event_utc: datetime,
               beam: int, local_beam: int, stream: int, dm: float, width: int,
               snr: float, members: list, window_s: float,
               registry: weights_registry.Registry | None) -> dict:
    """A synthetic trigger card for the replay, shaped like a t2d card."""
    sigma_ms = float(inj.get("sigma_ms") or 0.0)
    amp = float(inj.get("amp") or 0.0)
    est = inj.get("est_snr")
    rec_snr = (cluster or {}).get("snr", inj.get("rec_snr"))
    rec_dm = (cluster or {}).get("dm", inj.get("rec_dm"))
    source = (f"injection replay: injected FWHM {injection.fwhm_ms(sigma_ms):.1f} ms, "
              f"amp {amp:g} counts, injected S/N "
              + (f"{float(est):.1f}" if est is not None else "?"))
    if rec_snr is not None and rec_dm is not None:
        source += f"; hella reported S/N {float(rec_snr):.1f} at DM {float(rec_dm):.1f}"
    else:
        source += "; not recovered by hella"
    # The full sentence overflows the v2 suptitle (it is ~90 characters on top
    # of the name and the UTC), so the title says only what this is and the
    # parameters go in the waterfall panel title, wrapped to the panel width.
    note = "\n".join(textwrap.wrap(source, 52))

    sky = pointings = None
    if registry is not None:
        try:
            sky = registry.sky_for(event_utc, int(beam), sun=True)
            pointings = registry.pointings_for(event_utc)
        except Exception:                                   # noqa: BLE001
            logger.exception("weights registry lookup failed — plotting without sky")

    return {
        "candname": f"inj{int(inj['id'])}",
        "source": source,
        "panel_note": note,
        "event_utc": event_utc.isoformat(timespec="milliseconds"),
        "beam": int(beam),
        "local_beam": int(local_beam),
        "stream": int(stream),
        "snr": float(snr),
        "dm": float(dm),
        "width": int(width),
        "samp": int((cluster or {}).get("samp") or 0),
        "n_members": int((cluster or {}).get("n_members") or len(members)),
        "n_beams": int((cluster or {}).get("n_beams") or 1),
        "sky": sky,
        "pointings": pointings,
        "trigger_reason": "injection_replay",
        "injection": {k: inj.get(k) for k in
                      ("id", "inject_utc", "stream", "beam", "dm", "amp", "sigma_ms",
                       "est_snr", "file_id", "matched_cluster", "rec_snr", "rec_dm")},
        "context": {"window_s": window_s, "members": members},
    }


# ------------------------------------------------------------------ CLI

def _parse_utc(text: str) -> datetime:
    try:
        t = datetime.fromisoformat(text)
    except ValueError:
        t = timing.parse_dada_utc(text)
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--inject-id", type=int, required=True, help="row id in the injections ledger")
    p.add_argument("--dump", required=True, help=".dada file, or a directory of them")
    p.add_argument("--out", required=True, help="output PNG")
    p.add_argument("--layout", choices=["v2", "legacy"], default="v2")
    p.add_argument("--no-pulse", action="store_true",
                   help="render the dump untouched, for comparison")
    p.add_argument("--card-json", help="also write the synthetic card here")
    p.add_argument("--fil", help="also write the beam (+pulse) as a float32 single-beam .fil")
    p.add_argument("--db", default=DEFAULT_DB, help="T2 sqlite (opened read-only)")
    p.add_argument("--cands-dir", default=DEFAULT_CANDS_DIR,
                   help="hella cands directory for the T1 context panel")
    p.add_argument("--event-utc", help="arrival time at the top of the band; "
                                       "default: the matched cluster's event_utc")
    p.add_argument("--event-offset-s", type=float,
                   help="instead of --event-utc, place the pulse this many seconds "
                        "after the first sample of the dump")
    p.add_argument("--local-beam", type=int, help="override the beam index within the dump")
    p.add_argument("--dm", type=float, help="override the injected DM")
    p.add_argument("--amp", type=float, help="override the injected amplitude (counts)")
    p.add_argument("--sigma-ms", type=float, help="override the injected pulse sigma")
    p.add_argument("--context-window-s", type=float, default=DEFAULT_CONTEXT_WINDOW_S)
    p.add_argument("--no-registry", action="store_true",
                   help="skip the weights-registry sky lookup")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    inj, cluster = read_injection(args.db, args.inject_id)
    if cluster is None:
        logger.warning("injection %d has no matched cluster — using ledger values",
                       args.inject_id)

    dm = args.dm if args.dm is not None else float(inj.get("dm"))
    amp = args.amp if args.amp is not None else float(inj.get("amp"))
    sigma_ms = args.sigma_ms if args.sigma_ms is not None else float(inj.get("sigma_ms"))

    event_utc = None
    if args.event_utc:
        event_utc = _parse_utc(args.event_utc)
    elif args.event_offset_s is None and cluster and cluster.get("event_utc"):
        event_utc = _parse_utc(cluster["event_utc"])

    files = select_dump_files(Path(args.dump), event_utc)
    beam = int((cluster or {}).get("beam") or inj.get("beam") or 0)
    local_beam = (args.local_beam if args.local_beam is not None
                  else t2_beams.local_beam(beam))
    header, data = dump_reader.read_beams(files, [local_beam])
    beam2d = data[0]
    nchan, ntime = beam2d.shape

    if event_utc is None:
        if args.event_offset_s is None:
            raise SystemExit("no matched cluster: give --event-utc or --event-offset-s")
        event_utc = header.t0 + timedelta(seconds=args.event_offset_s)
    t_rel = (event_utc - header.t0).total_seconds()
    if not 0 <= t_rel <= ntime * header.tsamp_s:
        raise SystemExit(f"event at {t_rel:.2f}s falls outside the "
                         f"{ntime * header.tsamp_s:.2f}s dump — wrong dump?")
    c0 = int(round(t_rel / header.tsamp_s))

    width = int((cluster or {}).get("width") if cluster and cluster.get("width") is not None
                else _width_index(sigma_ms, header.tsamp_s))
    plot_dm = float((cluster or {}).get("dm") or dm)

    if not args.no_pulse:
        # The generator's own frequency grid, so the per-channel integer
        # delays are the ones that were actually injected.
        freqs = (injection.generator_freqs_mhz(nchan)
                 if nchan == injection.GEN_NCHAN else header.freqs_mhz)
        pulse = injection.injected_pulse(nchan, ntime, dm, amp, sigma_ms, c0,
                                         tsamp_s=header.tsamp_s, freqs_mhz=freqs)
        lit = int((pulse.max(axis=1) > 0).sum())
        logger.info("added pulse: DM %.2f amp %g sigma %.1f ms at sample %d "
                    "(peak %.0f counts, %d channels lit)", dm, amp, sigma_ms, c0,
                    pulse.max(), lit)
        if lit < nchan:
            sweep_s = timing.dispersion_sweep_s(dm)
            logger.warning("only %d of %d channels fit: the DM %.0f sweep is %.2f s "
                           "but the event sits %.2f s into a %.2f s dump — the replayed "
                           "S/N is a lower limit", lit, nchan, dm, sweep_s, t_rel,
                           ntime * header.tsamp_s)
        beam2d = beam2d + pulse

    stream = int(t2_beams.stream_for_beam(beam) if cluster
                 else (inj.get("stream") or t2_beams.stream_for_beam(beam)))
    members = []
    if cluster and cluster.get("obs_utc_start"):
        members = cands_members(Path(args.cands_dir), cluster["obs_utc_start"], stream,
                                event_utc, args.context_window_s)

    registry = None if args.no_registry else weights_registry.default_registry()
    snr = float((cluster or {}).get("snr") or inj.get("rec_snr") or inj.get("est_snr") or 0.0)
    card = build_card(inj, cluster, event_utc, beam, local_beam, stream, plot_dm,
                      width, snr, members, args.context_window_s, registry)

    if args.fil:
        try:
            event_archive.write_event_fil(beam2d, header, card, Path(args.fil))
        except (OSError, ValueError) as exc:
            logger.exception("could not write %s: %s", args.fil, exc)

    plot_card = dict(card, source="injection replay")
    png = plotting.make_candidate_figure(beam2d, header.freqs_mhz, header.tsamp_s,
                                         t_rel, plot_card, Path(args.out), layout=args.layout)

    measured = measure_snr(beam2d, header.freqs_mhz, header.tsamp_s, plot_dm, width, t_rel)
    card["replay"] = {"dump_files": [str(f) for f in files], "n_samples": int(ntime),
                      "pulse_added": not args.no_pulse, "pulse_sample": c0,
                      "measured_boxcar_snr": round(measured, 2), "plot": str(png)}
    if args.card_json:
        Path(args.card_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.card_json).write_text(json.dumps(card, indent=2))

    print(png)
    print(f"measured boxcar S/N (DM {plot_dm:.2f}, w = {2 ** width}) = {measured:.1f}")


def measure_snr(data2d: np.ndarray, freqs_mhz: np.ndarray, tsamp_s: float,
                dm: float, width: int, t_rel_event_s: float,
                search_s: float = 0.5) -> float:
    """Peak boxcar S/N near the event, the way the profile panel measures it."""
    prof = single_pulse.profile_snr(
        single_pulse.dedisperse(single_pulse.normalise(data2d), dm, freqs_mhz, tsamp_s
                                ).mean(axis=0), 2 ** int(width))
    t = np.arange(prof.size) * tsamp_s - t_rel_event_s
    near = np.abs(t) <= search_s
    return float(prof[near].max() if near.any() else prof.max())


if __name__ == "__main__":
    sys.exit(main())
