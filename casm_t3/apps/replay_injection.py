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
import copy
import json
import logging
import sqlite3
import sys
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
# t2d's injection gulp blocks (casm_t2 docs/CARD_SCHEMA.md, "Injection gulp
# blocks"): the v3 bottom row of the replay figure.
DEFAULT_GULP_DIR = "/mnt/nvme5/casm_pipeline/injection_gulps"


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


def select_dump_files(dump, event_utc: datetime | None,
                      sweep_s: float = 0.0, pad_s: float = 2.0) -> list[Path]:
    """The .dada file(s) to read: files as given, or the ones in a directory
    whose time span overlaps [event, event + sweep].

    ``dump`` is one path or several (list, or comma-separated string). The
    dump daemon rolls a requested window into consecutive files, so a sweep
    that starts near the end of one file finishes in the next: selecting only
    the file holding the top-of-band arrival cuts the pulse at the seam.
    """
    if isinstance(dump, (str, Path)):
        dumps = [Path(d) for d in str(dump).split(",") if d]
    else:
        dumps = [Path(d) for d in dump]

    given, dirs = [], []
    for d in dumps:
        (dirs if d.is_dir() else given).append(d)
    if given and not dirs:
        return given                                  # explicit files: all of them

    lo = None if event_utc is None else event_utc - timedelta(seconds=pad_s)
    hi = None if event_utc is None else event_utc + timedelta(seconds=sweep_s + pad_s)
    hits = list(given)
    for d in dirs:
        files = sorted(d.glob("*.dada"), key=lambda p: p.name)
        if not files:
            raise SystemExit(f"no .dada files in {d}")
        if event_utc is None:
            hits += files
            continue
        for p in files:
            try:
                hdr, span_s = _dump_span(p)
            except (OSError, ValueError) as exc:
                logger.warning("skipping %s: %s", p.name, exc)
                continue
            if hdr.t0 <= hi and hdr.t0 + timedelta(seconds=span_s) >= lo:
                hits.append(p)
    if not hits:
        window = "" if event_utc is None else f" covers {event_utc.isoformat()} +{sweep_s:.1f}s"
        raise SystemExit(f"no .dada in {', '.join(str(d) for d in dumps)}{window}")
    return hits


def order_dump_files(files: list[Path]) -> list[Path]:
    """Sort dump files by header start time and check they butt together.

    A gap or an overlap larger than one sample would shift every sample past
    the seam, so it is an error rather than a silent splice.
    """
    spans = []
    for p in files:
        hdr, span_s = _dump_span(p)
        spans.append((hdr.t0, span_s, hdr.tsamp_s, Path(p)))
    spans.sort(key=lambda s: s[0])
    for (t0, span_s, tsamp_s, a), (t1, _, _, b) in zip(spans, spans[1:]):
        gap_s = (t1 - (t0 + timedelta(seconds=span_s))).total_seconds()
        if abs(gap_s) > tsamp_s:
            raise SystemExit(
                f"dump files are not contiguous: {a.name} ends {gap_s * 1e3:+.1f} ms "
                f"before {b.name} starts (tsamp {tsamp_s * 1e3:.3f} ms) — "
                f"refusing to splice")
    return [s[3] for s in spans]


def truncation_freq_mhz(freqs_mhz: np.ndarray, dm: float, tsamp_s: float,
                        sigma_ms: float, c0_samp: int, ntime: int) -> float | None:
    """Lowest channel frequency (MHz) whose pulse fits entirely in the data.

    None when the whole sweep fits. ``injected_pulse`` drops channels whose
    delayed arrival falls past the last sample, so this is what the figure
    must declare instead of showing a band-truncated pulse.
    """
    delays = injection.channel_delay_samples(np.asarray(freqs_mhz), dm, tsamp_s)
    halfwin = int(np.ceil(4.0 * injection.sigma_samples(sigma_ms, tsamp_s)))
    fits = (int(c0_samp) + delays + halfwin) <= ntime - 1
    if fits.all():
        return None
    return float(np.asarray(freqs_mhz)[fits].min() if fits.any()
                 else np.asarray(freqs_mhz).max())


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


# ------------------------------------------------------------ gulp block

def find_gulp_block(gulp_dir, cluster: dict | None) -> dict | None:
    """t2d's injection gulp block for the matched cluster, or None.

    Files are ``<utc_start>_g<gulp>_c<cluster_id>.json``; among this gulp's
    files the one whose ``reference`` is the cluster (by id, else by
    observation, peak sample and beam) wins. None, never an error, when there
    is no cluster, no directory or no match: the figure then draws the legacy
    bottom row.
    """
    if not gulp_dir or not cluster or cluster.get("gulp") is None:
        return None
    d = Path(gulp_dir)
    if not d.is_dir():
        return None
    cid = cluster.get("id")
    for p in sorted(d.glob(f"*_g{int(cluster['gulp'])}_*.json")):
        try:
            block = json.loads(p.read_text())
        except (OSError, ValueError) as exc:
            logger.warning("unreadable gulp block %s: %s", p, exc)
            continue
        ref = block.get("reference") or {}
        if cid is not None and ref.get("cluster_id") == cid:
            return block
        if (block.get("utc_start") == cluster.get("obs_utc_start")
                and ref.get("samp") == cluster.get("samp")
                and ref.get("beam") == cluster.get("beam")):
            return block
    return None


def replay_gulp_block(block: dict, event_utc: datetime) -> tuple[dict, int | None]:
    """The block as the replay card carries it: (block, card samp).

    A copy with the injection cluster as the figure's red group (outcome
    "triggered"; any cluster that really triggered in the gulp becomes
    "also_triggerable") and every dt_s re-referenced to ``event_utc`` when it
    differs from the injection peak by half a sample or more. The card samp is
    the event's sample, so the figure's gulp span agrees with dt_s.
    """
    g = copy.deepcopy(block)
    ref = g.get("reference") or {}
    tsamp = float(g.get("tsamp_s") or timing.TSAMP_S)
    shift_s = 0.0
    if ref.get("event_utc"):
        shift_s = (_parse_utc(ref["event_utc"]) - event_utc).total_seconds()
        if abs(shift_s) < 0.5 * tsamp:
            shift_s = 0.0
    if shift_s:
        for row in g.get("trials") or []:
            row[0] = round(row[0] + shift_s, 4)
        for cl in g.get("clusters") or []:
            cl["peak"][0] = round(cl["peak"][0] + shift_s, 4)
    for cl in g.get("clusters") or []:
        if cl.get("id") == ref.get("id"):
            cl["outcome"] = "triggered"
        elif cl.get("outcome") == "triggered":
            cl["outcome"] = "also_triggerable"
    samp = ref.get("samp")
    if samp is not None:
        samp = int(round(int(samp) - shift_s / tsamp))
    return g, samp


# ----------------------------------------------------------------- card

def injection_summary(inj: dict, cluster: dict | None) -> str:
    """What went in and what hella made of it, in one line for the Slack thread:
    'injected DM 585.5, FWHM 11.8 ms, amp 5 counts, S/N 19.3; hella S/N 16.0 at DM 584.5'."""
    sigma_ms = float(inj.get("sigma_ms") or 0.0)
    amp = float(inj.get("amp") or 0.0)
    est = inj.get("est_snr")
    dm = inj.get("dm")
    rec_snr = (cluster or {}).get("snr", inj.get("rec_snr"))
    rec_dm = (cluster or {}).get("dm", inj.get("rec_dm"))
    text = ("injected " + (f"DM {float(dm):.1f}, " if dm is not None else "")
            + f"FWHM {injection.fwhm_ms(sigma_ms):.1f} ms, amp {amp:g} counts, S/N "
            + (f"{float(est):.1f}" if est is not None else "?"))
    if rec_snr is not None and rec_dm is not None:
        text += f"; hella S/N {float(rec_snr):.1f} at DM {float(rec_dm):.1f}"
    else:
        text += "; not recovered by hella"
    return text


def _width_index(sigma_ms: float, tsamp_s: float) -> int:
    """Boxcar index (ibox) closest to the injected pulse's FWHM."""
    n = injection.fwhm_ms(sigma_ms) / (1000.0 * tsamp_s)
    return int(max(0, min(9, round(np.log2(max(1.0, n))))))


def build_card(inj: dict, cluster: dict | None, event_utc: datetime,
               beam: int, local_beam: int, stream: int, dm: float, width: int,
               snr: float, members: list, window_s: float,
               registry: weights_registry.Registry | None) -> dict:
    """A synthetic trigger card for the replay, shaped like a t2d card.

    ``source`` is just "injection": the figure title carries no replay
    parameters (Vishnu, 2026-09-22). They are in ``injection.summary``, the
    one-line text for the Slack thread, and in the ``injection`` block.
    """
    source = "injection"

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
        "injection": dict({k: inj.get(k) for k in
                           ("id", "inject_utc", "stream", "beam", "dm", "amp", "sigma_ms",
                            "est_snr", "file_id", "matched_cluster", "rec_snr", "rec_dm")},
                          summary=injection_summary(inj, cluster)),
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
    p.add_argument("--dump", required=True,
                   help=".dada file, a directory of them, or a comma-separated list; "
                        "every file the DM sweep touches is read")
    p.add_argument("--out", required=True, help="output PNG")
    p.add_argument("--layout", choices=["v2", "legacy"], default="v2")
    p.add_argument("--subbands", type=int, default=None,
                   help="force the number of frequency rows in the waterfall and DM-time "
                        "panels (default: from the S/N, plotting.subbands_for)")
    p.add_argument("--no-pulse", action="store_true",
                   help="render the dump untouched, for comparison")
    p.add_argument("--card-json", help="also write the synthetic card here")
    p.add_argument("--fil", help="also write the beam (+pulse) as a float32 single-beam .fil")
    p.add_argument("--db", default=DEFAULT_DB, help="T2 sqlite (opened read-only)")
    p.add_argument("--cands-dir", default=DEFAULT_CANDS_DIR,
                   help="hella cands directory for the T1 context panel")
    p.add_argument("--label", help="display name for the title, e.g. "
                   "inj_20260910_0002; defaults to the ledger's file_id")
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
    p.add_argument("--gulp-dir", default=DEFAULT_GULP_DIR,
                   help="t2d's injection gulp blocks, for the v3 bottom row; "
                        "'' to draw the legacy bottom row")
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

    # The sweep decides how much data is needed: at DM 650 it is 6 s, more
    # than one dump file, and every file it touches has to be read.
    sweep_s = 0.0 if args.no_pulse else timing.dispersion_sweep_s(dm)
    files = order_dump_files(select_dump_files(args.dump, event_utc, sweep_s))
    logger.info("reading %d dump file(s): %s", len(files),
                ", ".join(f.name for f in files))
    beam = int((cluster or {}).get("beam") or inj.get("beam") or 0)
    local_beam = (args.local_beam if args.local_beam is not None
                  else t2_beams.local_beam(beam))
    header, data = dump_reader.read_beams(files, [local_beam], sort_by_name=False)
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
    # The figure dedisperses at, and line 2 of its title quotes, the ledger DM
    # (what was injected); hella's DM is in the card's injection block.
    plot_dm = float(dm)

    trunc_mhz = None
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
        trunc_mhz = truncation_freq_mhz(freqs, dm, header.tsamp_s, sigma_ms, c0, ntime)
        if trunc_mhz is not None:
            logger.warning("replay truncated below %.2f MHz: the DM %.0f sweep is "
                           "%.2f s but the event sits %.2f s into %.2f s of dump "
                           "(%d files, %d of %d channels lit) — the replayed S/N is "
                           "a lower limit", trunc_mhz, dm, sweep_s, t_rel,
                           ntime * header.tsamp_s, len(files), lit, nchan)
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
    # t2d's gulp block for the injection gulp gives the figure the v3 bottom
    # row with the injection as the red group. Without one (unrecovered shot,
    # shots before 2026-09-22) the legacy bottom row is drawn, as before.
    try:
        block = find_gulp_block(args.gulp_dir, cluster)
        if block is not None:
            card["gulp"], samp = replay_gulp_block(block, event_utc)
            if samp is not None:
                card["samp"] = samp
            logger.info("gulp block: gulp %s, %d trials", block.get("gulp"),
                        len(block.get("trials") or []))
        elif cluster:
            logger.info("no gulp block for cluster %s in %s: legacy bottom row",
                        cluster.get("id"), args.gulp_dir)
    except Exception:                                       # noqa: BLE001
        logger.exception("gulp block unusable: legacy bottom row")
        card.pop("gulp", None)

    if args.fil:
        try:
            event_archive.write_event_fil(beam2d, header, card, Path(args.fil))
        except (OSError, ValueError) as exc:
            logger.exception("could not write %s: %s", args.fil, exc)

    # The figure reads like any other event: only the name marks it as an
    # injection ("INJECTION: <id>   <UTC>"); line 2 is hella's recovered S/N
    # at the ledger DM. The injected parameters live in the card JSON and its
    # injection.summary line, never in the title (Vishnu, 2026-09-09 and
    # 2026-09-22).
    # The label a person reads: the ledger's file_id (inj_YYYYMMDD_NNNN),
    # overridable with --label, falling back to the integer row id.
    label = args.label or inj.get("file_id") or str(int(inj["id"]))
    suffix = "" if trunc_mhz is None else f" (replay truncated below {trunc_mhz:.0f} MHz)"
    plot_card = dict(card, candname=f"INJECTION: {label}{suffix}", source="blind")
    png = plotting.make_candidate_figure(beam2d, header.freqs_mhz, header.tsamp_s,
                                         t_rel, plot_card, Path(args.out), layout=args.layout,
                                         subbands=args.subbands)

    measured = measure_snr(beam2d, header.freqs_mhz, header.tsamp_s, plot_dm, width, t_rel)
    card["replay"] = {"dump_files": [str(f) for f in files], "n_samples": int(ntime),
                      "pulse_added": not args.no_pulse, "pulse_sample": c0,
                      "truncated_below_mhz": (None if trunc_mhz is None
                                              else round(trunc_mhz, 2)),
                      "measured_boxcar_snr": round(measured, 2), "plot": str(png)}
    if args.card_json:
        Path(args.card_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.card_json).write_text(json.dumps(card, indent=2))

    print(png)
    print(f"measured boxcar S/N (DM {plot_dm:.2f}, w = {2 ** width}) = {measured:.1f}")
    print(card["injection"]["summary"])


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
