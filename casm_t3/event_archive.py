"""Per-event archive: the small artifacts plus the detection beam as a .fil.

The bulk .dada dump holds all 64 beams and is deleted once the figure is
made. For the events that matter later — folding a known source, re-running
a search with a different DM step, handing a candidate to someone else —
what is actually needed is the one beam the detection was in. This module
writes that beam as a single-beam float32 SIGPROC filterbank (about 55 MB
for a typical dump) and gathers it with the PNG and result JSON into

    <events-root>/<candname>/{<candname>.png, <candname>.json, <candname>.fil}

on the node that made the dump. t3-collect pulls corr2's tree to corr1 and
drops the same ``.slack`` marker there after posting, so the two nodes'
events end up in one place.

The header recipe follows casm_io's beam_dump converter, including the
rawdatafile length limit: filtool crashes when rawdatafile is 80 characters
or longer, so only the basename is stored.

The schema-2 folder (casm_io docs/t3_events.md) adds the raw figure, the
hella mask, three CSVs, a summary JSON and a manifest written last. The helpers
below build the pieces the summary needs (provenance, dump span, figure
labels) and rebuild a trigger card from a stored folder for t3-replot.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
import subprocess
from datetime import timedelta, timezone
from pathlib import Path

import numpy as np
from casm_io.filterbank import write_filterbank

import casm_t3
from casm_t2 import timing

logger = logging.getLogger(__name__)

# Timeout per git call behind the t3_commit provenance key, s.
GIT_TIMEOUT_S = 5.0

# Clock behind every UTC label in an event folder (the fil's tstart, event_utc).
UTC_CLOCK = "TCS UTC_START from casm-corr1 system clock"

UNAVAILABLE_LABEL = "hella RFI flags unavailable"
SUB_INCOH_KEY = "BFCORR_SUB_INCOH"       # dump header key: IB subtraction on/off

# filtool crashes at rawdatafile >= 80 chars; keep a margin.
MAX_RAWDATAFILE = 78

# As used by casm_io/filterbank/beam_dump.py for CASM beam dumps.
TELESCOPE_ID = 0
MACHINE_ID = 0


def _tstart_mjd(t0) -> float:
    """MJD of a (tz-aware, UTC) datetime."""
    from astropy.time import Time

    if t0.tzinfo is not None:
        t0 = t0.astimezone(timezone.utc).replace(tzinfo=None)
    return float(Time(t0, scale="utc").mjd)


def fil_header_from_dump(header, card: dict, fil_path: Path) -> dict:
    """Build the SIGPROC header for one beam of a dump.

    ``header`` is a dump_reader.DumpHeader; ``card`` the T2 trigger card
    (``beam`` is the global beam number, ``local_beam`` the index within
    this node's dump).
    """
    freqs = header.freqs_mhz
    rawdatafile = Path(fil_path).name
    if len(rawdatafile) > MAX_RAWDATAFILE:
        raise ValueError(f"rawdatafile {rawdatafile!r} is {len(rawdatafile)} chars; "
                         f"filtool crashes at >= 80 — shorten the candidate name")
    return {
        "source_name": str(card.get("candname", "unknown")),
        "rawdatafile": rawdatafile,
        "telescope_id": TELESCOPE_ID,
        "machine_id": MACHINE_ID,
        "data_type": 1,
        "fch1": float(freqs[0]),
        "foff": float(freqs[1] - freqs[0]),
        "nchans": int(len(freqs)),
        "nbeams": 1,
        "ibeam": int(card.get("beam", card.get("local_beam", 0))),
        "nbits": 32,
        "tstart": _tstart_mjd(header.t0),
        "tsamp": float(header.tsamp_s),
        "nifs": 1,
    }


def write_event_fil(data2d: np.ndarray, header, card: dict, fil_path: Path) -> Path:
    """Write one beam, shape (nchan, ntime), as a float32 single-beam .fil."""
    fil_path = Path(fil_path)
    fil_hdr = fil_header_from_dump(header, card, fil_path)
    # SIGPROC is time-major: (ntime, nchan), C-contiguous.
    out = np.ascontiguousarray(np.asarray(data2d, dtype=np.float32).T)
    fil_path.parent.mkdir(parents=True, exist_ok=True)
    write_filterbank(str(fil_path), out, fil_hdr, nbits=32, backend="standalone")
    logger.info("wrote %s (%d chan x %d samp, %.0f MB)", fil_path,
                out.shape[1], out.shape[0], fil_path.stat().st_size / 1e6)
    return fil_path


def archive_event(events_root: Path, candname: str, files: list[Path]) -> Path:
    """Copy the given files into <events_root>/<candname>/ and return that dir.

    Files already inside the event directory (the .fil, which is written
    there directly to avoid copying ~55 MB) are left alone.
    """
    event_dir = Path(events_root) / candname
    event_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        f = Path(f)
        if f.parent.resolve() == event_dir.resolve():
            continue
        shutil.copy2(f, event_dir / f.name)
    return event_dir


# --------------------------------------------------------------------------
# schema-2 summary pieces
# --------------------------------------------------------------------------

def source_version(repo: str | Path | None = None) -> str:
    """Git commit of the casm_t3 checkout this process imports from.

    "<sha>" (12 hex digits) for a tree with no tracked changes, "<sha>+dirty"
    when tracked files differ from HEAD, "unknown" when git cannot answer.
    Never raises.
    """
    try:
        repo = str(repo or Path(casm_t3.__file__).resolve().parent.parent)
        head = subprocess.run(["git", "-C", repo, "rev-parse", "--short=12", "HEAD"],
                              capture_output=True, text=True, timeout=GIT_TIMEOUT_S)
        if head.returncode != 0 or not head.stdout.strip():
            return "unknown"
        status = subprocess.run(
            ["git", "-C", repo, "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, timeout=GIT_TIMEOUT_S)
        if status.returncode != 0:
            return "unknown"
    except Exception:  # provenance only: a failed lookup must not stop the daemon
        return "unknown"
    sha = head.stdout.strip()
    return f"{sha}+dirty" if status.stdout.strip() else sha


def sub_incoh_from_header(raw: dict) -> tuple[bool | None, str]:
    """(IB subtraction state, source) from a dump header's key/value map.

    BFCORR_SUB_INCOH "1"/"true" is on, "0"/"false" off, source "dump_header".
    An absent or unreadable key is (None, "unknown"): the state is never assumed.
    """
    value = str((raw or {}).get(SUB_INCOH_KEY, "")).strip().lower()
    if value in ("1", "true"):
        return True, "dump_header"
    if value in ("0", "false"):
        return False, "dump_header"
    return None, "unknown"


def sub_state_text(sub_incoh: bool | None) -> str:
    """"on", "off" or "unknown" for a figure label."""
    return "unknown" if sub_incoh is None else ("on" if sub_incoh else "off")


def raw_label(sub_incoh: bool | None) -> str:
    """Title note of the raw figure, which states the IB subtraction state."""
    return f"raw intensity, before hella RFI flagging; IB subtraction {sub_state_text(sub_incoh)}"


def unavailable_label(reason: str | None = None) -> str:
    """Title note of a figure drawn without hella's flags."""
    return f"{UNAVAILABLE_LABEL} ({reason})" if reason else UNAVAILABLE_LABEL


def hella_provenance(win) -> dict:
    """Binary md5, cfg sha256 and flag settings of the first record header of ``win``.

    Empty when there is no window or no header (the summary then holds nulls).
    """
    headers = getattr(win, "headers", None) or []
    if not headers:
        return {}
    h = headers[0]
    return {"hella_binary_md5": h.binary_md5, "hella_cfg_sha256": h.cfg_sha256,
            "hella_flag": {"flag1": h.flag1, "flag2": h.flag2, "st_mmin": h.st_mmin,
                           "st_nbin": h.st_nbin, "st_mmax": h.st_mmax, "st_nb": h.st_nb,
                           "st_chi1": h.st_chi1, "st_rho": h.st_rho,
                           "blank_cnt": h.blank_cnt}}


def dump_span(header, nsamp: int, host: str) -> dict:
    """The summary's ``dump`` entry for ``nsamp`` samples read from ``header``'s dump."""
    t0 = header.t0
    t1 = t0 + timedelta(seconds=nsamp * header.tsamp_s)
    return {"utc_start": t0.isoformat(timespec="milliseconds"),
            "utc_stop": t1.isoformat(timespec="milliseconds"),
            "sample0": header.sample0, "nsamp": int(nsamp), "host": host}


# --------------------------------------------------------------------------
# a card from a stored folder
# --------------------------------------------------------------------------

def _num(x):
    """float, or None for NaN (an empty CSV cell)."""
    x = float(x)
    return None if math.isnan(x) else x


def card_from_event(ev) -> dict:
    """The trigger card the figure needs, from a casm_io ``T3Event``.

    Schema 1: the card is the folder's JSON. Schema 2: the card is rebuilt from
    the summary and the three CSVs (the ``gulp`` block, ``context.members`` and
    ``pointings`` that the summary no longer carries). A card with no
    ``trigger.gulp`` gets no ``gulp`` block, as before.
    """
    if ev.schema == 1:
        return json.loads((ev.path / f"{ev.candname}.json").read_text())
    s = ev.summary
    prov, trig = s.get("provenance") or {}, s.get("trigger") or {}
    keys = ("candname", "event_utc", "snr", "dm", "width", "beam", "local_beam", "stream",
            "samp", "n_beams", "n_members", "sky_extent_deg", "trigger_reason", "source", "sky")
    card = {k: s[k] for k in keys if s.get(k) is not None}
    if len(ev.beams):
        card["pointings"] = {
            "weights_id": prov.get("weights_id"),
            "alt_deg": ev.beams["alt_deg"].tolist(), "az_deg": ev.beams["az_deg"].tolist(),
            "beam_fwhm_x_deg": prov.get("beam_fwhm_x_deg"),
            "beam_fwhm_y_deg": prov.get("beam_fwhm_y_deg")}
    t = ev.trials
    members = t[t["origin"].isin(["context", "both"])]
    card["context"] = {
        "window_s": trig.get("context_window_s"),
        "members": [[float(r.dt_s), int(r.beam), float(r.dm), float(r.snr), int(r.width_idx)]
                    for r in members.itertuples()]}
    if trig.get("gulp") is not None:
        from casm_io.candidates.t3_event import cluster_beams
        in_gulp = t[t["in_trigger_gulp"] == 1]
        clusters = []
        for r in ev.clusters.itertuples():
            clusters.append({
                "id": int(r.id),
                "peak": [float(r.peak_dt_s), int(r.peak_beam), float(r.peak_dm),
                         float(r.peak_snr), int(r.peak_width_idx)],
                "n_trials": int(r.n_trials),
                "beams": [[b, v] for b, v in cluster_beams(r.beams)],
                "sky_extent_deg": _num(r.sky_extent_deg), "dm_lo": _num(r.dm_lo),
                "dm_hi": _num(r.dm_hi), "outcome": r.outcome, "detail": r.detail})
        card["gulp"] = {
            "gulp": trig["gulp"], "utc_start": prov.get("obs_utc_start"),
            "samp_lo": trig.get("samp_lo"), "samp_hi": trig.get("samp_hi"),
            "tsamp_s": timing.TSAMP_S, "weights_id": prov.get("weights_id"),
            "n_raw": trig.get("n_raw"), "n_width_dropped": trig.get("n_width_dropped"),
            "n_shed": trig.get("n_shed"), "n_trials_total": trig.get("n_trials_total"),
            "trials_truncated": trig.get("trials_truncated"),
            "n_clusters_total": trig.get("n_clusters_total"),
            "clusters_truncated": trig.get("clusters_truncated"),
            "trials": [[float(r.dt_s), int(r.beam), float(r.dm), float(r.snr),
                        int(r.width_idx), int(r.cluster_id)] for r in in_gulp.itertuples()],
            "clusters": clusters, "rules": s.get("rules"), "storm": s.get("storm")}
    return card
