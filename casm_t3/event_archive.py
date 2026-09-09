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
"""

from __future__ import annotations

import logging
import shutil
from datetime import timezone
from pathlib import Path

import numpy as np
from casm_io.filterbank import write_filterbank

logger = logging.getLogger(__name__)

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
