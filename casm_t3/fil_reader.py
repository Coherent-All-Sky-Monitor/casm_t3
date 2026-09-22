"""Minimal SIGPROC filterbank reader for the archived detection beam.

``event_archive.write_event_fil`` writes one beam of a dump as a single-beam
float32 SIGPROC filterbank under ``<events-root>/<candname>/<candname>.fil``.
That file is all the offline replot needs: the .dada dump is deleted minutes
after the trigger, the .fil is kept.

Only what those files contain is supported: float32 (nbits 32), nifs 1, one
beam, time-major samples. Anything else raises ValueError rather than
returning a silently wrong array. Nothing here writes.
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# SIGPROC header keyword types. Every standard keyword is listed so an
# unexpected one is an error, not a misparsed byte offset for the rest of the
# header. ``signed`` is a single byte (SIGPROC header.c writes it as a char).
_DOUBLE_KEYS = {"tstart", "tsamp", "fch1", "foff", "az_start", "za_start",
                "src_raj", "src_dej", "period", "refdm"}
_INT_KEYS = {"machine_id", "telescope_id", "data_type", "nchans", "nbits",
             "nifs", "nbeams", "ibeam", "barycentric", "pulsarcentric",
             "nsamples"}
_BYTE_KEYS = {"signed"}
_STR_KEYS = {"source_name", "rawdatafile"}
_REQUIRED = {"nchans", "nbits", "tsamp", "fch1", "foff", "tstart"}

MJD_EPOCH = datetime(1858, 11, 17, tzinfo=timezone.utc)


@dataclass
class FilHeader:
    """What the figure needs out of a .fil header."""
    nchans: int
    nbits: int
    nifs: int
    tsamp_s: float
    fch1_mhz: float
    foff_mhz: float
    tstart_mjd: float
    keys: dict

    @property
    def freqs_mhz(self) -> np.ndarray:
        """Channel centre frequencies in file order (fch1 + i * foff)."""
        return self.fch1_mhz + self.foff_mhz * np.arange(self.nchans, dtype=float)

    @property
    def t0(self) -> datetime:
        """UTC of the first sample."""
        return MJD_EPOCH + timedelta(days=self.tstart_mjd)


def _read_exact(fh, n: int) -> bytes:
    raw = fh.read(n)
    if len(raw) < n:
        raise ValueError("truncated SIGPROC header")
    return raw


def _read_string(fh) -> str:
    (n,) = struct.unpack("<i", _read_exact(fh, 4))
    if not 0 <= n <= 4096:
        raise ValueError(f"implausible SIGPROC keyword length {n}: not a filterbank?")
    return _read_exact(fh, n).decode("ascii", "replace")


def read_header(path: str | Path) -> tuple[FilHeader, int]:
    """Parse the header; returns (header, byte offset of the first sample)."""
    path = Path(path)
    keys: dict = {}
    with path.open("rb") as fh:
        if _read_string(fh) != "HEADER_START":
            raise ValueError(f"{path} does not start with HEADER_START")
        while True:
            key = _read_string(fh)
            if key == "HEADER_END":
                break
            if key in _DOUBLE_KEYS:
                keys[key] = struct.unpack("<d", _read_exact(fh, 8))[0]
            elif key in _INT_KEYS:
                keys[key] = struct.unpack("<i", _read_exact(fh, 4))[0]
            elif key in _BYTE_KEYS:
                keys[key] = struct.unpack("<b", _read_exact(fh, 1))[0]
            elif key in _STR_KEYS:
                keys[key] = _read_string(fh)
            else:
                raise ValueError(f"unsupported SIGPROC keyword {key!r} in {path}")
        offset = fh.tell()
    missing = _REQUIRED - set(keys)
    if missing:
        raise ValueError(f"{path}: header is missing {sorted(missing)}")
    header = FilHeader(nchans=int(keys["nchans"]), nbits=int(keys["nbits"]),
                       nifs=int(keys.get("nifs", 1)), tsamp_s=float(keys["tsamp"]),
                       fch1_mhz=float(keys["fch1"]), foff_mhz=float(keys["foff"]),
                       tstart_mjd=float(keys["tstart"]), keys=keys)
    return header, offset


def read_fil(path: str | Path) -> tuple[np.ndarray, FilHeader]:
    """Read a single-beam float32 .fil as (nchan, ntime) float32 plus its header."""
    path = Path(path)
    header, offset = read_header(path)
    if header.nbits != 32:
        raise ValueError(f"{path}: nbits {header.nbits}, only 32-bit float is supported")
    if header.nifs != 1:
        raise ValueError(f"{path}: nifs {header.nifs}, only single-IF files are supported")
    if int(header.keys.get("nbeams", 1)) != 1:
        raise ValueError(f"{path}: nbeams {header.keys['nbeams']}, only single-beam files "
                         "are supported")
    raw = np.fromfile(path, dtype=np.float32, offset=offset)
    ntime, extra = divmod(raw.size, header.nchans)
    if extra:
        logger.warning("%s: %d trailing float32 values are not a whole sample; ignored",
                       path.name, extra)
        raw = raw[: ntime * header.nchans]
    data = raw.reshape(ntime, header.nchans).T      # SIGPROC is time-major
    logger.info("read %s (%d chan x %d samp, %.2f s)", path.name, header.nchans, ntime,
                ntime * header.tsamp_s)
    return np.ascontiguousarray(data), header
