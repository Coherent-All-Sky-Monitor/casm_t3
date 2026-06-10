"""Reader for casm_cand_dump beam intensity dumps.

A dump is one or more PSRDADA files: a 4096-byte ASCII header followed by
raw ring-buffer content from the post-corner-turn Beam block. The data are
little-endian float16 frames ordered (outer_time, beam, channel, inner_time)
with 64 beams, 3072 channels (descending frequency) and 64 inner samples
per frame at the native 1.048576 ms sampling.

Where multiple .dada files belong to one dump (dada_dbdisk rolls files),
pass them all; they are concatenated in filename order.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from casm_t2 import timing

logger = logging.getLogger(__name__)

DEFAULT_HDR_SIZE = 4096


@dataclass(slots=True)
class DumpHeader:
    """Relevant subset of the PSRDADA header, plus the raw key/value map."""

    nchan: int
    nbeam: int
    ninner: int
    nbit: int
    tsamp_s: float
    freq_top_mhz: float
    chan_width_mhz: float
    t0: datetime              # UTC of the first sample in this file
    hdr_size: int
    raw: dict[str, str] = field(default_factory=dict)

    @property
    def freqs_mhz(self) -> np.ndarray:
        """Channel centre frequencies, descending (native order)."""
        return self.freq_top_mhz - np.arange(self.nchan) * self.chan_width_mhz


def parse_header(text: str) -> DumpHeader:
    """Parse the ASCII header of a dump file."""
    raw: dict[str, str] = {}
    for line in text.split("\n"):
        parts = line.split(None, 1)
        if len(parts) == 2:
            raw[parts[0]] = parts[1].strip()

    nchan = int(raw.get("NCHAN", timing.NCHAN))
    nbeam = int(raw.get("NBEAM", 64))
    nbit = int(raw.get("NBIT", 16))
    tsamp_us = float(raw.get("TSAMP", timing.TSAMP_S * 1e6))

    if "RESOLUTION" in raw:
        ninner = int(raw["RESOLUTION"]) // (nbeam * nchan * nbit // 8)
    else:
        ninner = 64

    # Frequency axis: prefer explicit start/width keys, fall back to band constants.
    if "FREQ_START" in raw and "CHANBW" in raw:
        freq_top = float(raw["FREQ_START"])
        chan_width = abs(float(raw["CHANBW"]))
    elif "FREQ" in raw and "BW" in raw:
        bw = abs(float(raw["BW"]))
        freq_top = float(raw["FREQ"]) + bw / 2.0
        chan_width = bw / nchan
    else:
        freq_top, chan_width = timing.F_TOP_MHZ, timing.CHAN_WIDTH_MHZ

    # First-sample time: UTC_START (+PICOSECONDS) plus the byte offset of
    # this file into the observation, which is exact; DUMP_UTC_START is the
    # requested window and only used as a fallback.
    t0 = None
    if "UTC_START" in raw:
        t0 = timing.parse_dada_utc(raw["UTC_START"])
        t0 += timedelta(seconds=float(raw.get("PICOSECONDS", 0)) * 1e-12)
        if "OBS_OFFSET" in raw and "BYTES_PER_SECOND" in raw:
            t0 += timedelta(seconds=int(raw["OBS_OFFSET"]) / float(raw["BYTES_PER_SECOND"]))
        elif "DUMP_UTC_START" in raw:
            t0 = timing.parse_dada_utc(raw["DUMP_UTC_START"])
    elif "DUMP_UTC_START" in raw:
        t0 = timing.parse_dada_utc(raw["DUMP_UTC_START"])
    if t0 is None:
        raise ValueError("dump header has no usable start time")

    return DumpHeader(
        nchan=nchan, nbeam=nbeam, ninner=ninner, nbit=nbit,
        tsamp_s=tsamp_us * 1e-6,
        freq_top_mhz=freq_top, chan_width_mhz=chan_width,
        t0=t0, hdr_size=int(raw.get("HDR_SIZE", DEFAULT_HDR_SIZE)), raw=raw,
    )


def read_header(path: str | Path) -> DumpHeader:
    with open(path, "rb") as f:
        return parse_header(f.read(DEFAULT_HDR_SIZE).decode(errors="replace"))


def read_beams(paths: list[str | Path], beams: list[int]) -> tuple[DumpHeader, np.ndarray]:
    """Read selected local beams (0-63) from a dump.

    Returns the header of the first file and an array of shape
    (len(beams), nchan, ntime) in float32, channels in descending
    frequency order.
    """
    paths = sorted(paths, key=lambda p: Path(p).name)
    header = read_header(paths[0])
    dtype = np.float16 if header.nbit == 16 else np.float32
    frame_elems = header.nbeam * header.nchan * header.ninner

    per_file = []
    for path in paths:
        data = np.fromfile(path, dtype=dtype, offset=header.hdr_size)
        nframe = data.size // frame_elems
        if nframe == 0:
            logger.warning("%s holds less than one frame, skipping", path)
            continue
        data = data[:nframe * frame_elems].reshape(
            nframe, header.nbeam, header.nchan, header.ninner)
        # (frame, beam, chan, inner) -> (beam, chan, frame*inner)
        sel = data[:, beams, :, :].astype(np.float32)
        per_file.append(sel.transpose(1, 2, 0, 3).reshape(len(beams), header.nchan, -1))

    if not per_file:
        raise ValueError(f"no complete frames in {paths}")
    out = np.concatenate(per_file, axis=2)
    logger.info("read %s: beams=%s shape=%s t0=%s", Path(paths[0]).name, beams,
                out.shape, header.t0.isoformat(timespec="milliseconds"))
    return header, out
