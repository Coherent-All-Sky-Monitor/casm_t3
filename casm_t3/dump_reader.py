"""Reader for casm_cand_dump beam intensity dumps.

A dump is one or more PSRDADA files: a 4096-byte ASCII header followed by
raw ring-buffer content from the post-corner-turn Beam block. The data are
little-endian float16 frames ordered (outer_time, beam, channel, inner_time)
with 64 beams, 3072 channels (descending frequency) and 64 inner samples
per frame at the native 1.048576 ms sampling.

Where multiple .dada files belong to one dump (dada_dbdisk rolls files),
pass them all; they are ordered by each header's OBS_OFFSET (byte offsets in
the file names are not zero-padded, so a name sort splices them wrongly) and
concatenated. The returned header carries the absolute sample index of the
first sample and whether the files butt together exactly, which is what
aligns a dump with hella's per-gulp flag records.
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
    # Absolute sample of the first sample from UTC_START: OBS_OFFSET / bytes_per_sample.
    # None when OBS_OFFSET is absent or not a whole number of samples.
    sample0: int | None = None
    bytes_per_sample: int = 0        # nbeam * nchan * nbit / 8, bytes
    # True when every file starts where the previous one ends (always for one file).
    contiguous: bool = True
    contiguity_note: str = ""        # why not, when contiguous is False

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

    bytes_per_sample = nbeam * nchan * nbit // 8
    sample0 = None
    if "OBS_OFFSET" in raw:
        quot, rem = divmod(int(raw["OBS_OFFSET"]), bytes_per_sample)
        sample0 = quot if rem == 0 else None

    return DumpHeader(
        nchan=nchan, nbeam=nbeam, ninner=ninner, nbit=nbit,
        tsamp_s=tsamp_us * 1e-6,
        freq_top_mhz=freq_top, chan_width_mhz=chan_width,
        t0=t0, hdr_size=int(raw.get("HDR_SIZE", DEFAULT_HDR_SIZE)), raw=raw,
        sample0=sample0, bytes_per_sample=bytes_per_sample,
    )


def read_header(path: str | Path) -> DumpHeader:
    with open(path, "rb") as f:
        return parse_header(f.read(DEFAULT_HDR_SIZE).decode(errors="replace"))


def obs_offset(header: DumpHeader) -> int | None:
    """OBS_OFFSET of a file, bytes into the observation, or None when absent."""
    try:
        return int(header.raw["OBS_OFFSET"])
    except (KeyError, ValueError):
        return None


def check_contiguity(files: list[tuple[Path, DumpHeader]]) -> str | None:
    """Why ``files`` (in the order given) are not one unbroken run, or None.

    Each file must start at the previous file's OBS_OFFSET plus its data bytes,
    belong to the same observation, and the previous file must end on a whole
    frame (a trailing partial frame is dropped on read and would shift every
    later sample). A gap or overlap anywhere makes sample indices past it wrong.
    """
    for path, hdr in files:
        if obs_offset(hdr) is None:
            return f"{path.name} has no OBS_OFFSET"
    for (pa, ha), (pb, hb) in zip(files, files[1:]):
        if ha.raw.get("UTC_START") != hb.raw.get("UTC_START"):
            return f"{pa.name} and {pb.name} are from different observations"
        frame_bytes = ha.nbeam * ha.nchan * ha.ninner * ha.nbit // 8
        data_bytes = max(0, pa.stat().st_size - ha.hdr_size)
        if data_bytes % frame_bytes:
            return f"{pa.name} ends mid-frame ({data_bytes} data bytes)"
        want = obs_offset(ha) + data_bytes
        if obs_offset(hb) != want:
            return (f"{pb.name} starts at byte {obs_offset(hb)}, expected {want} "
                    f"after {pa.name}")
    return None


def read_beams(paths: list[str | Path], beams: list[int],
               sort_by_name: bool = True) -> tuple[DumpHeader, np.ndarray]:
    """Read selected local beams (0-63) from a dump.

    Returns the header of the first file and an array of shape
    (len(beams), nchan, ntime) in float32, channels in descending
    frequency order. The header's ``sample0`` is the absolute sample index of
    the first sample and ``contiguous`` says whether the files butt together
    (see check_contiguity); a non-contiguous dump is still read, in order.

    ``sort_by_name`` (historical name) True orders the files by the OBS_OFFSET
    in each header, numerically; the file names are not zero-padded, so a name
    sort puts 10000000 before 9000000. Files without OBS_OFFSET fall back to
    name order. False keeps the caller's order.
    """
    paths = [Path(p) for p in paths]
    pairs = [(p, read_header(p)) for p in paths]
    if sort_by_name:
        if all(obs_offset(h) is not None for _, h in pairs):
            pairs.sort(key=lambda ph: (obs_offset(ph[1]), ph[0].name))
        else:
            pairs.sort(key=lambda ph: ph[0].name)
    paths = [p for p, _ in pairs]
    header = pairs[0][1]
    header.contiguity_note = check_contiguity(pairs) or ""
    header.contiguous = not header.contiguity_note
    if header.contiguous and header.sample0 is None:
        header.contiguity_note = "OBS_OFFSET is not a whole number of samples"
        header.contiguous = False
    if not header.contiguous:
        logger.warning("dump is not contiguous: %s", header.contiguity_note)
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
    logger.info("read %s: beams=%s shape=%s t0=%s sample0=%s", Path(paths[0]).name, beams,
                out.shape, header.t0.isoformat(timespec="milliseconds"), header.sample0)
    return header, out
