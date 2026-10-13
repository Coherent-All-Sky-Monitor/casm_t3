"""dump_reader: files ordered by header OBS_OFFSET, contiguity, absolute sample0."""
from pathlib import Path

import numpy as np
import pytest

from casm_t3 import dump_reader

NCHAN, NBEAM, NINNER = 8, 2, 4
BPS = NBEAM * NCHAN * 2                      # bytes per sample, fp16
FRAME_BYTES = BPS * NINNER
TSAMP_S = 1.048576e-3
UTC = "2026-10-13-18:22:38"


def _write(path: Path, obs_offset, nframe: int, fill: float, utc: str = UTC, extra=()) -> Path:
    keys = [("HDR_SIZE", 4096), ("UTC_START", utc), ("PICOSECONDS", 0),
            ("BYTES_PER_SECOND", f"{BPS / TSAMP_S:.1f}"), ("NCHAN", NCHAN), ("NBEAM", NBEAM),
            ("NBIT", 16), ("TSAMP", f"{TSAMP_S * 1e6:.6f}"), ("RESOLUTION", FRAME_BYTES),
            ("FREQ_START", 484.375), ("CHANBW", -0.03)]
    if obs_offset is not None:
        keys.append(("OBS_OFFSET", obs_offset))
    keys.extend(extra)
    text = "\n".join(f"{k} {v}" for k, v in keys) + "\n"
    body = np.full((nframe, NBEAM, NCHAN, NINNER), fill, dtype=np.float16)
    with open(path, "wb") as f:
        f.write(text.encode().ljust(4096, b"\0"))
        body.tofile(f)
    return path


def _pair(tmp_path, off_a=10**10 - FRAME_BYTES, nframe=3, fills=(1.0, 2.0), gap_frames=0):
    """Two files whose names sort the wrong way: offset a has 10 digits, b has 11."""
    off_b = off_a + nframe * FRAME_BYTES + gap_frames * FRAME_BYTES
    assert len(str(off_a)) == 10 and len(str(off_b)) == 11
    a = _write(tmp_path / f"{UTC}_{off_a}.000000.dada", off_a, nframe, fills[0])
    b = _write(tmp_path / f"{UTC}_{off_b}.000000.dada", off_b, nframe, fills[1])
    assert sorted([a, b], key=lambda p: p.name) == [b, a]      # the name sort is wrong
    return a, b, off_a


def test_unpadded_names_are_ordered_by_obs_offset(tmp_path):
    a, b, off_a = _pair(tmp_path)
    for given in ([a, b], [b, a]):
        header, data = dump_reader.read_beams(given, [0])
        assert header.contiguous and header.contiguity_note == ""
        ntime = 3 * NINNER
        assert data.shape == (1, NCHAN, 2 * ntime)
        assert np.all(data[0, :, :ntime] == 1.0) and np.all(data[0, :, ntime:] == 2.0)
        assert header.t0 == dump_reader.read_header(a).t0


def test_sample0_is_obs_offset_over_bytes_per_sample(tmp_path):
    a, b, off_a = _pair(tmp_path)
    header, _ = dump_reader.read_beams([b, a], [0])
    assert header.bytes_per_sample == BPS
    assert header.sample0 == off_a // BPS
    assert dump_reader.read_header(b).sample0 == (off_a + 3 * FRAME_BYTES) // BPS


def test_gap_between_files_is_detected(tmp_path, caplog):
    a, b, _ = _pair(tmp_path, gap_frames=1)
    with caplog.at_level("WARNING"):
        header, data = dump_reader.read_beams([a, b], [0])
    assert header.contiguous is False
    assert b.name in header.contiguity_note and "expected" in header.contiguity_note
    assert "not contiguous" in caplog.text
    assert data.shape[2] == 6 * NINNER                  # still read, in order
    assert header.sample0 is not None


def test_overlap_between_files_is_detected(tmp_path):
    a = _write(tmp_path / "a.dada", 3200, 3, 1.0)
    b = _write(tmp_path / "b.dada", 3200 + 2 * FRAME_BYTES, 3, 2.0)       # starts inside a
    header, _ = dump_reader.read_beams([a, b], [0])
    assert header.contiguous is False


def test_different_observations_are_not_contiguous(tmp_path):
    a = _write(tmp_path / "a.dada", 3200, 3, 1.0)
    b = _write(tmp_path / "b.dada", 3200 + 3 * FRAME_BYTES, 3, 2.0, utc="2026-10-13-19:00:00")
    header, _ = dump_reader.read_beams([a, b], [0])
    assert header.contiguous is False and "different observations" in header.contiguity_note


def test_partial_frame_at_a_seam_is_not_contiguous(tmp_path):
    a = _write(tmp_path / "a.dada", 3200, 3, 1.0)
    with open(a, "ab") as f:
        f.write(b"\0" * (FRAME_BYTES // 2))             # a torn trailing frame
    b = _write(tmp_path / "b.dada", 3200 + 3 * FRAME_BYTES + FRAME_BYTES // 2, 3, 2.0)
    header, _ = dump_reader.read_beams([a, b], [0])
    assert header.contiguous is False and "mid-frame" in header.contiguity_note


def test_single_file_is_contiguous_and_aligned(tmp_path):
    a = _write(tmp_path / "a.dada", 3200, 3, 1.0)
    header, data = dump_reader.read_beams([a], [0])
    assert header.contiguous and header.sample0 == 3200 // BPS


def test_obs_offset_not_a_whole_number_of_samples(tmp_path):
    a = _write(tmp_path / "a.dada", 3200 + 1, 3, 1.0)
    header, _ = dump_reader.read_beams([a], [0])
    assert header.sample0 is None and header.contiguous is False
    assert "whole number of samples" in header.contiguity_note


def test_missing_obs_offset_falls_back_to_name_order(tmp_path):
    a = _write(tmp_path / "a.dada", None, 3, 1.0)
    b = _write(tmp_path / "b.dada", None, 3, 2.0)
    header, data = dump_reader.read_beams([b, a], [0])
    assert header.sample0 is None and header.contiguous is False
    assert "OBS_OFFSET" in header.contiguity_note
    assert np.all(data[0, :, :12] == 1.0)               # name order a, b


def test_sort_by_name_false_keeps_the_callers_order(tmp_path):
    a, b, _ = _pair(tmp_path)
    header, data = dump_reader.read_beams([b, a], [0], sort_by_name=False)
    assert np.all(data[0, :, :12] == 2.0)
    assert header.contiguous is False                    # b then a is not an unbroken run
