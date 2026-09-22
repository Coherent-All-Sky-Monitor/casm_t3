"""A high-DM replay must read every dump file its sweep touches.

The dump daemon rolls a requested window into consecutive .dada files. Taking
only the file that holds the top-of-band arrival cuts the sweep at the seam,
which is what left inj_20260911_0003 with a pulse above 447 MHz only. These
tests build a two-file synthetic dump whose sweep crosses the boundary.
"""
import json
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from casm_t3 import dump_reader, injection
from casm_t3.apps import replay_injection

NCHAN, NBEAM, NINNER, NFRAME = injection.GEN_NCHAN, 2, 64, 8
NSAMP = NINNER * NFRAME                                   # 512 samples per file
TSAMP_S = injection.GEN_TSAMP_S
CHAN_BW = 0.030517578125
BYTES_PER_SAMPLE = NBEAM * NCHAN * 2                      # float16
UTC_START = "2026-09-11-01:40:00.000000"
DM, AMP, SIGMA_MS, C0 = 42.0, 30.0, 2.0, 450              # pulse near the end of file 1


def _write_dump(path: Path, file_index: int, rng) -> None:
    """One .dada file of float16 noise, contiguous with its neighbours."""
    hdr = "\n".join([
        "HDR_SIZE 4096", f"UTC_START {UTC_START}", "PICOSECONDS 0",
        f"OBS_OFFSET {file_index * NSAMP * BYTES_PER_SAMPLE}",
        f"BYTES_PER_SECOND {BYTES_PER_SAMPLE / TSAMP_S:.0f}",
        f"NCHAN {NCHAN}", f"NBEAM {NBEAM}", "NBIT 16",
        f"TSAMP {TSAMP_S * 1e6:.6f}",
        f"RESOLUTION {NBEAM * NCHAN * NINNER * 2}",
        f"FREQ_START {injection.GEN_FCH1_MHZ}", f"CHANBW {CHAN_BW}", ""])
    body = rng.normal(100.0, 20.0, (NFRAME, NBEAM, NCHAN, NINNER)).astype(np.float16)
    with open(path, "wb") as f:
        f.write(hdr.encode().ljust(4096, b"\0"))
        body.tofile(f)


@pytest.fixture
def two_file_dump(tmp_path):
    d = tmp_path / "stream_1"
    d.mkdir()
    rng = np.random.default_rng(11)
    names = []
    for i in range(2):
        # byte offsets are not zero-padded in real dumps; keep that shape
        p = d / f"{UTC_START}_{i * NSAMP * BYTES_PER_SAMPLE}.000000.dada"
        _write_dump(p, i, rng)
        names.append(p)
    return d, names


def _expected_truncation_mhz(ntime: int) -> float:
    """Lowest frequency whose whole pulse fits in `ntime` samples, from the
    dispersion delay alone (no rounding to samples)."""
    halfwin = int(np.ceil(4.0 * injection.sigma_samples(SIGMA_MS, TSAMP_S)))
    delay_ms = (ntime - 1 - halfwin - C0) * TSAMP_S * 1e3
    fref_ghz = injection.GEN_FCH1_MHZ / 1000.0
    f_ghz = 1.0 / np.sqrt(1.0 / fref_ghz**2 + delay_ms / (injection.K_MS * DM))
    return float(f_ghz * 1000.0)


def test_sweep_selects_both_files(two_file_dump):
    d, names = two_file_dump
    event = dump_reader.read_header(names[0]).t0 + timedelta(seconds=C0 * TSAMP_S)
    sweep_s = 4.148808e3 * DM * (390.66**-2 - injection.GEN_FCH1_MHZ**-2)
    assert replay_injection.select_dump_files(d, event, 0.0, pad_s=0.0) == [names[0]]
    got = replay_injection.select_dump_files(d, event, sweep_s, pad_s=0.0)
    assert sorted(got) == sorted(names)
    assert replay_injection.order_dump_files(got) == names


def test_non_contiguous_files_raise(two_file_dump, tmp_path):
    d, names = two_file_dump
    # same data, but a second observation start: a 1 s gap at the seam
    bad = d / "2026-09-11-01:40:01.000000_0.000000.dada"
    raw = bytearray(names[1].read_bytes())
    raw[:4096] = raw[:4096].replace(UTC_START.encode(), b"2026-09-11-01:40:01.000000")
    raw[:4096] = raw[:4096].replace(b"OBS_OFFSET %d" % (NSAMP * BYTES_PER_SAMPLE),
                                    b"OBS_OFFSET 0")
    bad.write_bytes(bytes(raw))
    with pytest.raises(SystemExit, match="not contiguous"):
        replay_injection.order_dump_files([names[0], bad])


def test_concatenation_lights_every_channel(two_file_dump):
    d, names = two_file_dump
    files = replay_injection.order_dump_files(names)
    header, data = dump_reader.read_beams(files, [0], sort_by_name=False)
    ntime = data.shape[2]
    assert ntime == 2 * NSAMP
    freqs = injection.generator_freqs_mhz(NCHAN)
    pulse = injection.injected_pulse(NCHAN, ntime, DM, AMP, SIGMA_MS, C0,
                                     tsamp_s=header.tsamp_s, freqs_mhz=freqs)
    assert int((pulse.max(axis=1) > 0).sum()) == NCHAN
    assert replay_injection.truncation_freq_mhz(
        freqs, DM, header.tsamp_s, SIGMA_MS, C0, ntime) is None
    # one file only: the sweep runs off the end and the bottom of the band goes
    one = replay_injection.truncation_freq_mhz(
        freqs, DM, header.tsamp_s, SIGMA_MS, C0, NSAMP)
    assert one is not None
    # tolerance: the delays are rounded to whole samples, half a sample is
    # 0.15 MHz at this DM
    assert one == pytest.approx(_expected_truncation_mhz(NSAMP), abs=0.2)
    assert 390.66 < one < injection.GEN_FCH1_MHZ


def _ledger(tmp_path) -> Path:
    db = tmp_path / "t2.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE injections (id INTEGER PRIMARY KEY, file_id TEXT, "
                 "inject_utc TEXT, stream INT, beam INT, dm REAL, amp REAL, "
                 "sigma_ms REAL, est_snr REAL, matched_cluster INT, rec_snr REAL, "
                 "rec_dm REAL)")
    conn.execute("CREATE TABLE clusters (id INTEGER PRIMARY KEY, snr REAL, dm REAL, "
                 "beam INT, width INT, samp INT, event_utc TEXT, obs_utc_start TEXT, "
                 "n_members INT, n_beams INT)")
    conn.execute("INSERT INTO injections (id, file_id, inject_utc, stream, beam, dm, "
                 "amp, sigma_ms, est_snr) VALUES (3, 'inj_test_0003', ?, 1, 100, ?, "
                 "?, ?, 20.0)", (UTC_START, DM, AMP, SIGMA_MS))
    conn.commit()
    conn.close()
    return db


def _run(dump, out, db, tmp_path, extra=()):
    event = (datetime(2026, 9, 11, 1, 40, 0, tzinfo=timezone.utc)
             + timedelta(seconds=C0 * TSAMP_S))
    card = tmp_path / f"{Path(out).stem}.json"
    replay_injection.main(["--inject-id", "3", "--dump", str(dump), "--out", str(out),
                           "--db", str(db), "--no-registry", "--local-beam", "0",
                           "--card-json", str(card),
                           "--event-utc", event.isoformat(),
                           *extra])
    return json.loads(card.read_text())["replay"]


def test_main_reads_both_files_and_flags_a_single_file(two_file_dump, tmp_path, caplog):
    d, names = two_file_dump
    db = _ledger(tmp_path)

    both = _run(d, tmp_path / "both.png", db, tmp_path)
    assert len(both["dump_files"]) == 2
    assert both["n_samples"] == 2 * NSAMP
    assert both["truncated_below_mhz"] is None
    assert both["measured_boxcar_snr"] > 8.0

    with caplog.at_level("WARNING"):
        one = _run(names[0], tmp_path / "one.png", db, tmp_path)
    assert one["dump_files"] == [str(names[0])]
    assert one["truncated_below_mhz"] == pytest.approx(
        _expected_truncation_mhz(NSAMP), abs=0.2)
    assert "replay truncated below" in caplog.text
    assert one["measured_boxcar_snr"] < both["measured_boxcar_snr"]


def test_replay_title_is_hella_snr_at_the_ledger_dm(two_file_dump, tmp_path, monkeypatch):
    """The figure is named "INJECTION: <file_id>", quotes hella's recovered S/N
    and dedisperses at the ledger DM; no injection parameters reach it."""
    d, _names = two_file_dump
    db = _ledger(tmp_path)
    event = (datetime(2026, 9, 11, 1, 40, 0, tzinfo=timezone.utc)
             + timedelta(seconds=C0 * TSAMP_S))
    conn = sqlite3.connect(db)
    conn.execute("INSERT INTO clusters (id, snr, dm, beam, width, samp, event_utc, "
                 "n_members, n_beams) VALUES (77, 15.5, ?, 100, 1, 0, ?, 3, 1)",
                 (DM + 1.3, event.isoformat()))
    conn.execute("UPDATE injections SET matched_cluster = 77, rec_snr = 15.5, rec_dm = ? "
                 "WHERE id = 3", (DM + 1.3,))
    conn.commit()
    conn.close()

    seen = {}

    def fake_figure(data, freqs, tsamp, t_rel, card, out_png, **kw):
        seen.update(card)
        Path(out_png).write_bytes(b"png")
        return Path(out_png)

    monkeypatch.setattr(replay_injection.plotting, "make_candidate_figure", fake_figure)
    card_json = tmp_path / "c.json"
    replay_injection.main(["--inject-id", "3", "--dump", str(d), "--out",
                           str(tmp_path / "t.png"), "--db", str(db), "--no-registry",
                           "--local-beam", "0", "--card-json", str(card_json)])
    assert seen["candname"] == "INJECTION: inj_test_0003"
    assert seen["snr"] == 15.5                     # hella's recovered S/N
    assert seen["dm"] == DM                        # the ledger DM, not hella's DM + 1.3
    assert "FWHM" not in json.dumps({k: v for k, v in seen.items() if k != "injection"})
    card = json.loads(card_json.read_text())
    assert card["source"] == "injection"
    assert card["injection"]["summary"].endswith(f"; hella S/N 15.5 at DM {DM + 1.3:.1f}")
