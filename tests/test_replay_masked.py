"""t3-replay-injection with hella's flag records: masked and raw figures, synthetic labels.

The dump and records are synthetic_event's (4 beams x 512 channels); the ledger
is a one-row sqlite. Global beam 5 is local beam 1 of the dump.
"""
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

import event_env as ee
import synthetic_event as se
from casm_io.candidates import hella_mask as hm
from casm_t3 import fil_reader
from casm_t3.apps import replay_injection as ri

UTC_START = "2026-10-13-18:22:38"
MASKED = "synthetic replay; hella RFI flags applied"
RAW = "synthetic replay; raw intensity, before hella RFI flagging"


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
                 "amp, sigma_ms, est_snr) VALUES (3, 'inj_test_0003', ?, 1, ?, 29.5, "
                 "30.0, 4.0, 20.0)", (UTC_START, se.GLOBAL_BEAM))
    conn.commit()
    conn.close()
    return db


def _run(tmp_path, extra=(), name="inj3"):
    out = tmp_path / f"{name}.png"
    card = tmp_path / f"{name}.json"
    ri.main(["--inject-id", "3", "--dump", str(tmp_path / "dump"), "--out", str(out),
             "--db", str(_ledger(tmp_path)), "--no-registry", "--local-beam", str(se.LOCAL_BEAM),
             "--gulp-dir", "", "--event-utc", se.event_utc(), "--card-json", str(card), *extra])
    return out, json.loads(card.read_text())


@pytest.fixture
def scene(tmp_path):
    se.write_dump(tmp_path / "dump")
    se.write_records(tmp_path / "records")
    return tmp_path


def test_records_cover_the_replay_masked_figure_raw_figure_and_card(scene, monkeypatch):
    calls = ee.stub_figure(monkeypatch)
    raw = scene / "inj3_raw.png"
    out, card = _run(scene, ["--mask-records", str(scene / "records"), "--raw-out", str(raw),
                             "--clean", "hella", "--sumthreshold", "default",
                             "--fil", str(scene / "inj3.fil")])
    assert [c["out"] for c in calls] == [out.name, raw.name]
    main, rawcall = calls
    assert main["mask"].shape == (se.NCHAN, se.dump_samples()) and main["coverage"].all()
    assert main["mask_label"] == MASKED
    assert "clean" not in main                         # the real flags replace the offline replica
    assert rawcall["mask_label"] == RAW and "mask" not in rawcall
    assert out.is_file() and raw.is_file()
    assert card["kind"] == "synthetic_replay"
    hmb = card["hella_mask"]
    assert hmb["file"] == "inj3.mask.npz" and hmb["coverage"] == 1.0 and hmb["gulps"] == 3
    mask = hm.load_event_mask(scene / "inj3.mask.npz")
    data, _ = fil_reader.read_fil(scene / "inj3.fil")
    assert mask.combined.shape == data.shape and mask.sample0 == se.SAMPLE0
    assert np.array_equal(mask.combined, main["mask"])
    assert card["replay"]["raw_plot"] == str(raw) and card["replay"]["pulse_added"]
    assert card["source"] == "injection" and card["candname"] == "inj3"


def test_without_coverage_and_with_clean_hella_the_replica_path_is_kept(scene, monkeypatch):
    calls = ee.stub_figure(monkeypatch)
    out, card = _run(scene, ["--mask-records", str(scene / "empty"), "--clean", "hella",
                             "--raw-out", str(scene / "inj3_raw.png")])
    main, rawcall = calls
    assert main["clean"] == "hella" and "mask" not in main
    assert main["mask_label"] == "synthetic replay; offline replica of hella flagging"
    assert rawcall["mask_label"] == RAW
    assert card["kind"] == "synthetic_replay"
    assert card["hella_mask"] == {"file": None, "reason": "mask record dir missing"}


def test_records_given_but_nothing_covers_and_no_clean(scene, monkeypatch):
    calls = ee.stub_figure(monkeypatch)
    (scene / "empty").mkdir()
    _, card = _run(scene, ["--mask-records", str(scene / "empty")])
    assert len(calls) == 1
    assert calls[0]["mask_label"] == ("synthetic replay; hella RFI flags unavailable "
                                      "(no hella record covers the dump)")
    assert card["hella_mask"] == {"file": None, "reason": "no hella record covers the dump"}


def test_existing_invocation_is_unchanged(scene, monkeypatch):
    calls = ee.stub_figure(monkeypatch)
    out, card = _run(scene)
    assert len(calls) == 1 and calls[0]["out"] == out.name
    assert calls[0]["mask_label"] is None and calls[0]["clean"] is None
    assert "mask" not in calls[0] and card["replay"]["raw_plot"] is None
    assert card["kind"] == "synthetic_replay"
    assert card["hella_mask"] == {"file": None, "reason": "mask records not given"}


def test_masked_render_failure_falls_back_to_the_unavailable_figure(scene, monkeypatch):
    calls = ee.stub_figure(monkeypatch, fail=lambda name, kw: kw.get("mask") is not None)
    _, card = _run(scene, ["--mask-records", str(scene / "records")])
    assert calls[1]["mask_label"] == ("synthetic replay; hella RFI flags unavailable "
                                      "(mask render failed: ValueError)")
    assert card["hella_mask"]["file"] is None and "mask render failed" in card["hella_mask"]["reason"]


def test_legacy_layout_has_no_mask_support_and_no_labels(scene, monkeypatch):
    calls = ee.stub_figure(monkeypatch)
    _run(scene, ["--layout", "legacy", "--mask-records", str(scene / "records"),
                 "--raw-out", str(scene / "inj3_raw.png")])
    assert all(c["mask_label"] is None and "mask" not in c for c in calls)


def test_real_figures_carry_the_synthetic_titles(scene):
    out, card = _run(scene, ["--mask-records", str(scene / "records"),
                             "--raw-out", str(scene / "inj3_raw.png")])
    assert out.stat().st_size > 10_000 and (scene / "inj3_raw.png").stat().st_size > 10_000
    assert card["hella_mask"]["coverage"] == 1.0
