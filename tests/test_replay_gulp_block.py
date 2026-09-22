"""The injection replay's v3 bottom row, from t2d's injection gulp block.

t2d writes the gulp block of every injection gulp (casm_t2 CARD_SCHEMA.md,
"Injection gulp blocks"); t3-replay-injection finds it from the matched cluster,
draws the injection as the red group, and falls back to the legacy bottom row
when there is none.
"""
import json
import sqlite3
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

import synthetic_gulp as sg
from casm_t3.apps import replay_injection
from casm_t3.apps.replay_injection import _parse_utc
from test_plotting import _boxes_px, _drawn, _drawn_beams
from test_replay_span import C0, DM, TSAMP_S, _ledger, two_file_dump  # noqa: F401

CID = 77


def injection_block() -> dict:
    """sg's default gulp as t2d writes it for an injection: the event cluster
    (id 0) is the injection, and a real event (id 1) triggered in the gulp."""
    g = sg.build_card("default")["gulp"]
    for cl in g["clusters"]:
        if cl["id"] == 0:
            cl["outcome"] = "injection"
        elif cl["id"] == 1:
            cl["outcome"] = "triggered"
    g["reference"] = {"id": 0, "outcome": "injection", "cluster_id": CID,
                      "candname": "2609180001", "samp": sg.EVENT_SAMP, "beam": 336,
                      "event_utc": sg.event_utc()}
    return g


def write_block(d, block, cid=CID):
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{block['utc_start']}_g{block['gulp']}_c{cid}.json"
    p.write_text(json.dumps(block))
    return p


def cluster_row(**kw):
    row = {"id": CID, "gulp": sg.GULP, "obs_utc_start": sg.UTC_START,
           "samp": sg.EVENT_SAMP, "beam": 336, "event_utc": sg.event_utc()}
    row.update(kw)
    return row


def test_find_by_cluster_id_then_by_peak(tmp_path):
    block = injection_block()
    write_block(tmp_path, block)
    assert replay_injection.find_gulp_block(tmp_path, cluster_row()) == block
    # a block whose row id differs is still found by observation, sample, beam
    assert replay_injection.find_gulp_block(tmp_path, cluster_row(id=5)) == block
    # another cluster of the same gulp is not
    assert replay_injection.find_gulp_block(
        tmp_path, cluster_row(id=5, samp=sg.EVENT_SAMP - 3000)) is None


@pytest.mark.parametrize("gulp_dir, cluster", [
    ("missing", cluster_row()),               # no directory
    ("here", None),                           # not recovered
    ("here", cluster_row(gulp=None)),         # an old row
    ("here", cluster_row(gulp=sg.GULP + 1)),  # another gulp
    ("", cluster_row()),                      # --gulp-dir ''
])
def test_no_block_is_none_not_an_error(tmp_path, gulp_dir, cluster):
    write_block(tmp_path / "here", injection_block())
    d = str(tmp_path / gulp_dir) if gulp_dir else ""
    assert replay_injection.find_gulp_block(d, cluster) is None


def test_injection_becomes_the_red_group():
    block = injection_block()
    g, samp = replay_injection.replay_gulp_block(block, _parse_utc(sg.event_utc()))
    outcome = {cl["id"]: cl["outcome"] for cl in g["clusters"]}
    assert outcome[0] == "triggered" and outcome[1] == "also_triggerable"
    assert list(outcome.values()).count("triggered") == 1
    assert samp == sg.EVENT_SAMP
    assert g["trials"] == block["trials"]                 # no shift, no rounding drift
    assert block["clusters"][0]["outcome"] == "injection"  # the input is untouched


def test_dt_is_rereferenced_to_the_replay_event():
    block = injection_block()
    later = _parse_utc(sg.event_utc()) + timedelta(seconds=0.5)
    g, samp = replay_injection.replay_gulp_block(block, later)
    for a, b in zip(block["trials"], g["trials"]):
        assert b[0] == pytest.approx(a[0] - 0.5, abs=1e-4)
    assert g["clusters"][0]["peak"][0] == pytest.approx(-0.5, abs=1e-4)
    assert samp == sg.EVENT_SAMP + round(0.5 / sg.TSAMP_S)


def replay_card():
    card = sg.build_card("default", candname="INJECTION: inj_test_0001")
    card["gulp"], card["samp"] = replay_injection.replay_gulp_block(
        injection_block(), _parse_utc(card["event_utc"]))
    return card


def test_replay_card_draws_the_v3_row_with_the_injection_red():
    card = replay_card()
    inj = [cl for cl in card["gulp"]["clusters"] if cl["id"] == 0][0]
    with _drawn(card) as (fig, axes):
        boxes = _boxes_px(fig)
        red = _drawn_beams(axes["sky"], "sky_red", card["pointings"])
        red_trials = sum(len(c.get_offsets()) for c in axes["beams"].collections
                         if c.get_gid() == "trials_triggered")
    assert {"hist", "beams", "sky"} <= set(boxes) and "members" not in boxes
    assert red == {b for b, _ in inj["beams"]}
    assert red_trials == sum(1 for r in card["gulp"]["trials"] if r[5] == 0)
    # the same geometry as a live trigger card
    with _drawn(sg.build_card("default")) as (fig, _):
        assert _boxes_px(fig) == boxes


def _ledger_with_cluster(tmp_path, event):
    db = _ledger(tmp_path)
    conn = sqlite3.connect(db)
    conn.execute("ALTER TABLE clusters ADD COLUMN gulp INT")
    conn.execute("INSERT INTO clusters (id, snr, dm, beam, width, samp, event_utc, "
                 "obs_utc_start, n_members, n_beams, gulp) VALUES "
                 "(?, 15.5, ?, 100, 1, ?, ?, ?, 3, 1, ?)",
                 (CID, DM, sg.EVENT_SAMP, event, sg.UTC_START, sg.GULP))
    conn.execute("UPDATE injections SET matched_cluster = ? WHERE id = 3", (CID,))
    conn.commit()
    conn.close()
    return db


@pytest.mark.parametrize("with_block", [True, False])
def test_main_attaches_the_block_or_falls_back(two_file_dump, tmp_path, monkeypatch,  # noqa: F811
                                               with_block):
    d, _ = two_file_dump
    event = (datetime(2026, 9, 11, 1, 40, 0, tzinfo=timezone.utc)
             + timedelta(seconds=C0 * TSAMP_S)).isoformat(timespec="milliseconds")
    db = _ledger_with_cluster(tmp_path, event)
    block = injection_block()
    block["reference"].update(beam=100, event_utc=event)
    gdir = tmp_path / "gulps"
    gdir.mkdir()
    if with_block:
        write_block(gdir, block)
    seen = {}

    def fake_figure(data, freqs, tsamp, t_rel, card, out_png, **kw):
        seen.update(card)
        out_png.write_bytes(b"png")
        return out_png

    monkeypatch.setattr(replay_injection.plotting, "make_candidate_figure", fake_figure)
    card_json = tmp_path / "c.json"
    replay_injection.main(["--inject-id", "3", "--dump", str(d), "--out",
                           str(tmp_path / "t.png"), "--db", str(db), "--no-registry",
                           "--local-beam", "0", "--card-json", str(card_json),
                           "--gulp-dir", str(gdir)])
    card = json.loads(card_json.read_text())
    if with_block:
        red = [cl for cl in seen["gulp"]["clusters"] if cl["outcome"] == "triggered"]
        assert [cl["id"] for cl in red] == [0]
        assert seen["samp"] == sg.EVENT_SAMP
        assert np.allclose([r[0] for r in seen["gulp"]["trials"]],
                           [r[0] for r in block["trials"]])
        assert card["gulp"]["reference"]["cluster_id"] == CID
    else:
        assert "gulp" not in seen and "gulp" not in card
