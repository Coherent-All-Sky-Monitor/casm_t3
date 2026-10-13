"""t3-replot --event-dir: re-render an archived event folder (schema 1 or 2) from its .fil and mask."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

import event_env as ee
import synthetic_event as se
from casm_io.candidates import hella_mask as hm
from casm_io.candidates import t3_event
from casm_t3 import dump_reader, event_archive, fil_reader
from casm_t3.apps import replot

NAMES = t3_event.product_names(se.CANDNAME)
RAW = "before hella RFI flagging; IB subtraction off"


@pytest.fixture
def event(tmp_path, monkeypatch):
    """A processed schema-2 event folder (stubbed figures) and its environment."""
    env = ee.make_env(tmp_path, monkeypatch)
    ee.stub_figure(monkeypatch)
    env.process()
    return env


def test_both_views_use_the_stored_mask_and_label_the_raw_figure(event, tmp_path, monkeypatch):
    calls = ee.stub_figure(monkeypatch)
    out = tmp_path / "x.png"
    got = replot.render_event_dir(event.event_dir, out, "both")
    assert got == [out, tmp_path / "x_raw.png"]
    masked, raw = calls
    stored = t3_event.load_event(event.event_dir).mask
    assert np.array_equal(masked["mask"], stored.combined) and np.array_equal(masked["coverage"], stored.coverage)
    assert masked["shape"] == (se.NCHAN, se.dump_samples()) and "mask_label" not in masked
    assert raw["mask_label"] == RAW and "mask" not in raw
    # the event sits EVENT_S into the fil, whose clock is the dump's
    assert masked["t_rel"] == pytest.approx(se.EVENT_S, abs=2e-3)


def test_single_views(event, tmp_path, monkeypatch):
    calls = ee.stub_figure(monkeypatch)
    assert replot.render_event_dir(event.event_dir, tmp_path / "m.png", "masked") == [tmp_path / "m.png"]
    assert replot.render_event_dir(event.event_dir, tmp_path / "r.png", "raw") == [tmp_path / "r.png"]
    assert "mask" in calls[0] and calls[1]["mask_label"] == RAW and calls[1]["out"] == "r.png"
    with pytest.raises(ValueError):
        replot.render_event_dir(event.event_dir, tmp_path / "z.png", "sideways")


def test_card_is_rebuilt_from_the_summary_and_csvs(event):
    card = json.loads(event.card_path.read_text())          # the card as T2 spooled it
    ev = t3_event.load_event(event.event_dir)
    new = event_archive.card_from_event(ev)
    for k in ("candname", "event_utc", "snr", "dm", "width", "beam", "local_beam", "stream", "samp",
              "source", "sky", "trigger_reason"):
        assert new[k] == card[k], k
    g, g0 = new["gulp"], card["gulp"]
    for k in ("gulp", "utc_start", "samp_lo", "samp_hi", "weights_id", "n_raw", "n_width_dropped",
              "n_shed", "n_trials_total", "trials_truncated", "n_clusters_total",
              "clusters_truncated", "rules", "storm"):
        assert g[k] == g0[k], k
    assert g["trials"] == g0["trials"] and g["clusters"] == g0["clusters"]
    assert g["tsamp_s"] == pytest.approx(g0["tsamp_s"])
    assert new["pointings"]["alt_deg"] == card["pointings"]["alt_deg"]
    assert new["pointings"]["az_deg"] == card["pointings"]["az_deg"]
    assert new["pointings"]["beam_fwhm_x_deg"] == card["pointings"].get("beam_fwhm_x_deg")
    assert new["context"]["members"] == []


def test_context_members_and_card_without_a_gulp_block_round_trip(tmp_path, monkeypatch):
    env = ee.make_env(tmp_path, monkeypatch)
    card = dict(env.card, context={"window_s": 4.0, "members": [[-3.9982, 327, 675.895, 10.16, 5],
                                                                 [0.5, 285, 188.9, 12.0, 4]]})
    card.pop("gulp")
    env.card_path.write_text(json.dumps(card))
    ee.stub_figure(monkeypatch)
    env.process()
    new = event_archive.card_from_event(t3_event.load_event(env.event_dir))
    assert "gulp" not in new
    assert new["context"] == {"window_s": 4.0, "members": card["context"]["members"]}


def test_event_without_a_mask_gets_the_unavailable_label(tmp_path, monkeypatch):
    env = ee.make_env(tmp_path, monkeypatch, records=False)
    ee.stub_figure(monkeypatch)
    env.process()
    calls = ee.stub_figure(monkeypatch)
    replot.render_event_dir(env.event_dir, tmp_path / "x.png", "masked")
    assert calls[0]["mask_label"] == "hella RFI flags unavailable (no hella record covers the dump)"
    assert "mask" not in calls[0]


def test_schema_1_folder_re_renders_with_an_unknown_ib_state(tmp_path, monkeypatch):
    env = ee.make_env(tmp_path, monkeypatch)
    d = tmp_path / "old" / se.CANDNAME
    d.mkdir(parents=True)
    header, data = dump_reader.read_beams(env.dump_files, [se.LOCAL_BEAM])
    event_archive.write_event_fil(data[0], header, env.card, d / f"{se.CANDNAME}.fil")
    (d / f"{se.CANDNAME}.png").write_bytes(b"png")
    (d / f"{se.CANDNAME}.json").write_text(json.dumps(
        dict(env.card, host="casm-corr2", n_samples=int(data.shape[2]), data_available=False)))
    calls = ee.stub_figure(monkeypatch)
    replot.render_event_dir(d, tmp_path / "x.png", "both")
    assert calls[0]["mask_label"].startswith("hella RFI flags unavailable (schema 1 event")
    assert calls[1]["mask_label"] == "before hella RFI flagging; IB subtraction unknown"
    assert calls[0]["card"]["gulp"]["trials"] == json.loads(env.card_path.read_text())["gulp"]["trials"]


def test_mask_that_does_not_fit_the_fil_is_an_error(event, tmp_path, monkeypatch):
    real = hm.load_event_mask

    def short(path):
        w = real(path)
        w.combined = w.combined[:, :-1]
        return w
    monkeypatch.setattr(hm, "load_event_mask", short)
    ee.stub_figure(monkeypatch)
    with pytest.raises(ValueError, match="mask is"):
        replot.render_event_dir(event.event_dir, tmp_path / "x.png", "masked")


def test_folder_without_a_fil_is_an_error(event, tmp_path):
    (event.event_dir / NAMES["fil"]).unlink()
    with pytest.raises(FileNotFoundError, match="fil"):
        replot.render_event_dir(event.event_dir, tmp_path / "x.png")


def test_cli(event, tmp_path, monkeypatch, capsys):
    calls = ee.stub_figure(monkeypatch)
    out = tmp_path / "o.png"
    replot.main(["--event-dir", str(event.event_dir), "--out", str(out)])       # default view: both
    assert capsys.readouterr().out.split() == [str(out), str(tmp_path / "o_raw.png")]
    replot.main(["--event-dir", str(event.event_dir), "--view", "raw", "--out", str(out)])
    assert capsys.readouterr().out.split() == [str(out)]
    assert len(calls) == 3
    for bad in (["--event-dir", str(event.event_dir), "--clean", "hella"],
                ["--event-dir", str(event.event_dir), str(event.card_path)], []):
        with pytest.raises(SystemExit) as exc:
            replot.main(bad)
        assert exc.value.code == 2
    with pytest.raises(SystemExit, match="cannot re-render"):
        replot.main(["--event-dir", str(tmp_path / "nothing")])


def test_card_path_and_fil_options_are_unchanged(event, tmp_path, monkeypatch, capsys):
    calls = ee.stub_figure(monkeypatch)
    out = tmp_path / "c.png"
    replot.main([str(event.card_path), "--fil", str(event.event_dir / NAMES["fil"]), "--out", str(out)])
    assert capsys.readouterr().out.split() == [str(out)]
    assert len(calls) == 1 and calls[0]["out"] == "c.png"
    assert "mask" not in calls[0] and "mask_label" not in calls[0]


def test_real_figures_from_a_stored_folder(tmp_path, monkeypatch):
    env = ee.make_env(tmp_path, monkeypatch)
    env.process()                                       # real renderer end to end
    out = tmp_path / "re.png"
    got = replot.render_event_dir(env.event_dir, out, "both")
    assert [p.name for p in got] == ["re.png", "re_raw.png"]
    assert all(p.stat().st_size > 10_000 for p in got)
    # the stored folder reproduces the figures the daemon wrote, pixel for pixel
    assert got[0].read_bytes() == (env.event_dir / NAMES["png"]).read_bytes()
    assert got[1].read_bytes() == (env.event_dir / NAMES["raw_png"]).read_bytes()
