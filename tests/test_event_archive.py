"""Event archive: the single-beam .fil and the per-event directory."""
from datetime import datetime, timezone

import numpy as np
import pytest
from astropy.time import Time

from casm_t3 import event_archive
from casm_t3.apps import janitor
from casm_t3.dump_reader import DumpHeader

NCHAN, NTIME = 64, 100
TSAMP = 1.048576e-3
FTOP = 484.375
CHANW = 0.030517578125


def _header():
    return DumpHeader(
        nchan=NCHAN, nbeam=64, ninner=64, nbit=16, tsamp_s=TSAMP,
        freq_top_mhz=FTOP, chan_width_mhz=CHANW,
        t0=datetime(2026, 9, 2, 20, 0, 0, tzinfo=timezone.utc),
        hdr_size=4096, raw={})


def _card():
    return {"candname": "TESTcand", "beam": 336, "local_beam": 17}


def _data():
    rng = np.random.default_rng(3)
    return rng.normal(100.0, 10.0, (NCHAN, NTIME)).astype(np.float32)


def test_fil_roundtrips(tmp_path):
    header, card, data = _header(), _card(), _data()
    fil = event_archive.write_event_fil(data, header, card, tmp_path / "TESTcand.fil")
    assert fil.is_file()

    from casm_io.filterbank import FilterbankFile
    f = FilterbankFile(str(fil), verbose=False)
    hdr = f.header
    assert hdr["nchans"] == NCHAN
    assert hdr["nbits"] == 32
    assert f.nbeams == 1
    assert hdr["nifs"] == 1
    assert hdr["tsamp"] == pytest.approx(TSAMP, rel=1e-12)
    assert hdr["fch1"] == pytest.approx(FTOP, rel=1e-12)
    assert hdr["foff"] == pytest.approx(-CHANW, rel=1e-12)
    expect_mjd = Time(header.t0.replace(tzinfo=None), scale="utc").mjd
    assert abs(hdr["tstart"] - expect_mjd) < 1e-9
    # the reader hands back (nsamples, nchans)
    assert f.data.shape == (NTIME, NCHAN)
    np.testing.assert_allclose(f.data, data.T)


def test_fil_header_fields():
    hdr = event_archive.fil_header_from_dump(_header(), _card(), "/tmp/TESTcand.fil")
    assert hdr["source_name"] == "TESTcand"
    assert hdr["ibeam"] == 336          # global beam, not the local index
    assert hdr["nbeams"] == 1
    assert hdr["rawdatafile"] == "TESTcand.fil"   # basename only, filtool limit
    assert hdr["data_type"] == 1


def test_long_candname_rejected():
    card = dict(_card(), candname="X" * 90)
    with pytest.raises(ValueError):
        event_archive.fil_header_from_dump(_header(), card, "/tmp/" + "X" * 90 + ".fil")


def test_archive_event_copies_and_skips_in_place(tmp_path):
    root = tmp_path / "EVENTS"
    event_dir = root / "TESTcand"
    event_dir.mkdir(parents=True)
    fil = event_dir / "TESTcand.fil"
    fil.write_bytes(b"fil")
    png = tmp_path / "plots" / "TESTcand.png"
    png.parent.mkdir()
    png.write_bytes(b"png")
    meta = tmp_path / "plots" / "TESTcand.json"
    meta.write_text("{}")

    out = event_archive.archive_event(root, "TESTcand", [png, meta, fil])
    assert out == event_dir
    assert sorted(p.name for p in event_dir.iterdir()) == [
        "TESTcand.fil", "TESTcand.json", "TESTcand.png"]
    assert (event_dir / "TESTcand.png").read_bytes() == b"png"
    assert fil.read_bytes() == b"fil"       # untouched, never copied over itself


def test_janitor_patrol_patterns_reach_stream_dirs():
    # find runs with -maxdepth 1, so every patrol glob must name the
    # per-stream directories the .dada files actually live in.
    assert janitor.PATROL
    for _host, pattern in janitor.PATROL:
        assert pattern.endswith("/stream_*"), pattern


def test_render_card_writes_fil_and_records_it(tmp_path, monkeypatch):
    from casm_t3.apps import dump_plotter

    header, data = _header(), _data()
    card = dict(_card(), event_utc="2026-09-02T20:00:00.020+00:00")
    monkeypatch.setattr(dump_plotter.dump_reader, "read_beams",
                        lambda files, beams: (header, data[None, :, :]))
    monkeypatch.setattr(dump_plotter.plotting, "make_candidate_figure",
                        lambda *a, **k: a[5])

    fil = tmp_path / "TESTcand.fil"
    png, result = dump_plotter.render_card(card, [tmp_path / "x.dada"],
                                           tmp_path / "TESTcand.png", fil_path=fil)
    assert fil.is_file()
    assert result["fil"] == str(fil)

    # no fil_path: unchanged behaviour, no 'fil' key
    _, result2 = dump_plotter.render_card(card, [tmp_path / "x.dada"],
                                          tmp_path / "TESTcand.png")
    assert "fil" not in result2


def test_render_card_survives_a_failed_fil(tmp_path, monkeypatch):
    from casm_t3.apps import dump_plotter

    header, data = _header(), _data()
    card = dict(_card(), candname="Y" * 90, event_utc="2026-09-02T20:00:00.020+00:00")
    monkeypatch.setattr(dump_plotter.dump_reader, "read_beams",
                        lambda files, beams: (header, data[None, :, :]))
    monkeypatch.setattr(dump_plotter.plotting, "make_candidate_figure",
                        lambda *a, **k: a[5])

    png, result = dump_plotter.render_card(
        card, [tmp_path / "x.dada"], tmp_path / "c.png",
        fil_path=tmp_path / ("Y" * 90 + ".fil"))   # rejected: name too long
    assert png == tmp_path / "c.png"
    assert "fil" not in result
