"""t3-dump-plotter builds the schema-2 event folder (casm_io docs/t3_events.md).

End to end on a synthetic dump and synthetic hella records (tmp dirs only; the
archive host is this host, so shipping is a local copy). One test renders the
real figures; the rest record the figure calls and write a few bytes.
"""
import hashlib
import json
import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import event_env as ee
import synthetic_event as se
from casm_io.candidates import hella_mask as hm
from casm_io.candidates import t3_event
from casm_t3 import dump_reader, event_archive, fil_reader
from casm_t3.apps import dump_plotter

NAMES = t3_event.product_names(se.CANDNAME)
RAW_LABEL = "before hella RFI flagging; IB subtraction off"


def _present(d: Path) -> set:
    return {p.name for p in d.iterdir() if p.is_file()}


def test_event_folder_is_complete_aligned_and_shipped(tmp_path, monkeypatch):
    env = ee.make_env(tmp_path, monkeypatch)
    order = []                                           # what existed when the manifest was written
    real_manifest = t3_event.write_manifest

    def spy_manifest(d, cand):
        order.append(_present(Path(d)))
        return real_manifest(d, cand)
    monkeypatch.setattr(t3_event, "write_manifest", spy_manifest)
    copies = []
    real_copy = shutil.copy2

    def spy_copy(src, dst, *a, **k):
        copies.append((Path(src).name, Path(dst)))
        return real_copy(src, dst, *a, **k)
    monkeypatch.setattr(dump_plotter.shutil, "copy2", spy_copy)

    env.process()

    ed = env.event_dir
    for role in ("png", "raw_png", "fil", "mask", "json", "t1", "clusters", "beams", "manifest"):
        assert (ed / NAMES[role]).is_file(), role
    assert (ed / NAMES["png"]).stat().st_size > 10_000 and (ed / NAMES["raw_png"]).stat().st_size > 10_000
    # the manifest goes in last: every other product already existed, and it is the newest file
    assert order == [_present(ed) - {NAMES["manifest"]}]
    assert (ed / NAMES["manifest"]).stat().st_mtime_ns >= max(
        p.stat().st_mtime_ns for p in ed.iterdir() if p.name != NAMES["manifest"])

    ev = t3_event.load_event(ed, verify=True)
    assert ev.schema == 2 and ev.complete and ev.problems == []
    assert ev.mask is not None
    data, fh = fil_reader.read_fil(ev.fil)
    assert ev.mask.combined.shape == data.shape == (se.NCHAN, se.dump_samples())
    assert ev.mask.sample0 == se.SAMPLE0 and ev.mask.beam == se.GLOBAL_BEAM
    assert ev.mask.coverage.all()
    assert fh.freqs_mhz[0] == pytest.approx(484.375) and ev.mask.fch1 == pytest.approx(484.375)
    assert ev.mask.foff == pytest.approx(fh.foff_mhz)
    # hella's decisions land where the records put them: gulp 596 is samples 216..4312
    assert ev.mask.static[:se.STATIC_CHANS].all() and not ev.mask.static[se.STATIC_CHANS:].any()
    assert ev.mask.channel_flags[se.RFI_CHAN, 216:4312].all()
    assert not ev.mask.channel_flags[se.RFI_CHAN, :216].any()
    assert ev.mask.whole_channel[se.WHOLE_CHAN, 216:4312].all()
    assert ev.mask.sumthreshold[se.BLOB_CHANS[0]:se.BLOB_CHANS[1], 4312 + 160:4312 + 480].all()

    s = env.summary()
    assert s["schema"] == 2 and s["kind"] == "candidate"
    assert s["candname"] == se.CANDNAME and s["beam"] == se.GLOBAL_BEAM
    assert s["data_available"] is False                   # the dump went
    assert s["dump"] == {"utc_start": s["dump"]["utc_start"], "utc_stop": s["dump"]["utc_stop"],
                         "sample0": se.SAMPLE0, "nsamp": se.dump_samples(),
                         "host": dump_plotter.LOCAL_HOSTNAME}
    assert s["dump"]["utc_start"] < s["dump"]["utc_stop"]
    assert s["products"] == NAMES
    hmb = s["hella_mask"]
    assert hmb["file"] == NAMES["mask"] and hmb["coverage"] == 1.0 and hmb["gulps"] == 3
    prov = s["provenance"]
    assert prov["obs_utc_start"] == se.UTC_START
    assert prov["t2d_commit"] == "0123456789ab" and prov["t3_commit"] == ee.T3_COMMIT
    assert prov["sub_incoh"] is False and prov["sub_incoh_source"] == "dump_header"
    assert prov["hella_binary_md5"] == "ab" * 16
    assert prov["hella_cfg_sha256"] == hashlib.sha256(b"INPUT DADA\nFLAG2 60\n").hexdigest()
    assert prov["hella_flag"]["flag2"] == se.STATIC_CHANS and prov["hella_flag"]["st_mmin"] == se.MMIN
    assert prov["utc_clock"] == "TCS UTC_START from casm-corr1 system clock"
    assert prov["weights_id"] == se.WEIGHTS_ID
    assert (prov["beam_fwhm_x_deg"], prov["beam_fwhm_y_deg"]) == (19.326, 4.131)
    assert s["trigger"]["gulp"] == env.card["gulp"]["gulp"]

    # the plots-dir copy and the archive-tree copy are the same schema-2 summary
    jbytes = (ed / NAMES["json"]).read_bytes()
    assert (env.plots / NAMES["json"]).read_bytes() == jbytes
    assert (env.dest / NAMES["json"]).read_bytes() == jbytes
    assert _present(env.dest) == {NAMES[r] for r in ("png", "raw_png", "json", "manifest")}
    assert _present(env.plots) >= {NAMES[r] for r in ("png", "raw_png", "json", "manifest")}
    shipped = [n for n, dst in copies if dst == env.dest]
    assert shipped == [NAMES["png"], NAMES["raw_png"], NAMES["json"], NAMES["manifest"]]
    # the shipped manifest is the folder's, so it names the products that stayed behind
    assert json.loads((env.dest / NAMES["manifest"]).read_text())["files"].keys() >= {NAMES["fil"]}
    assert not [p for p in env.dump_dir.glob("*.dada")]    # deleted: everything was written


def test_main_figure_has_hella_flags_and_raw_figure_states_ib_subtraction(tmp_path, monkeypatch):
    env = ee.make_env(tmp_path, monkeypatch)
    calls = ee.stub_figure(monkeypatch)
    env.process()
    main = ee.call_for(calls, NAMES["png"])
    assert main["mask"].shape == (se.NCHAN, se.dump_samples())
    assert main["coverage"].all() and "mask_label" not in main
    assert main["layout"] is None and main["shape"] == (se.NCHAN, se.dump_samples())
    raw = ee.call_for(calls, NAMES["raw_png"])
    assert raw["mask_label"] == RAW_LABEL and "mask" not in raw
    ev = t3_event.load_event(env.event_dir)
    assert np.array_equal(main["mask"], ev.mask.combined)      # what was drawn is what was stored


@pytest.mark.parametrize("flag,state,source", [("1", True, "dump_header"), ("true", True, "dump_header"),
                                               ("0", False, "dump_header"), ("false", False, "dump_header"),
                                               (None, None, "unknown")])
def test_ib_subtraction_state_is_read_from_the_dump_header(tmp_path, monkeypatch, flag, state, source):
    env = ee.make_env(tmp_path, monkeypatch, dump_extra={"BFCORR_SUB_INCOH": flag})
    calls = ee.stub_figure(monkeypatch)
    env.process()
    text = {True: "on", False: "off", None: "unknown"}[state]
    assert ee.call_for(calls, NAMES["raw_png"])["mask_label"] == \
        f"before hella RFI flagging; IB subtraction {text}"
    prov = env.summary()["provenance"]
    assert prov["sub_incoh"] is state and prov["sub_incoh_source"] == source


@pytest.mark.parametrize("how,reason", [
    ("empty", "no hella record covers the dump"),
    ("off", "mask records off"),
    ("missing", "mask record dir missing"),
])
def test_no_records_gives_a_labelled_raw_figure_and_no_mask(tmp_path, monkeypatch, how, reason):
    env = ee.make_env(tmp_path, monkeypatch, records=False)
    if how == "off":
        env.args.mask_records = ""
    elif how == "missing":
        env.args.mask_records = str(tmp_path / "nowhere")
    calls = ee.stub_figure(monkeypatch)
    env.process()
    ed = env.event_dir
    assert not (ed / NAMES["mask"]).exists()
    main = ee.call_for(calls, NAMES["png"])
    assert "mask" not in main and main["mask_label"] == f"hella RFI flags unavailable ({reason})"
    assert ee.call_for(calls, NAMES["raw_png"])["mask_label"] == RAW_LABEL
    assert (ed / NAMES["raw_png"]).is_file()
    s = env.summary()
    assert s["hella_mask"] == {"file": None, "reason": reason}
    assert s["products"]["mask"] is None and s["provenance"]["hella_binary_md5"] is None
    ev = t3_event.load_event(ed)
    assert ev.complete and ev.mask is None
    assert not list(env.dump_dir.glob("*.dada"))           # an absent mask never keeps the dump


def test_non_contiguous_dump_gets_no_mask_and_says_why(tmp_path, monkeypatch, caplog):
    env = ee.make_env(tmp_path, monkeypatch, gap=True)
    calls = ee.stub_figure(monkeypatch)
    monkeypatch.setattr(hm, "window_mask", lambda *a, **k: pytest.fail("looked up a mask"))
    with caplog.at_level(logging.WARNING):
        env.process()
    assert "not contiguous" in caplog.text
    assert not (env.event_dir / NAMES["mask"]).exists()
    assert ee.call_for(calls, NAMES["png"])["mask_label"] == \
        "hella RFI flags unavailable (dump files not contiguous)"
    assert env.summary()["hella_mask"] == {"file": None, "reason": "dump files not contiguous"}
    # the record in the summary: the dump itself is still described
    assert env.summary()["dump"]["sample0"] == se.SAMPLE0


def test_partial_coverage_draws_what_is_covered(tmp_path, monkeypatch):
    env = ee.make_env(tmp_path, monkeypatch, ks=(0, 1))     # gulp 597 has no record
    calls = ee.stub_figure(monkeypatch)
    env.process()
    cov = ee.call_for(calls, NAMES["png"])["coverage"]
    assert cov[:4312].all() and not cov[4312:].any()
    s = env.summary()
    assert s["hella_mask"]["file"] == NAMES["mask"]
    assert s["hella_mask"]["coverage"] == pytest.approx(4312 / se.dump_samples())
    ev = t3_event.load_event(env.event_dir)
    assert ev.complete and np.array_equal(ev.mask.coverage, cov)


def test_mask_lookup_failure_never_blocks_the_figure(tmp_path, monkeypatch):
    env = ee.make_env(tmp_path, monkeypatch)
    calls = ee.stub_figure(monkeypatch)

    def boom(*a, **k):
        raise RuntimeError("corrupt record")
    monkeypatch.setattr(hm, "window_mask", boom)
    env.process()
    assert ee.call_for(calls, NAMES["png"])["mask_label"] == \
        "hella RFI flags unavailable (mask lookup failed: RuntimeError)"
    assert env.summary()["hella_mask"]["reason"] == "mask lookup failed: RuntimeError"
    assert not list(env.dump_dir.glob("*.dada"))


def test_masked_render_failure_falls_back_to_the_raw_figure(tmp_path, monkeypatch):
    env = ee.make_env(tmp_path, monkeypatch)
    calls = ee.stub_figure(monkeypatch, fail=lambda name, kw: kw.get("mask") is not None)
    env.process()
    assert [c["out"] for c in calls] == [NAMES["png"], NAMES["png"], NAMES["raw_png"]]
    assert calls[1]["mask_label"] == "hella RFI flags unavailable (mask render failed: ValueError)"
    # a mask that does not draw is not left in the folder, and the summary says why
    assert not (env.event_dir / NAMES["mask"]).exists()
    s = env.summary()
    assert s["hella_mask"] == {"file": None, "reason": "mask render failed: ValueError"}
    assert t3_event.load_event(env.event_dir).complete
    assert not list(env.dump_dir.glob("*.dada"))


def test_mask_shape_mismatch_falls_back_with_the_real_renderer(tmp_path, monkeypatch):
    """Records for another channel count: plotting raises ValueError, the Slack figure is raw."""
    env = ee.make_env(tmp_path, monkeypatch, records=False)
    se.write_records(env.records, nchan=320)
    env.process()
    ed = env.event_dir
    assert (ed / NAMES["png"]).stat().st_size > 10_000 and (ed / NAMES["raw_png"]).is_file()
    assert not (ed / NAMES["mask"]).exists()
    assert env.summary()["hella_mask"] == {"file": None, "reason": "mask render failed: ValueError"}


def test_mask_file_write_failure_keeps_the_masked_figure_and_the_dump(tmp_path, monkeypatch, caplog):
    env = ee.make_env(tmp_path, monkeypatch)
    calls = ee.stub_figure(monkeypatch)

    def boom(path, win):
        raise OSError("disk full")
    monkeypatch.setattr(hm, "save_event_mask", boom)
    with caplog.at_level(logging.WARNING):
        env.process()
    assert "mask" in ee.call_for(calls, NAMES["png"])
    h = env.summary()["hella_mask"]
    assert h["file"] is None and h["reason"] == "mask.npz could not be written" and h["coverage"] == 1.0
    # the masked figure cannot be reproduced without its mask: the dump stays
    assert _dump_kept(env) and "keeping the dump" in caplog.text and "mask.npz" in caplog.text
    assert env.summary()["data_available"] is True


# --- what keeps the dump -----------------------------------------------------------------

def _dump_kept(env) -> bool:
    return len(list(env.dump_dir.glob("*.dada"))) == 2


def test_failed_fil_write_keeps_the_dump_but_ships_the_event(tmp_path, monkeypatch, caplog):
    env = ee.make_env(tmp_path, monkeypatch)
    ee.stub_figure(monkeypatch)

    def boom(*a, **k):
        raise OSError("no space left")
    monkeypatch.setattr(event_archive, "write_event_fil", boom)
    with caplog.at_level(logging.WARNING):
        env.process()
    assert _dump_kept(env) and "keeping the dump" in caplog.text and "fil" in caplog.text
    s = env.summary()
    assert s["products"]["fil"] is None and s["data_available"] is True
    assert _present(env.dest) == {NAMES[r] for r in ("png", "raw_png", "json", "manifest")}
    assert not t3_event.load_event(env.event_dir).fil


def test_failed_raw_render_keeps_the_dump_and_ships_the_main_figure(tmp_path, monkeypatch, caplog):
    env = ee.make_env(tmp_path, monkeypatch)
    ee.stub_figure(monkeypatch, fail=lambda name, kw: name == NAMES["raw_png"])
    with caplog.at_level(logging.WARNING):
        env.process()
    assert _dump_kept(env) and "keeping the dump" in caplog.text and "raw figure" in caplog.text
    assert env.summary()["products"]["raw_png"] is None
    assert _present(env.dest) == {NAMES[r] for r in ("png", "json", "manifest")}
    assert t3_event.load_event(env.event_dir).complete


def test_failed_main_render_raises_and_keeps_the_dump(tmp_path, monkeypatch):
    env = ee.make_env(tmp_path, monkeypatch, records=False)
    ee.stub_figure(monkeypatch, fail=lambda name, kw: name == NAMES["png"])
    with pytest.raises(ValueError):
        env.process()
    assert _dump_kept(env) and not env.dest.exists()


def test_card_that_fails_before_any_product_leaves_no_empty_event_folder(tmp_path, monkeypatch):
    env = ee.make_env(tmp_path, monkeypatch)
    env.card["local_beam"] = 99                              # no such beam in the dump
    env.card_path.write_text(json.dumps(env.card))
    with pytest.raises(Exception):
        env.process()
    assert not env.event_dir.exists() and env.events.is_dir()
    assert _dump_kept(env) and not env.dest.exists()


def test_failed_card_keeps_an_event_folder_that_holds_files(tmp_path, monkeypatch):
    env = ee.make_env(tmp_path, monkeypatch)
    ee.stub_figure(monkeypatch, fail=lambda name, kw: name == NAMES["png"])
    with pytest.raises(ValueError):
        env.process()
    assert _present(env.event_dir) == {NAMES["fil"]}         # the fil written before the failure


@pytest.mark.parametrize("target", ["write_summary", "write_manifest"])
def test_failed_summary_or_manifest_ships_nothing_and_keeps_the_dump(tmp_path, monkeypatch, target):
    env = ee.make_env(tmp_path, monkeypatch)
    ee.stub_figure(monkeypatch)

    def boom(*a, **k):
        raise OSError("read-only file system")
    monkeypatch.setattr(t3_event, target, boom)
    with pytest.raises(OSError):
        env.process()
    assert _dump_kept(env) and not env.dest.exists()


def test_failed_sidecars_do_not_block_the_event(tmp_path, monkeypatch):
    env = ee.make_env(tmp_path, monkeypatch)
    ee.stub_figure(monkeypatch)

    def boom(*a, **k):
        raise ValueError("pointings alt_deg (3) and az_deg (2) differ")
    monkeypatch.setattr(t3_event, "write_sidecars", boom)
    env.process()
    s = env.summary()
    assert s["products"]["t1"] is None and s["products"]["beams"] is None
    assert not list(env.dump_dir.glob("*.dada"))              # sidecars are not a delete condition


def test_failed_shipping_keeps_the_dump(tmp_path, monkeypatch):
    env = ee.make_env(tmp_path, monkeypatch)
    ee.stub_figure(monkeypatch)
    monkeypatch.setattr(dump_plotter.subprocess, "run",
                        lambda *a, **k: (_ for _ in ()).throw(subprocess.CalledProcessError(1, "scp")))
    env.args.archive_host = "casm-elsewhere"
    with pytest.raises(subprocess.CalledProcessError):
        env.process()
    assert _dump_kept(env)


def test_unwritable_event_root_keeps_the_dump_and_builds_in_the_plots_dir(tmp_path, monkeypatch):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x")
    env = ee.make_env(tmp_path, monkeypatch, events_root=str(blocker / "EVENTS"))
    ee.stub_figure(monkeypatch)
    env.process()
    assert _dump_kept(env)                                    # no fil could be stored
    assert _present(env.plots) >= {NAMES[r] for r in ("png", "raw_png", "json", "manifest")}
    assert _present(env.dest) == {NAMES[r] for r in ("png", "raw_png", "json", "manifest")}


def test_events_root_off_builds_in_the_plots_dir_and_still_deletes(tmp_path, monkeypatch):
    env = ee.make_env(tmp_path, monkeypatch, events_root="")
    ee.stub_figure(monkeypatch)
    env.process()
    assert not (tmp_path / "EVENTS").exists()
    assert _present(env.plots) >= {NAMES[r] for r in ("png", "raw_png", "json", "manifest", "mask")}
    assert not (env.plots / NAMES["fil"]).exists()
    assert not list(env.dump_dir.glob("*.dada"))
    assert json.loads((env.plots / NAMES["json"]).read_text())["products"]["fil"] is None


def test_delete_dumps_off_and_known_source_dumps_are_kept(tmp_path, monkeypatch):
    env = ee.make_env(tmp_path, monkeypatch, delete_dumps=False)
    ee.stub_figure(monkeypatch)
    env.process()
    assert _dump_kept(env) and env.summary()["data_available"] is True

    env2 = ee.make_env(tmp_path / "ks", monkeypatch)
    env2.card_path.write_text(json.dumps(dict(env2.card, trigger_reason="known_source_J0000")))
    env2.process()
    assert _dump_kept(env2) and env2.summary()["data_available"] is True


def test_reprocessing_a_card_does_not_keep_a_stale_mask(tmp_path, monkeypatch):
    env = ee.make_env(tmp_path, monkeypatch, delete_dumps=False)
    ee.stub_figure(monkeypatch)
    env.process()
    assert (env.event_dir / NAMES["mask"]).is_file()
    shutil.rmtree(env.records)
    env.records.mkdir()
    env.process()
    assert not (env.event_dir / NAMES["mask"]).exists()
    manifest = json.loads((env.event_dir / NAMES["manifest"]).read_text())
    assert NAMES["mask"] not in manifest["files"]
    assert t3_event.load_event(env.event_dir).complete


# --- shipping ------------------------------------------------------------------------------

def test_remote_ship_sends_the_manifest_in_its_own_last_scp(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(dump_plotter.subprocess, "run", lambda argv, **k: calls.append(argv))
    files = [tmp_path / "a.png", tmp_path / "a_raw.png", tmp_path / "a.json"]
    dump_plotter.ship_artifacts(files, "a", "casm-elsewhere", "/archive", final=[tmp_path / "a.manifest.json"])
    assert [c[0] for c in calls] == ["ssh", "scp", "scp"]
    assert calls[1][2:5] == [str(f) for f in files] and calls[1][-1] == "casm-elsewhere:/archive/a/"
    assert calls[2][2:] == [str(tmp_path / "a.manifest.json"), "casm-elsewhere:/archive/a/"]


def test_local_ship_keeps_the_old_two_file_form(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    for n in ("a.png", "a.json"):
        (src / n).write_text(n)
    dump_plotter.ship_artifacts([src / "a.png", src / "a.json"], "a",
                                dump_plotter.LOCAL_HOSTNAME, str(tmp_path / "arch"))
    assert _present(tmp_path / "arch" / "a") == {"a.png", "a.json"}


# --- helpers ---------------------------------------------------------------------------------

def test_sub_incoh_from_header():
    f = event_archive.sub_incoh_from_header
    assert f({"BFCORR_SUB_INCOH": "1"}) == (True, "dump_header")
    assert f({"BFCORR_SUB_INCOH": " True "}) == (True, "dump_header")
    assert f({"BFCORR_SUB_INCOH": "0"}) == (False, "dump_header")
    assert f({"BFCORR_SUB_INCOH": "FALSE"}) == (False, "dump_header")
    assert f({}) == (None, "unknown")
    assert f({"BFCORR_SUB_INCOH": "maybe"}) == (None, "unknown")


def test_source_version_never_raises(tmp_path, monkeypatch):
    import re
    assert re.fullmatch(r"[0-9a-f]{12}(\+dirty)?|unknown", event_archive.source_version())
    assert event_archive.source_version(tmp_path) == "unknown"          # not a git checkout

    def boom(*a, **k):
        raise FileNotFoundError("git")
    monkeypatch.setattr(event_archive.subprocess, "run", boom)
    assert event_archive.source_version() == "unknown"


def test_dump_span_and_hella_provenance_without_a_window(tmp_path, monkeypatch):
    se.write_dump(tmp_path / "d")
    header, data = dump_reader.read_beams(sorted((tmp_path / "d").glob("*.dada")), [0])
    span = event_archive.dump_span(header, data.shape[2], "casm-corr2")
    assert span["sample0"] == se.SAMPLE0 and span["nsamp"] == se.dump_samples()
    assert span["host"] == "casm-corr2" and span["utc_start"].endswith("+00:00")
    assert event_archive.hella_provenance(None) == {}


def test_records_are_named_as_the_writer_names_them(tmp_path):
    path = se.write_records(tmp_path)
    assert path.name == f"hmask_b{se.BEAM0:03d}_g{se.FIRST_GULP:07d}_p4242.hmr"
    assert hm.read_file(path).records[0].first_sample // se.GULP == se.FIRST_GULP


def test_a_file_the_hella_writer_produced_is_found_and_read(tmp_path):
    """tests/data holds a 2-gulp, 64-beam record file written by the C++ writer."""
    import gzip
    from types import SimpleNamespace
    d = tmp_path / "2026-09-02-00:40:49"
    d.mkdir()
    src = Path(__file__).parent / "data" / "hmask_b000_g0000100_p2872845.hmr.gz"
    (d / src.name[:-3]).write_bytes(gzip.decompress(src.read_bytes()))
    header = SimpleNamespace(contiguous=True, contiguity_note="", sample0=824200 + 100,
                             raw={"UTC_START": "2026-09-02-00:40:49"},
                             freqs_mhz=484.375 - np.arange(3072) * 0.030517578125)
    win, reason = dump_plotter.lookup_mask(str(tmp_path), header, {"beam": 5}, 9000)
    assert reason is None and win.combined.shape == (3072, 9000)
    assert win.coverage.all() and len(win.gulps) == 2
    assert win.static[:472].all() and not win.static[472:].any()
    prov = event_archive.hella_provenance(win)
    assert prov["hella_binary_md5"] == "a7fb5259dc7849efea5786d617c2bcd8"
    assert prov["hella_flag"]["flag2"] == 472 and win.headers[0].pid == 2872845
    # a window before the file's first gulp is not covered
    header.sample0 = 824200 - 20000
    none, why = dump_plotter.lookup_mask(str(tmp_path), header, {"beam": 5}, 9000)
    assert why == "no hella record covers the dump"
