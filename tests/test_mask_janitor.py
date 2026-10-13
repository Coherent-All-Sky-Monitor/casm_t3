"""t3-janitor hella_masks target: age, quota, pins, orphans, dry run (local tmp dirs, no ssh)."""
import argparse
import json
import os
import sys
import time

import pytest

from casm_t3.apps import janitor

HOST = "testhost"
NOW = 1_800_000_000.0
OBS = "2026-10-13-18:22:38"


def _args(root, spool, **kw):
    ns = dict(mask_root=str(root), spool=str(spool), mask_max_age_h=24.0,
              mask_max_gb=100.0, mask_pin_max_age_h=72.0, dry_run=False)
    ns.update(kw)
    return argparse.Namespace(**ns)


def _chunk(root, obs, gulp, age_h, beam0=0, size=1000, part=False, pid=None):
    d = root / obs
    d.mkdir(parents=True, exist_ok=True)
    tag = "" if pid is None else f"_p{pid}"
    f = d / f"hmask_b{beam0:03d}_g{gulp:07d}{tag}.hmr{'.part' if part else ''}"
    f.write_bytes(b"x" * size)
    os.utime(f, (NOW - age_h * 3600, NOW - age_h * 3600))
    return f


def _age_dir(path, age_h):
    os.utime(path, (NOW - age_h * 3600, NOW - age_h * 3600))


def _card(spool, name, suffix=".json", age_h=0.0, **fields):
    spool.mkdir(parents=True, exist_ok=True)
    f = spool / f"{name}{suffix}"
    f.write_text(json.dumps({"candname": name, **fields}))
    os.utime(f, (NOW - age_h * 3600, NOW - age_h * 3600))
    return f


def _gulp_card(spool, name, gulp, utc_start=OBS, **kw):
    return _card(spool, name, gulp={"gulp": gulp, "utc_start": utc_start},
                 samp=gulp * 8192, **kw)


def _sweep(args):
    janitor.sweep_masks(args, HOST, hosts=(HOST,), now=NOW)


def _names(root):
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def test_age_rule_deletes_only_old_closed_chunks(tmp_path):
    root, spool = tmp_path / "masks", tmp_path / "spool"
    spool.mkdir()
    _chunk(root, OBS, 0, age_h=30)                  # old, closed: goes
    _chunk(root, OBS, 420, age_h=23)                # young, closed: stays
    _chunk(root, OBS, 840, age_h=30, part=True)     # open, under twice the limit: stays
    _sweep(_args(root, spool))
    assert _names(root) == [f"{OBS}/hmask_b000_g0000420.hmr",
                            f"{OBS}/hmask_b000_g0000840.hmr.part"]


def test_empty_observation_dir_is_removed_and_nonempty_kept(tmp_path):
    root, spool = tmp_path / "masks", tmp_path / "spool"
    spool.mkdir()
    old, keep = "2026-10-10-00:00:00", "2026-10-13-00:00:00"
    _chunk(root, old, 0, age_h=50)
    _chunk(root, keep, 0, age_h=1)
    _age_dir(root / old, 50)
    _age_dir(root / keep, 1)
    _sweep(_args(root, spool))
    assert not (root / old).exists()
    assert (root / keep / "hmask_b000_g0000000.hmr").is_file()
    assert root.is_dir()


def test_dir_changed_within_grace_is_not_removed(tmp_path):
    root, spool = tmp_path / "masks", tmp_path / "spool"
    spool.mkdir()
    _chunk(root, OBS, 0, age_h=30)
    _age_dir(root / OBS, 0.0)       # hella just touched the dir
    _sweep(_args(root, spool))
    assert (root / OBS).is_dir() and _names(root) == []


def test_quota_deletes_oldest_closed_first_and_counts_open_chunks(tmp_path):
    root, spool = tmp_path / "masks", tmp_path / "spool"
    spool.mkdir()
    for i, age in enumerate((5, 4, 3, 2)):          # oldest first
        _chunk(root, OBS, i * 420, age_h=age)
    _chunk(root, OBS, 4 * 420, age_h=0.1, part=True)
    # 5 x 1000 B on disk, quota 3000 B: the two oldest closed chunks go
    _sweep(_args(root, spool, mask_max_gb=3000 / 1e9))
    assert _names(root) == [f"{OBS}/hmask_b000_g0000840.hmr",
                            f"{OBS}/hmask_b000_g0001260.hmr",
                            f"{OBS}/hmask_b000_g0001680.hmr.part"]


def test_quota_credits_chunks_already_deleted_by_age(tmp_path):
    root, spool = tmp_path / "masks", tmp_path / "spool"
    spool.mkdir()
    _chunk(root, OBS, 0, age_h=40)                  # age rule
    _chunk(root, OBS, 420, age_h=3)
    _chunk(root, OBS, 840, age_h=2)
    # 3000 B on disk, quota 2000 B: the age deletion already brings it to 2000 B
    _sweep(_args(root, spool, mask_max_gb=2000 / 1e9))
    assert _names(root) == [f"{OBS}/hmask_b000_g0000420.hmr",
                            f"{OBS}/hmask_b000_g0000840.hmr"]


def test_pending_card_pins_its_chunk_and_both_neighbours(tmp_path):
    root, spool = tmp_path / "masks", tmp_path / "spool"
    for i in range(5):
        _chunk(root, OBS, i * 420, age_h=40 + i)
    _gulp_card(spool, "261013aaaaaa", gulp=900)     # chunk g840; neighbours g420, g1260
    _card(spool, "261013bbbbbb", suffix=".json.done", gulp={"gulp": 0, "utc_start": OBS})
    _card(spool, "261013cccccc", suffix=".json.failed", gulp={"gulp": 0, "utc_start": OBS})
    _sweep(_args(root, spool))
    assert _names(root) == [f"{OBS}/hmask_b000_g0000420.hmr",
                            f"{OBS}/hmask_b000_g0000840.hmr",
                            f"{OBS}/hmask_b000_g0001260.hmr"]


def test_pin_applies_to_every_stream_in_the_observation(tmp_path):
    root, spool = tmp_path / "masks", tmp_path / "spool"
    for beam0 in (0, 64):
        for i in range(4):
            _chunk(root, OBS, i * 420, age_h=40, beam0=beam0)
    _gulp_card(spool, "261013aaaaaa", gulp=10)       # first chunk; neighbour g420
    _sweep(_args(root, spool))
    assert _names(root) == [f"{OBS}/hmask_b000_g0000000.hmr", f"{OBS}/hmask_b000_g0000420.hmr",
                            f"{OBS}/hmask_b064_g0000000.hmr", f"{OBS}/hmask_b064_g0000420.hmr"]


def test_pin_ignores_other_observations(tmp_path):
    root, spool = tmp_path / "masks", tmp_path / "spool"
    other = "2026-10-12-00:00:00"
    _chunk(root, other, 0, age_h=40)
    _chunk(root, OBS, 0, age_h=40)
    _gulp_card(spool, "261013aaaaaa", gulp=0)
    _sweep(_args(root, spool))
    assert _names(root) == [f"{OBS}/hmask_b000_g0000000.hmr"]


def test_card_without_gulp_block_pins_by_dump_time_and_peak_sample(tmp_path):
    root, spool = tmp_path / "masks", tmp_path / "spool"
    earlier = "2026-10-12-00:00:00"
    for obs in (earlier, OBS):
        for i in range(4):
            _chunk(root, obs, i * 420, age_h=40)
    # fast-path card: no gulp block; dump at 19:44 UTC falls in the OBS observation,
    # peak sample 4696190 -> gulp (4696190 + 960) // 8192 = 573, chunk g420
    _card(spool, "261013fastff", samp=4696190, dump_utc_start="2026-10-13-19:44:40.312")
    _sweep(_args(root, spool))
    assert _names(root) == [f"{OBS}/hmask_b000_g0000000.hmr", f"{OBS}/hmask_b000_g0000420.hmr",
                            f"{OBS}/hmask_b000_g0000840.hmr"]


def test_card_with_unknown_observation_or_unreadable_pins_everything(tmp_path):
    root, spool = tmp_path / "masks", tmp_path / "spool"
    _chunk(root, OBS, 0, age_h=40)
    _chunk(root, "2026-10-12-00:00:00", 0, age_h=40)
    _card(spool, "261013nonone")                     # no gulp block, no dump time
    _sweep(_args(root, spool))
    assert len(_names(root)) == 2
    (spool / "261013nonone.json").unlink()
    bad = _card(spool, "261013badbad")
    bad.write_text("{not json")
    os.utime(bad, (NOW, NOW))
    _sweep(_args(root, spool))
    assert len(_names(root)) == 2


def test_card_with_gulp_block_but_no_gulp_number_pins_its_observation(tmp_path):
    root, spool = tmp_path / "masks", tmp_path / "spool"
    other = "2026-10-12-00:00:00"
    _chunk(root, OBS, 0, age_h=40)
    _chunk(root, OBS, 420, age_h=40)
    _chunk(root, other, 0, age_h=40)
    _card(spool, "261013nogulp", gulp={"utc_start": OBS})
    _sweep(_args(root, spool))
    assert _names(root) == [f"{OBS}/hmask_b000_g0000000.hmr", f"{OBS}/hmask_b000_g0000420.hmr"]


def test_stuck_card_stops_pinning_after_the_pin_age(tmp_path):
    root, spool = tmp_path / "masks", tmp_path / "spool"
    _chunk(root, OBS, 0, age_h=40)
    _gulp_card(spool, "261013aaaaaa", gulp=0, age_h=80)
    _sweep(_args(root, spool))
    assert _names(root) == []


def test_pins_hold_against_the_quota_too(tmp_path):
    root, spool = tmp_path / "masks", tmp_path / "spool"
    for i, age in enumerate((5, 4, 3, 2)):
        _chunk(root, OBS, i * 420, age_h=age)
    _gulp_card(spool, "261013aaaaaa", gulp=0)        # pins g0 and g420
    _sweep(_args(root, spool, mask_max_gb=1000 / 1e9))
    assert _names(root) == [f"{OBS}/hmask_b000_g0000000.hmr", f"{OBS}/hmask_b000_g0000420.hmr"]


def test_dry_run_deletes_nothing(tmp_path, caplog):
    root, spool = tmp_path / "masks", tmp_path / "spool"
    spool.mkdir()
    _chunk(root, OBS, 0, age_h=40)
    _chunk(root, OBS, 420, age_h=39)
    _age_dir(root / OBS, 39)
    before = _names(root)
    with caplog.at_level("INFO", logger="t3.janitor"):
        _sweep(_args(root, spool, dry_run=True))
    assert _names(root) == before
    assert (root / OBS).is_dir()
    assert "[DRY RUN]" in caplog.text and "delete" in caplog.text


def test_missing_root_lists_nothing_and_logs_the_empty_patrol(tmp_path, caplog):
    spool = tmp_path / "spool"
    spool.mkdir()
    with caplog.at_level("INFO", logger="t3.janitor"):
        _sweep(_args(tmp_path / "absent", spool))
    assert "0 chunks" in caplog.text


def test_failed_spool_scan_skips_the_host(tmp_path, monkeypatch):
    root, spool = tmp_path / "masks", tmp_path / "spool"
    _chunk(root, OBS, 0, age_h=40)

    def boom(*a, **k):
        raise OSError("ssh down")
    monkeypatch.setattr(janitor, "list_pending_cards", boom)
    _sweep(_args(root, spool))
    assert len(_names(root)) == 1


def test_spool_scan_failure_status_is_an_error(tmp_path):
    with pytest.raises(OSError):
        janitor._run_on(HOST, HOST, "exit 3", check=True)


def test_pid_suffix_is_parsed_and_old_names_still_are(tmp_path):
    root = tmp_path / "masks"
    _chunk(root, OBS, 420, age_h=1, beam0=64, pid=4242)
    _chunk(root, OBS, 840, age_h=1, beam0=64, pid=17, part=True)
    _chunk(root, OBS, 0, age_h=1)                          # no _p<pid>
    chunks, _ = janitor.list_mask_chunks(HOST, str(root), HOST)
    got = sorted((c.beam0, c.first_gulp, c.pid, c.closed) for c in chunks)
    assert got == [(0, 0, 0, True), (64, 420, 4242, True), (64, 840, 17, False)]


def test_chunks_with_a_pid_suffix_follow_the_age_rule_and_pins(tmp_path):
    root, spool = tmp_path / "masks", tmp_path / "spool"
    for i in range(4):
        _chunk(root, OBS, i * 420, age_h=40, pid=4242)
    _gulp_card(spool, "261013aaaaaa", gulp=500)         # chunk g420, neighbours g0 and g840
    _sweep(_args(root, spool))
    assert _names(root) == [f"{OBS}/hmask_b000_g0000000_p4242.hmr",
                            f"{OBS}/hmask_b000_g0000420_p4242.hmr",
                            f"{OBS}/hmask_b000_g0000840_p4242.hmr"]


def test_two_chunks_starting_at_one_gulp_are_both_pinned(tmp_path):
    # a hella restart at gulp 420: the old and the new process each opened a chunk there
    chunks = [janitor.MaskChunk(HOST, f"/m/{OBS}/p{pid}_g{g}", 1, 0.0, True, OBS, 0, g, pid)
              for g, pid in ((0, 7), (420, 7), (420, 9), (840, 9))]
    got = janitor.pinned_chunks(chunks, [_pending(gulp=430)])
    assert got == {f"/m/{OBS}/p7_g420", f"/m/{OBS}/p9_g420", f"/m/{OBS}/p9_g840"}


def test_orphaned_open_chunk_is_deleted_with_a_warning(tmp_path, caplog):
    root, spool = tmp_path / "masks", tmp_path / "spool"
    spool.mkdir()
    _chunk(root, OBS, 0, age_h=50, part=True, pid=1)     # untouched over 2 x 24 h: orphan
    _chunk(root, OBS, 420, age_h=47, part=True, pid=2)   # a live writer appends every gulp
    _chunk(root, OBS, 840, age_h=0.1, part=True, pid=3)
    with caplog.at_level("WARNING", logger="t3.janitor"):
        _sweep(_args(root, spool))
    assert _names(root) == [f"{OBS}/hmask_b000_g0000420_p2.hmr.part",
                            f"{OBS}/hmask_b000_g0000840_p3.hmr.part"]
    assert "orphaned open chunk" in caplog.text and "g0000000_p1.hmr.part" in caplog.text


def test_orphan_limit_scales_with_the_age_limit(tmp_path):
    root, spool = tmp_path / "masks", tmp_path / "spool"
    spool.mkdir()
    _chunk(root, OBS, 0, age_h=5, part=True)
    _chunk(root, OBS, 420, age_h=3, part=True)
    _sweep(_args(root, spool, mask_max_age_h=2.0))       # orphan after 4 h
    assert _names(root) == [f"{OBS}/hmask_b000_g0000420.hmr.part"]


def test_orphan_pinned_by_a_pending_card_stays(tmp_path, caplog):
    root, spool = tmp_path / "masks", tmp_path / "spool"
    _chunk(root, OBS, 0, age_h=50, part=True)
    _gulp_card(spool, "261013aaaaaa", gulp=0, age_h=50)
    with caplog.at_level("WARNING", logger="t3.janitor"):
        _sweep(_args(root, spool))
    assert len(_names(root)) == 1 and "is pinned" in caplog.text


def test_orphan_dry_run_deletes_nothing_and_quota_credits_it(tmp_path, caplog):
    root, spool = tmp_path / "masks", tmp_path / "spool"
    spool.mkdir()
    _chunk(root, OBS, 0, age_h=50, part=True)
    with caplog.at_level("INFO", logger="t3.janitor"):
        _sweep(_args(root, spool, dry_run=True))
    assert len(_names(root)) == 1 and "[DRY RUN]" in caplog.text
    # live run: the orphan's bytes count against the quota before closed chunks go
    _chunk(root, OBS, 420, age_h=3)
    _sweep(_args(root, spool, mask_max_gb=1000 / 1e9))
    assert _names(root) == [f"{OBS}/hmask_b000_g0000420.hmr"]


def test_listing_ignores_files_that_are_not_chunks(tmp_path):
    root = tmp_path / "masks"
    _chunk(root, OBS, 0, age_h=1)
    _chunk(root, OBS, 420, age_h=1, part=True)
    (root / OBS / "hmask_b000_g0000840.hmr.tmp").write_bytes(b"x")
    (root / OBS / "notes.txt").write_bytes(b"x")
    (root / "stray.hmr").write_bytes(b"x")
    chunks, dirs = janitor.list_mask_chunks(HOST, str(root), HOST)
    assert sorted((c.first_gulp, c.closed) for c in chunks) == [(0, True), (420, False)]
    assert list(dirs) == [OBS]
    assert {c.obs for c in chunks} == {OBS} and {c.beam0 for c in chunks} == {0}


def test_pending_card_listing_reads_bare_json_only(tmp_path):
    spool = tmp_path / "spool"
    _gulp_card(spool, "261013aaaaaa", gulp=573)
    _card(spool, "261013bbbbbb", suffix=".json.done")
    (spool / "261013cccccc.json.tmp").write_text("{}")
    cards = janitor.list_pending_cards(HOST, str(spool), HOST)
    assert [(c.name, c.gulp, c.utc_start, c.bad) for c in cards] == [
        ("261013aaaaaa.json", 573, OBS, False)]
    assert janitor.list_pending_cards(HOST, str(tmp_path / "none"), HOST) == []


def _chunks_for(gulps, beam0=0, obs=OBS):
    return [janitor.MaskChunk(HOST, f"/m/{obs}/b{beam0}_g{g}", 1, 0.0, True, obs, beam0, g)
            for g in gulps]


def _pending(**kw):
    base = dict(host=HOST, name="c.json", mtime=0.0, utc_start=OBS, gulp=None,
                samp=None, dump_utc_start=None)
    base.update(kw)
    return janitor.PendingCard(**base)


def test_pinned_chunks_boundaries():
    chunks = _chunks_for([0, 420, 840, 1260])
    path = lambda g: f"/m/{OBS}/b0_g{g}"  # noqa: E731
    assert janitor.pinned_chunks(chunks, [_pending(gulp=419)]) == {path(0), path(420)}
    assert janitor.pinned_chunks(chunks, [_pending(gulp=420)]) == {path(0), path(420), path(840)}
    assert janitor.pinned_chunks(chunks, [_pending(gulp=99999)]) == {path(840), path(1260)}
    assert janitor.pinned_chunks(chunks, []) == set()


def test_cli_defaults_leave_dump_behaviour_and_set_mask_defaults(monkeypatch):
    seen = {}

    def grab(self, *a, **k):
        seen["p"] = self
        raise SystemExit(0)
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", grab)
    with pytest.raises(SystemExit):
        janitor.main()
    d = {a.dest: a.default for a in seen["p"]._actions}
    assert d["targets"] == "dumps"
    assert d["max_gb"] == 150.0 and d["max_age_days"] == 7.0
    assert d["mask_root"] == "/mnt/nvme4/data/casm/hella_masks"
    assert d["mask_max_age_h"] == 24.0 and d["mask_max_gb"] == 100.0
    assert d["spool"] == "/mnt/nvme4/data/casm/t2_spool"


def test_masks_target_does_not_open_the_t2_database(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(janitor.t2db, "connect",
                        lambda *a: calls.append("db") or pytest.fail("db opened"))
    monkeypatch.setattr(janitor, "sweep", lambda *a: calls.append("dumps"))
    monkeypatch.setattr(janitor, "sweep_masks", lambda *a, **k: calls.append("masks"))
    monkeypatch.setattr(sys, "argv", ["t3-janitor", "--targets", "hella_masks", "--once"])
    janitor.main()
    assert calls == ["masks"]


def test_default_target_runs_dumps_only(monkeypatch):
    calls = []
    monkeypatch.setattr(janitor.t2db, "connect", lambda *a: object())
    monkeypatch.setattr(janitor, "sweep", lambda *a: calls.append("dumps"))
    monkeypatch.setattr(janitor, "sweep_masks", lambda *a, **k: calls.append("masks"))
    monkeypatch.setattr(sys, "argv", ["t3-janitor", "--once"])
    janitor.main()
    assert calls == ["dumps"]


def test_unknown_target_is_rejected(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["t3-janitor", "--targets", "hella_masks,bogus", "--once"])
    with pytest.raises(SystemExit) as exc:
        janitor.main()
    assert exc.value.code == 2


def test_sweep_time_defaults_to_now(tmp_path):
    root, spool = tmp_path / "masks", tmp_path / "spool"
    spool.mkdir()
    f = root / OBS / "hmask_b000_g0000000.hmr"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"x")
    old = time.time() - 30 * 3600
    os.utime(f, (old, old))
    janitor.sweep_masks(_args(root, spool), HOST, hosts=(HOST,))
    assert not f.exists()


def _fake_ssh(monkeypatch, calls, rc=None):
    """Route `ssh <host> <cmd>` to a local bash so the remote branch runs offline."""
    real = janitor.subprocess.run

    def run(argv, *a, **k):
        if argv[0] == "ssh":
            calls.append(argv)
            if rc is not None:
                return janitor.subprocess.CompletedProcess(argv, rc, "", "unreachable")
            argv = ["bash", "-c", argv[2]]
        return real(argv, *a, **k)
    monkeypatch.setattr(janitor.subprocess, "run", run)


def test_remote_host_is_listed_and_swept_over_ssh(tmp_path, monkeypatch):
    root, spool = tmp_path / "masks", tmp_path / "spool"
    _chunk(root, OBS, 0, age_h=40)
    _chunk(root, OBS, 420, age_h=40)
    _chunk(root, OBS, 840, age_h=40, part=True)
    _gulp_card(spool, "261013aaaaaa", gulp=0)          # pins g0 and g420
    _chunk(root, "2026-10-10-00:00:00", 0, age_h=50)
    _age_dir(root / "2026-10-10-00:00:00", 50)
    calls = []
    _fake_ssh(monkeypatch, calls)
    janitor.sweep_masks(_args(root, spool), "other-host", hosts=("casm-corr2",), now=NOW)
    assert _names(root) == [f"{OBS}/hmask_b000_g0000000.hmr", f"{OBS}/hmask_b000_g0000420.hmr",
                            f"{OBS}/hmask_b000_g0000840.hmr.part"]
    assert not (root / "2026-10-10-00:00:00").exists()
    assert all(c[1] == "casm-corr2" for c in calls)
    assert any("rm -f --" in c[2] for c in calls) and any("rmdir --" in c[2] for c in calls)


def test_unreachable_remote_host_loses_nothing(tmp_path, monkeypatch):
    root, spool = tmp_path / "masks", tmp_path / "spool"
    _chunk(root, OBS, 0, age_h=40)
    _fake_ssh(monkeypatch, [], rc=255)
    janitor.sweep_masks(_args(root, spool), "other-host", hosts=("casm-corr2",), now=NOW)
    assert len(_names(root)) == 1
