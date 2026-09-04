"""weights_watch: log parsing and the upload / defaults / unknown decision.

The log lines quoted here are verbatim from /data/casm/logs on corr1 around the
2026-09-02 21:07 UTC and 2026-09-04 16:41 UTC obs restarts.
"""
import json
from datetime import datetime, timedelta, timezone

from casm_t2 import weights_registry as wr
from casm_t3.apps import weights_watch as ww


def _reg(tmp_path):
    reg = wr.Registry(tmp_path)
    alt = [30.0] * 512; az = [10.0] * 512
    reg.record_product(h5_path="/a.h5", stream_md5={s: f"a{s}" for s in range(6)}, alt_deg=alt, az_deg=az)
    reg.record_product(h5_path="/b.h5", stream_md5={s: f"b{s}" for s in range(6)}, alt_deg=alt, az_deg=az)
    return reg


def _alert_kinds(reg):
    if not reg.alerts_path.exists():
        return []
    return [json.loads(l)["kind"] for l in open(reg.alerts_path)]


def test_parse_real_lines():
    s, t, n = ww.parse_transfer("4 [2026-09-02 14:08:04.177] [info] Transferred 67108864 bytes, finishing")
    assert (s, n) == (4, 67108864) and t == datetime(2026, 9, 2, 21, 8, 4, 177000, tzinfo=timezone.utc)
    s, t = ww.parse_start("3 [2026-09-02-14:08:04.205] START casm_bfcorr -a 64 -f 512 -i a006 -m corr")
    assert s == 3 and t == datetime(2026, 9, 2, 21, 8, 4, 205000, tzinfo=timezone.utc)
    assert ww.parse_transfer("4 [2026-09-02 14:08:04.177] [info] Transferring 67108864 bytes") is None
    assert ww.parse_start("2 [2026-09-04 09:41:54.357] [info] Read input header: UTC_START=2026-09-04-16:42:39") is None


def test_parse_is_dst_aware():
    """PDT in September is UTC+7, PST in January is UTC+8; no fixed offset."""
    _, t, _ = ww.parse_transfer("1 [2026-09-04 09:41:40.041] [info] Transferred 67108864 bytes, finishing")
    assert t == datetime(2026, 9, 4, 16, 41, 40, 41000, tzinfo=timezone.utc)
    _, t = ww.parse_start("1 [2026-09-04-09:41:40.133] START casm_bfcorr -a 64 -f 512 -i a002 -m corr")
    assert t == datetime(2026, 9, 4, 16, 41, 40, 133000, tzinfo=timezone.utc)
    _, t = ww.parse_start("1 [2026-01-15-09:41:40.133] START casm_bfcorr -a 64 -f 512 -i a002 -m corr")
    assert t == datetime(2026, 1, 15, 17, 41, 40, 133000, tzinfo=timezone.utc)


def test_restart_transfer_precedes_start(tmp_path, monkeypatch):
    """The 2026-09-04 defect: the FIFO push finishes 92 ms BEFORE the START line,
    so a START-then-transfer test filed all six streams as source=unknown."""
    reg = _reg(tmp_path)
    w = ww.Watcher(reg, alert=False)
    monkeypatch.setattr(ww, "defaults_payload_md5", lambda stream: f"a{stream}")
    _, t_tr, n = ww.parse_transfer("1 [2026-09-04 09:41:40.041] [info] Transferred 67108864 bytes, finishing")
    _, t_st = ww.parse_start("1 [2026-09-04-09:41:40.133] START casm_bfcorr -a 64 -f 512 -i a002 -m corr")
    w.note_start(1, t_st)
    ev = w.handle_transfer(1, t_tr, n)
    assert ev["source"] == "defaults" and ev["product_id"] == reg.lookup_payload("a1")
    assert "bfcorr START within 120 s" in ev["evidence"]
    assert reg.product_at(t_tr + timedelta(seconds=1))[0]["h5_path"] == "/a.h5"


def test_defaults_reload_is_identified_and_revert_alerted(tmp_path, monkeypatch):
    reg = _reg(tmp_path)
    t0 = datetime(2026, 9, 2, 20, 0, tzinfo=timezone.utc)
    for s in range(6):
        reg.record_live_event(utc=t0, stream=s, payload_md5=f"b{s}", source="upload")   # newest upload = b
    w = ww.Watcher(reg, alert=False)
    monkeypatch.setattr(ww, "defaults_payload_md5", lambda stream: f"a{stream}")   # defaults on disk = a
    t1 = t0 + timedelta(minutes=5)
    w.note_start(2, t1)
    ev = w.handle_transfer(2, t1 + timedelta(seconds=3), ww.CB_BYTES)
    assert ev["source"] == "defaults" and ev["product_id"] == reg.lookup_payload("a2")
    assert "reverted_to_defaults" in _alert_kinds(reg) and "partial_deploy" in _alert_kinds(reg)
    assert reg.product_at(t1 + timedelta(seconds=10))[1] == "partial"


def test_defaults_without_any_start(tmp_path, monkeypatch):
    """casm-lmc re-pushes the same defaults with no bfcorr START: still identified."""
    reg = _reg(tmp_path)
    w = ww.Watcher(reg, alert=False)
    monkeypatch.setattr(ww, "defaults_payload_md5", lambda stream: f"a{stream}")
    t = datetime(2026, 9, 4, 16, 41, 40, tzinfo=timezone.utc)
    ev = w.handle_transfer(3, t, ww.CB_BYTES)
    assert ev["source"] == "defaults" and ev["product_id"] == reg.lookup_payload("a3")
    assert "no START seen" in ev["evidence"]
    assert _alert_kinds(reg) == []
    assert reg.product_at(t + timedelta(seconds=1))[0]["h5_path"] == "/a.h5"


def test_unregistered_defaults_keeps_the_md5_and_alerts(tmp_path, monkeypatch):
    reg = _reg(tmp_path)
    w = ww.Watcher(reg, alert=False)
    monkeypatch.setattr(ww, "defaults_payload_md5", lambda stream: "deadbeef" * 4)
    t = datetime(2026, 9, 4, 16, 41, 40, tzinfo=timezone.utc)
    ev = w.handle_transfer(0, t, ww.CB_BYTES)
    assert ev["source"] == "unknown_defaults" and ev["payload_md5"] == "deadbeef" * 4
    assert ev["product_id"] is None
    assert _alert_kinds(reg) == ["unknown_defaults"]


def test_hash_failure_is_the_only_null_md5(tmp_path, monkeypatch):
    reg = _reg(tmp_path)
    w = ww.Watcher(reg, alert=False)
    monkeypatch.setattr(ww, "defaults_payload_md5", lambda stream: None)   # corr2 unreachable
    t = datetime(2026, 9, 4, 16, 41, 40, tzinfo=timezone.utc)
    ev = w.handle_transfer(4, t, ww.CB_BYTES)
    assert ev["payload_md5"] is None and ev["source"] == "unknown"
    assert _alert_kinds(reg) == ["defaults_hash_failed"]


def test_upload_match_and_ib_transfer_ignored(tmp_path, monkeypatch):
    reg = _reg(tmp_path)
    monkeypatch.setattr(ww, "defaults_payload_md5", lambda stream: f"a{stream}")
    t0 = datetime(2026, 9, 2, 20, 0, tzinfo=timezone.utc)
    reg.record_live_event(utc=t0, stream=1, payload_md5="a1", source="upload")
    w = ww.Watcher(reg, alert=False)
    assert w.handle_transfer(1, t0 + timedelta(seconds=4), ww.CB_BYTES) is None      # the upload itself
    assert w.handle_transfer(1, t0 + timedelta(seconds=4), 67584) is None            # IB transfer ignored
