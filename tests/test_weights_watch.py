"""weights_watch: log parsing and the upload / defaults / unknown decision."""
from datetime import datetime, timedelta, timezone

from casm_t2 import weights_registry as wr
from casm_t3.apps import weights_watch as ww


def _reg(tmp_path):
    reg = wr.Registry(tmp_path)
    alt = [30.0] * 512; az = [10.0] * 512
    reg.record_product(h5_path="/a.h5", stream_md5={s: f"a{s}" for s in range(6)}, alt_deg=alt, az_deg=az)
    reg.record_product(h5_path="/b.h5", stream_md5={s: f"b{s}" for s in range(6)}, alt_deg=alt, az_deg=az)
    return reg


def test_parse_real_lines():
    s, t, n = ww.parse_transfer("4 [2026-09-02 14:08:04.177] [info] Transferred 67108864 bytes, finishing")
    assert (s, n) == (4, 67108864) and t == datetime(2026, 9, 2, 21, 8, 4, 177000, tzinfo=timezone.utc)
    s, t = ww.parse_start("3 [2026-09-02-14:08:04.205] START casm_bfcorr -a 64 -f 512 -i a006 -m corr")
    assert s == 3 and t.hour == 21
    assert ww.parse_transfer("4 [2026-09-02 14:08:04.177] [info] Transferring 67108864 bytes") is None


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
    kinds = [a["kind"] for a in map(__import__("json").loads, open(reg.alerts_path))]
    assert "reverted_to_defaults" in kinds and "partial_deploy" in kinds
    assert reg.product_at(t1 + timedelta(seconds=10))[1] == "partial"


def test_upload_match_and_unknown(tmp_path):
    reg = _reg(tmp_path)
    t0 = datetime(2026, 9, 2, 20, 0, tzinfo=timezone.utc)
    reg.record_live_event(utc=t0, stream=1, payload_md5="a1", source="upload")
    w = ww.Watcher(reg, alert=False)
    assert w.handle_transfer(1, t0 + timedelta(seconds=4), ww.CB_BYTES) is None      # the upload itself
    assert w.handle_transfer(1, t0 + timedelta(seconds=4), 67584) is None            # IB transfer ignored
    ev = w.handle_transfer(1, t0 + timedelta(minutes=10), ww.CB_BYTES)              # no upload, no restart
    assert ev["source"] == "unknown" and ev["product_id"] is None
    assert reg.product_at(t0 + timedelta(minutes=11))[1] == "unknown"
