import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from casm_t2.weights_registry import Registry

spec = importlib.util.spec_from_file_location("repair", Path(__file__).parents[1] / "scripts/repair_20261006_upload_events.py")
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


def test_targeted_repair_preserves_originals_and_backup(tmp_path, monkeypatch):
    monkeypatch.setattr(Registry, "lookup_payload", lambda self, md5: repair.PRODUCT)
    rows = []
    for s in range(6):
        rows.append(dict(utc=f"2026-10-06T{repair.TIMES[s]}+00:00", stream=s,
                         product_id="7f00805b81f13763" if s < 3 else None,
                         payload_md5="old" if s < 3 else repair.PAYLOADS[s],
                         source="defaults" if s < 3 else "unknown_defaults",
                         recorded_utc="2026-10-06T20:32:42.510+00:00",
                         evidence="FIFO transfer matching on-disk defaults, no START seen"))
    for s in range(6):
        rows.append(dict(utc="2026-10-06T20:32:49.113+00:00", stream=s,
                         product_id=repair.PRODUCT, payload_md5=repair.PAYLOADS[s], source="upload"))
    path = tmp_path / "live_events.jsonl"
    original = ("\n".join(json.dumps(r) for r in rows) + "\n").encode()
    path.write_bytes(original)
    assert not repair.repair(tmp_path)["applied"]
    assert path.read_bytes() == original
    result = repair.repair(tmp_path, True)
    assert Path(result["backup"]).read_bytes() == original
    assert result["original_sha256"] == hashlib.sha256(original).hexdigest()
    fixed = [json.loads(l) for l in path.read_bytes().splitlines()]
    assert fixed[6:] == rows[6:]
    for s, row in enumerate(fixed[:6]):
        assert row["original"] == rows[s]
        assert row["payload_md5"] == repair.PAYLOADS[s]
        assert row["source"] == "upload"
        assert row["utc"] < fixed[-1]["utc"]
    assert {ev["product_id"] for ev in Registry(tmp_path).live_at("2026-10-06T22:00:00+00:00").values()} == {repair.PRODUCT}
    with pytest.raises(AssertionError, match="Expected exactly six"):
        repair.repair(tmp_path, True)


def test_wrong_or_missing_evidence_refused(tmp_path):
    with pytest.raises(AssertionError, match="Expected exactly six"):
        repair.corrected_bytes(b'{}\n', Registry(tmp_path))
