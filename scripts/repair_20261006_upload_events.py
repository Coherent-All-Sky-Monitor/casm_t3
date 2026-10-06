"""Repair only the six independently verified October 6 upload misclassifications.

Dry-run by default. Stop t3-weights-watch before --apply. No hardware, defaults,
Slack, or product writes. Preserve the original file and every corrected row.
Remote event time is the corr1 observation upper bound, not an inferred exact
activation time. The canonical uploader's later successful events stay intact.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from casm_t2.weights_registry import Registry

PRODUCT = "a7658dba9b34376a"
TIMES = ["20:32:33.924", "20:32:34.352", "20:32:34.731",
         "20:35:05.454", "20:35:06.195", "20:35:06.963"]
PAYLOADS = ["6f193edca5fabf5bfa33a0d56bd078d2", "c7717c3e268309fcd5b5093605c6b982",
            "cf1038c051f0fd2c6bcab66be1d0bf79", "a64714a0b67917b60dc522f8b7e70fc3",
            "cb9f4a46d67b70da340962f40fe833d2", "60305fb0f4420c45986d05a95eeadc41"]
REASON = ("Verified monitor upload #1 / job 53; consumer receive/apply logs with "
          "UTC_START 2026-10-06-20:32:32; all six CB payloads independently match "
          "the reviewed stage, registry, and restart defaults. Watcher inferred "
          "defaults before uploader registration; corr2 log clock is about 150 s "
          "ahead. Remote UTC uses corr1 observation upper bound. No weights changed.")


def corrected_bytes(original: bytes, registry: Registry) -> bytes:
    rows = [json.loads(line) for line in original.splitlines()]
    matched = set()
    for row in rows:
        stream = row.get("stream")
        if stream not in range(6) or row.get("utc") != f"2026-10-06T{TIMES[stream]}+00:00":
            continue
        if row.get("source") != ("defaults" if stream < 3 else "unknown_defaults"):
            continue
        assert stream not in matched, "Duplicate target event"
        assert row["recorded_utc"].startswith("2026-10-06T20:32:"), "Unexpected observation time"
        assert row["product_id"] == ("7f00805b81f13763" if stream < 3 else None)
        assert row["evidence"].startswith("FIFO transfer matching on-disk defaults, no START seen")
        assert registry.lookup_payload(PAYLOADS[stream]) == PRODUCT, "Product/index mismatch"
        assert any(e.get("source") == "upload" and e.get("stream") == stream
                   and e.get("product_id") == PRODUCT and e.get("payload_md5") == PAYLOADS[stream]
                   and e.get("utc") == "2026-10-06T20:32:49.113+00:00" for e in rows), "Missing upload evidence"
        old = dict(row)
        row.update(original=old, correction=REASON, source="upload", product_id=PRODUCT,
                   payload_md5=PAYLOADS[stream], evidence=REASON,
                   utc=old["utc"] if stream < 3 else old["recorded_utc"],
                   time_basis="corr1_log" if stream < 3 else "corr1_observation_upper_bound")
        matched.add(stream)
    assert matched == set(range(6)), f"Expected exactly six original events, found {matched}"
    return ("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n").encode()


def repair(root: Path, apply: bool = False) -> dict:
    registry = Registry(root)
    path = registry.events_path
    original = path.read_bytes()
    replacement = corrected_bytes(original, registry)
    sha = hashlib.sha256(original).hexdigest()
    backup = path.with_name(f"live_events.before-20261006-upload-repair-{sha[:12]}.jsonl.bak")
    result = dict(changed=6, backup=str(backup), original_sha256=sha, applied=apply)
    if not apply:
        return result
    with backup.open("xb") as out:
        out.write(original)
        out.flush()
        os.fsync(out.fileno())
    fd, temporary = tempfile.mkstemp(prefix=".events-repair-", dir=root)
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(replacement)
            out.flush()
            os.fsync(out.fileno())
        os.chmod(temporary, path.stat().st_mode & 0o777)
        if path.read_bytes() != original:
            raise RuntimeError("Registry changed during repair; backup retained, original untouched")
        os.replace(temporary, path)
        registry.record_alert("events_corrected", REASON, **result)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(json.dumps(repair(args.registry, args.apply), indent=2))
