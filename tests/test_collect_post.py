"""t3-collect posting: manifest gating, main then raw thread reply, retry, markers."""
import json
import os
import re
import sys
import time

import pytest

from casm_t3 import alerts
from casm_t3.apps import collect

NAME = "261013nnuclj"
CHANNEL = "C0TEST"
MAIN_TS = "1760380000.000100"
RAW_CAPTION = "raw intensity, before hella RFI flagging"


class Slack:
    """Stands in for alerts.post_candidate: records calls, scripted results."""

    def __init__(self):
        self.calls = []
        self.results = []          # popped per call; default is a resolved ts

    def __call__(self, png, text, channel=None, thread_ts=None):
        self.calls.append({"png": os.path.basename(str(png)), "text": text,
                           "channel": channel, "thread_ts": thread_ts})
        if self.results:
            return self.results.pop(0)
        return MAIN_TS if len(self.calls) == 1 else "1760380001.000200"


@pytest.fixture
def slack(monkeypatch):
    s = Slack()
    monkeypatch.setattr(alerts, "configured", lambda: True)
    monkeypatch.setattr(alerts, "load_channel", lambda: CHANNEL)
    monkeypatch.setattr(alerts, "post_candidate", s)
    return s


def _cand(root, name=NAME, schema=2, manifest=True, raw=True, age_h=0.0):
    d = root / name
    d.mkdir(parents=True)
    meta = {"candname": name, "snr": 20.7, "dm": 247.6,
            "event_utc": "2026-10-13T00:57:02.250+00:00"}
    if schema:
        meta["schema"] = schema
    (d / f"{name}.json").write_text(json.dumps(meta))
    (d / f"{name}.png").write_bytes(b"png-main")
    files = {f"{name}.png": {"bytes": 8, "sha256": "0" * 64}}
    if raw:
        (d / f"{name}_raw.png").write_bytes(b"png-raw!")
        files[f"{name}_raw.png"] = {"bytes": 8, "sha256": "1" * 64}
    if manifest:
        (d / f"{name}.manifest.json").write_text(json.dumps({"schema": 2, "files": files}))
    t = time.time() - age_h * 3600
    for f in d.iterdir():
        os.utime(f, (t, t))
    return d


def _post(root, events=None, max_age_h=24.0):
    collect._post_new(root, "http://localhost:8061", max_age_h, events)


def _marker(d):
    return (d / ".slack").read_text()


def test_schema2_waits_for_the_manifest(tmp_path, slack):
    d = _cand(tmp_path, manifest=False)
    _post(tmp_path)
    assert slack.calls == [] and not (d / ".slack").exists()
    (d / f"{NAME}.manifest.json").write_text(json.dumps(
        {"schema": 2, "files": {f"{NAME}.png": {"bytes": 8}, f"{NAME}_raw.png": {"bytes": 8}}}))
    _post(tmp_path)
    assert [c["png"] for c in slack.calls] == [f"{NAME}.png", f"{NAME}_raw.png"]


def test_unparsable_manifest_is_incomplete(tmp_path, slack):
    d = _cand(tmp_path)
    (d / f"{NAME}.manifest.json").write_text('{"schema": 2, "fil')
    _post(tmp_path)
    assert slack.calls == [] and not (d / ".slack").exists()


def test_manifest_size_mismatch_waits_for_the_sync(tmp_path, slack):
    d = _cand(tmp_path)
    (d / f"{NAME}.png").write_bytes(b"old")            # manifest says 8 bytes
    _post(tmp_path)
    assert slack.calls == [] and not (d / ".slack").exists()


def test_legacy_dir_posts_once_with_the_old_marker_and_no_raw(tmp_path, slack):
    d = _cand(tmp_path, schema=0, manifest=False, raw=True)
    _post(tmp_path)
    _post(tmp_path)
    assert [c["png"] for c in slack.calls] == [f"{NAME}.png"]
    assert slack.calls[0]["thread_ts"] is None and slack.calls[0]["channel"] is None
    assert re.fullmatch(rf"posted \S+ ts={re.escape(MAIN_TS)}\n", _marker(d))


def test_legacy_marker_never_triggers_a_raw_post(tmp_path, slack):
    d = _cand(tmp_path, schema=0, manifest=False, raw=True)
    (d / ".slack").write_text("posted 2026-09-30T00:00:00+00:00 ts=1.2\n")
    _post(tmp_path)
    assert slack.calls == []


def test_main_then_raw_as_thread_reply(tmp_path, slack):
    d = _cand(tmp_path)
    _post(tmp_path)
    main, raw = slack.calls
    assert main["png"] == f"{NAME}.png" and main["thread_ts"] is None
    assert f"<http://localhost:8061/cands/{NAME}|{NAME}>" in main["text"]
    assert raw["png"] == f"{NAME}_raw.png" and raw["thread_ts"] == MAIN_TS
    assert raw["text"] == RAW_CAPTION and raw["channel"] == CHANNEL
    lines = _marker(d).splitlines()
    assert re.fullmatch(rf"posted \S+ ts={re.escape(MAIN_TS)} channel={CHANNEL} kind=main", lines[0])
    assert re.fullmatch(rf"posted \S+ ts=\S+ channel={CHANNEL} kind=raw", lines[1])
    _post(tmp_path)
    assert len(slack.calls) == 2


def test_raw_failure_retries_without_reposting_main(tmp_path, slack):
    d = _cand(tmp_path)
    slack.results = [MAIN_TS, False]                   # raw upload fails
    _post(tmp_path)
    assert [c["png"] for c in slack.calls] == [f"{NAME}.png", f"{NAME}_raw.png"]
    assert _marker(d).count("kind=") == 1 and "kind=main" in _marker(d)
    _post(tmp_path)                                    # next cycle: only the raw
    assert [c["png"] for c in slack.calls] == [f"{NAME}.png", f"{NAME}_raw.png", f"{NAME}_raw.png"]
    assert slack.calls[2]["thread_ts"] == MAIN_TS
    assert "kind=raw" in _marker(d)
    _post(tmp_path)
    assert len(slack.calls) == 3


def test_main_failure_leaves_no_marker_and_skips_raw(tmp_path, slack):
    d = _cand(tmp_path)
    slack.results = [False]
    _post(tmp_path)
    assert [c["png"] for c in slack.calls] == [f"{NAME}.png"]
    assert not (d / ".slack").exists()
    _post(tmp_path)                                    # retried whole next cycle
    assert [c["png"] for c in slack.calls][1:] == [f"{NAME}.png", f"{NAME}_raw.png"]


def test_unknown_main_ts_posts_raw_unthreaded_with_the_candidate_named(tmp_path, slack):
    d = _cand(tmp_path)
    slack.results = [True, "1760380002.000300"]        # main ok but ts unresolved
    _post(tmp_path)
    raw = slack.calls[1]
    assert raw["thread_ts"] is None
    assert NAME in raw["text"] and "raw intensity" in raw["text"]
    lines = _marker(d).splitlines()
    assert "ts=unknown" in lines[0] and "kind=main" in lines[0]
    assert lines[1].endswith("kind=raw_unthreaded")
    _post(tmp_path)
    assert len(slack.calls) == 2


def test_unknown_ts_read_back_from_the_marker_on_a_later_cycle(tmp_path, slack):
    d = _cand(tmp_path)
    slack.results = [True, False, "1760380002.000300"]
    _post(tmp_path)
    assert "kind=raw" not in _marker(d)
    _post(tmp_path)
    assert slack.calls[2]["thread_ts"] is None
    assert _marker(d).splitlines()[1].endswith("kind=raw_unthreaded")


def test_raw_that_arrives_after_the_main_post_is_threaded_later(tmp_path, slack):
    d = _cand(tmp_path, raw=False)
    _post(tmp_path)
    assert [c["png"] for c in slack.calls] == [f"{NAME}.png"]
    (d / f"{NAME}_raw.png").write_bytes(b"png-raw!")
    _post(tmp_path)
    assert slack.calls[1]["png"] == f"{NAME}_raw.png" and slack.calls[1]["thread_ts"] == MAIN_TS


def test_marker_is_mirrored_with_full_content(tmp_path, slack):
    cands, events = tmp_path / "cands", tmp_path / "events"
    cands.mkdir()
    (events / NAME).mkdir(parents=True)
    d = _cand(cands)
    slack.results = [MAIN_TS, False]
    _post(cands, events)
    assert (events / NAME / ".slack").read_text() == _marker(d)
    assert _marker(d).count("kind=") == 1
    _post(cands, events)
    assert _marker(d).count("kind=") == 2
    assert (events / NAME / ".slack").read_text() == _marker(d)


def test_missing_event_dir_is_not_created_by_mirroring(tmp_path, slack):
    cands, events = tmp_path / "cands", tmp_path / "events"
    cands.mkdir()
    events.mkdir()
    _cand(cands)
    _post(cands, events)
    assert not (events / NAME).exists()


def test_candidate_json_is_never_rewritten(tmp_path, slack):
    d = _cand(tmp_path)
    path = d / f"{NAME}.json"
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    _post(tmp_path)
    _post(tmp_path)
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


@pytest.mark.parametrize("schema", [0, 2])
def test_stale_candidate_is_marked_and_not_posted(tmp_path, slack, schema):
    d = _cand(tmp_path, schema=schema, manifest=bool(schema), age_h=30)
    _post(tmp_path)
    assert slack.calls == []
    assert re.fullmatch(r"skipped stale \S+\n", _marker(d))


def test_raw_stale_after_main_is_given_up(tmp_path, slack):
    d = _cand(tmp_path)
    (d / ".slack").write_text(f"posted 2026-10-12T00:00:00+00:00 ts={MAIN_TS} "
                              f"channel={CHANNEL} kind=main\n")
    old = time.time() - 30 * 3600
    os.utime(d / f"{NAME}_raw.png", (old, old))
    _post(tmp_path)
    assert slack.calls == []
    assert _marker(d).splitlines()[-1].startswith("skipped stale")
    _post(tmp_path)
    assert slack.calls == []


def test_unconfigured_slack_leaves_everything_unmarked(tmp_path, slack, monkeypatch):
    monkeypatch.setattr(alerts, "configured", lambda: False)
    d = _cand(tmp_path)
    _post(tmp_path)
    assert slack.calls == [] and not (d / ".slack").exists()


def test_raw_reply_uses_the_channel_of_the_main_post(tmp_path, slack):
    d = _cand(tmp_path)
    (d / ".slack").write_text(f"posted 2026-10-13T00:00:00+00:00 ts={MAIN_TS} "
                              "channel=COTHER kind=main\n")
    _post(tmp_path)
    assert slack.calls[0]["channel"] == "COTHER" and slack.calls[0]["thread_ts"] == MAIN_TS


# alerts.post_candidate: thread_ts reaches files.completeUploadExternal

class _Resp:
    def __init__(self, doc):
        self._doc = doc

    def raise_for_status(self):
        pass

    def json(self):
        return self._doc


class _FakeRequests:
    def __init__(self):
        self.complete = []

    def get(self, url, headers=None, params=None, timeout=None):
        if url.endswith("files.getUploadURLExternal"):
            return _Resp({"ok": True, "upload_url": "http://upload.invalid/x", "file_id": "F1"})
        assert url.endswith("files.info")
        return _Resp({"ok": True, "file": {"shares": {"public": {CHANNEL: [{"ts": "5.5"}]}}}})

    def post(self, url, headers=None, json=None, files=None, timeout=None):
        if url.endswith("files.completeUploadExternal"):
            self.complete.append(json)
        return _Resp({"ok": True})


@pytest.fixture
def fake_requests(monkeypatch, tmp_path):
    fake = _FakeRequests()
    monkeypatch.setitem(sys.modules, "requests", fake)
    token, channel = tmp_path / "slack_api", tmp_path / "slack_channel"
    token.write_text("xoxb-test\n")
    channel.write_text(CHANNEL + "\n")
    monkeypatch.setattr(alerts, "TOKEN_PATH", token)
    monkeypatch.setattr(alerts, "CHANNEL_PATH", channel)
    return fake


def test_post_candidate_passes_thread_ts_and_returns_the_reply_ts(tmp_path, fake_requests):
    png = tmp_path / "x_raw.png"
    png.write_bytes(b"png")
    ts = alerts.post_candidate(png, "raw", thread_ts="1760380000.000100")
    assert ts == "5.5"
    body = fake_requests.complete[0]
    assert body["thread_ts"] == "1760380000.000100"
    assert body["channel_id"] == CHANNEL and body["initial_comment"] == "raw"


def test_post_candidate_without_thread_ts_is_unchanged(tmp_path, fake_requests):
    png = tmp_path / "x.png"
    png.write_bytes(b"png")
    assert alerts.post_candidate(png, "main") == "5.5"
    assert "thread_ts" not in fake_requests.complete[0]
