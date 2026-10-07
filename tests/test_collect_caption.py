"""Slack caption link: candidate posts open the casm_monitor candidate page."""
import pytest

from casm_t3.apps import collect


def test_caption_links_the_8061_candidate_page():
    meta = {"candname": "260922eukylj", "snr": 20.7, "dm": 247.6,
            "event_utc": "2026-09-22T00:57:02.250+00:00"}
    text = collect._caption(meta, "http://localhost:8061")
    assert "*<http://localhost:8061/cands/260922eukylj|260922eukylj>*" in text
    assert "Open in dashboard" not in text
    assert "/event/" not in text
    assert text.endswith("2026-09-22 00:57:02 UTC · 2026-09-21 17:57:02 PDT")
    assert "\n" not in text


def test_approved_caption_and_trailing_slash():
    meta = {"candname": "261007vlstlx", "snr": 20.6, "dm": 24.9,
            "event_utc": "2026-10-07T22:47:40+00:00"}
    assert collect._caption(meta, "http://localhost:8061/") == (
        "*<http://localhost:8061/cands/261007vlstlx|261007vlstlx>* — "
        "20.6σ, DM 24.9 pc cm⁻³, 2026-10-07 22:47:40 UTC · 15:47:40 PDT")


@pytest.mark.parametrize("value,expected", [
    ("2026-01-07T22:47:40Z", "2026-01-07 22:47:40 UTC · 14:47:40 PST"),
    ("2026-10-07T22:47:40", "2026-10-07 22:47:40 UTC · 15:47:40 PDT"),
    ("2026-10-07T15:47:40-07:00", "2026-10-07 22:47:40 UTC · 15:47:40 PDT"),
    ("2026-03-08T09:59:59Z", "2026-03-08 09:59:59 UTC · 01:59:59 PST"),
    ("2026-03-08T10:00:00Z", "2026-03-08 10:00:00 UTC · 03:00:00 PDT"),
    ("2026-11-01T08:59:59Z", "2026-11-01 08:59:59 UTC · 01:59:59 PDT"),
    ("2026-11-01T09:00:00Z", "2026-11-01 09:00:00 UTC · 01:00:00 PST"),
    ("2027-01-01T01:00:00Z", "2027-01-01 01:00:00 UTC · 2026-12-31 17:00:00 PST"),
])
def test_event_clocks_follow_event_date_and_dst(value, expected):
    assert collect._event_clocks(value) == expected


@pytest.mark.parametrize("value", [None, "", "broken timestamp"])
def test_bad_time_does_not_block_candidate_posting(value):
    text = collect._caption({"event_utc": value}, "http://localhost:8061")
    assert text.endswith("time unavailable")
    assert "\n" not in text


def test_default_web_base_is_8061():
    parser_defaults = {a.dest: a.default for a in _parser()._actions}
    assert parser_defaults["web_base"] == "http://localhost:8061"


def _parser():
    """The argparse parser main() builds, captured without running the loop."""
    import argparse
    captured = {}
    orig = argparse.ArgumentParser.parse_args

    def grab(self, *a, **k):
        captured["p"] = self
        raise SystemExit(0)
    argparse.ArgumentParser.parse_args = grab
    try:
        collect.main()
    except SystemExit:
        pass
    finally:
        argparse.ArgumentParser.parse_args = orig
    return captured["p"]
