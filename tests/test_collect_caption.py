"""Slack caption link: candidate posts open the casm_monitor candidate page."""
from casm_t3.apps import collect


def test_caption_links_the_8061_candidate_page():
    meta = {"candname": "260922eukylj", "snr": 20.7, "dm": 247.6,
            "event_utc": "2026-09-22T00:57:02.250+00:00"}
    text = collect._caption(meta, "http://localhost:8061")
    assert "<http://localhost:8061/cands/260922eukylj|Open in dashboard>" in text
    assert "/event/" not in text


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
