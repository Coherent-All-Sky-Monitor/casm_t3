"""t3-collect: pull corr2 artifacts to corr1 and post new candidates to Slack.

Replaces the old bash rsync loop (deploy/systemd/t3-collect.service) and adds
the ONE Slack posting point for the whole pipeline. Rationale: casm-corr2 has
no internet, so its plotter can never reach Slack; and the corr1 plotter
posting for itself while the collector posts for corr2 would mean two code
paths to keep honest. Instead neither plotter posts; this daemon watches the
corr1 archive (which receives corr1 artifacts directly from the local plotter
and corr2 artifacts via the rsync pull below) and posts anything new, from
either node, exactly once.

Posting state is a ``.slack`` marker file inside each candidate directory --
written on success OR on permanent skip (too old), so a candidate is never
posted twice and a backlog from before Slack was configured is not replayed
onto the channel. Slack stays a silent no-op until both
``~/.config/slack_api`` (bot token) and ``~/.config/slack_channel``
(channel ID) exist; see alerts.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from casm_t2 import logsetup

from casm_t3 import alerts

logger = logging.getLogger("t3.collect")


def _rsync(remote: str, dest: Path) -> None:
    """One rsync pull; failures are logged and retried next cycle."""
    try:
        r = subprocess.run(["rsync", "-a", remote, str(dest)],
                           capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            logger.warning("rsync pull failed (rc %d): %s",
                           r.returncode, r.stderr.strip()[:300])
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("rsync pull failed: %s", exc)


def _caption(meta: dict, web_base: str) -> str:
    """DSA-110 card style: *name* — 12.9σ, DM 239.7 pc cm⁻³, <UTC> UTC."""
    name = meta.get("candname", "?")
    utc = str(meta.get("event_utc", "")).replace("T", " ")[:19]
    return (f"*{name}* — {meta.get('snr', 0):.1f}σ, "
            f"DM {meta.get('dm', 0):.1f} pc cm⁻³, {utc} UTC | "
            f"<{web_base}/event/{name}|Open in dashboard>")


def _post_new(candidates: Path, web_base: str, max_age_h: float) -> None:
    """Post every candidate dir that has artifacts but no .slack marker."""
    if not alerts.configured():
        return                       # leave unmarked: post once configured
    now = time.time()
    for d in sorted(candidates.iterdir()):
        if not d.is_dir():
            continue
        marker = d / ".slack"
        png, meta_json = d / f"{d.name}.png", d / f"{d.name}.json"
        if marker.exists() or not (png.is_file() and meta_json.is_file()):
            continue
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if now - png.stat().st_mtime > max_age_h * 3600:
            marker.write_text(f"skipped stale {stamp}\n")
            continue
        try:
            meta = json.loads(meta_json.read_text())
        except (OSError, ValueError) as exc:
            logger.warning("unreadable %s: %s", meta_json, exc)
            continue                 # retry next cycle; maybe mid-copy
        if alerts.post_candidate(png, _caption(meta, web_base)):
            marker.write_text(f"posted {stamp}\n")
        # on failure: no marker, retried next cycle (fail-soft)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--remote",
                   default="casm-corr2:/mnt/nvme4/data/casm/t3_artifacts/")
    p.add_argument("--dest", default="/mnt/nvme5/casm_pipeline/candidates/")
    p.add_argument("--interval-s", type=float, default=20.0)
    p.add_argument("--web-base", default="http://127.0.0.1:8050",
                   help="event-page link base used in Slack captions "
                        "(readers use the standard ssh tunnel)")
    p.add_argument("--max-age-h", type=float, default=24.0,
                   help="never post candidates older than this; they get a "
                        "'skipped stale' marker instead")
    p.add_argument("--log-file", default=None)
    args = p.parse_args()

    logsetup.setup(args.log_file)
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    logger.info("collecting %s -> %s every %.0fs; slack %s", args.remote,
                dest, args.interval_s,
                "configured" if alerts.configured() else
                "unconfigured (dotfiles absent; posting disabled)")
    while True:
        _rsync(args.remote, dest)
        try:
            _post_new(dest, args.web_base, args.max_age_h)
        except Exception:  # noqa: BLE001 - posting must not stop collection
            logger.exception("slack posting pass failed")
        time.sleep(args.interval_s)


if __name__ == "__main__":
    main()
