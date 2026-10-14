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

Schema-2 candidate folders (summary JSON has ``"schema": 2``, casm_io
docs/t3_events.md) are posted only once ``<name>.manifest.json`` exists, which
the plotter writes last. The main figure and the ``_raw.png`` thread reply are
separate marker lines (``kind=main``, ``kind=raw``), so a failed reply is
retried without reposting the main figure. Schema-1 folders keep the single
``posted <iso> ts=<ts>`` line. A candidate's JSON is never rewritten here: it
is mirrored from corr2 by ``rsync -a``.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from casm_t2 import logsetup

from casm_t3 import alerts

logger = logging.getLogger("t3.collect")
PACIFIC = ZoneInfo("America/Los_Angeles")


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


def _mirror_marker(events_dest: Path | None, name: str, text: str) -> None:
    """Write the same .slack marker into the event dir, if one exists."""
    if events_dest is None:
        return
    event_dir = events_dest / name
    try:
        if event_dir.is_dir():
            (event_dir / ".slack").write_text(text)
    except OSError as exc:
        logger.warning("could not mark %s: %s", event_dir, exc)


def _event_clocks(value) -> str:
    """UTC and Pacific event clocks; show the local date when it differs."""
    try:
        event = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return "time unavailable"
    # Legacy event_utc values without an offset still describe UTC, not the
    # collector host's local timezone.
    if event.tzinfo is None:
        event = event.replace(tzinfo=timezone.utc)
    utc = event.astimezone(timezone.utc)
    local = utc.astimezone(PACIFIC)
    local_format = "%H:%M:%S %Z" if local.date() == utc.date() else "%Y-%m-%d %H:%M:%S %Z"
    return f"{utc:%Y-%m-%d %H:%M:%S} UTC · {local.strftime(local_format)}"


def _caption(meta: dict, web_base: str) -> str:
    """One text line: linked candidate ID, search S/N, DM, UTC and PST/PDT."""
    name = meta.get("candname", "?")
    link = f"<{web_base.rstrip('/')}/cands/{name}|{name}>"
    return (f"*{link}* — {meta.get('snr', 0):.1f}σ, "
            f"DM {meta.get('dm', 0):.1f} pc cm⁻³, "
            f"{_event_clocks(meta.get('event_utc'))}")


@dataclass(slots=True)
class _Marker:
    """What a ``.slack`` marker says has been done for one candidate."""

    text: str = ""
    main: bool = False           # a kind=main line exists
    main_ts: str | None = None   # message ts of the main post ("unknown" if unresolved)
    channel: str | None = None   # channel ID of the main post
    raw: bool = False            # a kind=raw or kind=raw_unthreaded line exists
    stale: bool = False          # a "skipped stale" line exists


def _read_marker(path: Path) -> _Marker:
    """Parse a marker; schema-1 lines carry no kind= and set neither flag."""
    try:
        text = path.read_text()
    except OSError:
        return _Marker()
    m = _Marker(text=text)
    for line in text.splitlines():
        words = line.split()
        if words[:2] == ["skipped", "stale"]:
            m.stale = True
        elif words[:1] == ["posted"]:
            fields = dict(w.split("=", 1) for w in words[2:] if "=" in w)
            kind = fields.get("kind")
            if kind == "main":
                m.main = True
                m.main_ts = fields.get("ts")
                m.channel = fields.get("channel")
            elif kind in ("raw", "raw_unthreaded"):
                m.raw = True
    return m


def _is_schema2(meta: dict) -> bool:
    schema = meta.get("schema")
    return isinstance(schema, int) and schema >= 2


def _manifest_ready(d: Path) -> bool:
    """True when the manifest exists, parses, and agrees with the PNG sizes.

    A listed file whose size on disk differs is still being synced, so wait.
    """
    name = d.name
    try:
        manifest = json.loads((d / f"{name}.manifest.json").read_text())
        files = manifest["files"]
    except (OSError, ValueError, KeyError, TypeError):
        return False
    if not isinstance(files, dict):
        return False
    for png in (d / f"{name}.png", d / f"{name}_raw.png"):
        entry = files.get(png.name)
        if png.is_file() and isinstance(entry, dict) and "bytes" in entry:
            if entry["bytes"] != png.stat().st_size:
                return False
    return True


def _post_raw(d: Path, marker: _Marker, web_base: str, max_age_h: float,
              events_dest: Path | None) -> None:
    """Post ``<name>_raw.png`` once, as a reply to the main post when its ts is known."""
    raw_png = d / f"{d.name}_raw.png"
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        age_s = time.time() - raw_png.stat().st_mtime
    except OSError:
        return                       # vanished mid-pass; retry next cycle
    if age_s > max_age_h * 3600:
        marker.text += f"skipped stale {stamp}\n"
        (d / ".slack").write_text(marker.text)
        _mirror_marker(events_dest, d.name, marker.text)
        return
    if not _manifest_ready(d):
        return
    threaded = marker.main_ts not in (None, "", "unknown")
    if threaded:
        caption = "raw intensity, before hella RFI flagging"
    else:
        link = f"<{web_base.rstrip('/')}/cands/{d.name}|{d.name}>"
        caption = f"*{link}* raw intensity, before hella RFI flagging"
    channel = marker.channel if marker.channel not in (None, "", "unknown") else None
    res = alerts.post_candidate(raw_png, caption, channel=channel,
                                thread_ts=marker.main_ts if threaded else None)
    if res:
        ts = res if isinstance(res, str) else "unknown"
        kind = "raw" if threaded else "raw_unthreaded"
        channel_id = channel or alerts.load_channel() or "unknown"
        marker.text += f"posted {stamp} ts={ts} channel={channel_id} kind={kind}\n"
        marker.raw = True
        (d / ".slack").write_text(marker.text)
        _mirror_marker(events_dest, d.name, marker.text)
    # on failure: no marker line, retried next cycle without reposting the main figure


def _post_new(candidates: Path, web_base: str, max_age_h: float,
              events_dest: Path | None = None) -> None:
    """Post every candidate dir that has artifacts but no .slack marker.

    The marker is mirrored into the matching event dir (if there is one) so
    the per-event archive records that the candidate was posted. A schema-2
    dir waits for its manifest, and its raw view is posted as a thread reply
    after the main figure.
    """
    if not alerts.configured():
        return                       # leave unmarked: post once configured
    now = time.time()
    for d in sorted(candidates.iterdir()):
        if not d.is_dir():
            continue
        marker_path = d / ".slack"
        png, meta_json = d / f"{d.name}.png", d / f"{d.name}.json"
        if marker_path.exists():
            # Only a schema-2 marker (kind=main) can still owe the raw reply.
            if (d / f"{d.name}_raw.png").is_file():
                marker = _read_marker(marker_path)
                if marker.main and not marker.raw and not marker.stale:
                    _post_raw(d, marker, web_base, max_age_h, events_dest)
            continue
        if not (png.is_file() and meta_json.is_file()):
            continue
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if now - png.stat().st_mtime > max_age_h * 3600:
            text = f"skipped stale {stamp}\n"
            marker_path.write_text(text)
            _mirror_marker(events_dest, d.name, text)
            continue
        try:
            meta = json.loads(meta_json.read_text())
        except (OSError, ValueError) as exc:
            logger.warning("unreadable %s: %s", meta_json, exc)
            continue                 # retry next cycle; maybe mid-copy
        schema2 = _is_schema2(meta)
        if schema2 and not _manifest_ready(d):
            continue                 # incomplete bundle: retry next cycle, no marker
        res = alerts.post_candidate(png, _caption(meta, web_base))
        if res:
            # res is the Slack message ts when resolvable (True otherwise);
            # keeping it makes the post editable/threadable later.
            ts = res if isinstance(res, str) else "unknown"
            if schema2:
                channel = alerts.load_channel() or "unknown"
                text = f"posted {stamp} ts={ts} channel={channel} kind=main\n"
            else:
                text = f"posted {stamp} ts={ts}\n"
            marker_path.write_text(text)
            _mirror_marker(events_dest, d.name, text)
            if schema2 and (d / f"{d.name}_raw.png").is_file():
                _post_raw(d, _read_marker(marker_path), web_base, max_age_h,
                          events_dest)
        # on failure: no marker, retried next cycle (fail-soft)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--remote",
                   default="casm-corr2:/mnt/nvme4/data/casm/t3_artifacts/")
    p.add_argument("--dest", default="/mnt/nvme5/casm_pipeline/candidates/")
    p.add_argument("--events-remote",
                   default="casm-corr2:/mnt/nvme3/T3/EVENTS/",
                   help="per-event archive on the remote node ('' disables)")
    p.add_argument("--events-dest", default="/mnt/nvme3/T3/EVENTS/",
                   help="local per-event archive the remote tree is pulled into")
    p.add_argument("--interval-s", type=float, default=20.0)
    p.add_argument("--web-base", default="http://localhost:8061",
                   help="candidate-page link base used in Slack captions, <base>/cands/<name> "
                        "(readers use the standard ssh tunnel)")
    p.add_argument("--max-age-h", type=float, default=24.0,
                   help="never post candidates older than this; they get a "
                        "'skipped stale' marker instead")
    p.add_argument("--log-file", default=None)
    args = p.parse_args()

    logsetup.setup(args.log_file)
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    events_dest = Path(args.events_dest) if args.events_dest else None
    if events_dest is not None:
        try:
            events_dest.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("event archive %s unusable: %s", events_dest, exc)
            events_dest = None
    logger.info("collecting %s -> %s every %.0fs; slack %s", args.remote,
                dest, args.interval_s,
                "configured" if alerts.configured() else
                "unconfigured (dotfiles absent; posting disabled)")
    while True:
        _rsync(args.remote, dest)
        if args.events_remote and events_dest is not None:
            _rsync(args.events_remote, events_dest)
        try:
            _post_new(dest, args.web_base, args.max_age_h, events_dest)
        except Exception:  # noqa: BLE001 - posting must not stop collection
            logger.exception("slack posting pass failed")
        time.sleep(args.interval_s)


if __name__ == "__main__":
    main()
