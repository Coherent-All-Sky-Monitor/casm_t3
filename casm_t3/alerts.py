"""Slack alerting over the plain Web API. Degrades to a no-op when unconfigured.

Two dotfiles configure it (both single-line; create them and posting turns on,
no service edits needed):

- ``~/.config/slack_api``      the bot token (xoxb-...)
- ``~/.config/slack_channel``  the target channel ID (C..., not the name --
  the file-upload API requires the ID; channel *names* only work for plain
  chat.postMessage)

Uses ``requests`` directly (the DSA-110 slack_notify.py pattern): the
external-upload flow files.getUploadURLExternal -> POST bytes ->
files.completeUploadExternal. No slack_sdk dependency -- it is not installed
in casm_offline_env and corr2 could not pip-install it anyway.

Every failure (missing token/channel, network, API error) is logged and
swallowed: alerting must never take down the pipeline. Only casm-corr1 can
reach Slack (corr2 has no internet), so the single poster in production is
t3-collect on corr1; see apps/collect.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

TOKEN_PATH = Path.home() / ".config" / "slack_api"
CHANNEL_PATH = Path.home() / ".config" / "slack_channel"

_SLACK_API = "https://slack.com/api"
_TIMEOUT_S = 15.0


def _read_first_word(path: Path) -> str | None:
    try:
        return path.read_text().split()[0]
    except (OSError, IndexError):
        return None


def _load_token() -> str | None:
    return _read_first_word(TOKEN_PATH)


def load_channel() -> str | None:
    """The configured channel ID, or None when alerting is unconfigured."""
    return _read_first_word(CHANNEL_PATH)


def configured() -> bool:
    return _load_token() is not None and load_channel() is not None


def post_text(text: str, channel: str | None = None) -> bool:
    """Post a plain message. Returns True on success, never raises."""
    token = _load_token()
    channel = channel or load_channel()
    if token is None or channel is None:
        logger.info("slack unconfigured (token %s, channel %s); skipping: %s",
                    TOKEN_PATH, CHANNEL_PATH, text)
        return False
    try:
        import requests
        r = requests.post(f"{_SLACK_API}/chat.postMessage",
                          headers={"Authorization": f"Bearer {token}"},
                          json={"channel": channel, "text": text},
                          timeout=_TIMEOUT_S)
        r.raise_for_status()
        doc = r.json()
        if not doc.get("ok"):
            logger.error("slack chat.postMessage failed: %s", doc.get("error"))
            return False
        return True
    except Exception as exc:  # noqa: BLE001 - alerting is strictly best-effort
        logger.error("slack post failed: %s", exc)
        return False


def post_candidate(png_path: str | Path, text: str,
                   channel: str | None = None) -> bool:
    """Upload a candidate plot with a caption. Returns True on success.

    ``channel`` must be a channel ID (C...); defaults to ~/.config/slack_channel.
    The bot must be a member of the channel (/invite it once).
    """
    token = _load_token()
    channel = channel or load_channel()
    if token is None or channel is None:
        logger.info("slack unconfigured (token %s, channel %s); skipping: %s",
                    TOKEN_PATH, CHANNEL_PATH, text)
        return False
    png = Path(png_path)
    if not png.is_file():
        logger.error("slack post: no such plot %s", png)
        return False
    try:
        import requests
        auth = {"Authorization": f"Bearer {token}"}

        r1 = requests.get(f"{_SLACK_API}/files.getUploadURLExternal",
                          headers=auth,
                          params={"filename": png.name,
                                  "length": png.stat().st_size},
                          timeout=_TIMEOUT_S)
        r1.raise_for_status()
        d1 = r1.json()
        if not d1.get("ok"):
            logger.error("slack getUploadURLExternal failed: %s", d1.get("error"))
            return False

        with png.open("rb") as fh:
            r2 = requests.post(d1["upload_url"], files={"file": fh},
                               timeout=_TIMEOUT_S)
        r2.raise_for_status()

        r3 = requests.post(f"{_SLACK_API}/files.completeUploadExternal",
                           headers=auth,
                           json={"files": [{"id": d1["file_id"],
                                            "title": png.stem}],
                                 "channel_id": channel,
                                 "initial_comment": text},
                           timeout=_TIMEOUT_S)
        r3.raise_for_status()
        d3 = r3.json()
        if not d3.get("ok"):
            logger.error("slack completeUploadExternal failed: %s",
                         d3.get("error"))
            return False
        logger.info("posted %s to %s", png.name, channel)
        return _share_ts(d1["file_id"], channel, auth) or True
    except Exception as exc:  # noqa: BLE001 - alerting is strictly best-effort
        logger.error("slack post failed: %s", exc)
        return False


def _share_ts(file_id: str, channel: str, auth: dict,
              timeout_s: float = 10.0) -> str | None:
    """Message ts of the file's share into ``channel`` (DSA post-map trick).

    Slack materialises the upload->message share asynchronously; poll
    files.info briefly. Needs the files:read scope (granted 2026-08-13);
    returns None on timeout or missing scope — the post itself already
    succeeded, the caller just loses later editability for this one.
    """
    import time as _time
    import requests
    deadline = _time.monotonic() + timeout_s
    while _time.monotonic() < deadline:
        try:
            d = requests.get(f"{_SLACK_API}/files.info", headers=auth,
                             params={"file": file_id}, timeout=_TIMEOUT_S).json()
        except Exception:  # noqa: BLE001
            return None
        if d.get("ok"):
            shares = (d.get("file") or {}).get("shares") or {}
            for vis in ("public", "private"):
                entries = (shares.get(vis) or {}).get(channel)
                if entries and entries[0].get("ts"):
                    return entries[0]["ts"]
        elif d.get("error") == "missing_scope":
            return None
        _time.sleep(1.0)
    return None
