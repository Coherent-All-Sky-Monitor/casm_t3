"""Slack alerting. Degrades to a no-op when no token is configured.

The bot token is read from ~/.config/slack_api (first line). Posting uses
slack_sdk when available; failures are logged and never raised, since
alerting must not take down the pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

TOKEN_PATH = Path.home() / ".config" / "slack_api"


def _load_token() -> str | None:
    try:
        return TOKEN_PATH.read_text().split()[0]
    except (OSError, IndexError):
        return None


def post_candidate(png_path: str | Path, text: str, channel: str) -> bool:
    """Upload a candidate plot with a caption. Returns True on success."""
    token = _load_token()
    if token is None:
        logger.info("no slack token at %s; skipping post (%s)", TOKEN_PATH, text)
        return False
    try:
        from slack_sdk import WebClient
    except ImportError:
        logger.warning("slack_sdk not installed; skipping post")
        return False
    try:
        client = WebClient(token=token)
        client.files_upload_v2(channel=channel, file=str(png_path),
                               initial_comment=text, title=Path(png_path).name)
        logger.info("posted %s to #%s", png_path, channel)
        return True
    except Exception as exc:  # noqa: BLE001 - alerting is strictly best-effort
        logger.error("slack post failed: %s", exc)
        return False
