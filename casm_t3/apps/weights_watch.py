"""t3-weights-watch: record every beamformer weights load that was not an upload.

bfcorr receives weights only through the per-stream FIFO that the medusa weights
daemon feeds into the weights ring. Two writers exist: ``deploy_bf_weights`` (an
upload, which records itself in the registry) and ``casm_bfcorr.py`` writing the
on-disk defaults at daemon start (a restart). The daemon logs every transfer
("Transferred 67108864 bytes, finishing", one line per stream, local time) into
/data/casm/logs/antenna_bf_weights.log on corr1, which is the merged log for both
nodes. This service tails that log and, for each CB-sized transfer with no upload
event within ``match_window_s``:

* if a bfcorr START on that stream preceded it within ``restart_window_s`` (the
  merged antenna_bfcorr.log), it is a defaults reload: hash the defaults file on
  the owning node (read-only; corr2 over ssh), look the payload up in the
  registry, record a live event with source=defaults;
* otherwise it is an unrecorded upload: record source=unknown with the payload
  unidentified, so T2 stores NULL coordinates for that period, and raise an alert.

Alerts (defaults differ from the newest upload = the array reverted; streams
disagree = partial deploy; unknown payload) go to the registry's alerts.jsonl and
to Slack through casm_t3.alerts when configured. Nothing in fourier-space is
touched: this reads two log files and hashes files on disk.
"""
from __future__ import annotations

import argparse
import logging
import re
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from casm_t2 import weights_registry as wr

from .. import alerts

logger = logging.getLogger("t3.weights_watch")

WEIGHTS_LOG = Path("/data/casm/logs/antenna_bf_weights.log")
BFCORR_LOG = Path("/data/casm/logs/antenna_bfcorr.log")
LOG_TZ = ZoneInfo("America/Los_Angeles")
CB_BYTES = 67108864
TRANSFER_RE = re.compile(r"^(\d) \[(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}\.\d+)\] \[info\] Transferred (\d+) bytes, finishing")
START_RE = re.compile(r"^(\d) \[(\d{4}-\d{2}-\d{2})-(\d{2}:\d{2}:\d{2}\.\d+)\] START casm_bfcorr ")


def _to_utc(date: str, clock: str) -> datetime:
    local = datetime.fromisoformat(f"{date}T{clock}").replace(tzinfo=LOG_TZ)
    return local.astimezone(timezone.utc)


def parse_transfer(line: str) -> tuple[int, datetime, int] | None:
    m = TRANSFER_RE.match(line)
    if not m:
        return None
    return int(m.group(1)), _to_utc(m.group(2), m.group(3)), int(m.group(4))


def parse_start(line: str) -> tuple[int, datetime] | None:
    m = START_RE.match(line)
    if not m:
        return None
    return int(m.group(1)), _to_utc(m.group(2), m.group(3))


def defaults_payload_md5(stream: int, timeout_s: float = 120.0) -> str | None:
    """md5 of the defaults payload bfcorr reads for ``stream`` on its own node."""
    host = wr.STREAM_HOST[stream]
    path = wr.DEFAULTS_FILE_FMT.format(stream=stream)
    try:
        if host == "casm-corr1":
            return wr.payload_md5_of_file(path)
        out = subprocess.run(["ssh", "-o", "ConnectTimeout=10", host,
                              f"tail -c +{wr.HDR_SIZE + 1} {path} | md5sum"],
                             capture_output=True, text=True, timeout=timeout_s)
        if out.returncode != 0:
            logger.error("md5 on %s failed: %s", host, out.stderr.strip())
            return None
        return out.stdout.split()[0]
    except (OSError, subprocess.SubprocessError) as exc:
        logger.error("md5 on %s failed: %s", host, exc)
        return None


class Tail:
    """Follow a log file from an offset, surviving rotation and truncation."""

    def __init__(self, path: Path, start_at_end: bool = True):
        self.path = path
        self.pos = path.stat().st_size if (start_at_end and path.exists()) else 0

    def lines(self):
        try:
            size = self.path.stat().st_size
        except OSError:
            return
        if size < self.pos:
            self.pos = 0
        with open(self.path, "rb") as f:
            f.seek(self.pos)
            chunk = f.read()
            self.pos = f.tell()
        for raw in chunk.split(b"\n"):
            if raw:
                yield raw.decode("ascii", "ignore")


class Watcher:
    def __init__(self, registry: wr.Registry, match_window_s: float = 20.0,
                 restart_window_s: float = 120.0, alert: bool = True):
        self.reg = registry
        self.match_window = timedelta(seconds=match_window_s)
        self.restart_window = timedelta(seconds=restart_window_s)
        self.alert = alert
        self.recent_starts: dict[int, datetime] = {}

    def note_start(self, stream: int, utc: datetime) -> None:
        self.recent_starts[stream] = utc

    def upload_matches(self, stream: int, utc: datetime) -> bool:
        for ev in reversed(self.reg.events()):
            if ev["source"] != "upload" or int(ev["stream"]) != stream:
                continue
            t = datetime.fromisoformat(ev["utc"])
            if abs(t - utc) <= self.match_window:
                return True
            if t < utc - self.match_window:
                break
        return False

    def handle_transfer(self, stream: int, utc: datetime, nbytes: int) -> dict | None:
        if nbytes != CB_BYTES:
            return None                     # IB weights or partial; not a pointing change
        if self.upload_matches(stream, utc):
            logger.info("stream %d transfer at %s matches an upload event", stream, utc.isoformat())
            return None
        started = self.recent_starts.get(stream)
        if started is not None and timedelta(0) <= utc - started <= self.restart_window:
            md5 = defaults_payload_md5(stream)
            ev = self.reg.record_live_event(utc=utc, stream=stream, payload_md5=md5,
                                            source="defaults", evidence="bfcorr START then FIFO transfer")
            self._check_after_defaults(stream, utc, ev)
            return ev
        ev = self.reg.record_live_event(utc=utc, stream=stream, payload_md5=None,
                                        source="unknown", evidence="FIFO transfer with no upload event and no bfcorr START")
        self._alert("unregistered_weights",
                    f"weights loaded on bfcorr stream {stream} at {utc.isoformat(timespec='seconds')} "
                    "with no registry upload and no bfcorr restart: unidentified payload, "
                    "T2 will store no coordinates until a registered upload")
        return ev

    def _check_after_defaults(self, stream: int, utc: datetime, ev: dict) -> None:
        if ev.get("product_id") is None:
            self._alert("unknown_defaults",
                        f"bfcorr stream {stream} reloaded defaults at {utc.isoformat(timespec='seconds')} "
                        f"whose payload md5 {ev.get('payload_md5')} is not a registered product")
            return
        uploads = [e for e in self.reg.events() if e["source"] == "upload" and int(e["stream"]) == stream]
        if uploads and uploads[-1].get("product_id") not in (None, ev["product_id"]):
            newest = self.reg.product(uploads[-1]["product_id"]) or {}
            self._alert("reverted_to_defaults",
                        f"bfcorr stream {stream} restarted at {utc.isoformat(timespec='seconds')} and loaded "
                        f"defaults product {ev['product_id']} ({Path((self.reg.product(ev['product_id']) or {}).get('h5_path','?')).name}), "
                        f"not the newest upload {uploads[-1]['product_id']} ({Path(newest.get('h5_path','?')).name}): "
                        "the last upload was made without --save-defaults")
        prod, status = self.reg.product_at(utc + timedelta(seconds=1))
        if status == "partial":
            self._alert("partial_deploy", f"bfcorr streams carry different weights products after stream {stream} "
                                          f"loaded at {utc.isoformat(timespec='seconds')}; no single beam pointing exists")

    def _alert(self, kind: str, text: str) -> None:
        logger.error("%s: %s", kind, text)
        self.reg.record_alert(kind, text)
        if self.alert and alerts.configured():
            alerts.post_text(f":warning: WEIGHTS {kind}: {text}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Record bfcorr weights reloads in the weights registry")
    p.add_argument("--registry", default=str(wr.REGISTRY_DIR))
    p.add_argument("--weights-log", default=str(WEIGHTS_LOG))
    p.add_argument("--bfcorr-log", default=str(BFCORR_LOG))
    p.add_argument("--interval-s", type=float, default=5.0)
    p.add_argument("--from-start", action="store_true", help="replay the whole logs instead of tailing from now")
    p.add_argument("--no-slack", action="store_true")
    p.add_argument("--log-file")
    args = p.parse_args(argv)
    logging.basicConfig(filename=args.log_file, level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    watcher = Watcher(wr.Registry(args.registry), alert=not args.no_slack)
    wtail = Tail(Path(args.weights_log), start_at_end=not args.from_start)
    btail = Tail(Path(args.bfcorr_log), start_at_end=not args.from_start)
    logger.info("watching %s and %s, registry %s", args.weights_log, args.bfcorr_log, args.registry)
    while True:
        for line in btail.lines():
            st = parse_start(line)
            if st:
                watcher.note_start(*st)
        for line in wtail.lines():
            tr = parse_transfer(line)
            if tr:
                watcher.handle_transfer(*tr)
        time.sleep(args.interval_s)


if __name__ == "__main__":
    main()
