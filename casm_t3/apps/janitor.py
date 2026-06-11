"""Disk janitor for triggered dumps on both backend nodes.

Runs on corr1 (which can ssh to corr2; not vice versa) and enforces a size
quota plus a maximum age on every dump directory. Dumps belonging to events
a human has labeled frb or pulsar are never deleted; everything else is fair
game once it is old enough or the quota is exceeded (oldest first). Each
deletion is recorded against its trigger row (cleaned_utc) when the file can
be matched to one.

Intensity dump files are named <obs_utc>_<byte_offset>.000000.dada, so the
file's sky-time span is recoverable from the name and size
(BYTES_PER_SECOND = 375e6 per stream); a file is matched to a trigger when
its span overlaps the trigger's dump window.

Always try --dry-run first.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from casm_t2 import db as t2db
from casm_t2 import timing

logger = logging.getLogger("t3.janitor")

BYTES_PER_SECOND = 375e6

# (host, glob) pairs the janitor patrols. Voltage dirs are included even
# while voltage triggering is disabled - smoke tests land files there too.
PATROL = [
    ("casm-corr1", "/mnt/nvme4/data/casm/cand_beam_dumps/stream_*"),
    ("casm-corr2", "/mnt/nvme4/data/casm/cand_beam_dumps/stream_*"),
    ("casm-corr1", "/mnt/nvme4/data/casm/cand_dumps"),
    ("casm-corr2", "/mnt/nvme4/data/casm/cand_dumps"),
]


@dataclass(slots=True)
class DumpFile:
    host: str
    path: str
    size: int
    mtime: float

    def span(self) -> tuple[datetime, datetime] | None:
        """Sky-time interval covered by the file, from its name, or None."""
        name = Path(self.path).name
        try:
            obs_s, rest = name.split("_", 1)
            offset = int(rest.split(".")[0])
            t0 = timing.parse_dada_utc(obs_s) + timedelta(seconds=offset / BYTES_PER_SECOND)
            return t0, t0 + timedelta(seconds=self.size / BYTES_PER_SECOND)
        except (ValueError, IndexError):
            return None


def list_files(host: str, pattern: str, local_host: str) -> list[DumpFile]:
    cmd = f"find {pattern} -maxdepth 1 -name '*.dada' -printf '%T@ %s %p\\n' 2>/dev/null"
    if host == local_host:
        out = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True).stdout
    else:
        out = subprocess.run(["ssh", host, cmd], capture_output=True, text=True,
                             timeout=60).stdout
    files = []
    for line in out.splitlines():
        mtime, size, path = line.split(" ", 2)
        files.append(DumpFile(host, path, int(size), float(mtime)))
    return files


def protected_windows(conn) -> list[tuple[datetime, datetime]]:
    """Dump windows of events labeled frb/pulsar - never deleted."""
    rows = conn.execute(
        "SELECT t.dump_utc_start, t.dump_utc_stop FROM triggers t"
        " WHERE t.action IN ('triggered','partial') AND t.dump_utc_start IS NOT NULL"
        " AND t.candname IN (SELECT name FROM labels WHERE label IN ('frb','pulsar'))"
    ).fetchall()
    out = []
    for a, b in rows:
        try:
            out.append((timing.parse_dada_utc(a), timing.parse_dada_utc(b)))
        except ValueError:
            pass
    return out


def mark_cleaned(conn, f: DumpFile) -> None:
    span = f.span()
    if span is None:
        return
    a = timing.format_dada_utc(span[0])
    b = timing.format_dada_utc(span[1])
    conn.execute(
        "UPDATE triggers SET cleaned_utc = ? WHERE action IN ('triggered','partial')"
        " AND dump_utc_start <= ? AND dump_utc_stop >= ? AND cleaned_utc IS NULL",
        (datetime.now(timezone.utc).isoformat(timespec="milliseconds"), b, a))
    conn.commit()


def sweep(conn, args, local_host: str) -> None:
    prot = protected_windows(conn)
    now = time.time()

    def is_protected(f: DumpFile) -> bool:
        span = f.span()
        if span is None:  # unparseable: keep, fail safe
            return True
        return any(s <= span[1] and span[0] <= e for s, e in prot)

    for host, pattern in PATROL:
        try:
            files = list_files(host, pattern, local_host)
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.warning("listing %s:%s failed: %s", host, pattern, exc)
            continue
        if not files:
            continue
        files.sort(key=lambda f: f.mtime)
        total_gb = sum(f.size for f in files) / 1e9
        doomed: list[DumpFile] = []
        # age rule
        for f in files:
            if (now - f.mtime) > args.max_age_days * 86400 and not is_protected(f):
                doomed.append(f)
        # quota rule, oldest first
        excess = total_gb - args.max_gb
        for f in files:
            if excess <= 0:
                break
            if f in doomed or is_protected(f):
                continue
            doomed.append(f)
            excess -= f.size / 1e9
        logger.info("%s:%s -> %d files / %.0f GB, deleting %d (%.0f GB)%s",
                    host, pattern, len(files), total_gb, len(doomed),
                    sum(f.size for f in doomed) / 1e9,
                    " [DRY RUN]" if args.dry_run else "")
        for f in doomed:
            logger.info("  delete %s:%s (%.1f GB, %.1f d old)%s", f.host, f.path,
                        f.size / 1e9, (now - f.mtime) / 86400,
                        " [DRY RUN]" if args.dry_run else "")
            if args.dry_run:
                continue
            try:
                if f.host == local_host:
                    Path(f.path).unlink()
                else:
                    subprocess.run(["ssh", f.host, f"rm -f '{f.path}'"],
                                   check=True, timeout=30)
                mark_cleaned(conn, f)
            except (OSError, subprocess.SubprocessError) as exc:
                logger.error("delete failed for %s:%s: %s", f.host, f.path, exc)


def main() -> None:
    p = argparse.ArgumentParser(description="Dump-directory janitor (run on corr1)")
    p.add_argument("--db", default=t2db.DEFAULT_PATH)
    p.add_argument("--max-gb", type=float, default=150.0,
                   help="per-directory-tree quota")
    p.add_argument("--max-age-days", type=float, default=7.0)
    p.add_argument("--interval-s", type=float, default=3600.0)
    p.add_argument("--once", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    import socket
    local_host = socket.gethostname().split(".")[0]
    conn = t2db.connect(args.db)
    while True:
        try:
            sweep(conn, args, local_host)
        except Exception:
            logger.exception("sweep failed")
        if args.once or args.dry_run:
            break
        time.sleep(args.interval_s)


if __name__ == "__main__":
    main()
