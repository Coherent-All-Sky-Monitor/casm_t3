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

A second target, ``hella_masks`` (``--targets hella_masks``), holds hella's
flag-record chunks (casm_io docs/hella_masks.md) to an age limit and a size
quota on both nodes. It deletes closed ``hmask_b*_g*_p*.hmr`` chunks and, as an
orphan, an open ``.hmr.part`` untouched for twice the age limit (a live writer
appends every gulp), and keeps every chunk a pending trigger card in the T2
spool could still need (see ``pinned_chunks``). It does not touch the T2
database. The default target is ``dumps`` alone, so existing invocations are
unchanged.

Always try --dry-run first.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from casm_t2 import beams as t2_beams
from casm_t2 import db as t2db
from casm_t2 import logsetup
from casm_t2 import timing

logger = logging.getLogger("t3.janitor")

BYTES_PER_SECOND = 375e6

# (host, glob) pairs the janitor patrols. Voltage dirs are included even
# while voltage triggering is disabled - smoke tests land files there too.
PATROL = [
    ("casm-corr1", "/mnt/nvme4/data/casm/cand_beam_dumps/stream_*"),
    ("casm-corr2", "/mnt/nvme4/data/casm/cand_beam_dumps/stream_*"),
    # The .dada files sit one level down, in per-stream subdirectories, and
    # find is run with -maxdepth 1: the glob must reach the stream dirs or
    # the patrol silently matches nothing (fixed 2026-09-09).
    ("casm-corr1", "/mnt/nvme4/data/casm/cand_dumps/stream_*"),
    ("casm-corr2", "/mnt/nvme4/data/casm/cand_dumps/stream_*"),
]


# Hella mask records: <root>/<UTC_START>/hmask_b<beam0>_g<first_gulp>_p<pid>.hmr, one
# directory per observation. An open chunk is .hmr.part and is deleted only as
# an orphan (MASK_ORPHAN_AGE_FACTOR x the age limit untouched). The _p<pid>
# part is optional so names without it still parse.
MASK_HOSTS = ("casm-corr1", "casm-corr2")
MASK_CHUNK_RE = re.compile(r"^hmask_b(\d+)_g(\d+)(?:_p(\d+))?\.hmr(\.part)?$")
MASK_ORPHAN_AGE_FACTOR = 2.0   # open chunk untouched this many age limits is orphaned
GULP_SAMP = 8192          # samples per hella gulp
GULP_LEAD_SAMP = 960      # hella gulp g carries samples g*8192-960 .. g*8192+7231
MASK_DIR_GRACE_S = 600.0  # an observation dir changed this recently is not removed
TARGETS = ("dumps", "hella_masks")


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
            logger.debug("%s:%s -> no .dada files", host, pattern)
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


@dataclass(slots=True)
class MaskChunk:
    host: str
    path: str
    size: int
    mtime: float
    closed: bool       # .hmr, not .hmr.part
    obs: str           # UTC_START directory name
    beam0: int         # global index of the stream's first beam
    first_gulp: int
    pid: int = 0       # writing process, 0 for a name without _p<pid>


@dataclass(slots=True)
class PendingCard:
    """A trigger card in a T2 spool with no .done/.failed rename yet."""
    host: str
    name: str
    mtime: float
    utc_start: str | None       # card["gulp"]["utc_start"]
    gulp: int | None            # card["gulp"]["gulp"]
    samp: int | None
    dump_utc_start: str | None  # card["dump_utc_start"], sky time of the dump
    bad: bool = False           # unreadable card: pins every chunk


# Run with python3 on the spool's host (local or over ssh): one JSON line per
# card that is still a bare .json file.
_SPOOL_SCAN = """
import json, os, sys
d = sys.argv[1]
try:
    names = sorted(os.listdir(d))
except FileNotFoundError:
    names = []
for n in names:
    if not n.endswith(".json"):
        continue
    p = os.path.join(d, n)
    try:
        mtime = os.path.getmtime(p)
    except OSError:
        continue
    try:
        with open(p) as fh:
            c = json.load(fh)
        g = c.get("gulp") or {}
        row = {"name": n, "mtime": mtime, "utc_start": g.get("utc_start"),
               "gulp": g.get("gulp"), "samp": c.get("samp"),
               "dump_utc_start": c.get("dump_utc_start")}
    except Exception:
        row = {"name": n, "mtime": mtime, "bad": True}
    print(json.dumps(row))
"""


def _run_on(host: str, local_host: str, cmd: str, stdin: str | None = None,
            check: bool = False, timeout: float = 60) -> str:
    """Stdout of a shell command on ``host``.

    An ssh failure (rc 255) always raises OSError; any non-zero rc does with
    ``check``.
    """
    if host == local_host:
        r = subprocess.run(["bash", "-c", cmd], input=stdin, capture_output=True,
                           text=True, timeout=timeout)
    else:
        r = subprocess.run(["ssh", host, cmd], input=stdin, capture_output=True,
                           text=True, timeout=timeout)
        if r.returncode == 255:
            raise OSError(f"ssh {host} failed: {r.stderr.strip()[:200]}")
    if check and r.returncode != 0:
        raise OSError(f"{host}: command failed (rc {r.returncode}): {r.stderr.strip()[:200]}")
    return r.stdout


def list_mask_chunks(host: str, root: str, local_host: str) -> tuple[list[MaskChunk], dict[str, float]]:
    """Chunk files (open and closed) and observation dirs under ``root`` on ``host``.

    Returns (chunks, {obs dir name: dir mtime}); a missing root lists nothing.
    """
    cmd = (f"find {shlex.quote(root)} -mindepth 1 -maxdepth 2 "
           r"\( -type d -o -type f -name 'hmask_b*_g*.hmr*' \) "
           r"-printf '%y %T@ %s %p\n' 2>/dev/null")
    out = _run_on(host, local_host, cmd)
    root_path = Path(root)
    chunks: list[MaskChunk] = []
    dirs: dict[str, float] = {}
    for line in out.splitlines():
        try:
            kind, mtime, size, path = line.split(" ", 3)
            mtime_f, size_i = float(mtime), int(size)
        except ValueError:
            continue
        path_p = Path(path)
        if kind == "d" and path_p.parent == root_path:
            dirs[path_p.name] = mtime_f
        elif kind == "f" and path_p.parent.parent == root_path:
            m = MASK_CHUNK_RE.match(path_p.name)
            if m:
                chunks.append(MaskChunk(host, path, size_i, mtime_f,
                                        closed=m.group(4) is None,
                                        obs=path_p.parent.name,
                                        beam0=int(m.group(1)),
                                        first_gulp=int(m.group(2)),
                                        pid=int(m.group(3) or 0)))
    return chunks, dirs


def list_pending_cards(host: str, spool: str, local_host: str) -> list[PendingCard]:
    """Cards in ``host``'s T2 spool that the plotter has not yet renamed."""
    py = shlex.quote(sys.executable) if host == local_host else "python3"
    out = _run_on(host, local_host, f"{py} - {shlex.quote(spool)}",
                  stdin=_SPOOL_SCAN, check=True)
    cards = []
    for line in out.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        cards.append(PendingCard(
            host=host, name=row.get("name", "?"), mtime=float(row.get("mtime", 0.0)),
            utc_start=row.get("utc_start"), gulp=row.get("gulp"), samp=row.get("samp"),
            dump_utc_start=row.get("dump_utc_start"), bad=bool(row.get("bad"))))
    return cards


def _card_gulp(card: PendingCard) -> int | None:
    """Hella gulp number of the card: the gulp block's, else from the peak sample."""
    if isinstance(card.gulp, int):
        return card.gulp
    if isinstance(card.samp, int):
        return (card.samp + GULP_LEAD_SAMP) // GULP_SAMP
    return None


def _card_obs(card: PendingCard, obs_names: list[str]) -> str | None:
    """Observation directory the card's data sits in, or None when unknown.

    The gulp block names it. Without one (fast-path and pre-v2 cards) it is the
    latest observation that started at or before the dump's sky time.
    """
    if card.utc_start:
        return card.utc_start
    if card.dump_utc_start:
        earlier = [o for o in obs_names if o[:19] <= card.dump_utc_start[:19]]
        return max(earlier) if earlier else None
    return None


def pinned_chunks(chunks: list[MaskChunk], cards: list[PendingCard]) -> set[str]:
    """Paths of chunks that pending trigger cards could still need.

    Per card: the chunk of the card's observation that contains the card's
    gulp, plus the chunk before and the chunk after it (a dump spans one to two
    gulps and may cross a chunk boundary). A chunk covers its own first gulp up
    to the first gulp of the next chunk of the same stream. A card with no gulp
    number pins every chunk of its observation; a card with no known
    observation, or an unreadable one, pins every chunk.
    """
    pinned: set[str] = set()
    obs_names = sorted({c.obs for c in chunks})
    for card in cards:
        obs = None if card.bad else _card_obs(card, obs_names)
        if obs is None:
            return {c.path for c in chunks}
        in_obs = [c for c in chunks if c.obs == obs]
        gulp = _card_gulp(card)
        if gulp is None:
            pinned.update(c.path for c in in_obs)
            continue
        by_stream: dict[int, list[MaskChunk]] = {}
        for c in in_obs:
            by_stream.setdefault(c.beam0, []).append(c)
        for group in by_stream.values():
            group.sort(key=lambda c: (c.first_gulp, c.pid))
            hit = max((i for i, c in enumerate(group) if c.first_gulp <= gulp), default=0)
            pinned.update(c.path for c in group[max(0, hit - 1):hit + 2])
    return pinned


def _delete_remote(host: str, paths: list[str], cmd: str) -> None:
    subprocess.run(["ssh", host, cmd + " " + " ".join(shlex.quote(p) for p in paths)],
                   check=True, timeout=60)


def sweep_masks(args, local_host: str, hosts=MASK_HOSTS, now: float | None = None) -> None:
    """Age and quota sweep of the hella mask records on each host.

    Per host: closed chunks older than ``--mask-max-age-h`` go, then the oldest
    closed chunks go until the tree (open chunks included) is within
    ``--mask-max-gb``. An open chunk untouched for MASK_ORPHAN_AGE_FACTOR times
    the age limit has no live writer and goes too, with a warning. Pinned
    chunks stay for every rule. A host whose chunk list or spool scan fails is
    skipped whole: with the pins unknown nothing may be deleted there.
    """
    now = time.time() if now is None else now
    tag = " [DRY RUN]" if args.dry_run else ""
    for host in hosts:
        try:
            chunks, dirs = list_mask_chunks(host, args.mask_root, local_host)
            cards = list_pending_cards(host, args.spool, local_host)
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning("mask sweep of %s:%s skipped: %s", host, args.mask_root, exc)
            continue
        live = []
        for card in cards:
            if now - card.mtime > args.mask_pin_max_age_h * 3600:
                logger.warning("pending card %s:%s is %.0f h old; it no longer pins",
                               host, card.name, (now - card.mtime) / 3600)
            else:
                live.append(card)
        orphan_s = MASK_ORPHAN_AGE_FACTOR * args.mask_max_age_h * 3600
        total_b = sum(c.size for c in chunks)
        closed = sorted((c for c in chunks if c.closed), key=lambda c: (c.mtime, c.path))
        pinned = pinned_chunks(chunks, live)
        doomed: list[MaskChunk] = []
        for c in chunks:
            if c.closed or now - c.mtime <= orphan_s:
                continue
            if c.path in pinned:
                logger.warning("orphaned open chunk %s:%s (untouched %.0f h) is pinned; "
                               "left in place", host, c.path, (now - c.mtime) / 3600)
                continue
            logger.warning("orphaned open chunk %s:%s untouched for %.0f h (limit %.0f h): "
                           "no live writer, deleting", host, c.path,
                           (now - c.mtime) / 3600, orphan_s / 3600)
            doomed.append(c)
        for c in closed:
            if now - c.mtime > args.mask_max_age_h * 3600 and c.path not in pinned:
                doomed.append(c)
        # bytes: the quota test must not flip on float residue near zero
        excess = total_b - sum(c.size for c in doomed) - args.mask_max_gb * 1e9
        for c in closed:
            if excess < 1:
                break
            if c in doomed or c.path in pinned:
                continue
            doomed.append(c)
            excess -= c.size
        logger.info("%s:%s -> %d chunks (%d closed) / %.1f GB, %d pending cards, "
                    "%d pinned, deleting %d (%.1f GB)%s", host, args.mask_root,
                    len(chunks), len(closed), total_b / 1e9, len(live), len(pinned),
                    len(doomed), sum(c.size for c in doomed) / 1e9, tag)
        if excess >= 1:
            logger.warning("%s:%s still %.1f GB over quota: the rest is open or pinned",
                           host, args.mask_root, excess / 1e9)
        for c in doomed:
            logger.info("  delete %s:%s (%.2f GB, %.1f h old)%s", host, c.path,
                        c.size / 1e9, (now - c.mtime) / 3600, tag)
        removed = _delete_chunks(host, local_host, doomed, args.dry_run)
        _remove_empty_dirs(host, local_host, args.mask_root, chunks, dirs, removed,
                           now, args.mask_max_age_h, args.dry_run)


def _delete_chunks(host: str, local_host: str, doomed: list[MaskChunk],
                   dry_run: bool) -> set[str]:
    """Delete chunks; returns the paths that are gone (or would be, in a dry run)."""
    if dry_run:
        return {c.path for c in doomed}
    gone: set[str] = set()
    if host == local_host:
        for c in doomed:
            try:
                Path(c.path).unlink()
                gone.add(c.path)
            except FileNotFoundError:
                gone.add(c.path)
            except OSError as exc:
                logger.error("delete failed for %s:%s: %s", host, c.path, exc)
    elif doomed:
        paths = [c.path for c in doomed]
        try:
            _delete_remote(host, paths, "rm -f --")
            gone.update(paths)
        except (OSError, subprocess.SubprocessError) as exc:
            logger.error("delete failed on %s (%d files): %s", host, len(paths), exc)
    return gone


def _remove_empty_dirs(host: str, local_host: str, root: str, chunks: list[MaskChunk],
                       dirs: dict[str, float], removed: set[str], now: float,
                       max_age_h: float, dry_run: bool) -> None:
    """rmdir observation dirs left with no chunk files.

    rmdir refuses a non-empty directory, so a stray file keeps its dir. A dir
    changed in the last MASK_DIR_GRACE_S is left alone: hella may be between
    closing one chunk and opening the next.
    """
    for obs, dir_mtime in sorted(dirs.items()):
        left = [c for c in chunks if c.obs == obs and c.path not in removed]
        touched = any(c.obs == obs and c.path in removed for c in chunks)
        quiet_s = now - dir_mtime
        if left or quiet_s < MASK_DIR_GRACE_S:
            continue
        if not touched and quiet_s < max_age_h * 3600:
            continue          # already empty but young: leave it to a later pass
        path = f"{root.rstrip('/')}/{obs}"
        logger.info("  remove empty dir %s:%s%s", host, path, " [DRY RUN]" if dry_run else "")
        if dry_run:
            continue
        try:
            if host == local_host:
                Path(path).rmdir()
            else:
                _delete_remote(host, [path], "rmdir --")
        except (OSError, subprocess.SubprocessError) as exc:
            logger.debug("rmdir %s:%s refused: %s", host, path, exc)


def main() -> None:
    p = argparse.ArgumentParser(description="Dump-directory janitor (run on corr1)")
    p.add_argument("--db", default=t2db.DEFAULT_PATH)
    p.add_argument("--max-gb", type=float, default=150.0,
                   help="per-directory-tree quota")
    p.add_argument("--max-age-days", type=float, default=7.0)
    p.add_argument("--targets", default="dumps",
                   help="comma list of dumps, hella_masks; hella_masks sweeps the "
                        "hella flag-record chunks on both nodes (no T2 database)")
    p.add_argument("--mask-root", default="/mnt/nvme4/data/casm/hella_masks",
                   help="hella mask record root on each node")
    p.add_argument("--mask-max-age-h", type=float, default=24.0,
                   help="delete closed mask chunks older than this")
    p.add_argument("--mask-max-gb", type=float, default=100.0,
                   help="per-node mask tree quota; oldest closed chunks go first")
    p.add_argument("--mask-pin-max-age-h", type=float, default=72.0,
                   help="a pending trigger card older than this stops pinning chunks")
    p.add_argument("--spool", default=t2_beams.T2_SPOOL_DIR,
                   help="T2 spool dir on each node; its pending cards pin mask chunks")
    p.add_argument("--interval-s", type=float, default=3600.0)
    p.add_argument("--once", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--log-file", default=None)
    args = p.parse_args()
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    bad = [t for t in targets if t not in TARGETS]
    if bad or not targets:
        p.error(f"--targets: unknown {bad or args.targets!r}; choose from {', '.join(TARGETS)}")

    logsetup.setup(args.log_file)
    import socket
    local_host = socket.gethostname().split(".")[0]
    conn = t2db.connect(args.db) if "dumps" in targets else None
    while True:
        if "dumps" in targets:
            try:
                sweep(conn, args, local_host)
            except Exception:
                logger.exception("sweep failed")
        if "hella_masks" in targets:
            try:
                sweep_masks(args, local_host)
            except Exception:
                logger.exception("mask sweep failed")
        if args.once or args.dry_run:
            break
        time.sleep(args.interval_s)


if __name__ == "__main__":
    main()
