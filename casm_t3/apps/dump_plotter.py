"""Per-node worker: turn fresh beam dumps into candidate plots and alerts.

Runs on each backend node. T2 drops a JSON trigger card in the local spool
directory when it requests a dump from one of this node's casm_cand_dump
daemons; this daemon picks the card up, waits for the .dada file(s) to
finish landing, extracts the detection beam, renders the candidate figure,
posts it to Slack, and ships the small artifacts (PNG + JSON) to the
archive directory on corr1. Bulk dump data never leaves the node.

Processed cards are renamed to .done (or .failed) in place, which doubles
as the restart bookkeeping.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path

from casm_t2 import beams as t2_beams
from casm_t2 import logsetup

from casm_t3 import alerts, dump_reader, plotting

logger = logging.getLogger("t3.dump_plotter")

LOCAL_HOSTNAME = socket.gethostname().split(".")[0]


def find_dump_files(dump_dir: Path, not_before: float,
                    not_after: float | None = None) -> list[Path]:
    """List .dada files in dump_dir whose mtime falls in the given window.

    Triggers are rate-limited to >=120 s apart per stream, so a window of a
    minute or so around a card's mtime isolates that card's dump from its
    neighbours in the shared stream directory.
    """
    hits = [p for p in dump_dir.glob("*.dada")
            if p.stat().st_mtime >= not_before
            and (not_after is None or p.stat().st_mtime <= not_after)]
    return sorted(hits, key=lambda p: p.name)


def wait_for_dump(dump_dir: Path, after_mtime: float, timeout_s: float,
                  settle_s: float = 3.0) -> list[Path]:
    """Wait for new .dada files in dump_dir and for their sizes to settle."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        cands = [p for p in dump_dir.glob("*.dada") if p.stat().st_mtime >= after_mtime]
        if cands:
            sizes = {p: p.stat().st_size for p in cands}
            time.sleep(settle_s)
            again = [p for p in dump_dir.glob("*.dada") if p.stat().st_mtime >= after_mtime]
            if {p: p.stat().st_size for p in again} == sizes and again:
                return sorted(again, key=lambda p: p.name)
        time.sleep(2.0)
    return []


def ship_artifacts(files: list[Path], candname: str, archive_host: str, archive_dir: str) -> None:
    """Copy small result files to the archive tree on the archive host."""
    dest = f"{archive_dir}/{candname}"
    if archive_host == LOCAL_HOSTNAME:
        Path(dest).mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(f, dest)
    else:
        subprocess.run(["ssh", archive_host, f"mkdir -p {dest}"], check=True, timeout=30)
        subprocess.run(["scp", "-q", *map(str, files), f"{archive_host}:{dest}/"],
                       check=True, timeout=120)
    logger.info("archived %s -> %s:%s", candname, archive_host, dest)


def render_card(card: dict, dump_files: list[Path], out_png: Path) -> tuple[Path, dict]:
    """Read the detection beam from dump_files and render the candidate figure.

    Pure transform — no dump triggers, no artifact shipping, no alerting —
    so the offline t3-replot CLI can share it safely with the daemon.
    Returns the PNG path and the result record (card + provenance) for the
    caller to persist or discard.
    """
    header, data = dump_reader.read_beams(dump_files, [card["local_beam"]])
    event_utc = datetime.fromisoformat(card["event_utc"])
    t_rel = (event_utc - header.t0).total_seconds()
    if not 0 <= t_rel <= data.shape[2] * header.tsamp_s:
        logger.warning("event time %.2fs falls outside dump of %.2fs — timing offset?",
                       t_rel, data.shape[2] * header.tsamp_s)

    png = plotting.make_candidate_figure(
        data[0], header.freqs_mhz, header.tsamp_s, t_rel, card, out_png)

    result = dict(card)
    result.update(host=LOCAL_HOSTNAME, dump_files=[str(f) for f in dump_files],
                  n_samples=int(data.shape[2]), plot=str(png))
    return png, result


def process_card(card_path: Path, args: argparse.Namespace) -> None:
    card = json.loads(card_path.read_text())
    candname = card["candname"]
    dump_dir = Path(card["dump_dir"])
    logger.info("processing %s (beam %d, snr %.1f)", candname, card["beam"], card["snr"])

    files = wait_for_dump(dump_dir, after_mtime=card_path.stat().st_mtime - 10,
                          timeout_s=args.dump_timeout)
    if not files:
        raise RuntimeError(f"no dump appeared in {dump_dir} within {args.dump_timeout}s")

    plots_dir = Path(args.plots_dir)
    png, result = render_card(card, files, plots_dir / f"{candname}.png")
    # Plot-then-delete keeps the dump budget disk-neutral: once the figure
    # and result JSON are archived, the bulk .dada has served its purpose.
    # Known-source dumps are kept for folding (the janitor ages them out),
    # and a failed render never reaches this point, so its dump survives.
    delete_after = (args.delete_dumps
                    and not str(card.get("trigger_reason", "")).startswith("known_source"))
    result["data_available"] = not delete_after
    result_json = plots_dir / f"{candname}.json"
    result_json.write_text(json.dumps(result, indent=2))

    ship_artifacts([png, result_json], candname, args.archive_host, args.archive_dir)
    alerts.post_candidate(
        png,
        f"*{card['source']}* single-pulse candidate `{candname}`: "
        f"S/N {card['snr']:.1f}, DM {card['dm']:.2f}, beam {card['beam']}, "
        f"width {card['width']} samp ({LOCAL_HOSTNAME})",
        channel=args.slack_channel)

    if delete_after:
        for f in files:
            try:
                size_gb = f.stat().st_size / 1e9
                f.unlink()
                logger.info("deleted dump %s (%.1f GB) after plotting", f.name, size_gb)
            except OSError as exc:
                logger.warning("could not delete %s: %s", f, exc)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot and alert on fresh beam dumps")
    p.add_argument("--spool", default=t2_beams.T2_SPOOL_DIR)
    p.add_argument("--plots-dir", default="/mnt/nvme4/data/casm/t3_plots")
    p.add_argument("--archive-host", default="casm-corr1")
    p.add_argument("--archive-dir", default="/mnt/nvme5/casm_pipeline/candidates")
    p.add_argument("--slack-channel", default="casm-alerts")
    p.add_argument("--dump-timeout", type=float, default=180.0)
    p.add_argument("--poll", type=float, default=2.0)
    p.add_argument("--log-file", default=None,
                   help="also log to this file (journald always gets a copy)")
    p.add_argument("--delete-dumps", action="store_true",
                   help="delete .dada files after a successful plot (known-source dumps kept)")
    args = p.parse_args()

    logsetup.setup(args.log_file)
    spool = Path(args.spool)
    spool.mkdir(parents=True, exist_ok=True)
    Path(args.plots_dir).mkdir(parents=True, exist_ok=True)
    logger.info("watching %s on %s", spool, LOCAL_HOSTNAME)

    while True:
        for card_path in sorted(spool.glob("*.json")):
            try:
                process_card(card_path, args)
            except Exception as exc:  # noqa: BLE001 - one bad card must not stop the loop
                logger.exception("failed on %s: %s", card_path.name, exc)
                card_path.rename(card_path.with_suffix(".json.failed"))
            else:
                card_path.rename(card_path.with_suffix(".json.done"))
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
