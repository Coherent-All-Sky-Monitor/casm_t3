"""Offline re-render of a candidate figure from an existing trigger card.

The iteration loop for plotter development: point it at a spool card
(.json or .json.done) on the node that holds the dump and it re-renders
the figure from the .dada already on disk. It never triggers dumps, ships
artifacts, posts to Slack, or renames spool cards — unlike requeueing a
card for the live daemon, which does all four.

    t3-replot /mnt/nvme4/data/casm/t2_spool/<cand>.json.done --out /tmp/x.png
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from casm_t3.apps import dump_plotter

# Dump .dada files land within seconds of the card being written, and
# triggers on a stream are >=120 s apart, so a one-sided minute window
# around the card mtime picks out exactly this card's dump.
MTIME_SLACK_BEFORE_S = 10.0
MTIME_SLACK_AFTER_S = 60.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("card", help="trigger card path (.json or .json.done)")
    p.add_argument("--out", help="output PNG (default: ./<candname>_replot.png)")
    p.add_argument("--dump-dir", help="override the dump directory in the card")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    card_path = Path(args.card)
    card = json.loads(card_path.read_text())
    dump_dir = Path(args.dump_dir or card["dump_dir"])

    mtime = card_path.stat().st_mtime
    files = dump_plotter.find_dump_files(
        dump_dir, mtime - MTIME_SLACK_BEFORE_S, mtime + MTIME_SLACK_AFTER_S)
    if not files:
        sys.exit(f"no .dada in {dump_dir} within "
                 f"[-{MTIME_SLACK_BEFORE_S:.0f}, +{MTIME_SLACK_AFTER_S:.0f}] s "
                 f"of the card mtime — wrong node, or dump cleaned up?")

    out_png = Path(args.out or f"{card['candname']}_replot.png")
    png, _ = dump_plotter.render_card(card, files, out_png)
    print(png)


if __name__ == "__main__":
    main()
