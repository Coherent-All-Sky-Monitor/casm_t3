"""Offline re-render of a candidate figure from an existing trigger card.

The iteration loop for plotter development: point it at a spool card
(.json or .json.done) on the node that holds the dump and it re-renders
the figure from the .dada already on disk. It never triggers dumps, ships
artifacts, posts to Slack, or renames spool cards (requeueing a card for
the live daemon does all four).

    t3-replot /mnt/nvme4/data/casm/t2_spool/<cand>.json.done --out /tmp/x.png

When the dump is gone (every archived event after a few minutes, and every
T2 backtest card) the same figure is rendered from the per-event single-beam
filterbank instead: the card's own ``fil`` if that file exists, else
<events-root>/<candname>/<candname>.fil.

    t3-replot /mnt/nvme3/T3/EVENTS/<cand>/<cand>.json --out /tmp/x.png
    t3-replot <card> --fil /path/to/<cand>.fil --out /tmp/x.png

``--events-root ''`` turns that fallback off. A card with no dump directory
never falls back to the current directory: this deployment's working
directory can itself be a dump directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from casm_t3 import plotting
from casm_t3.apps import dump_plotter

# Dump .dada files land within seconds of the card being written, and
# triggers on a stream are >=120 s apart, so a one-sided minute window
# around the card mtime picks out exactly this card's dump.
MTIME_SLACK_BEFORE_S = 10.0
MTIME_SLACK_AFTER_S = 60.0

# Where the per-event archive keeps <candname>/<candname>.fil (dump_plotter
# --events-root); the .fil outlives the .dada, so it is the offline source.
DEFAULT_EVENTS_ROOT = "/mnt/nvme3/T3/EVENTS"


def archived_fil(card: dict, events_root: str | None) -> Path | None:
    """The .fil to fall back to, or None when there is none or the fallback is off.

    The card's own ``fil`` if it names an existing file, else
    <events_root>/<candname>/<candname>.fil if that exists. An empty or None
    ``events_root`` disables the fallback entirely.
    """
    if not events_root:
        return None
    own = card.get("fil")
    if own and Path(own).is_file():
        return Path(own)
    candname = card["candname"]
    path = Path(events_root) / candname / f"{candname}.fil"
    return path if path.is_file() else None


def card_dump_files(card: dict, card_path: Path, dump_dir: str | None) -> list[Path]:
    """This card's .dada files, or [] when its dump directory is unset or gone.

    Only an explicit directory is searched (``--dump-dir`` or the card's
    ``dump_dir``); an empty value never means the current directory.
    """
    dump_dir = dump_dir or card.get("dump_dir")
    if not dump_dir or not str(dump_dir).strip():
        return []
    d = Path(dump_dir)
    if not d.is_dir():
        return []
    mtime = card_path.stat().st_mtime
    return dump_plotter.find_dump_files(d, mtime - MTIME_SLACK_BEFORE_S,
                                        mtime + MTIME_SLACK_AFTER_S)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("card", help="trigger card path (.json or .json.done)")
    p.add_argument("--out", help="output PNG (default: ./<candname>_replot.png)")
    p.add_argument("--dump-dir", help="override the dump directory in the card")
    p.add_argument("--fil", help="render from this single-beam .fil; no dump is looked for")
    p.add_argument("--events-root", default=DEFAULT_EVENTS_ROOT,
                   help="per-event archive searched for <candname>/<candname>.fil when "
                        "the card has no dump on this node (default: %(default)s); "
                        "--events-root '' disables the .fil fallback")
    p.add_argument("--layout", choices=list(plotting.LAYOUTS), default=None,
                   help="figure layout (default: v2, which draws the approved v3 figure)")
    p.add_argument("--subbands", type=int, default=None,
                   help="force the number of frequency rows in the waterfall and DM-time "
                        "panels (default: from the S/N, plotting.subbands_for)")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    card_path = Path(args.card)
    card = json.loads(card_path.read_text())
    out_png = Path(args.out or f"{card['candname']}_replot.png")

    if args.fil:
        fil = Path(args.fil)
        if not fil.is_file():
            sys.exit(f"{fil} does not exist")
        png, _ = dump_plotter.render_card_from_fil(card, fil, out_png, layout=args.layout,
                                                   subbands=args.subbands)
        print(png)
        return

    files = card_dump_files(card, card_path, args.dump_dir)
    if files:
        png, _ = dump_plotter.render_card(card, files, out_png, layout=args.layout,
                                          subbands=args.subbands)
        print(png)
        return

    fil = archived_fil(card, args.events_root)
    if fil is None:
        where = args.dump_dir or card.get("dump_dir") or "(no dump_dir in the card)"
        fallback = ("the .fil fallback is off (--events-root '')" if not args.events_root
                    else f"no archived .fil under {args.events_root}")
        sys.exit(f"nothing to plot for {card['candname']}: no .dada in {where} within "
                 f"[-{MTIME_SLACK_BEFORE_S:.0f}, +{MTIME_SLACK_AFTER_S:.0f}] s of the card "
                 f"mtime (wrong node, or dump cleaned up) and {fallback}")
    logging.getLogger("t3.replot").info("no dump on this node; rendering from %s", fil)
    png, _ = dump_plotter.render_card_from_fil(card, fil, out_png, layout=args.layout,
                                               subbands=args.subbands)
    print(png)


if __name__ == "__main__":
    main()
