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

An archived event folder (schema 1 or 2, casm_io ``load_event``) is re-rendered
from its own .fil and, for the masked view, its stored hella mask:

    t3-replot --event-dir /mnt/nvme3/T3/EVENTS/<cand> --view both --out /tmp/x.png

``--view masked`` draws hella's flags applied (or the raw beam labelled
"hella RFI flags unavailable" when the folder has no mask), ``raw`` the beam
before flagging, ``both`` the two (the raw one at ``<out>_raw.png``).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from casm_io.candidates import t3_event

from casm_t3 import event_archive, fil_reader, single_pulse, plotting
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


VIEWS = ("masked", "raw", "both")


def render_event_dir(event_dir, out_png, view: str = "both", layout: str | None = None,
                     subbands: int | None = None) -> list[Path]:
    """Re-render an archived event folder from its .fil and stored mask.

    ``masked`` is drawn with hella's flags from ``<event>.mask.npz`` (the raw
    beam labelled "hella RFI flags unavailable (reason)" when the folder has no
    mask), ``raw`` is the beam before flagging, labelled with the IB subtraction
    state the folder recorded. ``both`` writes the masked figure to ``out_png``
    and the raw one to ``<out_png stem>_raw.png``. Returns the PNG paths written.
    A mask whose shape differs from the .fil is an error, never a silent resize.
    """
    if view not in VIEWS:
        raise ValueError(f"view must be one of {VIEWS}, not {view!r}")
    ev = t3_event.load_event(event_dir)
    if ev.fil is None:
        raise FileNotFoundError(f"{ev.path}: no {ev.candname}.fil to render from")
    card = event_archive.card_from_event(ev)
    data, fh = fil_reader.read_fil(ev.fil)
    t_rel = (datetime.fromisoformat(card["event_utc"]) - fh.t0).total_seconds()
    span_s = data.shape[1] * fh.tsamp_s
    if not 0 <= t_rel <= span_s:
        logging.getLogger("t3.replot").warning(
            "event time %.2fs falls outside %s of %.2fs; timing offset?",
            t_rel, ev.fil.name, span_s)
    out_png = Path(out_png)

    def draw(out: Path, **kw) -> Path:
        return plotting.make_candidate_figure(data, fh.freqs_mhz, fh.tsamp_s, t_rel, card, out,
                                              layout=layout, subbands=subbands, **kw)

    written = []
    if view in ("masked", "both"):
        if ev.mask is not None:
            if ev.mask.combined.shape != data.shape:
                raise ValueError(f"{ev.candname}: mask is {ev.mask.combined.shape} "
                                 f"(chan, samp), the .fil is {data.shape}")
            written.append(draw(out_png, mask=ev.mask.combined, coverage=ev.mask.coverage))
        else:
            reason = (ev.summary.get("hella_mask") or {}).get("reason") or "no mask in the folder"
            written.append(draw(out_png, mask_label=event_archive.unavailable_label(reason)))
    if view in ("raw", "both"):
        sub = (ev.summary.get("provenance") or {}).get("sub_incoh")
        raw_out = (out_png.with_name(out_png.stem + "_raw.png") if view == "both" else out_png)
        written.append(draw(raw_out, mask_label=event_archive.raw_label(sub)))
    return written


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("card", nargs="?", help="trigger card path (.json or .json.done)")
    p.add_argument("--event-dir", metavar="DIR",
                   help="re-render this archived event folder (schema 1 or 2) from its .fil "
                        "and stored hella mask instead of a card; no dump is looked for")
    p.add_argument("--view", choices=VIEWS, default="both",
                   help="with --event-dir: the figure with hella's flags applied, the raw "
                        "figure, or both (raw at <out>_raw.png) (default: %(default)s)")
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
    p.add_argument("--clean", choices=["none", "hella"], default="none",
                   help="'hella': pass the cutout through the deployed hella flagging stage "
                        "(single_pulse.hella_clean) before plotting, to see roughly what "
                        "the search saw (default: as recorded)")
    p.add_argument("--scrunch", action="append", default=None, metavar="T,F,SIGMA",
                   help="with --clean hella: replace the deployed (never-firing) SCRUNCH "
                        "stage by these, in order, e.g. --scrunch 1,1,5 --scrunch 16,1,4; "
                        "hella semantics (box-average, cut at SIGMA/sqrt(T*F), grow, replace)")
    p.add_argument("--sumthreshold", metavar="KEY=VAL,...", default=None,
                   help="with --clean hella: add a SumThreshold stage against a channel-"
                        "neighbour reference after the scrunch stages; 'default' or e.g. "
                        "chi1=8,rho=1.5,mmin=16,mmax=512,nb=4,grow_t=0 (single_pulse.sumthreshold_flag)")
    args = p.parse_args(argv)
    if bool(args.card) == bool(args.event_dir):
        p.error("give a trigger card or --event-dir, not both and not neither")
    if args.event_dir and (args.clean != "none" or args.scrunch or args.sumthreshold
                           or args.fil or args.dump_dir):
        p.error("--event-dir uses the folder's .fil and stored mask; "
                "--clean/--scrunch/--sumthreshold/--fil/--dump-dir do not apply")
    clean = None if args.clean == "none" else args.clean
    scrunches = [single_pulse.parse_scrunch(x) for x in args.scrunch] if args.scrunch else None
    sumthreshold = (single_pulse.parse_sumthreshold(args.sumthreshold)
                    if args.sumthreshold is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.event_dir:
        name = Path(args.event_dir).resolve().name
        try:
            pngs = render_event_dir(args.event_dir, args.out or f"{name}_replot.png",
                                    view=args.view, layout=args.layout, subbands=args.subbands)
        except (OSError, ValueError) as exc:
            sys.exit(f"cannot re-render {args.event_dir}: {exc}")
        for png in pngs:
            print(png)
        return

    card_path = Path(args.card)
    card = json.loads(card_path.read_text())
    out_png = Path(args.out or f"{card['candname']}_replot.png")

    if args.fil:
        fil = Path(args.fil)
        if not fil.is_file():
            sys.exit(f"{fil} does not exist")
        png, _ = dump_plotter.render_card_from_fil(card, fil, out_png, layout=args.layout,
                                                   subbands=args.subbands, clean=clean,
                                                   scrunches=scrunches, sumthreshold=sumthreshold)
        print(png)
        return

    files = card_dump_files(card, card_path, args.dump_dir)
    if files:
        png, _ = dump_plotter.render_card(card, files, out_png, layout=args.layout,
                                          subbands=args.subbands, clean=clean,
                                                   scrunches=scrunches, sumthreshold=sumthreshold)
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
                                               subbands=args.subbands, clean=clean,
                                                   scrunches=scrunches, sumthreshold=sumthreshold)
    print(png)


if __name__ == "__main__":
    main()
