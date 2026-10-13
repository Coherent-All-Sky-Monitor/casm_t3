"""Shared setup for the event-folder tests: a spool card, a synthetic dump, hella records.

``make_env`` lays out tmp dirs, writes the dump (synthetic_event) and, optionally, the
record file, and returns an ``Env`` whose ``args`` run ``dump_plotter.process_card``
offline: the archive host is this host (a local copy), no sleeps, no network.
``stub_figure`` replaces the figure renderer with a recorder that writes a few bytes.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import synthetic_event as se
from casm_t3 import plotting
from casm_t3.apps import dump_plotter

T3_COMMIT = "abc123def456"


@dataclass
class Env:
    tmp: Path
    dump_dir: Path
    dump_files: list
    records: Path
    events: Path
    plots: Path
    archive: Path
    card: dict
    card_path: Path
    args: argparse.Namespace

    @property
    def event_dir(self) -> Path:
        return self.events / se.CANDNAME

    @property
    def dest(self) -> Path:
        return self.archive / se.CANDNAME

    def process(self) -> None:
        dump_plotter.process_card(self.card_path, self.args)

    def summary(self) -> dict:
        return json.loads((self.event_dir / f"{se.CANDNAME}.json").read_text())


def make_env(tmp_path, monkeypatch, records: bool = True, ks=(0, 1, 2), gap: bool = False,
             dump_extra: dict | None = None, **args_over) -> Env:
    dump_dir = tmp_path / "cand_beam_dumps" / "stream_1"
    files = se.write_dump(dump_dir, gap=gap, **(dump_extra or {}))
    root = tmp_path / "hella_masks"
    root.mkdir()
    if records:
        se.write_records(root, ks=ks)
    card = se.card(dump_dir)
    card_path = tmp_path / "spool" / f"{se.CANDNAME}.json"
    card_path.parent.mkdir()
    card_path.write_text(json.dumps(card))
    args = argparse.Namespace(
        plots_dir=str(tmp_path / "plots"), archive_host=dump_plotter.LOCAL_HOSTNAME,
        archive_dir=str(tmp_path / "candidates"), events_root=str(tmp_path / "EVENTS"),
        dump_timeout=1.0, delete_dumps=True, mask_records=str(root), t3_commit=T3_COMMIT)
    for k, v in args_over.items():
        setattr(args, k, v)
    monkeypatch.setattr(dump_plotter, "wait_for_dump",
                        lambda d, after_mtime, timeout_s, settle_s=3.0:
                        sorted(Path(d).glob("*.dada")))
    return Env(tmp_path, dump_dir, files, root, tmp_path / "EVENTS", tmp_path / "plots",
               tmp_path / "candidates", card, card_path, args)


def stub_figure(monkeypatch, fail=None) -> list:
    """Replace make_candidate_figure by a recorder; ``fail(name, kwargs)`` True raises ValueError."""
    calls: list = []

    def fake(data, freqs, tsamp, t_rel, card, out_png, **kw):
        out = Path(out_png)
        calls.append(dict(kw, out=out.name, shape=data.shape, card=card, t_rel=t_rel))
        if fail is not None and fail(out.name, kw):
            raise ValueError("stub figure failure")
        out.write_bytes(b"PNG stub " + out.name.encode())
        return out

    monkeypatch.setattr(plotting, "make_candidate_figure", fake)
    return calls


def call_for(calls: list, name: str) -> dict:
    """The last recorded figure call that wrote the file ``name``."""
    return [c for c in calls if c["out"] == name][-1]
