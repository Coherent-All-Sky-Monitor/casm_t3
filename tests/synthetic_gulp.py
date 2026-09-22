"""Synthetic trigger cards with a ``gulp`` block (casm_t2 CARD_SCHEMA) for the
candidate-figure tests.

Timing is self-consistent: the event UTC is utc_start + samp * tsamp, and the
event sample lies inside [samp_lo, samp_hi]. The cases cover what the bottom
row has to survive:

    default             compact triggered cluster near the gulp's trailing
                        edge, a few other clusters (one sky-wide), trials on
                        the gulp's first and last samples, one trial whose
                        cluster is not listed (t2d's cluster cap)
    storm               thousands of trials in hundreds of beams, stormy,
                        trial list capped (trials_truncated)
    empty               no trials, no clusters
    single_beam         one trial, one beam, the triggered cluster
    sky_wide_triggered  the triggered cluster spans 40 beams across the array

    python tests/synthetic_gulp.py /tmp/outdir     # renders every case
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

TSAMP_S = 1.048576e-3
GULP_SAMP = 8192
GULP = 1000
UTC_START = "2026-09-18-05:27:45"
SAMP_LO = GULP * GULP_SAMP
SAMP_HI = SAMP_LO + GULP_SAMP - 1              # 8192 samples, 8.59 s
EVENT_SAMP = SAMP_HI - 300                     # near the trailing gulp edge
NBEAM = 512

CASES = ("default", "storm", "empty", "single_beam", "sky_wide_triggered")

RULES = {"snr": 18.0, "dm_floor": 20.0, "max_beams": 8, "max_extent_deg": 25.0,
         "storm_skip": 20000, "quiet_sum": 2000, "injections_dump": False}

# Every source the legend can list, including the "no beam" case that sizes it.
SOURCES = {"Sun": (38.0, 210.0), "Cas A": (17.6, 30.0), "Cyg A": (-5.0, 300.0),
           "Tau A": (50.0, 100.0)}


def event_utc(samp: int = EVENT_SAMP) -> str:
    """ISO UTC of a gulp sample: utc_start + samp * tsamp."""
    t0 = datetime.strptime(UTC_START, "%Y-%m-%d-%H:%M:%S").replace(tzinfo=timezone.utc)
    return (t0 + timedelta(seconds=samp * TSAMP_S)).isoformat(timespec="milliseconds")


def _dt(samp: float) -> float:
    return round((float(samp) - EVENT_SAMP) * TSAMP_S, 4)


class _Builder:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.clusters: list[dict] = []
        self.trials: list[list] = []

    def cluster(self, outcome: str, samp: float, dm: float, beams: list[tuple[int, float]],
                n_trials: int, extent: float = 1.0, cid: int | None = None,
                listed: bool = True) -> int:
        """Add a cluster whose trials sit within ~30 ms of ``samp`` in its beams."""
        rng = self.rng
        cid = len(self.clusters) if cid is None else cid
        best = max(beams, key=lambda bs: bs[1])
        for i in range(n_trials):
            b, s_max = beams[i % len(beams)] if i < len(beams) else beams[rng.integers(len(beams))]
            s = float(np.clip(samp + rng.normal(0, 30), SAMP_LO, SAMP_HI))
            snr = s_max if i < len(beams) else max(12.0, s_max - abs(rng.normal(0, 2.0)))
            self.trials.append([_dt(s), int(b), round(float(dm + rng.normal(0, 1.0)), 2),
                                round(float(snr), 2), int(rng.integers(1, 6)), cid])
        if listed:
            self.clusters.append({
                "id": cid, "peak": [_dt(samp), int(best[0]), float(dm), float(best[1]), 3],
                "n_trials": n_trials, "beams": [[int(b), float(s)] for b, s in beams],
                "sky_extent_deg": float(extent), "dm_lo": max(0.0, dm - 10), "dm_hi": dm + 10,
                "outcome": outcome, "detail": ""})
        return cid

    def block(self, **extra) -> dict:
        self.trials.sort(key=lambda r: r[0])        # t2d writes them time-ordered
        gulp = {"gulp": GULP, "utc_start": UTC_START, "samp_lo": SAMP_LO,
                "samp_hi": SAMP_HI, "tsamp_s": TSAMP_S, "weights_id": None,
                "n_raw": len(self.trials) + 40, "n_width_dropped": 40, "n_shed": 0,
                "trials": self.trials, "clusters": self.clusters, "rules": RULES,
                "storm": {"raw_recent": [len(self.trials)], "stormy": False}}
        gulp.update(extra)
        return gulp


def _default(b: _Builder) -> dict:
    b.cluster("triggered", EVENT_SAMP, 29.5, [(335, 19.8), (336, 24.1), (337, 17.4)], 40)
    b.cluster("snr_below", EVENT_SAMP - 3000, 121.0, [(99, 15.1), (100, 16.9)], 12)
    # shares beam 336 with the triggered cluster: that beam must stay red only
    b.cluster("dm_below", EVENT_SAMP - 1100, 8.4, [(336, 21.0), (50, 22.3)], 8)
    wide = [(int(x), float(12 + 10 * b.rng.random()))
            for x in b.rng.choice(np.arange(NBEAM), size=24, replace=False)]
    b.cluster("too_many_beams", EVENT_SAMP - 6000, 148.0, wide, 60, extent=61.0)
    # a trial whose cluster t2d did not list (clusters_truncated)
    b.cluster("snr_below", EVENT_SAMP - 4000, 300.0, [(420, 13.0)], 1, cid=999, listed=False)
    # trials on the gulp's first and last samples; dt_s is rounded to 0.1 ms in
    # a card, and here the first one is rounded down, a hair before samp_lo
    b.trials.append([math.floor((SAMP_LO - EVENT_SAMP) * TSAMP_S * 1e4) / 1e4, 7, 55.0, 12.5, 2, 1])
    b.trials.append([_dt(SAMP_HI), 8, 55.0, 12.5, 2, 1])
    return b.block(n_trials_total=len(b.trials), trials_truncated=False,
                   n_clusters_total=len(b.clusters) + 1, clusters_truncated=True)


def _storm(b: _Builder) -> dict:
    b.cluster("triggered", EVENT_SAMP, 60.0, [(64, 30.0), (65, 22.0)], 30)
    for k in range(24):
        beams = [(int(x), float(12 + 20 * b.rng.random()))
                 for x in b.rng.choice(np.arange(NBEAM), size=12, replace=False)]
        b.cluster("storm", SAMP_LO + b.rng.integers(0, GULP_SAMP), float(20 + 300 * b.rng.random()),
                  beams, 120, extent=40.0)
    return b.block(n_trials_total=4 * len(b.trials), trials_truncated=True,
                   storm={"raw_recent": [18000, 21000, 25000], "stormy": True})


def _empty(b: _Builder) -> dict:
    return b.block(n_raw=0, n_width_dropped=0)


def _single_beam(b: _Builder) -> dict:
    b.cluster("triggered", EVENT_SAMP, 587.0, [(4, 16.0)], 1)
    return b.block()


def _sky_wide_triggered(b: _Builder) -> dict:
    wide = [(int(x), float(14 + 10 * b.rng.random()))
            for x in b.rng.choice(np.arange(NBEAM), size=40, replace=False)]
    b.cluster("triggered", EVENT_SAMP, 45.0, wide, 80, extent=70.0)
    b.cluster("snr_below", EVENT_SAMP - 2000, 70.0, [(500, 13.0), (501, 12.5)], 4)
    b.cluster("snr_below", EVENT_SAMP - 5000, 90.0, [(10, 14.0)], 2)
    return b.block()


def build_card(case: str = "default", candname: str | None = None, seed: int = 11) -> dict:
    """A trigger card for one of CASES, with the triggered cluster's peak as
    the card's beam, DM and S/N."""
    if case not in CASES:
        raise ValueError(f"case must be one of {CASES}")
    b = _Builder(seed)
    gulp = {"default": _default, "storm": _storm, "empty": _empty,
            "single_beam": _single_beam, "sky_wide_triggered": _sky_wide_triggered}[case](b)
    trig = next((c for c in gulp["clusters"] if c["outcome"] == "triggered"), None)
    beam, dm, snr = ((trig["peak"][1], trig["peak"][2], trig["peak"][3]) if trig
                     else (336, 29.5, 20.0))
    grid = pointings()
    return {
        "candname": candname or f"SYNTH{case}", "source": "blind",
        "event_utc": event_utc(), "samp": EVENT_SAMP,
        "beam": int(beam), "local_beam": int(beam) % 64, "stream": int(beam) // 64,
        "snr": float(snr), "dm": float(dm), "width": 3,
        "sky": {"alt_deg": grid["alt_deg"][beam], "az_deg": grid["az_deg"][beam],
                "ra_deg": 150.0, "dec_deg": -5.0, "sun_alt_deg": SOURCES["Sun"][0],
                "sun_az_deg": SOURCES["Sun"][1], "sources": dict(SOURCES)},
        "pointings": grid,
        "gulp": gulp,
    }


def pointings(nbeam: int = NBEAM) -> dict:
    """A 512-beam alt/az grid, one distinct pointing per beam."""
    rows, per = 16, nbeam // 16
    alt, az = [], []
    for i in range(nbeam):
        r, c = divmod(i, per)
        alt.append(25.0 + 60.0 * (r + 0.5) / rows)
        az.append(90.0 + 150.0 * (c + 0.5) / per)
    return {"alt_deg": alt, "az_deg": az}


def synthetic_data(dm: float, nchan: int = 512, ntime: int = 4096, t_event_samp: int = 2048,
                   width_samp: int = 8, seed: int = 1):
    """(data, freqs_mhz, tsamp_s, t_rel_event_s): CASM band in ``nchan``
    channels, noise plus one dispersed pulse arriving at ``t_event_samp``."""
    rng = np.random.default_rng(seed)
    freqs = 484.375 - np.arange(nchan) * (0.030517578125 * 3072 / nchan)
    data = rng.normal(100, 10, (nchan, ntime)).astype(np.float32)
    delays = np.round(4.148808e3 * dm * (freqs ** -2 - freqs.max() ** -2) / TSAMP_S).astype(int)
    for c in range(nchan):
        lo = t_event_samp + delays[c]
        data[c, lo: lo + width_samp] += 6.0
    return data, freqs, TSAMP_S, t_event_samp * TSAMP_S


def main() -> None:
    from casm_t3 import plotting

    out = Path(sys.argv[1] if len(sys.argv) > 1 else "synthetic_cards")
    out.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        card = build_card(case)
        data, freqs, tsamp, t_rel = synthetic_data(card["dm"])
        png = plotting.make_candidate_figure(data, freqs, tsamp, t_rel, card,
                                             out / f"{case}.png")
        (out / f"{case}.json").write_text(json.dumps(card, indent=1))
        print(png)


if __name__ == "__main__":
    main()
