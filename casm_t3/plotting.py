"""Candidate diagnostic figure: waterfall, dedispersed waterfall and profile."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from . import single_pulse


def make_candidate_figure(data: np.ndarray, freqs_mhz: np.ndarray, tsamp_s: float,
                          t_rel_event_s: float, card: dict, out_png: str | Path,
                          ffactor: int = 8) -> Path:
    """Render the standard three-panel candidate plot.

    Parameters
    ----------
    data : (nchan, ntime) float32, raw (dispersed) cutout for the detection beam.
    t_rel_event_s : time of the candidate (top-of-band arrival) relative to
        the first sample of ``data``.
    card : trigger-card dict with candname/source/snr/dm/width/beam/event_utc.
    """
    dm, width = card["dm"], int(card["width"])
    norm = single_pulse.normalise(data)
    dedis = single_pulse.dedisperse(norm, dm, freqs_mhz, tsamp_s)

    tfactor = max(1, width // 2)
    wf_raw = single_pulse.downsample(norm, ffactor, tfactor)
    wf_dd = single_pulse.downsample(dedis, ffactor, tfactor)
    prof = single_pulse.profile_snr(dedis.mean(axis=0), width)

    t_axis = (np.arange(wf_raw.shape[1]) * tfactor + tfactor / 2) * tsamp_s - t_rel_event_s
    t_prof = np.arange(prof.size) * tsamp_s - t_rel_event_s
    extent = [t_axis[0], t_axis[-1], freqs_mhz[-1], freqs_mhz[0]]
    vmin, vmax = np.percentile(wf_raw, [2, 98])

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True,
                             gridspec_kw={"height_ratios": [1.2, 2, 2]})
    ax_prof, ax_dd, ax_raw = axes

    ax_prof.plot(t_prof, prof, "k-", lw=0.8)
    ax_prof.axvline(0, color="r", alpha=0.3, lw=1)
    ax_prof.set_ylabel("S/N")
    ax_prof.set_title(
        f"{card['candname']}   {card['source']}   "
        f"S/N={card['snr']:.1f}  DM={dm:.2f}  width={width}  "
        f"beam {card['beam']}\n{card['event_utc']}", fontsize=10)

    ax_dd.imshow(wf_dd, aspect="auto", extent=extent, vmin=vmin, vmax=vmax, cmap="viridis")
    ax_dd.set_ylabel("Freq (MHz)  [dedispersed]")

    ax_raw.imshow(wf_raw, aspect="auto", extent=extent, vmin=vmin, vmax=vmax, cmap="viridis")
    ax_raw.set_ylabel("Freq (MHz)  [raw]")
    ax_raw.set_xlabel("Time - event (s)")

    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png
