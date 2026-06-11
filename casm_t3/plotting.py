"""Candidate diagnostic figure.

Layout follows the DSA-110 candidate plotter and transientX conventions:

    dedispersed profile   | DM=0 raw timeseries
    dedispersed freq-time | DM-time bowtie
    beam vs time of T1 context candidates (full width)

The beam panel is driven by the T1 "context" candidates that casm_t2's
source watcher embeds in the trigger card: every candidate from every beam
within a few seconds of the event. RFI lights up many beams at unrelated
DMs; a real pulse stays compact in beam and DM.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors

from . import single_pulse

NBEAM_TOTAL = 512

# transientX uses viridis for both image panels (candplot.cpp: ax_ft and
# ax_dmt both pcolor with "viridis"); match it here.
WATERFALL_CMAP = "viridis"

# DM colour scale for the beam scatter: plasma with the bright-yellow top
# cut off — full plasma is unreadable on the white panel background.
BEAM_DM_CMAP = mcolors.ListedColormap(
    plt.get_cmap("plasma")(np.linspace(0.0, 0.8, 256)), name="plasma_dark")


def _block_mean_freqs(freqs_mhz: np.ndarray, ffactor: int) -> np.ndarray:
    n = (freqs_mhz.size // ffactor) * ffactor
    return freqs_mhz[:n].reshape(-1, ffactor).mean(axis=1)


def _beam_panel(ax, card: dict, fig) -> None:
    """Beam-time scatter of T1 context candidates, coloured by DM."""
    ctx = card.get("context") or {}
    members = ctx.get("members") or []
    beam = int(card["beam"])

    if not members:
        ax.text(0.5, 0.5, "no context candidates", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="0.4")
    else:
        m = np.asarray(members, dtype=float)  # columns: dt, beam, dm, snr, width
        dt, mbeam, mdm, msnr = m[:, 0], m[:, 1], m[:, 2], m[:, 3]

        sc = ax.scatter(dt, mbeam, c=mdm, s=4 + 2 * np.clip(msnr - 10, 0, 20),
                        cmap=BEAM_DM_CMAP, vmin=0, vmax=max(50.0, 1.5 * card["dm"]),
                        alpha=0.85, linewidths=0)
        # Colorbar in an inset just outside the axes: fig.colorbar(ax=...)
        # would shrink this panel and break its alignment with the row above.
        cax = ax.inset_axes((1.008, 0.0, 0.012, 1.0))
        fig.colorbar(sc, cax=cax, label=r"DM (pc cm$^{-3}$)")
        for edge in range(64, NBEAM_TOTAL, 64):
            ax.axhline(edge, color="0.85", lw=0.4, zorder=0)

    ax.scatter([0], [beam], marker="s", s=90, facecolors="none",
               edgecolors="red", linewidths=1.2, zorder=5)
    window_s = float(ctx.get("window_s", 4.0))
    ax.set_xlim(-window_s, window_s)
    ax.set_ylim(-5, NBEAM_TOTAL + 4)
    ax.set_xlabel("Time - event (s)")
    ax.set_ylabel("Beam")


def make_candidate_figure(data: np.ndarray, freqs_mhz: np.ndarray, tsamp_s: float,
                          t_rel_event_s: float, card: dict, out_png: str | Path,
                          ffactor: int = 8) -> Path:
    """Render the candidate plot.

    Parameters
    ----------
    data : (nchan, ntime) float32, raw (dispersed) cutout for the detection beam.
    t_rel_event_s : candidate time (top-of-band arrival) relative to the
        first sample of ``data``.
    card : trigger-card dict (candname/source/snr/dm/width/beam/event_utc,
        optional context member list).

    The waterfall and dedispersed profile use per-channel normalised data
    (the bandpass would otherwise drown the pulse), so the profile y-axis is
    band-averaged power in those normalised units; a boxcar S/N at the
    candidate width is annotated on the panel rather than used as the axis.
    The DM=0 panel is the raw band-averaged timeseries, no normalisation.
    """
    dm, width = float(card["dm"]), int(card["width"])
    norm = single_pulse.normalise(data)
    dedis = single_pulse.dedisperse(norm, dm, freqs_mhz, tsamp_s)

    tfactor = max(1, width // 2)
    wf_dd = single_pulse.downsample(dedis, ffactor, tfactor)
    prof_dd = wf_dd.mean(axis=0)

    raw_dm0 = data.mean(axis=0)
    n = (raw_dm0.size // tfactor) * tfactor
    raw_dm0 = raw_dm0[:n].reshape(-1, tfactor).mean(axis=1)

    # Matched-boxcar S/N near the event, quoted on the profile panel.
    prof_full = single_pulse.profile_snr(dedis.mean(axis=0), width)
    t_full = np.arange(prof_full.size) * tsamp_s - t_rel_event_s
    near = np.abs(t_full) <= 0.5
    snr_box = prof_full[near].max() if near.any() else prof_full.max()

    small = single_pulse.downsample(norm, ffactor, 1)
    f_small = _block_mean_freqs(freqs_mhz, ffactor)[: small.shape[0]]
    dms = single_pulse.dm_grid(dm)
    dmt = single_pulse.dm_time(small, f_small, tsamp_s, dms, width)
    dmt_disp = single_pulse.downsample(dmt, 1, tfactor)

    t_wf = (np.arange(wf_dd.shape[1]) * tfactor + tfactor / 2) * tsamp_s - t_rel_event_s

    fig = plt.figure(figsize=(12, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=(1.0, 1.5, 1.1))
    ax_prof = fig.add_subplot(gs[0, 0])
    ax_dm0 = fig.add_subplot(gs[0, 1])
    ax_wf = fig.add_subplot(gs[1, 0])
    ax_dmt = fig.add_subplot(gs[1, 1])
    ax_bt = fig.add_subplot(gs[2, :])

    ax_prof.plot(t_wf, prof_dd, "k-", lw=0.7)
    ax_prof.axvline(0, color="red", alpha=0.4, lw=1)
    ax_prof.set_ylabel("Power (arb.)")
    ax_prof.set_title(f"dedispersed at DM={dm:.2f}", fontsize=9)
    ax_prof.text(0.02, 0.92, f"boxcar S/N = {snr_box:.1f} (w = {width})",
                 transform=ax_prof.transAxes, fontsize=8, va="top")

    ax_dm0.plot(t_wf, raw_dm0, "-", color="0.3", lw=0.7)
    ax_dm0.axvline(0, color="red", alpha=0.4, lw=1)
    ax_dm0.set_ylabel("Power (arb.)")
    ax_dm0.set_title("DM = 0 raw timeseries", fontsize=9)

    # Anchor the stretch to the noise so it sits in the dark end of the ramp
    # and only RFI/pulse climb to green-yellow (the transientX look); a
    # percentile stretch lets pure noise span the full colormap.
    med = np.median(wf_dd)
    sigma = 1.4826 * np.median(np.abs(wf_dd - med))
    if sigma <= 0:
        sigma = wf_dd.std() or 1.0
    ax_wf.imshow(wf_dd, aspect="auto", interpolation="nearest",
                 extent=[t_wf[0], t_wf[-1], freqs_mhz[-1], freqs_mhz[0]],
                 vmin=med - sigma, vmax=med + 7 * sigma, cmap=WATERFALL_CMAP)
    ax_wf.set_ylabel("Freq (MHz)")
    ax_wf.set_xlabel("Time - event (s)")

    # dmt is already in S/N units: pin the floor at 0 so noise stays dark.
    ax_dmt.imshow(dmt_disp, aspect="auto", origin="lower", interpolation="nearest",
                  extent=[t_wf[0], t_wf[-1], dms[0], dms[-1]],
                  vmin=0, vmax=max(8.0, np.percentile(dmt_disp, 99.9)),
                  cmap=WATERFALL_CMAP)
    ax_dmt.plot(0, dm, "o", ms=16, mfc="none", mec="red", mew=1.2)
    ax_dmt.set_ylabel(r"DM (pc cm$^{-3}$)")
    ax_dmt.set_xlabel("Time - event (s)")

    for ax in (ax_prof, ax_dm0):
        ax.set_xlim(t_wf[0], t_wf[-1])

    _beam_panel(ax_bt, card, fig)

    source = "" if card.get("source") == "blind" else f"{card.get('source', '')}   "
    fig.suptitle(
        f"{card['candname']}   {source}S/N={card['snr']:.1f}   "
        f"DM={dm:.2f}   width={width * tsamp_s * 1e3:.1f} ms   "
        f"beam {card['beam']}   {card['event_utc']}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png
