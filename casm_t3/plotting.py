"""Candidate diagnostic figure.

Layout (2026-09-02 revision, reviewed by Vishnu):

    dedispersed profile          | DM = 0 band-mean timeseries
    dedispersed waterfall        | boxcar S/N vs trial DM (bow-tie)
    T1 members, beam vs time     | sky footprint: lit beams on the alt/az grid

The bottom row shares one DM colour scale and one S/N size scale. The sky panel
draws every beam that has a T1 member within the coincidence window (the same
+-256 samples casm_t2 uses for its occupancy veto) at its true alt/az, so an
array-wide event is visible at a glance where the beam-index axis hides it.

Coordinates come ONLY from the trigger card's ``sky`` block, which t2d fills from
the weights live at the event time (casm_t2.weights_registry). No static table:
the one used until 2026-09-02 put a 45 deg median error on every posted
coordinate (casm-wiki incidents.md). A card without ``sky`` gets no coordinates
and no sky panel, never a guess.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors

from casm_t2 import weights_registry

from . import single_pulse

NBEAM_TOTAL = 512
COINCIDENCE_S = 256 * 1.048576e-3     # casm_t2 occupancy window_samp 256

# Event-physics panels on inferno (casm-wiki display-conventions.md).
WATERFALL_CMAP = "inferno"
# DM colour scale for the member panels: plasma with the bright top cut off,
# unreadable otherwise on a white panel.
BEAM_DM_CMAP = mcolors.ListedColormap(
    plt.get_cmap("plasma")(np.linspace(0.0, 0.85, 256)), name="plasma_dark")


def _block_mean_freqs(freqs_mhz: np.ndarray, ffactor: int) -> np.ndarray:
    n = (freqs_mhz.size // ffactor) * ffactor
    return freqs_mhz[:n].reshape(-1, ffactor).mean(axis=1)


def _snr_size(snr: np.ndarray) -> np.ndarray:
    return 8.0 + 3.0 * np.clip(snr - 12.0, 0.0, 40.0) ** 1.5


def _member_panel(ax, members: np.ndarray, card: dict, window_s: float,
                  dm_norm: mcolors.Normalize):
    """Beam index vs time of the T1 context members; returns the scatter mappable."""
    beam = int(card["beam"])
    sc = None
    ax.axvspan(-COINCIDENCE_S, COINCIDENCE_S, color="#c0392b", alpha=0.08, lw=0, zorder=0)
    if members.size:
        sc = ax.scatter(members[:, 0], members[:, 1], c=members[:, 2], s=_snr_size(members[:, 3]),
                        cmap=BEAM_DM_CMAP, norm=dm_norm, alpha=0.85, linewidths=0)
    else:
        ax.text(0.5, 0.5, "no context candidates", ha="center", va="center",
                transform=ax.transAxes, color="0.4")
    for edge in range(64, NBEAM_TOTAL, 64):
        ax.axhline(edge - 0.5, color="0.88", lw=0.6, zorder=0)
    ax.plot(0, beam, "s", mfc="none", mec="#c0392b", ms=11, mew=1.4, zorder=5)
    ax.set_xlim(-window_s, window_s)
    ax.set_ylim(-8, NBEAM_TOTAL + 8)
    ax.set_yticks(np.arange(0, NBEAM_TOTAL + 1, 64))
    ax.set_xlabel("time - event (s)")
    ax.set_ylabel("beam")
    ax.set_title(f"T1 candidates within \N{PLUS-MINUS SIGN}{window_s:g} s of the event (size = S/N)")
    return sc


def _sky_panel(ax, members: np.ndarray, card: dict, pointings: dict | None,
               dm_norm: mcolors.Normalize) -> int | None:
    """Lit beams on the alt/az grid. Returns the footprint count, or None when the
    weights live at the event are unknown (panel then says so and draws nothing)."""
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlim(0, 70)
    ax.set_yticks([20, 40, 60])
    ax.set_yticklabels(["alt 70", "50", "30"], fontsize=9, color="#555555")
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(["N", "E", "S", "W"])
    ax.tick_params(axis="x", pad=6)
    ax.grid(alpha=0.35)
    if pointings is None:
        ax.text(0.5, 0.5, "beam pointings unknown\n(no weights registered for this time)",
                ha="center", va="center", transform=ax.transAxes, color="0.4", fontsize=10)
        return None
    alt = np.asarray(pointings["alt_deg"]); az = np.asarray(pointings["az_deg"])
    ax.scatter(np.radians(az), 90 - alt, s=5, color="#d0d0d0", zorder=1)
    best: dict[int, tuple[float, float]] = {}
    if members.size:
        near = members[np.abs(members[:, 0]) <= COINCIDENCE_S]
        for dt, b, dm, snr, w in near:
            b = int(b)
            if 0 <= b < NBEAM_TOTAL and (b not in best or snr > best[b][0]):
                best[b] = (snr, dm)
    if best:
        bb = np.array(sorted(best)); ss = np.array([best[b][0] for b in bb]); dd = np.array([best[b][1] for b in bb])
        ax.scatter(np.radians(az[bb]), 90 - alt[bb], c=dd, cmap=BEAM_DM_CMAP, norm=dm_norm,
                   s=25 + 6 * np.clip(ss - 12, 0, 40) ** 1.5, edgecolors="k", linewidths=0.4, zorder=3)
    beam = int(card["beam"])
    ax.scatter(np.radians(az[beam]), 90 - alt[beam], s=230, facecolors="none",
               edgecolors="#c0392b", linewidths=1.5, zorder=4)
    sky = card.get("sky") or {}
    if sky.get("sun_alt_deg") is not None and sky["sun_alt_deg"] > 0:
        ax.scatter(np.radians(sky["sun_az_deg"]), 90 - sky["sun_alt_deg"], marker="*", s=170,
                   color="#f4a300", edgecolors="k", linewidths=0.5, zorder=5)
    n_lit = len(best)
    ax.text(0.5, -0.10, f"{n_lit} of {NBEAM_TOTAL} beams with a T1 candidate within {COINCIDENCE_S:.2f} s",
            transform=ax.transAxes, ha="center", va="top", fontsize=11)
    return n_lit


def _coord_line(card: dict, tsamp_s: float) -> str:
    sky = card.get("sky") or {}
    beam = card["beam"]
    if not sky or sky.get("alt_deg") is None:
        return f"beam {beam}: pointing unknown (no weights registered for the event time)"
    line = f"beam {beam}: alt {sky['alt_deg']:.1f}\N{DEGREE SIGN}  az {sky['az_deg']:.1f}\N{DEGREE SIGN}"
    if sky.get("ra_deg") is not None and sky.get("dec_deg") is not None:
        try:
            hms, dms = weights_registry.hms_dms(sky["ra_deg"], sky["dec_deg"])
            line += f"   RA {hms}  Dec {dms}"
        except Exception:
            line += f"   RA {sky['ra_deg']:.3f}\N{DEGREE SIGN}  Dec {sky['dec_deg']:+.3f}\N{DEGREE SIGN}"
    return line


def make_candidate_figure(data: np.ndarray, freqs_mhz: np.ndarray, tsamp_s: float,
                          t_rel_event_s: float, card: dict, out_png: str | Path,
                          ffactor: int = 8, registry: weights_registry.Registry | None = None) -> Path:
    """Render the candidate plot.

    data : (nchan, ntime) float32, raw (dispersed) cutout for the detection beam.
    t_rel_event_s : candidate time (top-of-band arrival) relative to the first
        sample of ``data``.
    card : trigger-card dict (candname/source/snr/dm/width/beam/event_utc, the
        ``sky`` block written by t2d, optional context member list).
    """
    dm, width = float(card["dm"]), 2 ** int(card["width"])
    norm = single_pulse.normalise(data)
    dedis = single_pulse.dedisperse(norm, dm, freqs_mhz, tsamp_s)
    tfactor = max(1, width // 2)
    wf_dd = single_pulse.downsample(dedis, ffactor, tfactor)
    prof_dd = wf_dd.mean(axis=0)
    # Waterfall pixels: 64 sub-bands x one boxcar width. A pulse of total S/N 17
    # over 3072 channels is ~0.8 sigma per pixel at 8 channels x half a boxcar;
    # at 48 channels x a full boxcar it is ~2 sigma and visible.
    ffactor_wf = 48
    tfactor_wf = max(1, width)
    wf_show = single_pulse.downsample(dedis, ffactor_wf, tfactor_wf)
    t_show = (np.arange(wf_show.shape[1]) * tfactor_wf + tfactor_wf / 2) * tsamp_s - t_rel_event_s
    raw_dm0 = data.mean(axis=0)
    n = (raw_dm0.size // tfactor) * tfactor
    raw_dm0 = raw_dm0[:n].reshape(-1, tfactor).mean(axis=1)
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
    half_prof = max(1.0, 30 * width * tsamp_s)
    wing_s = 4.148808e3 * 0.5 * (dms[-1] - dms[0]) * (freqs_mhz.min() ** -2 - freqs_mhz.max() ** -2)
    half_dmt = max(half_prof, 0.75 * wing_s)
    xlim_prof = (max(-half_prof, t_wf[0]), min(half_prof, t_wf[-1]))
    xlim_dmt = (max(-half_dmt, t_wf[0]), min(half_dmt, t_wf[-1]))

    ctx = card.get("context") or {}
    members = np.asarray(ctx.get("members") or [], dtype=float).reshape(-1, 5)
    window_s = float(ctx.get("window_s", 4.0))
    dm_norm = mcolors.Normalize(vmin=15.0, vmax=max(60.0, 1.5 * dm, float(np.percentile(members[:, 2], 95)) if members.size else 0.0))
    sky = card.get("sky") or {}
    reg = registry or weights_registry.default_registry()
    pointings = reg.product(sky.get("weights_id")) if sky.get("weights_id") else None

    plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 11, "axes.labelsize": 11,
                         "xtick.labelsize": 10, "ytick.labelsize": 10})
    fig = plt.figure(figsize=(13, 12.5))
    gs = fig.add_gridspec(3, 12, height_ratios=(1.0, 1.5, 1.45), hspace=0.55, wspace=1.6, top=0.895)
    # column 6 stays empty as a gutter so the right column's y-labels never
    # touch the left column's frames
    ax_prof = fig.add_subplot(gs[0, 0:6]); ax_dm0 = fig.add_subplot(gs[0, 7:12])
    ax_wf = fig.add_subplot(gs[1, 0:6]); ax_dmt = fig.add_subplot(gs[1, 7:12])
    ax_bt = fig.add_subplot(gs[2, 0:7])
    ax_sky = fig.add_subplot(gs[2, 7:12], projection="polar")
    ax_sky.set_position([0.565, 0.075, 0.30, 0.245])

    ax_prof.plot(t_wf, prof_dd, "k-", lw=1.0)
    ax_prof.axvline(0, color="#c0392b", alpha=0.8, lw=0.9)
    ax_prof.set_ylabel("power (arb.)")
    ax_prof.set_title(f"dedispersed at DM = {dm:.2f}")
    ax_prof.text(0.02, 0.92, f"boxcar S/N = {snr_box:.1f} (w = {width})",
                 transform=ax_prof.transAxes, fontsize=9, va="top")
    ax_prof.set_xlim(xlim_prof)

    ax_dm0.plot(t_wf, raw_dm0, "-", color="0.3", lw=0.8)
    ax_dm0.axvline(0, color="#c0392b", alpha=0.8, lw=0.9)
    ax_dm0.set_ylabel("power (arb.)")
    ax_dm0.set_title("DM = 0 band-mean timeseries")
    ax_dm0.set_xlim(t_wf[0], t_wf[-1])

    med = np.median(wf_show)
    sigma = 1.4826 * np.median(np.abs(wf_show - med)) or (wf_show.std() or 1.0)
    ax_wf.imshow(wf_show, aspect="auto", interpolation="nearest",
                 extent=[t_show[0], t_show[-1], freqs_mhz[-1], freqs_mhz[0]],
                 vmin=med - 1.5 * sigma, vmax=med + 4 * sigma, cmap=WATERFALL_CMAP)
    ax_wf.set_xlim(xlim_prof)
    ax_wf.set_ylabel("frequency (MHz)")
    ax_wf.set_xlabel("time - event (s)")
    ax_wf.set_title(f"waterfall, dedispersed at DM = {dm:.2f}")

    im_dmt = ax_dmt.imshow(dmt_disp, aspect="auto", origin="lower", interpolation="nearest",
                           extent=[t_wf[0], t_wf[-1], dms[0], dms[-1]],
                           vmin=0, vmax=max(8.0, np.percentile(dmt_disp, 99.9)), cmap=WATERFALL_CMAP)
    cb = fig.colorbar(im_dmt, ax=ax_dmt, pad=0.02, fraction=0.05)
    cb.set_label("boxcar S/N")
    ax_dmt.plot(0, dm, "o", ms=14, mfc="none", mec="#c0392b", mew=1.3)
    ax_dmt.set_xlim(xlim_dmt)
    ax_dmt.set_ylabel(r"DM (pc cm$^{-3}$)")
    ax_dmt.set_xlabel("time - event (s)")
    ax_dmt.set_title(f"boxcar S/N vs trial DM (w = {width})")

    sc = _member_panel(ax_bt, members, card, window_s, dm_norm)
    _sky_panel(ax_sky, members, card, pointings, dm_norm)
    if sc is not None:
        p = cb.ax.get_position()
        cax = fig.add_axes([p.x0, 0.10, p.width, 0.22])
        fig.colorbar(sc, cax=cax).set_label(r"DM (pc cm$^{-3}$)")

    source = "" if card.get("source", "blind") == "blind" else f"{card.get('source')}   "
    lines = [
        f"{card['candname']}   {source}{card['event_utc']}",
        f"S/N = {card['snr']:.1f}   DM = {dm:.2f} pc cm$^{{-3}}$   width = {width * tsamp_s * 1e3:.1f} ms",
        _coord_line(card, tsamp_s),
    ]
    fig.suptitle("\n".join(lines), y=0.975, fontsize=12)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_png
