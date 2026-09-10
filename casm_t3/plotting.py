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

import logging
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors

from casm_t2 import weights_registry

from . import single_pulse

logger = logging.getLogger(__name__)

NBEAM_TOTAL = 512
COINCIDENCE_S = 256 * 1.048576e-3     # casm_t2 occupancy window_samp 256

# Event-physics panels on inferno (casm-wiki display-conventions.md).
WATERFALL_CMAP = "viridis"    # Vishnu 2026-09-03: viridis over inferno on the image panels
# The legacy (live) layout keeps its transientX look until the v2 layout is approved.
LEGACY_CMAP = "viridis"
SHOW_SOURCES = True    # sun, Cas A, Cyg A, Tau A markers + legend on the sky panel (approved by Vishnu 2026-09-03)
DEFAULT_LAYOUT = "v2"       # approved by Vishnu 2026-09-03: unified layout with sky footprint is what Slack posts
# DM colour scale for the member panels: plasma with the bright top cut off,
# unreadable otherwise on a white panel.
BEAM_DM_CMAP = mcolors.ListedColormap(
    plt.get_cmap("plasma")(np.linspace(0.0, 0.85, 256)), name="plasma_dark")


# --- adaptive frequency averaging for the image panels -----------------------
# The waterfall and DM-time panels average channels into rows. With the time bin
# matched to the boxcar, a pulse of total S/N spreads over the rows as roughly
# S/N / sqrt(nrows) per pixel, so a fixed row count either buries a weak
# candidate in noise or wastes band resolution on a strong one. Pick the row
# count that puts about SUBBAND_TARGET_SIGMA in each pixel, clamped to a sane
# range and snapped to a divisor of the band so no channels are dropped.
# 4.2 sigma per pixel (S/N 23 -> 32 rows, S/N 43 -> 96) was chosen on the
# 2026-09-10 injection renders.
SUBBAND_TARGET_SIGMA = 4.2
SUBBAND_MIN = 16
SUBBAND_MAX = 384                     # 3072 / 8, the fixed value used until 2026-09-10
SUBBAND_CHOICES = (16, 24, 32, 48, 64, 96, 128, 192, 384)
DEFAULT_SUBBANDS = SUBBAND_MAX        # fallback when the card carries no usable S/N


def subbands_for(snr: float | None, nchan: int = 3072,
                 target_sigma: float = SUBBAND_TARGET_SIGMA) -> int:
    """Number of frequency rows to average the band into for the image panels.

    nrows = clamp(round((snr / target_sigma) ** 2), SUBBAND_MIN, SUBBAND_MAX),
    snapped to the nearest entry of SUBBAND_CHOICES that divides ``nchan``, so
    that per-pixel S/N ~ snr / sqrt(nrows) lands near ``target_sigma``.
    A missing or non-finite S/N falls back to DEFAULT_SUBBANDS.
    """
    hi = min(SUBBAND_MAX, max(1, int(nchan)))
    choices = [n for n in SUBBAND_CHOICES if n <= hi and int(nchan) % n == 0]
    if not choices:
        choices = [n for n in SUBBAND_CHOICES if n <= hi] or [hi]
    try:
        snr = float(snr)
    except (TypeError, ValueError):
        snr = float("nan")
    if not math.isfinite(snr) or snr <= 0:
        ideal = DEFAULT_SUBBANDS
    else:
        ideal = round((snr / float(target_sigma)) ** 2)
    ideal = min(max(ideal, SUBBAND_MIN), hi)
    return min(choices, key=lambda n: (abs(n - ideal), n))


# --- channel averaging for the DM-time panel --------------------------------
# The DM-time panel must NOT reuse the display subbands. Its channels are
# averaged BEFORE dedispersion, so every trial DM smears within a subband:
# t_smear = 8.3e-3 ms x DM x BW_MHz / f_GHz^3 at the bottom of the band. At
# DM 300 a 96-channel subband (2.9 MHz) smears by ~118 ms, and even the old
# fixed ffactor 8 (0.244 MHz) smears ~10 ms at DM 300 and ~30 ms at DM 900 —
# far past the boxcar, which is why the bow-tie faded on narrow candidates.
# Average only as far as the smear stays within half the boxcar width.
DMT_FFACTOR_CHOICES = (1, 2, 3, 4, 6, 8, 12, 16)
DMT_SMEAR_F_GHZ = 0.3906              # bottom of the band, worst case
DMT_SMEAR_K_MS = 8.3e-3               # ms per (pc cm^-3) per MHz at 1 GHz
DMT_SMEAR_FRACTION = 0.5              # allowed smear, in boxcar widths


def dmt_ffactor_for(dm: float, boxcar_samples: int, tsamp_s: float,
                    chan_mhz: float, nchan: int = 3072) -> int:
    """Largest channel-average factor whose intra-subband smear stays within
    DMT_SMEAR_FRACTION of the boxcar width; 1 (full resolution) if none does."""
    limit_ms = DMT_SMEAR_FRACTION * max(1, int(boxcar_samples)) * float(tsamp_s) * 1e3
    chan_mhz = abs(float(chan_mhz))
    best = 1
    for f in DMT_FFACTOR_CHOICES:
        if int(nchan) % f:
            continue
        smear_ms = DMT_SMEAR_K_MS * abs(float(dm)) * (f * chan_mhz) / DMT_SMEAR_F_GHZ ** 3
        if smear_ms <= limit_ms:
            best = max(best, f)
    return best


def _dmt_plan(dm: float, width: int, freqs_mhz: np.ndarray, tsamp_s: float) -> int:
    chan_mhz = abs(float(np.median(np.diff(freqs_mhz)))) if freqs_mhz.size > 1 else 1.0
    f_dmt = dmt_ffactor_for(dm, width, tsamp_s, chan_mhz, nchan=freqs_mhz.size)
    smear_ms = DMT_SMEAR_K_MS * abs(dm) * (f_dmt * chan_mhz) / DMT_SMEAR_F_GHZ ** 3
    logger.info("DM-time channel factor %d (%d channels, smear %.1f ms vs boxcar %.1f ms)",
                f_dmt, freqs_mhz.size // f_dmt, smear_ms, width * tsamp_s * 1e3)
    return f_dmt


DMT_MAX_COLUMNS = 400


def _dmt_tfactor(width: int, xlim_dmt: tuple[float, float], tsamp_s: float) -> int:
    """Time binning for the DM-time display: one boxcar width, or coarser.

    Two constraints. (1) A pixel narrower than the boxcar shows a fraction of
    the pulse, exactly as in the waterfall, so the floor is ``width``.
    (2) imshow(interpolation="nearest") resamples by dropping columns, so a
    panel ~450 px wide fed ~950 columns can drop the one column holding the
    peak: a S/N 24.7 narrow candidate rendered as a faint dot. Cap the drawn
    columns inside the plotted window at DMT_MAX_COLUMNS. Pooling is by
    maximum (single_pulse.downsample_max), the array is already a boxcar S/N.
    """
    n_window = max(1, int(math.ceil((xlim_dmt[1] - xlim_dmt[0]) / tsamp_s)))
    return int(max(max(1, width), math.ceil(n_window / DMT_MAX_COLUMNS)))


def _subband_plan(card: dict, nchan: int, ffactor: int | None,
                  subbands: int | None) -> tuple[int, int]:
    """(nrows, ffactor) for the image panels: explicit override wins, else adaptive."""
    nchan = max(1, int(nchan))
    if subbands:
        nrows = max(1, min(int(subbands), nchan))
        ffactor = max(1, nchan // nrows)
    elif ffactor:
        ffactor = max(1, int(ffactor))
    else:
        nrows = subbands_for(card.get("snr"), nchan=nchan)
        ffactor = max(1, nchan // nrows)
    nrows = nchan // ffactor
    logger.info("display averaging: %d subbands (ffactor %d) for S/N %s over %d channels",
                nrows, ffactor, card.get("snr"), nchan)
    return nrows, ffactor


def _block_mean_freqs(freqs_mhz: np.ndarray, ffactor: int) -> np.ndarray:
    n = (freqs_mhz.size // ffactor) * ffactor
    return freqs_mhz[:n].reshape(-1, ffactor).mean(axis=1)


def _snr_floor(members: np.ndarray, card: dict) -> float:
    """Size scale origin: the lowest member S/N, i.e. the T1 threshold in force
    (it changes with the search settings, so nothing is hard-coded)."""
    if members.size:
        return float(np.floor(members[:, 3].min()))
    return float(np.floor(min(card.get("snr", 12.0), 12.0)))


def _snr_size(snr: np.ndarray, floor: float, base: float = 8.0, k: float = 3.0) -> np.ndarray:
    return base + k * np.clip(np.asarray(snr, dtype=float) - floor, 0.0, 40.0) ** 1.5


def _member_panel(ax, members: np.ndarray, card: dict, window_s: float,
                  dm_norm: mcolors.Normalize, floor: float):
    """Beam index vs time of the T1 context members; returns the scatter mappable."""
    beam = int(card["beam"])
    sc = None
    ax.axvspan(-COINCIDENCE_S, COINCIDENCE_S, color="#c0392b", alpha=0.08, lw=0, zorder=0)
    if members.size:
        sc = ax.scatter(members[:, 0], members[:, 1], c=members[:, 2], s=_snr_size(members[:, 3], floor),
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
    ax.set_title(f"T1 candidates within \N{PLUS-MINUS SIGN}{window_s:g} s of the event")
    # size legend: three reference S/N values spanning what this card contains
    top = float(members[:, 3].max()) if members.size else floor + 20
    refs = sorted({int(round(v)) for v in (floor + 2, (floor + top) / 2, top)})
    handles = [ax.scatter([], [], s=_snr_size(np.array([v]), floor)[0], color="0.45", alpha=0.85,
                          linewidths=0, label=f"S/N {v}") for v in refs]
    ax.legend(handles=handles, loc="upper left", fontsize=9, frameon=False,
              labelspacing=1.4, borderpad=0.6, handletextpad=1.2)
    return sc


def _sky_panel(ax, members: np.ndarray, card: dict, pointings: dict | None,
               dm_norm: mcolors.Normalize, floor: float) -> int | None:
    """Lit beams on the alt/az grid. Returns the footprint count, or None when the
    weights live at the event are unknown (panel then says so and draws nothing)."""
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlim(0, 72)                         # edge just below the beam-grid floor (alt ~20)
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
                   s=_snr_size(ss, floor, base=25.0, k=6.0), edgecolors="k", linewidths=0.4, zorder=3)
    beam = int(card["beam"])
    ax.scatter(np.radians(az[beam]), 90 - alt[beam], s=230, facecolors="none",
               edgecolors="#c0392b", linewidths=1.5, zorder=4)
    sky = card.get("sky") or {}
    if SHOW_SOURCES:
        srcs = dict(sky.get("sources") or {})
        if not srcs and sky.get("sun_alt_deg") is not None:
            srcs["Sun"] = (sky["sun_alt_deg"], sky["sun_az_deg"])
        style = {"Sun": ("*", "#f4a300", 190), "Cas A": ("P", "#2e86c1", 110),
                 "Cyg A": ("X", "#27ae60", 110), "Tau A": ("D", "#8e44ad", 80)}
        handles = []
        floor_alt = 90 - ax.get_rmax()
        for name, (m, col, size) in style.items():
            if name not in srcs:
                continue
            s_alt, s_az = srcs[name]
            if s_alt >= floor_alt:                       # inside the drawn sky: plot it
                h = ax.scatter(np.radians(s_az), 90 - s_alt, marker=m, s=size, color=col,
                               edgecolors="k", linewidths=0.5, zorder=5, label=name, clip_on=False)
            elif s_alt > 0:                              # up, but below the beam grid: say so
                h = ax.scatter([], [], marker=m, s=size, color=col, edgecolors="k", linewidths=0.5,
                               alpha=0.6, label=f"{name} (alt {s_alt:.0f}\N{DEGREE SIGN}, no beam)")
            else:
                h = ax.scatter([], [], marker=m, s=size, color=col, edgecolors="k", linewidths=0.5,
                               alpha=0.3, label=f"{name} (set)")
            handles.append(h)
        if handles:
            ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.86, 1.10), fontsize=8.5,
                      frameon=False, labelspacing=0.9, handletextpad=0.6)
    elif sky.get("sun_alt_deg") is not None and sky["sun_alt_deg"] > 0:
        ax.scatter(np.radians(sky["sun_az_deg"]), 90 - sky["sun_alt_deg"], marker="*", s=170,
                   color="#f4a300", edgecolors="k", linewidths=0.5, zorder=5)
    n_lit = len(best)
    ax.text(0.5, -0.10, f"{n_lit} of {NBEAM_TOTAL} beams with a T1 candidate within {COINCIDENCE_S:.2f} s",
            transform=ax.transAxes, ha="center", va="top", fontsize=11)
    return n_lit


def _dm0_curve(ax, dm: float, freqs_mhz: np.ndarray, color: str = "#e41a1c", lw: float = 1.4,
               alpha: float = 1.0) -> None:
    """Trace where a DM = 0 (undispersed) burst lands in the dedispersed waterfall.

    Dedispersion advances each channel by its DM delay, so a zero-DM burst at
    the event time appears as the reversed sweep t(f) = -k DM (f^-2 - f_top^-2),
    anchored at t = 0 at the top of the band. A feature that follows this red
    curve is terrestrial; the real pulse is the vertical line at t = 0.
    """
    f = np.linspace(freqs_mhz.min(), freqs_mhz.max(), 256)
    t = -4.148808e3 * dm * (f ** -2 - freqs_mhz.max() ** -2)
    ax.plot(t, f, color=color, lw=lw, alpha=alpha, clip_on=True)


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


def make_candidate_figure_v2(data: np.ndarray, freqs_mhz: np.ndarray, tsamp_s: float,
                             t_rel_event_s: float, card: dict, out_png: str | Path,
                             ffactor: int | None = None, registry: weights_registry.Registry | None = None,
                             subbands: int | None = None) -> Path:
    """Render the candidate plot.

    data : (nchan, ntime) float32, raw (dispersed) cutout for the detection beam.
    t_rel_event_s : candidate time (top-of-band arrival) relative to the first
        sample of ``data``.
    card : trigger-card dict (candname/source/snr/dm/width/beam/event_utc, the
        ``sky`` block written by t2d, optional context member list).
    """
    dm, width = float(card["dm"]), 2 ** int(card["width"])
    nsub, ffactor = _subband_plan(card, freqs_mhz.size, ffactor, subbands)
    norm = single_pulse.normalise(data)
    dedis = single_pulse.dedisperse(norm, dm, freqs_mhz, tsamp_s)
    tfactor = max(1, width // 2)
    wf_dd = single_pulse.downsample(dedis, ffactor, tfactor)
    prof_dd = wf_dd.mean(axis=0)
    # Waterfall pixels: one subband x one boxcar width. The subband count comes
    # from the candidate S/N (see subbands_for): a pulse of total S/N 17 over
    # 3072 channels is ~0.9 sigma per pixel at 384 rows whatever the time
    # averaging, and reads as a coherent line only once the rows are coarse
    # enough to put ~3 sigma in each.
    tfactor_wf = max(1, width)
    wf_show = single_pulse.downsample(dedis, ffactor, tfactor_wf)
    nsub = wf_show.shape[0]     # what the panel actually shows, after truncation
    t_show = (np.arange(wf_show.shape[1]) * tfactor_wf + tfactor_wf / 2) * tsamp_s - t_rel_event_s
    raw_dm0 = data.mean(axis=0)
    n = (raw_dm0.size // tfactor) * tfactor
    raw_dm0 = raw_dm0[:n].reshape(-1, tfactor).mean(axis=1)
    prof_full = single_pulse.profile_snr(dedis.mean(axis=0), width)
    t_full = np.arange(prof_full.size) * tsamp_s - t_rel_event_s
    near = np.abs(t_full) <= 0.5
    snr_box = prof_full[near].max() if near.any() else prof_full.max()
    # DM-time: its own, much finer channel averaging (see dmt_ffactor_for) —
    # the display subbands would smear the pulse away before dedispersion.
    f_dmt = _dmt_plan(dm, width, freqs_mhz, tsamp_s)
    small = single_pulse.downsample(norm, f_dmt, 1)
    f_small = _block_mean_freqs(freqs_mhz, f_dmt)[: small.shape[0]]
    dms = single_pulse.dm_grid(dm)
    dmt = single_pulse.dm_time(small, f_small, tsamp_s, dms, width)
    t_wf = (np.arange(wf_dd.shape[1]) * tfactor + tfactor / 2) * tsamp_s - t_rel_event_s
    half_prof = max(1.0, 30 * width * tsamp_s)
    wing_s = 4.148808e3 * 0.5 * (dms[-1] - dms[0]) * (freqs_mhz.min() ** -2 - freqs_mhz.max() ** -2)
    half_dmt = max(half_prof, 0.75 * wing_s)
    xlim_prof = (max(-half_prof, t_wf[0]), min(half_prof, t_wf[-1]))
    xlim_dmt = (max(-half_dmt, t_wf[0]), min(half_dmt, t_wf[-1]))
    tfactor_dmt = _dmt_tfactor(width, xlim_dmt, tsamp_s)
    dmt_disp = single_pulse.downsample_max(dmt, tfactor_dmt)
    t_dmt = (np.arange(dmt_disp.shape[1]) * tfactor_dmt + tfactor_dmt / 2) * tsamp_s - t_rel_event_s

    ctx = card.get("context") or {}
    members = np.asarray(ctx.get("members") or [], dtype=float).reshape(-1, 5)
    window_s = float(ctx.get("window_s", 4.0))
    dm_norm = mcolors.Normalize(vmin=15.0, vmax=max(60.0, 1.5 * dm, float(np.percentile(members[:, 2], 95)) if members.size else 0.0))
    sky = card.get("sky") or {}
    # pointing table: from the card itself (t2d embeds the live weights' table so
    # corr2, which has no registry, can plot); registry only as a fallback
    pointings = card.get("pointings") if (card.get("pointings") or {}).get("alt_deg") else None
    if pointings is None and sky.get("weights_id"):
        reg = registry or weights_registry.default_registry()
        pointings = reg.product(sky.get("weights_id"))

    plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 11, "axes.labelsize": 11,
                         "xtick.labelsize": 10, "ytick.labelsize": 10})
    # Fixed canvas 1510 x 1517 px at 120 dpi (the framing of 260903bembjy, approved):
    # every Slack post has identical pixel size, so previews render at one width.
    fig = plt.figure(figsize=(1510 / 120, 1517 / 120))
    # two equal columns; colour bars hang outside their axes as insets so the
    # right-hand panels keep the same width as the left-hand ones
    gs = fig.add_gridspec(3, 2, height_ratios=(0.85, 1.75, 1.5), hspace=0.38, wspace=0.28,
                          left=0.06, right=0.885, top=0.915, bottom=0.06)
    ax_prof = fig.add_subplot(gs[0, 0]); ax_dm0 = fig.add_subplot(gs[0, 1])
    ax_wf = fig.add_subplot(gs[1, 0]); ax_dmt = fig.add_subplot(gs[1, 1])
    ax_bt = fig.add_subplot(gs[2, 0])
    ax_sky = fig.add_subplot(gs[2, 1], projection="polar")
    # bottom row: members panel wider (7:5), polar panel a square whose circle
    # fills the row height (figure is 13 x 12.5 in, so width = height x 12.5/13)
    bt = ax_bt.get_position(); rt = ax_dmt.get_position()
    fw, fh = fig.get_size_inches()
    ax_bt.set_position([bt.x0, bt.y0, 0.55 - bt.x0, bt.height])
    sky_h = bt.height + 0.04
    sky_w = sky_h * fh / fw
    ax_sky.set_position([0.585, bt.y0 - 0.02, sky_w, sky_h])

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
                 vmin=med - 1.5 * sigma, vmax=med + 3.5 * sigma, cmap=WATERFALL_CMAP)
    _dm0_curve(ax_wf, dm, freqs_mhz)
    ax_wf.set_xlim(xlim_prof)
    ax_wf.set_ylim(freqs_mhz.min(), freqs_mhz.max())
    ax_wf.set_ylabel("frequency (MHz)")
    ax_wf.set_xlabel("time - event (s)")
    ax_wf.set_title(f"waterfall, dedispersed at DM = {dm:.2f}, red: DM = 0 curve"
                    f" ({nsub} subbands)")

    im_dmt = ax_dmt.imshow(dmt_disp, aspect="auto", origin="lower", interpolation="nearest",
                           extent=[t_dmt[0], t_dmt[-1], dms[0], dms[-1]],
                           vmin=0, vmax=max(8.0, np.percentile(dmt_disp, 99.9)), cmap=WATERFALL_CMAP)
    cb = fig.colorbar(im_dmt, cax=ax_dmt.inset_axes((1.02, 0.0, 0.03, 1.0)))
    cb.set_label("boxcar S/N")
    ax_dmt.plot(0, dm, "o", ms=14, mfc="none", mec="#c0392b", mew=1.3)
    ax_dmt.set_xlim(xlim_dmt)
    ax_dmt.set_ylabel(r"DM (pc cm$^{-3}$)")
    ax_dmt.set_xlabel("time - event (s)")
    ax_dmt.set_title(f"boxcar S/N vs trial DM (w = {width})")

    floor = _snr_floor(members, card)
    sc = _member_panel(ax_bt, members, card, window_s, dm_norm, floor)
    _sky_panel(ax_sky, members, card, pointings, dm_norm, floor)
    if sc is not None:
        p = cb.ax.get_position()
        cax = fig.add_axes([p.x0, bt.y0 + 0.02, p.width, bt.height - 0.12])   # leaves room for the source legend above
        fig.colorbar(sc, cax=cax).set_label(r"DM (pc cm$^{-3}$)")

    source = "" if card.get("source", "blind") == "blind" else f"{card.get('source')}   "
    lines = [
        f"{card['candname']}   {source}{card['event_utc']}",
        f"S/N = {card['snr']:.1f}   DM = {dm:.2f} pc cm$^{{-3}}$   width = {width * tsamp_s * 1e3:.1f} ms",
        _coord_line(card, tsamp_s),
    ]
    fig.suptitle("\n".join(lines), y=0.985, fontsize=12)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)      # fixed canvas: identical pixel size for every candidate
    plt.close(fig)
    return out_png


def _legacy_beam_panel(ax, card: dict, fig) -> None:
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


def make_candidate_figure_legacy(data: np.ndarray, freqs_mhz: np.ndarray, tsamp_s: float,
                          t_rel_event_s: float, card: dict, out_png: str | Path,
                          ffactor: int | None = None, subbands: int | None = None) -> Path:
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
    # hella reports width as log2 of the boxcar length in samples
    dm, width = float(card["dm"]), 2 ** int(card["width"])
    nsub, ffactor = _subband_plan(card, freqs_mhz.size, ffactor, subbands)
    norm = single_pulse.normalise(data)
    dedis = single_pulse.dedisperse(norm, dm, freqs_mhz, tsamp_s)

    tfactor = max(1, width // 2)
    wf_dd = single_pulse.downsample(dedis, ffactor, tfactor)
    nsub = wf_dd.shape[0]
    prof_dd = wf_dd.mean(axis=0)

    raw_dm0 = data.mean(axis=0)
    n = (raw_dm0.size // tfactor) * tfactor
    raw_dm0 = raw_dm0[:n].reshape(-1, tfactor).mean(axis=1)

    # Matched-boxcar S/N near the event, quoted on the profile panel.
    prof_full = single_pulse.profile_snr(dedis.mean(axis=0), width)
    t_full = np.arange(prof_full.size) * tsamp_s - t_rel_event_s
    near = np.abs(t_full) <= 0.5
    snr_box = prof_full[near].max() if near.any() else prof_full.max()

    # DM-time keeps its own channel resolution, not the display subbands.
    f_dmt = _dmt_plan(dm, width, freqs_mhz, tsamp_s)
    small = single_pulse.downsample(norm, f_dmt, 1)
    f_small = _block_mean_freqs(freqs_mhz, f_dmt)[: small.shape[0]]
    dms = single_pulse.dm_grid(dm)
    dmt = single_pulse.dm_time(small, f_small, tsamp_s, dms, width)

    t_wf = (np.arange(wf_dd.shape[1]) * tfactor + tfactor / 2) * tsamp_s - t_rel_event_s

    # Fixed framing: every candidate gets the same window around the pulse
    # instead of wherever it happened to fall in the dump, so plots are
    # directly comparable. The dedispersed panels need ~a second of context
    # (more for wide pulses); the DM-time bowtie must also fit its wings,
    # which extend in time as the trial-DM mismatch grows. The DM=0 panel
    # keeps the full dump span — its job is RFI context.
    half_prof = max(1.0, 30 * width * tsamp_s)
    wing_s = 4.148808e3 * 0.5 * (dms[-1] - dms[0]) * (
        freqs_mhz.min() ** -2 - freqs_mhz.max() ** -2)
    half_dmt = max(half_prof, 0.75 * wing_s)
    xlim_prof = (max(-half_prof, t_wf[0]), min(half_prof, t_wf[-1]))
    xlim_dmt = (max(-half_dmt, t_wf[0]), min(half_dmt, t_wf[-1]))

    tfactor_dmt = _dmt_tfactor(width, xlim_dmt, tsamp_s)
    dmt_disp = single_pulse.downsample_max(dmt, tfactor_dmt)
    t_dmt = (np.arange(dmt_disp.shape[1]) * tfactor_dmt + tfactor_dmt / 2) * tsamp_s - t_rel_event_s

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
                 vmin=med - sigma, vmax=med + 7 * sigma, cmap=LEGACY_CMAP)
    _dm0_curve(ax_wf, dm, freqs_mhz)
    ax_wf.set_xlim(xlim_prof)
    ax_wf.set_ylim(freqs_mhz.min(), freqs_mhz.max())
    ax_wf.set_ylabel("Freq (MHz)")
    ax_wf.set_xlabel("Time - event (s)")
    ax_wf.set_title(f"waterfall ({nsub} subbands)", fontsize=9)

    # dmt is already in S/N units: pin the floor at 0 so noise stays dark.
    im_dmt = ax_dmt.imshow(dmt_disp, aspect="auto", origin="lower", interpolation="nearest",
                           extent=[t_dmt[0], t_dmt[-1], dms[0], dms[-1]],
                           vmin=0, vmax=max(8.0, np.percentile(dmt_disp, 99.9)),
                           cmap=LEGACY_CMAP)
    cax_dmt = ax_dmt.inset_axes((1.015, 0.0, 0.018, 1.0))
    fig.colorbar(im_dmt, cax=cax_dmt, label=f"boxcar S/N (w = {width})")
    ax_dmt.plot(0, dm, "o", ms=16, mfc="none", mec="red", mew=1.2)
    ax_dmt.set_xlim(xlim_dmt)
    ax_dmt.set_ylabel(r"DM (pc cm$^{-3}$)")
    ax_dmt.set_xlabel("Time - event (s)")

    ax_prof.set_xlim(xlim_prof)
    ax_dm0.set_xlim(t_wf[0], t_wf[-1])

    _legacy_beam_panel(ax_bt, card, fig)

    source = "" if card.get("source") == "blind" else f"{card.get('source', '')}   "
    lines = [
        f"{card['candname']}   {source}{card['event_utc']}",
        f"S/N={card['snr']:.1f}   DM={dm:.2f} pc cm$^{{-3}}$   "
        f"width={width * tsamp_s * 1e3:.1f} ms",
    ]
    lines.append(_coord_line(card, tsamp_s))
    fig.suptitle("\n".join(lines), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png


def make_candidate_figure(data, freqs_mhz, tsamp_s, t_rel_event_s, card, out_png,
                          ffactor: int | None = None, registry=None, layout: str | None = None,
                          subbands: int | None = None) -> Path:
    """Dispatch on layout: 'legacy' is what Slack shows until v2 is approved.

    ``subbands`` (or the lower-level ``ffactor``) forces the row count of the
    image panels; left unset, it follows the card S/N through subbands_for.
    """
    layout = layout or DEFAULT_LAYOUT
    if layout == "v2":
        return make_candidate_figure_v2(data, freqs_mhz, tsamp_s, t_rel_event_s, card, out_png,
                                        ffactor=ffactor, registry=registry, subbands=subbands)
    return make_candidate_figure_legacy(data, freqs_mhz, tsamp_s, t_rel_event_s, card, out_png,
                                        ffactor=ffactor, subbands=subbands)
