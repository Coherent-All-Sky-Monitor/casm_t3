"""Candidate diagnostic figure.

Layout v3 (approved by Vishnu 2026-09-22, ``layout="v2"``, the default):

    pulse profile dedispersed at the card DM   | DM = 0 band-mean timeseries
    waterfall dedispersed at the card DM       | boxcar S/N vs trial DM
    T1 candidates in the T2 gulp (count, beam) | sky: beams with a T1 candidate

Both top panels are boxcar S/N at the card width on one shared y range, so a
DM = 0 spike under the event is read against the same scale as the event.
The bottom row is drawn from the card's ``gulp`` block (every T1 candidate
T2 clustered in the gulp it triggered on); a card without that block gets
the pre-v2 bottom row instead (T1 context members against beam and the
beams lit within the coincidence window), in the rectangles it has always
had.

Every axes rectangle is a fixed figure fraction on a fixed 1510 x 1517 px
canvas: no tight_layout, nothing measured from the data, so every candidate
has the same geometry to the pixel and a wrapped title never moves a panel.
The approved prototype this was ported from is
/mnt/nvme3/vishnu/candidate_plot_v3/render_proto_prodgeom.py (flags
``--polish --beam-panel scatter --top-window 1 --top-right dm0``).

Coordinates come ONLY from the trigger card's ``sky`` block, which t2d fills from
the weights live at the event time (casm_t2.weights_registry). No static table:
the one used until 2026-09-02 put a 45 deg median error on every posted
coordinate (casm-wiki incidents.md). A card without ``sky`` gets no coordinates,
never a guess. The pointing grid for the sky panel comes from the card's
``pointings``, else from the registry by weights id, else by event time.
"""

from __future__ import annotations

import logging
import math
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from casm_t2 import weights_registry

from . import single_pulse

logger = logging.getLogger(__name__)

NBEAM_TOTAL = 512
COINCIDENCE_S = 256 * 1.048576e-3     # casm_t2 occupancy window_samp 256
TSAMP_DEFAULT_S = 1.048576e-3
K_DM = 4.148808e3                     # dispersion constant, s MHz^2 (pc cm^-3)^-1

# Event-physics panels on inferno (casm-wiki display-conventions.md).
WATERFALL_CMAP = "viridis"    # Vishnu 2026-09-03: viridis over inferno on the image panels
# The legacy (live) layout keeps its transientX look until the v2 layout is approved.
LEGACY_CMAP = "viridis"
SHOW_SOURCES = True    # sun, Cas A, Cyg A, Tau A markers + legend on the sky panel (approved by Vishnu 2026-09-03)
# "v2" is the name the daemons and CLIs pass; it draws the v3 layout
# approved on 2026-09-22. "legacy" is the pre-2026-09-03 figure.
DEFAULT_LAYOUT = "v2"
LAYOUTS = ("v2", "legacy")
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
# Row counts are restricted to powers of two: 3072 = 3 x 1024, so every power
# of two from 16 to 1024 divides the band, and 256 rows (12 channels/row) is
# the fine limit we keep. 4.2 sigma per pixel (S/N 23 -> 32 rows, S/N 43 ->
# 128 rows) was chosen on the 2026-09-10 injection renders.
SUBBAND_TARGET_SIGMA = 4.2
SUBBAND_MIN = 16
SUBBAND_MAX = 256
SUBBAND_CHOICES = (16, 32, 64, 128, 256)
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
    weights live at the event are unknown (panel then says so and draws nothing).

    The bottom-row sky panel for a card without a ``gulp`` block.
    """
    _polar_grid(ax)
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
    _source_markers(ax, card)
    n_lit = len(best)
    ax.text(0.5, -0.10, f"{n_lit} of {NBEAM_TOTAL} beams with a T1 candidate within {COINCIDENCE_S:.2f} s",
            transform=ax.transAxes, ha="center", va="top", fontsize=11)
    return n_lit


def _polar_grid(ax) -> None:
    """Alt/az polar frame shared by both sky panels: N up, E right, rim at alt 18."""
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlim(0, 72)                         # edge just below the beam-grid floor (alt ~20)
    ax.set_yticks([20, 40, 60])
    ax.set_yticklabels(["alt 70", "50", "30"], fontsize=9, color="#555555")
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(["N", "E", "S", "W"])
    ax.tick_params(axis="x", pad=6)
    ax.grid(alpha=0.35)


SOURCE_STYLE = {"Sun": ("*", "#f4a300", 190), "Cas A": ("P", "#2e86c1", 110),
                "Cyg A": ("X", "#27ae60", 110), "Tau A": ("D", "#8e44ad", 80)}


def _source_markers(ax, card: dict):
    """Sun and calibrator markers plus their legend, shared by both sky panels.

    Returns the legend (None when there is nothing to list). A caller that
    draws a second legend on the same axes must re-attach this one with
    ``ax.add_artist`` first: ``ax.legend`` replaces the previous legend.
    """
    sky = card.get("sky") or {}
    if not SHOW_SOURCES:
        if sky.get("sun_alt_deg") is not None and sky["sun_alt_deg"] > 0:
            ax.scatter(np.radians(sky["sun_az_deg"]), 90 - sky["sun_alt_deg"], marker="*",
                       s=170, color="#f4a300", edgecolors="k", linewidths=0.5, zorder=5)
        return None
    srcs = dict(sky.get("sources") or {})
    if not srcs and sky.get("sun_alt_deg") is not None:
        srcs["Sun"] = (sky["sun_alt_deg"], sky["sun_az_deg"])
    handles = []
    floor_alt = 90 - ax.get_rmax()
    for name, (m, col, size) in SOURCE_STYLE.items():
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
    if not handles:
        return None
    return ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.86, 1.10), fontsize=8.5,
                     frameon=False, labelspacing=0.9, handletextpad=0.6)


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


# =============================================================================
# Layout v3 (approved 2026-09-22)
# =============================================================================

# --- geometry ----------------------------------------------------------------
# Canvas 1510 x 1517 px at 120 dpi (every Slack post has the same pixel size).
# Every rectangle below is a figure fraction; nothing is measured from the
# data, so two candidates get the same axes to the pixel.
FIG_W_PX, FIG_H_PX, FIG_DPI = 1510, 1517, 120
FIG_RC = {"font.size": 10.5, "axes.titlesize": 11, "axes.labelsize": 11,
          "xtick.labelsize": 10, "ytick.labelsize": 10}
TITLE_TOP = 0.985                # figure fraction: top of the three-line title block
TITLE_FONTSIZE = 12
TITLE_WRAP_CHARS = 110
LEFT_X0, LEFT_X1 = 0.060, 0.455           # left column: 91 px gutter for the y labels
RIGHT_X0, RIGHT_X1 = 0.5232, 0.885        # right column: 103 px column gutter
# Each row reserves two title lines (44 px) above its axes; row 1 starts 24 px
# under the title block.
ROW1 = (0.7412, 0.9097 - 44.0 / 1517 + 24.0 / 1517)       # 0.7412 .. 0.8965
ROW2 = (0.3795, 0.6744)
ROW3 = (0.0600 - 12.0 / 1517, 0.3127 - 15.0 / 1517)       # 0.052089 .. 0.302812
# Bottom-left: two stacked axes sharing x. They start 12 px right of the
# panels above so the two-line y label of the histogram keeps a margin.
STACK_X0 = 0.0735
STACK_GAP = 0.012                # figure fraction between the two stacked axes
HIST_H_RATIO = 1.0               # candidate histogram height, against the beam panel
BEAM_H_RATIO = 1.4               # beam panel: 512 rows need the room
# Sky panel: square, 0.91 x the live rectangle, centred on the right column,
# its bottom 15 px under the live one so the two-line title fits above it.
SKY_SCALE = 0.91
SKY_DROP = 15.0 / 1517
DMT_CBAR_INSET = (1.02, 0.0, 0.03, 1.0)   # colour bar, axes fractions of the DM-time panel

# The pre-v2 bottom row, for cards without a ``gulp`` block: the rectangles the
# live figure has always drawn it in (gridspec row 3 of the 2026-09-03 layout).
LIVE_ROW3 = (0.06, 0.3095783601453035)
LIVE_MEMBERS_X1 = 0.55
LIVE_SKY_X0 = 0.585
# v3 row 2 sits 17 px lower than the live one, so the live sky panel's N label
# would land on the DM-time x label: it is shrunk 6% about its bottom centre.
LIVE_SKY_SCALE = 0.94

# --- row 1 ------------------------------------------------------------------
TICK_NBINS = 5
DMT_TICK_NBINS = 6
YLABEL_PAD = 4.0
CORNER_X = 0.02                  # corner text left edge, axes fraction
CORNER_Y = 0.92
CORNER_LINE_DY = 0.076           # line pitch, axes fraction (18 px of 236)
CORNER_MAX_FRAC = 0.46           # right edge budget: t = 0 sits at 0.5 in the 1 s window
CORNER_ASSERT_FRAC = 0.47        # rendered right edge must stay inside this
CORNER_PT = 9.0
CORNER_PT_SMALL = 8.5            # only if a single word will not fit at 9 pt
# The y range can be 1.1 x a spike near the left edge, which then reaches the
# corner text: a white backing keeps the text readable over the trace.
CORNER_BBOX = dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.5)
TOP_Y_EVENT_MULT = 3.0           # an unrelated spike gets at most 3 x the event's peak

# --- row 3 ------------------------------------------------------------------
TRIGGER_COLOR = "#c0392b"
OTHER_COLOR = "0.55"
OTHER_EDGE = "0.25"
HIST_BIN_SAMP = 96               # 100.66 ms, the whole number of samples closest to 0.1 s
HIST_FACE = "0.35"
HIST_EDGE = "0.15"
HIST_TICKS = (1, 10, 100, 1000)
JOB_LABEL_PT = 8.5
JOB_LABEL_X = 1.006              # axes fractions, just right of the spine
CORR_LABEL_X = 1.0910            # rotated, right of the widest job label
BEAM_SIZE_MIN_PT2 = 40.0         # beam scatter: marker area, linear in S/N
BEAM_SIZE_MAX_PT2 = 250.0
SKY_SIZE_MIN_PT2 = 20.0          # sky panel: marker area, linear in S/N
SKY_SIZE_MAX_PT2 = 250.0
GREY_BEAM_MIN_PT2 = 40.0         # grey sky beams stay bigger than a grid dot
TRIGGER_RING_PT2 = 320.0
# N/E/S/W sit this far outside the rim, in the panel's radius units
# (r = 90 - alt, rim at 72, so one unit is about 3 px).
COMPASS_PAD_R = 4.0
COMPASS = ((0.0, "N", "center", "bottom"), (90.0, "E", "left", "center"),
           (180.0, "S", "center", "top"), (270.0, "W", "right", "center"))
COMPASS_TEXT_PX = 17.0           # height of a compass label at the tick-label size
SKY_TITLE_SIZE = 11.0            # the other panels' title size
SKY_TITLE_CLEAR_PX = 6.0         # clear space between the title and the N label
SKY_LEGEND_ANCHOR = (0.89, 1.045)       # source legend: clear of the title and the rim
SKY_KEY_ANCHOR = (0.88, 0.02)           # S/N size key: lower right, outside the rim


class CornerTextError(RuntimeError):
    """A corner annotation reaches the t = 0 line; the render is aborted."""


# --- small helpers ------------------------------------------------------------

def _figure_px(ax) -> tuple[float, float]:
    """Axes width and height in pixels on the fixed canvas."""
    pos = ax.get_position()
    return pos.width * FIG_W_PX, pos.height * FIG_H_PX


def wrap_samples(dm: float, freqs_mhz: np.ndarray, tsamp_s: float) -> int:
    """Samples at the end of a dedispersed array that single_pulse.dedisperse
    wrapped round from the start of the dump (np.roll), i.e. the sweep length."""
    delay_s = K_DM * abs(float(dm)) * (freqs_mhz.min() ** -2 - freqs_mhz.max() ** -2)
    return int(math.ceil(delay_s / tsamp_s))


def step_fmt(ticks) -> str:
    """Tick label format from the tick STEP: decimals = clamp(-floor(log10(step)), 0, 2).

    138..146 by 2 prints 138, 0.04..0.16 by 0.04 prints 0.04, 0.0..1.0 by 0.2
    prints 0.2, so a label is never wider than the step needs.
    """
    t = np.asarray(ticks, dtype=float)
    step = float(np.median(np.diff(t))) if t.size > 1 else 1.0
    if not np.isfinite(step) or step <= 0:
        return "%.1f"
    return "%%.%df" % max(0, min(2, int(-math.floor(math.log10(abs(step))))))


def fmt_snr(value: float) -> str:
    """S/N for the off-scale corner line: 1071 without a decimal, 62.4 with one."""
    return f"{value:.0f}" if abs(value) >= 100 else f"{value:.1f}"


def event_utc_text(event_utc: str) -> str:
    """'2026-09-22 00:57:02.250 UTC' from the card's ISO event time."""
    t = datetime.fromisoformat(event_utc)
    if t.tzinfo is not None:
        t = t.astimezone(timezone.utc)
    return t.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " UTC"


def set_wrapped_title(ax, text: str, **kw):
    """Panel title, wrapped onto two lines when it is wider than its axes.

    The break goes after the first comma when that fits, otherwise at the
    most even word boundary. Never shrunk: every row reserves two title
    lines, so a wrap does not move a panel.
    """
    art = ax.set_title(text, **kw)
    renderer = ax.figure.canvas.get_renderer()
    ax_px = _figure_px(ax)[0]
    if art.get_window_extent(renderer).width <= ax_px:
        return art
    head, sep, tail = text.partition(", ")
    if sep:
        art.set_text(head + ",\n" + tail)
        if art.get_window_extent(renderer).width <= ax_px:
            return art
    words = text.split()
    best = None
    for i in range(1, len(words)):
        art.set_text(" ".join(words[:i]) + "\n" + " ".join(words[i:]))
        w = art.get_window_extent(renderer).width
        if best is None or w < best[0]:
            best = (w, art.get_text())
    if best is not None:
        art.set_text(best[1])
    return art


def place_corner_text(ax, lines: list[str]) -> list:
    """Corner annotation that stops short of the red t = 0 line.

    Every line must end at or before CORNER_MAX_FRAC of the axes width. A line
    that is too wide is split at the most even word boundary that fits (never
    leaving "DM" apart from its value), and the pieces stack CORNER_LINE_DY
    apart. If a single word still does not fit, the whole block drops to
    CORNER_PT_SMALL for that panel only.
    """
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    ax_w = _figure_px(ax)[0]
    pad_px = 2.0 * CORNER_BBOX["pad"] * FIG_DPI / 72.0
    budget = CORNER_MAX_FRAC - CORNER_X

    def width_frac(text: str, size: float) -> float:
        probe = ax.text(CORNER_X, CORNER_Y, text, transform=ax.transAxes,
                        fontsize=size, va="top")
        w = probe.get_window_extent(renderer).width
        probe.remove()
        return (w + pad_px) / ax_w

    def greedy(words: list, size: float) -> list:
        out, cur = [], ""
        for word in words:
            trial = f"{cur} {word}".strip()
            if cur and width_frac(trial, size) > budget:
                out.append(cur)
                cur = word
            else:
                cur = trial
        if cur:
            out.append(cur)
        return out

    def wrap(text: str, size: float) -> list:
        words = text.split()
        if width_frac(text, size) <= budget or len(words) < 2:
            return [text]
        best = None
        for k in range(1, len(words)):
            if words[k - 1] == "DM":          # never leave "DM" without its value
                continue
            a, b = " ".join(words[:k]), " ".join(words[k:])
            wa, wb = width_frac(a, size), width_frac(b, size)
            if max(wa, wb) > budget:
                continue
            key = (max(wa, wb), abs(wa - wb))
            if best is None or key < best[0]:
                best = (key, [a, b])
        return best[1] if best else greedy(words, size)

    size = CORNER_PT
    laid = [seg for line in lines for seg in wrap(line, size)]
    if any(width_frac(seg, size) > budget for seg in laid):
        size = CORNER_PT_SMALL
        laid = [seg for line in lines for seg in wrap(line, size)]
    return [ax.text(CORNER_X, CORNER_Y - i * CORNER_LINE_DY, seg, transform=ax.transAxes,
                    fontsize=size, va="top", bbox=CORNER_BBOX, gid="corner")
            for i, seg in enumerate(laid)]


def corner_right_fracs(ax, arts, renderer) -> list[float]:
    """Rendered right edge of each corner line (backing box included) as a
    fraction of the axes width."""
    box = ax.get_window_extent(renderer)
    return [float((a.get_window_extent(renderer).x1 - box.x0) / box.width) for a in arts]


# --- the gulp block -------------------------------------------------------------

def _trials_array(gulp: dict) -> np.ndarray:
    """gulp.trials as (n, 6): dt_s, beam, dm, snr, width_idx, cluster_id."""
    rows = gulp.get("trials") or []
    if not rows:
        return np.empty((0, 6), dtype=float)
    return np.asarray([r[:6] for r in rows], dtype=float).reshape(-1, 6)


def _triggered_cluster(gulp: dict) -> dict | None:
    """The cluster T2 dumped (outcome "triggered"), or None."""
    for cl in gulp.get("clusters") or []:
        if str(cl.get("outcome")) == "triggered":
            return cl
    return None


def gulp_span_s(gulp: dict, card: dict) -> tuple[float, float]:
    """The gulp's sample range in seconds relative to the event.

    The event sits wherever it falls in the block, so a trigger at a gulp edge
    reads as one. Falls back to the trials' own span when the card carries no
    ``samp`` or the block no sample range.
    """
    tsamp = float(gulp.get("tsamp_s") or TSAMP_DEFAULT_S)
    lo, hi, samp = gulp.get("samp_lo"), gulp.get("samp_hi"), card.get("samp")
    if lo is not None and hi is not None and samp is not None:
        span = ((float(lo) - float(samp)) * tsamp, (float(hi) - float(samp)) * tsamp)
    else:
        trials = _trials_array(gulp)
        span = ((float(trials[:, 0].min()), float(trials[:, 0].max())) if trials.size
                else (-4.0, 4.0))
    if not span[1] > span[0]:                     # one trial: give it room
        span = (span[0] - 4.0, span[1] + 4.0)
    return span


def gulp_utc_span(gulp: dict) -> tuple[str, str] | None:
    """UTC of the gulp's first and last sample as HH:MM:SS.s (the date is in
    the figure title). None when the block has no utc_start or sample range."""
    try:
        t0 = datetime.strptime(str(gulp["utc_start"]), "%Y-%m-%d-%H:%M:%S")
        tsamp = float(gulp["tsamp_s"])
        lo = t0 + timedelta(seconds=float(gulp["samp_lo"]) * tsamp + 0.05)
        hi = t0 + timedelta(seconds=float(gulp["samp_hi"]) * tsamp + 0.05)
    except (KeyError, TypeError, ValueError):
        return None
    return (lo.strftime("%H:%M:%S.%f")[:-5], hi.strftime("%H:%M:%S.%f")[:-5])


def gulp_span_seconds(gulp: dict) -> float | None:
    """Length of the gulp in seconds (8.6 s for 8192 samples), None if unstated."""
    lo, hi = gulp.get("samp_lo"), gulp.get("samp_hi")
    if lo is None or hi is None:
        return None
    return (float(hi) - float(lo) + 1) * float(gulp.get("tsamp_s") or TSAMP_DEFAULT_S)


def gulp_hist_title(gulp: dict) -> str:
    """Title over the candidate histogram, broken after the gulp number's comma.

    The count is len(gulp.trials): the candidates T2 clustered, after the
    widest-boxcar drop, not n_raw. When t2d capped the list
    (``trials_truncated``) it reads "<drawn> of <n_trials_total>".
    """
    n = len(gulp.get("trials") or [])
    total = gulp.get("n_trials_total")
    if gulp.get("trials_truncated") and total is not None and int(total) > n:
        count, noun_n = f"{n} of {int(total)}", int(total)
    else:
        count, noun_n = f"{n}", n
    text = (f"{count} T1 candidate{'' if noun_n == 1 else 's'} clustered in gulp "
            f"{gulp.get('gulp', '?')}")
    span = gulp_utc_span(gulp)
    if span:
        text += f",\n{span[0]} to {span[1]} UTC"
    if (gulp.get("storm") or {}).get("stormy"):
        text += " (storm gulp)"
    return text


def sky_title(gulp: dict, n_red: int, n_grey: int) -> str:
    """Two lines over the sky panel, counting the beam markers it draws."""
    span_s = gulp_span_seconds(gulp)
    span = f"{span_s:.1f} s gulp" if span_s else "gulp"
    parts = [f"{n_red} in the triggered group (red)"]
    if n_grey:
        parts.append(f"{n_grey} other{'' if n_grey == 1 else 's'} (grey)")
    return f"Beams with a T1 candidate in this {span}:\n" + ", ".join(parts)


def sky_beam_sets(gulp: dict, nbeam: int = NBEAM_TOTAL) -> tuple[dict[int, float], dict[int, float]]:
    """(red, grey) beam -> best S/N maps for the sky panel.

    Red is every member beam of the triggered cluster, grey every member beam
    of any other cluster; a beam in both is red only. One entry per beam, at
    its best S/N.
    """
    trig = _triggered_cluster(gulp)
    red: dict[int, float] = {}
    for b, v in ((trig or {}).get("beams") or []):
        b = int(b)
        if 0 <= b < nbeam:
            red[b] = max(red.get(b, -np.inf), float(v))
    grey: dict[int, float] = {}
    for cl in gulp.get("clusters") or []:
        if cl is trig:
            continue
        for b, v in (cl.get("beams") or []):
            b = int(b)
            if 0 <= b < nbeam and b not in red:
                grey[b] = max(grey.get(b, -np.inf), float(v))
    return red, grey


def _linear_area(snr, lo: float, hi: float, a_min: float, a_max: float) -> np.ndarray:
    """Marker area in pt^2, linear in S/N from lo -> a_min to hi -> a_max, clipped."""
    snr = np.asarray(snr, dtype=float)
    span = max(float(hi) - float(lo), 1e-6)
    return a_min + (a_max - a_min) * np.clip((snr - float(lo)) / span, 0.0, 1.0)


# --- bottom row panels ------------------------------------------------------------

def _gulp_trials_panel(ax_hist, ax_beam, gulp: dict, card: dict) -> None:
    """Every T1 candidate in the gulp, in two stacked axes sharing x.

    Top: candidates per HIST_BIN_SAMP-sample bin, log y; bins are cut on the
    gulp's own sample grid from samp_lo. Bottom: beam index against time, one
    marker per candidate sized by S/N, the triggered cluster's in red.
    """
    trials = _trials_array(gulp)
    tsamp = float(gulp.get("tsamp_s") or TSAMP_DEFAULT_S)
    lo, hi = gulp_span_s(gulp, card)
    s_lo, s_hi, samp = gulp.get("samp_lo"), gulp.get("samp_hi"), card.get("samp")
    if s_lo is not None and s_hi is not None and samp is not None:
        n_bins = max(1, int(math.ceil((float(s_hi) + 1 - float(s_lo)) / HIST_BIN_SAMP)))
        edges = (float(s_lo) + np.arange(n_bins + 1) * HIST_BIN_SAMP - float(samp)) * tsamp
    else:
        n_bins = max(1, int(round((hi - lo) / (HIST_BIN_SAMP * tsamp))))
        edges = np.linspace(lo, hi, n_bins + 1)
    dt = trials[:, 0]
    # dt_s is rounded in the card, so a trial on the gulp's first sample can land
    # a hair outside the first edge: clip, so every trial is counted once.
    counts, _ = np.histogram(np.clip(dt, edges[0], edges[-1]), bins=edges)
    centres = 0.5 * (edges[:-1] + edges[1:])

    ax_hist.bar(centres, counts, width=HIST_BIN_SAMP * tsamp, align="center", color=HIST_FACE,
                edgecolor=HIST_EDGE, linewidth=0.4, zorder=2, gid="hist")
    ax_hist.axvline(0.0, color=TRIGGER_COLOR, lw=1.1, alpha=0.9, zorder=3)
    top = max(10.0, float(counts.max()) * 1.6) if counts.size else 10.0
    ax_hist.set_ylim(0.7, top)            # before the log scale: an empty gulp has no
    ax_hist.set_yscale("log")             # positive bar to autoscale on
    ax_hist.set_yticks([t for t in HIST_TICKS if t <= top])
    ax_hist.set_yticklabels([str(t) for t in HIST_TICKS if t <= top])
    ax_hist.minorticks_off()
    ax_hist.tick_params(labelbottom=False)
    ax_hist.set_ylabel("candidates\nper 0.1 s")
    set_wrapped_title(ax_hist, gulp_hist_title(gulp))

    for edge in range(64, NBEAM_TOTAL, 64):
        ax_beam.axhline(edge, color="0.55" if edge == 256 else "0.85",
                        lw=1.0 if edge == 256 else 0.5, zorder=0)
    trig = _triggered_cluster(gulp)
    trig_id = None if trig is None or trig.get("id") is None else int(trig["id"])
    is_trig = trials[:, 5].astype(int) == trig_id if trig_id is not None \
        else np.zeros(trials.shape[0], dtype=bool)
    if trials.size:
        snr = trials[:, 3]
        sizes = _linear_area(snr, snr.min(), snr.max(), BEAM_SIZE_MIN_PT2, BEAM_SIZE_MAX_PT2)
        ax_beam.scatter(dt[~is_trig], trials[~is_trig, 1], s=sizes[~is_trig],
                        color=OTHER_COLOR, alpha=0.6, edgecolors=OTHER_EDGE, linewidths=0.8,
                        zorder=2, gid="trials_other")
        if is_trig.any():
            ax_beam.scatter(dt[is_trig], trials[is_trig, 1], s=sizes[is_trig],
                            color=TRIGGER_COLOR, alpha=0.9, edgecolors=TRIGGER_COLOR,
                            linewidths=0.8, zorder=3, gid="trials_triggered")
    ax_beam.axvline(0.0, color=TRIGGER_COLOR, lw=1.1, alpha=0.9, zorder=4)
    ax_beam.set_ylim(-8, NBEAM_TOTAL + 8)
    ax_beam.set_yticks(np.arange(0, NBEAM_TOTAL + 1, 64))
    ax_beam.set_ylabel("beam")
    ax_beam.set_xlabel("time - event (s)")
    ax_beam.set_xlim(lo, hi)
    # Job and correlator labels live in the strip outside the right spine:
    # "job N" against the 64-beam band it marks, then corr1 / corr2 rotated in
    # their own column, each centred on its half of the array.
    for j in range(8):
        ax_beam.text(JOB_LABEL_X, 64 * j + 32, f"job {j}", transform=ax_beam.get_yaxis_transform(),
                     ha="left", va="center", fontsize=JOB_LABEL_PT, color="0.35", clip_on=False,
                     zorder=5, gid="job_label")
    for label, centre in (("corr1", 128), ("corr2", 384)):
        ax_beam.text(CORR_LABEL_X, centre, label, transform=ax_beam.get_yaxis_transform(),
                     ha="center", va="center", rotation=90, fontsize=JOB_LABEL_PT, color="0.2",
                     clip_on=False, zorder=5, gid="corr_label")


def _sky_size_key(ax, snr_lo: float, snr_hi: float):
    """S/N size key for the sky panel: the smallest and largest beam S/N drawn
    (one entry when they print the same), with the grey markers' look."""
    values = [snr_lo] if f"{snr_lo:.1f}" == f"{snr_hi:.1f}" else [snr_lo, snr_hi]
    handles = [ax.scatter([], [], s=max(_linear_area([v], snr_lo, snr_hi, SKY_SIZE_MIN_PT2,
                                                     SKY_SIZE_MAX_PT2)[0], GREY_BEAM_MIN_PT2),
                          color=OTHER_COLOR, alpha=0.75, edgecolors=OTHER_EDGE, linewidths=0.8,
                          label=f"S/N {v:.1f}") for v in values]
    return ax.legend(handles=handles, fontsize=9.0, handletextpad=0.8, loc="lower left",
                     bbox_to_anchor=SKY_KEY_ANCHOR, frameon=False, labelspacing=1.5,
                     borderpad=0.3)


def _gulp_sky_panel(ax, gulp: dict, card: dict, pointings: dict | None) -> tuple[int, int]:
    """Alt/az panel: one marker per beam that had a T1 candidate in the gulp.

    Red: a member beam of the triggered cluster (ring on its peak beam). Grey:
    a member beam of any other cluster. One marker per beam at its best S/N;
    a beam in both sets is red. Returns (red beams, grey beams) drawn.
    """
    _polar_grid(ax)
    # the N/E/S/W tick labels would sit on the rim and the grid dots: drop
    # them and draw the letters just outside the rim instead
    ax.set_xticklabels([])
    for az_deg, label, ha, va in COMPASS:
        ax.text(np.radians(az_deg), ax.get_rmax() + COMPASS_PAD_R, label, ha=ha, va=va,
                fontsize=plt.rcParams["xtick.labelsize"], clip_on=False, zorder=7)
    if pointings is None:
        ax.text(0.5, 0.5, "beam pointings unknown\n(no weights registered for this time)",
                ha="center", va="center", transform=ax.transAxes, color="0.4", fontsize=10)
        src_legend = _source_markers(ax, card)
        if src_legend is not None:
            src_legend.set_bbox_to_anchor(SKY_LEGEND_ANCHOR, transform=ax.transAxes)
        return 0, 0
    alt = np.asarray(pointings["alt_deg"], dtype=float)
    az = np.asarray(pointings["az_deg"], dtype=float)
    ax.scatter(np.radians(az), 90 - alt, s=5, color="#d0d0d0", zorder=1)
    red, grey = sky_beam_sets(gulp, min(NBEAM_TOTAL, alt.size))
    pool = list(red.values()) + list(grey.values())
    snr_lo = float(min(pool)) if pool else 0.0
    snr_hi = float(max(pool)) if pool else 1.0
    for beams, colour, alpha, edge, lw, floor, z, gid in (
            (grey, OTHER_COLOR, 0.75, OTHER_EDGE, 0.8, GREY_BEAM_MIN_PT2, 2, "sky_grey"),
            (red, TRIGGER_COLOR, 0.95, TRIGGER_COLOR, 0.3, 0.0, 4, "sky_red")):
        if not beams:
            continue
        idx = sorted(beams)
        sizes = np.maximum(_linear_area([beams[b] for b in idx], snr_lo, snr_hi,
                                        SKY_SIZE_MIN_PT2, SKY_SIZE_MAX_PT2), floor)
        ax.scatter(np.radians(az[idx]), 90 - alt[idx], s=sizes, color=colour, alpha=alpha,
                   edgecolors=edge, linewidths=lw, zorder=z, gid=gid)
    trig = _triggered_cluster(gulp)
    if trig and trig.get("peak"):
        pb = int(trig["peak"][1])
        if 0 <= pb < alt.size:
            ax.scatter(np.radians(az[pb]), 90 - alt[pb], s=TRIGGER_RING_PT2, facecolors="none",
                       edgecolors="#111111", linewidths=1.4, zorder=6, gid="sky_ring")
    src_legend = _source_markers(ax, card)
    if src_legend is not None:
        src_legend.set_bbox_to_anchor(SKY_LEGEND_ANCHOR, transform=ax.transAxes)
        # re-attach: the size key's ax.legend() would otherwise replace it
        ax.add_artist(src_legend)
    if pool:
        _sky_size_key(ax, snr_lo, snr_hi)
    return len(red), len(grey)


def _resolve_pointings(card: dict, gulp: dict | None,
                       registry: weights_registry.Registry | None) -> dict | None:
    """Beam pointing grid for the sky panel.

    The card's own ``pointings`` first (t2d embeds the live weights' table so
    corr2, which has no registry, can plot), then the registry by the weights
    id in the ``sky`` or ``gulp`` block, then the registry by event time.
    """
    if (card.get("pointings") or {}).get("alt_deg"):
        return card["pointings"]
    weights_id = (card.get("sky") or {}).get("weights_id") or (gulp or {}).get("weights_id")
    try:
        reg = registry or weights_registry.default_registry()
        pointings = reg.product(weights_id) if weights_id else None
        if pointings is None and card.get("event_utc"):
            pointings = reg.pointings_for(datetime.fromisoformat(card["event_utc"]))
    except Exception:                       # no registry on this node, or no match
        logger.warning("no beam pointings for %s", card.get("candname"), exc_info=True)
        return None
    return pointings if (pointings or {}).get("alt_deg") else None


# --- the figure ---------------------------------------------------------------

def new_canvas():
    """An empty figure on the fixed canvas (call inside plt.rc_context(FIG_RC))."""
    fig = plt.figure(figsize=(FIG_W_PX / FIG_DPI, FIG_H_PX / FIG_DPI))
    fig.set_dpi(FIG_DPI)          # measure text in the pixels the PNG will have
    return fig


def draw_candidate_figure(fig, data: np.ndarray, freqs_mhz: np.ndarray, tsamp_s: float,
                          t_rel_event_s: float, card: dict, ffactor: int | None = None,
                          registry: weights_registry.Registry | None = None,
                          subbands: int | None = None) -> dict:
    """Draw the v3 candidate figure into ``fig`` (from new_canvas()).

    data : (nchan, ntime) float32, raw (dispersed) cutout for the detection beam.
    t_rel_event_s : candidate time (top-of-band arrival) relative to the first
        sample of ``data``.
    card : trigger card (candname/snr/dm/width/beam/samp/event_utc, the ``sky``
        block written by t2d, and the ``gulp`` block or the ``context`` members).

    Returns {"axes": {role: Axes}}. Raises CornerTextError when a corner
    annotation would reach the t = 0 line.
    """
    tsamp = float(tsamp_s)
    t_rel = float(t_rel_event_s)
    dm, width = float(card["dm"]), 2 ** int(card["width"])
    gulp = card.get("gulp") or None
    nsub, ffactor = _subband_plan(card, freqs_mhz.size, ffactor, subbands)

    norm = single_pulse.normalise(data)
    dedis = single_pulse.dedisperse(norm, dm, freqs_mhz, tsamp)
    ntime = dedis.shape[1]
    # samples past n_valid are the start of the dump rolled round by dedisperse
    n_valid = max(width * 4, ntime - wrap_samples(dm, freqs_mhz, tsamp))
    n_ok = min(n_valid, ntime)

    # Framing, as in the live figure: about a second either side of the event
    # on a half-boxcar grid, wider for wide pulses; DM-time fits its wings.
    tfactor = max(1, width // 2)
    n_wf = max(1, ntime // tfactor)
    t_wf0 = (tfactor / 2) * tsamp - t_rel
    t_wf1 = ((n_wf - 1) * tfactor + tfactor / 2) * tsamp - t_rel

    # Row 1: both panels are profile_snr (boxcar at the card width, robust
    # noise) of the band mean of the SAME per-channel normalised array, left
    # dedispersed at the card DM, right unshifted. The noise is measured on the
    # whole series, the wrap tail included: on 260922lruuie a burst 1.7 s after
    # the event covers most of the unwrapped part, and a MAD over that alone
    # read S/N 29 as 0.7. Only the drawn dedispersed trace stops at the wrap.
    prof_dd = single_pulse.profile_snr(dedis.mean(axis=0), width)
    prof_dm0 = single_pulse.profile_snr(norm.mean(axis=0), width)
    t_nat = np.arange(ntime) * tsamp - t_rel
    # the peak is looked for within two boxcars of the sample the card names,
    # so a brighter unrelated burst elsewhere cannot define the on-pulse window
    i_event = int(round(t_rel / tsamp))
    s_lo = max(0, i_event - 2 * width)
    s_hi = min(ntime, i_event + 2 * width + 1)
    if s_hi > s_lo:
        i_peak = s_lo + int(np.argmax(prof_dd[s_lo:s_hi]))
        peak_dd = float(prof_dd[i_peak])
        peak_dm0 = float(prof_dm0[s_lo:s_hi].max())
    else:                                        # event outside the data
        i_peak = int(np.clip(i_event, 0, ntime - 1))
        peak_dd = peak_dm0 = float("nan")
    # the boxcar that peak refers to: profile_snr convolves with mode="same",
    # so profile sample i is the window [i - width + 1 + (width - 1) // 2, +width)
    i_on = int(np.clip(i_peak - width + 1 + (width - 1) // 2, 0, max(0, n_valid - width)))

    # Row 2 left: one subband x one boxcar per pixel, the display grid phased
    # so the boxcar around the profile peak is exactly one column.
    start = int(i_on % width)
    wf_show = single_pulse.downsample(dedis[:, start:], ffactor, width)
    nsub = wf_show.shape[0]      # what the panel shows, after truncation
    t_show = (np.arange(wf_show.shape[1]) * width + start + width / 2) * tsamp - t_rel

    # Row 2 right: DM-time on its own, much finer channel averaging (the
    # display subbands would smear the pulse before dedispersion).
    f_dmt = _dmt_plan(dm, width, freqs_mhz, tsamp)
    small = single_pulse.downsample(norm, f_dmt, 1)
    f_small = _block_mean_freqs(freqs_mhz, f_dmt)[: small.shape[0]]
    dms = single_pulse.dm_grid(dm)
    dmt = single_pulse.dm_time(small, f_small, tsamp, dms, width)
    half_prof = max(1.0, 30 * width * tsamp)
    wing_s = K_DM * 0.5 * (dms[-1] - dms[0]) * (freqs_mhz.min() ** -2 - freqs_mhz.max() ** -2)
    half_dmt = max(half_prof, 0.75 * wing_s)
    xlim_prof = (max(-half_prof, t_wf0), min(half_prof, t_wf1))
    xlim_dmt = (max(-half_dmt, t_wf0), min(half_dmt, t_wf1))
    tfactor_dmt = _dmt_tfactor(width, xlim_dmt, tsamp)
    dmt_disp = single_pulse.downsample_max(dmt, tfactor_dmt)
    t_dmt = (np.arange(dmt_disp.shape[1]) * tfactor_dmt + tfactor_dmt / 2) * tsamp - t_rel

    # --- axes: every rectangle from the constants, nothing measured
    lw_ = LEFT_X1 - LEFT_X0
    rw_ = RIGHT_X1 - RIGHT_X0
    ax_prof = fig.add_axes([LEFT_X0, ROW1[0], lw_, ROW1[1] - ROW1[0]], label="profile")
    ax_dm0 = fig.add_axes([RIGHT_X0, ROW1[0], rw_, ROW1[1] - ROW1[0]], label="dm0")
    ax_wf = fig.add_axes([LEFT_X0, ROW2[0], lw_, ROW2[1] - ROW2[0]], label="waterfall")
    ax_dmt = fig.add_axes([RIGHT_X0, ROW2[0], rw_, ROW2[1] - ROW2[0]], label="dm_time")
    axes = {"profile": ax_prof, "dm0": ax_dm0, "waterfall": ax_wf, "dm_time": ax_dmt}

    # --- row 1 left: dedispersed profile
    sn_dd = prof_dd[:n_ok]
    in_dd = (t_nat[:n_ok] >= xlim_prof[0]) & (t_nat[:n_ok] <= xlim_prof[1])
    in_dm0 = (t_nat >= xlim_prof[0]) & (t_nat <= xlim_prof[1])
    pmax_dd = float(sn_dd[in_dd].max()) if in_dd.any() else float(sn_dd.max())
    pmax_dm0 = float(prof_dm0[in_dm0].max()) if in_dm0.any() else float(prof_dm0.max())
    # fit both drawn traces, but never give an unrelated spike more than three
    # times the event's own peak: past that it runs off the top and says so
    top_y = 1.1 * min(max(pmax_dd, pmax_dm0, 5.0), TOP_Y_EVENT_MULT * peak_dd)
    if not top_y > 0.0:             # non-detection card: local peak at or below 0
        top_y = 1.1 * max(pmax_dd, pmax_dm0, 5.0)
    top_ylim = (0.0, top_y)

    ax_prof.plot(t_nat[:n_ok], sn_dd, "k-", lw=0.7)
    ax_prof.axvline(0, color=TRIGGER_COLOR, alpha=0.8, lw=0.9)
    ax_prof.set_ylabel(f"boxcar S/N (w = {width})")
    set_wrapped_title(ax_prof, f"Pulse profile dedispersed at DM = {dm:.2f}")
    left_lines = [f"peak S/N near event: {peak_dd:.1f}"]
    if pmax_dd > top_ylim[1]:
        left_lines.append(f"max in window: {fmt_snr(pmax_dd)} (off scale)")
    corner_left = place_corner_text(ax_prof, left_lines)
    ax_prof.set_xlim(*xlim_prof)
    ax_prof.set_ylim(*top_ylim)
    # integer=True as well as the integer format: on a short range MaxNLocator
    # would otherwise pick 1.5-wide steps and "%.0f" would label 1.5 as 2
    ax_prof.yaxis.set_major_locator(MaxNLocator(nbins=TICK_NBINS, integer=True))
    ax_prof.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax_prof.yaxis.labelpad = YLABEL_PAD

    # --- row 1 right: DM = 0 band mean, same window, ticks and y range
    ax_dm0.plot(t_nat, prof_dm0, "-", color="0.3", lw=0.7)
    ax_dm0.axvline(0, color=TRIGGER_COLOR, alpha=0.8, lw=0.9)
    ax_dm0.set_ylabel(f"boxcar S/N (w = {width})")
    set_wrapped_title(ax_dm0, "DM = 0 band-mean timeseries")
    right_lines = [f"peak S/N at DM 0 near event: {peak_dm0:.1f}"]
    if pmax_dm0 > top_ylim[1]:
        right_lines.append(f"max in window: {fmt_snr(pmax_dm0)} (off scale)")
    corner_right = place_corner_text(ax_dm0, right_lines)
    ax_dm0.set_xlim(*xlim_prof)
    ax_dm0.set_xticks(ax_prof.get_xticks())
    ax_dm0.set_xlim(*xlim_prof)
    ax_dm0.set_yticks(ax_prof.get_yticks())
    ax_dm0.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax_dm0.yaxis.labelpad = YLABEL_PAD
    ax_dm0.set_ylim(*top_ylim)

    # --- row 2 left: dedispersed waterfall
    med = np.median(wf_show)
    sigma = 1.4826 * np.median(np.abs(wf_show - med)) or (wf_show.std() or 1.0)
    # extent is the outer EDGES of the first and last column, not their centres
    half_col = width * tsamp / 2
    ax_wf.imshow(wf_show, aspect="auto", interpolation="nearest",
                 extent=[t_show[0] - half_col, t_show[-1] + half_col, freqs_mhz[-1], freqs_mhz[0]],
                 vmin=med - 1.5 * sigma, vmax=med + 3.5 * sigma, cmap=WATERFALL_CMAP)
    _dm0_curve(ax_wf, dm, freqs_mhz)
    ax_wf.set_xlim(xlim_prof)
    ax_wf.set_ylim(freqs_mhz.min(), freqs_mhz.max())
    ax_wf.set_ylabel("frequency (MHz)")
    ax_wf.set_xlabel("time - event (s)")
    wf_title = f"Waterfall, dedispersed at DM = {dm:.2f}, red: DM = 0 curve"
    if subbands:
        wf_title += f" ({nsub} subbands)"
    set_wrapped_title(ax_wf, wf_title)

    # --- row 2 right: boxcar S/N vs trial DM
    half_dmt_col = tfactor_dmt * tsamp / 2          # column edges, as on the waterfall
    im_dmt = ax_dmt.imshow(dmt_disp, aspect="auto", origin="lower", interpolation="nearest",
                           extent=[t_dmt[0] - half_dmt_col, t_dmt[-1] + half_dmt_col,
                                   dms[0], dms[-1]],
                           vmin=0, vmax=max(8.0, np.percentile(dmt_disp, 99.9)),
                           cmap=WATERFALL_CMAP)
    cax = ax_dmt.inset_axes(DMT_CBAR_INSET)
    cax.set_label("dm_time_cbar")
    cb = fig.colorbar(im_dmt, cax=cax)
    cb.set_label("boxcar S/N")
    ax_dmt.plot(0, dm, "o", ms=14, mfc="none", mec=TRIGGER_COLOR, mew=1.3)
    ax_dmt.set_xlim(xlim_dmt)
    ax_dmt.set_ylim(dms[0], dms[-1])
    ax_dmt.set_ylabel(r"DM (pc cm$^{-3}$)")
    ax_dmt.set_xlabel("time - event (s)")
    ax_dmt.xaxis.labelpad = 0.5      # room for the sky title under it
    ax_dmt.yaxis.set_major_locator(MaxNLocator(nbins=DMT_TICK_NBINS))
    ax_dmt.yaxis.set_major_formatter(FormatStrFormatter(step_fmt(ax_dmt.get_yticks())))
    ax_dmt.yaxis.labelpad = YLABEL_PAD
    cb.locator = MaxNLocator(nbins=DMT_TICK_NBINS)
    cb.formatter = FormatStrFormatter("%.0f")
    cb.update_ticks()
    set_wrapped_title(ax_dmt, f"Boxcar S/N vs trial DM (w = {width})")
    axes["dm_time_cbar"] = cax

    # --- row 3
    pointings = _resolve_pointings(card, gulp, registry)
    if gulp:
        axes.update(_draw_gulp_row(fig, gulp, card, pointings))
    else:
        axes.update(_draw_live_row(fig, card, pointings, dm))

    source_lines = [
        textwrap.fill(f"{card['candname']}   {event_utc_text(card['event_utc'])}",
                      TITLE_WRAP_CHARS),
        f"hella S/N = {float(card['snr']):.1f}   DM = {dm:.2f} pc cm$^{{-3}}$   "
        f"width = {width * tsamp * 1e3:.1f} ms",
        _coord_line(card, tsamp),
    ]
    fig.suptitle("\n".join(source_lines), y=TITLE_TOP, fontsize=TITLE_FONTSIZE)

    # a draw first: before it, text extents are not final
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for panel, ax, arts in (("profile", ax_prof, corner_left), ("dm0", ax_dm0, corner_right)):
        for frac in corner_right_fracs(ax, arts, renderer):
            if frac >= CORNER_ASSERT_FRAC:
                raise CornerTextError(f"{panel} corner text reaches x fraction {frac:.3f} "
                                      f"(limit {CORNER_ASSERT_FRAC})")
    return {"axes": axes}


def _draw_gulp_row(fig, gulp: dict, card: dict, pointings: dict | None) -> dict:
    """Bottom row for a card with a ``gulp`` block: candidates left, sky right."""
    bt_y0, bt_h = ROW3[0], ROW3[1] - ROW3[0]
    bl_w = LEFT_X1 - STACK_X0
    h_unit = (bt_h - STACK_GAP) / (HIST_H_RATIO + BEAM_H_RATIO)
    h_beam, h_hist = h_unit * BEAM_H_RATIO, h_unit * HIST_H_RATIO
    ax_beam = fig.add_axes([STACK_X0, bt_y0, bl_w, h_beam], label="beams")
    ax_hist = fig.add_axes([STACK_X0, bt_y0 + h_beam + STACK_GAP, bl_w, h_hist],
                           sharex=ax_beam, label="hist")
    sky_h = (bt_h + 0.04) * SKY_SCALE
    sky_w = sky_h * FIG_H_PX / FIG_W_PX
    sky_cx = (RIGHT_X0 + RIGHT_X1) / 2
    sky_y0 = bt_y0 - 0.02 - SKY_DROP
    ax_sky = fig.add_axes([sky_cx - sky_w / 2, sky_y0, sky_w, sky_h], projection="polar",
                          label="sky")

    _gulp_trials_panel(ax_hist, ax_beam, gulp, card)
    n_red, n_grey = _gulp_sky_panel(ax_sky, gulp, card, pointings)
    # The sky title sits in the strip between the DM-time x label and the N
    # compass label, SKY_TITLE_CLEAR_PX above N; placed from the constants.
    sky_top_px = (1.0 - (sky_y0 + sky_h)) * FIG_H_PX
    n_pad_px = COMPASS_PAD_R * (sky_h * FIG_H_PX / 2) / 72
    title_bottom_px = sky_top_px - n_pad_px - COMPASS_TEXT_PX - SKY_TITLE_CLEAR_PX
    fig.text(sky_cx, 1.0 - title_bottom_px / FIG_H_PX, sky_title(gulp, n_red, n_grey),
             ha="center", va="bottom", fontsize=SKY_TITLE_SIZE, linespacing=1.0, gid="sky_title")
    return {"hist": ax_hist, "beams": ax_beam, "sky": ax_sky}


def _draw_live_row(fig, card: dict, pointings: dict | None, dm: float) -> dict:
    """Bottom row for a card without a ``gulp`` block: the pre-v2 panels
    (T1 context members, beams lit within the coincidence window) in the
    rectangles the live figure has always drawn them in, the sky panel 6%
    smaller (LIVE_SKY_SCALE)."""
    y0, h = LIVE_ROW3[0], LIVE_ROW3[1] - LIVE_ROW3[0]
    ax_bt = fig.add_axes([LEFT_X0, y0, LIVE_MEMBERS_X1 - LEFT_X0, h], label="members")
    sky_h0 = h + 0.04
    sky_h = sky_h0 * LIVE_SKY_SCALE
    sky_w0, sky_w = sky_h0 * FIG_H_PX / FIG_W_PX, sky_h * FIG_H_PX / FIG_W_PX
    ax_sky = fig.add_axes([LIVE_SKY_X0 + (sky_w0 - sky_w) / 2, y0 - 0.02, sky_w, sky_h],
                          projection="polar", label="sky")
    ctx = card.get("context") or {}
    members = np.asarray(ctx.get("members") or [], dtype=float).reshape(-1, 5)
    window_s = float(ctx.get("window_s", 4.0))
    dm_norm = mcolors.Normalize(vmin=15.0, vmax=max(
        60.0, 1.5 * dm, float(np.percentile(members[:, 2], 95)) if members.size else 0.0))
    floor = _snr_floor(members, card)
    sc = _member_panel(ax_bt, members, card, window_s, dm_norm, floor)
    _sky_panel(ax_sky, members, card, pointings, dm_norm, floor)
    out = {"members": ax_bt, "sky": ax_sky}
    if sc is not None:
        # under the DM-time colour bar, leaving room for the source legend above
        rw_ = RIGHT_X1 - RIGHT_X0
        cax = fig.add_axes([RIGHT_X0 + DMT_CBAR_INSET[0] * rw_, y0 + 0.02,
                            DMT_CBAR_INSET[2] * rw_, h - 0.12], label="members_cbar")
        fig.colorbar(sc, cax=cax).set_label(r"DM (pc cm$^{-3}$)")
        out["members_cbar"] = cax
    return out


def make_candidate_figure_v3(data: np.ndarray, freqs_mhz: np.ndarray, tsamp_s: float,
                             t_rel_event_s: float, card: dict, out_png: str | Path,
                             ffactor: int | None = None,
                             registry: weights_registry.Registry | None = None,
                             subbands: int | None = None) -> Path:
    """Render the v3 candidate figure to ``out_png`` (1510 x 1517 px).

    See draw_candidate_figure for the arguments. The figure is closed whether
    or not the render succeeds.
    """
    out_png = Path(out_png)
    with plt.rc_context(FIG_RC):
        fig = new_canvas()
        try:
            draw_candidate_figure(fig, data, freqs_mhz, tsamp_s, t_rel_event_s, card,
                                  ffactor=ffactor, registry=registry, subbands=subbands)
            out_png.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_png, dpi=FIG_DPI)      # fixed canvas: same pixel size every time
        finally:
            plt.close(fig)
    return out_png


# the layout name the daemons pass ("v2") draws v3
make_candidate_figure_v2 = make_candidate_figure_v3


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
    try:
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
        if subbands:
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
    finally:
        plt.close(fig)
    return out_png


def make_candidate_figure(data, freqs_mhz, tsamp_s, t_rel_event_s, card, out_png,
                          ffactor: int | None = None, registry=None, layout: str | None = None,
                          subbands: int | None = None) -> Path:
    """Dispatch on layout: "v2" (the default) draws the v3 figure, "legacy" the
    pre-2026-09-03 one.

    ``subbands`` (or the lower-level ``ffactor``) forces the row count of the
    image panels; left unset, it follows the card S/N through subbands_for.
    """
    layout = layout or DEFAULT_LAYOUT
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, not {layout!r}")
    if layout == "v2":
        return make_candidate_figure_v3(data, freqs_mhz, tsamp_s, t_rel_event_s, card, out_png,
                                        ffactor=ffactor, registry=registry, subbands=subbands)
    return make_candidate_figure_legacy(data, freqs_mhz, tsamp_s, t_rel_event_s, card, out_png,
                                        ffactor=ffactor, subbands=subbands)
