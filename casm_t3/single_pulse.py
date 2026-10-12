"""Single-pulse processing for candidate plots: dedispersion and profiles."""

from __future__ import annotations

import math

import numpy as np


def dedisperse(data: np.ndarray, dm: float, freqs_mhz: np.ndarray, tsamp_s: float) -> np.ndarray:
    """Incoherently dedisperse a (nchan, ntime) array in place-order.

    Channels are rolled so a pulse with the given DM aligns with its arrival
    time at the highest frequency. Samples wrapped from the end are harmless
    for plotting purposes as long as the cutout is wider than the sweep.
    """
    f_ref = freqs_mhz.max()
    delays_s = 4.148808e3 * dm * (freqs_mhz**-2 - f_ref**-2)
    shifts = np.round(delays_s / tsamp_s).astype(int)
    out = np.empty_like(data)
    for i, shift in enumerate(shifts):
        out[i] = np.roll(data[i], -shift)
    return out


def normalise(data: np.ndarray) -> np.ndarray:
    """Subtract per-channel median and scale by a robust per-channel std.

    The result is forced C-contiguous. Callers often pass a transposed view
    (an archived .fil is stored time-major), and numpy would keep that F-order
    through the arithmetic; every per-channel np.roll in dedisperse would then
    ravel a strided row and a full-resolution DM-time panel took 51 s instead
    of 3 s.
    """
    med = np.median(data, axis=1, keepdims=True)
    mad = np.median(np.abs(data - med), axis=1, keepdims=True)
    std = 1.4826 * mad
    std[std == 0] = 1.0
    return np.ascontiguousarray((data - med) / std)


def downsample(data: np.ndarray, ffactor: int = 8, tfactor: int = 1) -> np.ndarray:
    """Block-average a (nchan, ntime) array in frequency and time."""
    nchan, ntime = data.shape
    nchan -= nchan % ffactor
    ntime -= ntime % tfactor
    d = data[:nchan, :ntime].reshape(nchan // ffactor, ffactor, ntime // tfactor, tfactor)
    return d.mean(axis=(1, 3))


def downsample_max(data: np.ndarray, tfactor: int) -> np.ndarray:
    """Block-MAXIMUM a (nrow, ntime) array in time only.

    For the DM-time bowtie the array is already a boxcar S/N map, so the value
    a display pixel should report is the peak boxcar S/N inside its time bin,
    not the average. Averaging a width-long bin over a boxcar response costs
    25-50% of the peak depending on where the pulse falls in the bin, which is
    what made a narrow high-S/N candidate show as a faint dot. A trailing
    partial bin is dropped, as in downsample.
    """
    tfactor = max(1, int(tfactor))
    nrow, ntime = data.shape
    ntime -= ntime % tfactor
    return data[:, :ntime].reshape(nrow, ntime // tfactor, tfactor).max(axis=2)


def profile_snr(profile: np.ndarray, width: int) -> np.ndarray:
    """Boxcar-smoothed profile in units of its own robust noise."""
    width = max(1, width)
    kernel = np.ones(width) / width
    smooth = np.convolve(profile - np.median(profile), kernel, mode="same")
    mad = np.median(np.abs(smooth - np.median(smooth)))
    sigma = 1.4826 * mad if mad > 0 else smooth.std() or 1.0
    return smooth / sigma


def dm_grid(dm: float, ndm: int = 65) -> np.ndarray:
    """Trial DMs bracketing a candidate DM for the DM-time bowtie.

    The grid is uniform with step ``max(0.4 dm, 15) / (ndm // 2)`` and always
    contains ``dm`` itself: ``ndm`` is forced odd and the trials are laid out as
    offsets from ``dm``. An even count left the candidate DM half a step (0.6%
    of DM) from every trial, which for a narrow high-DM pulse costs most of the
    bowtie peak on its own. If the low end would go negative the grid slides up
    (fewer trials below ``dm``, the same number in total, same step), so it
    stays non-negative and ``dm`` stays exact.
    """
    half = max(0.4 * dm, 15.0)
    ndm = max(3, int(ndm) | 1)              # odd: dm sits on a trial, not between two
    nhalf = ndm // 2
    step = half / nhalf
    below = min(nhalf, int(math.floor(dm / step + 1e-9)))
    return dm + (np.arange(ndm) - below) * step


def dm_time(data: np.ndarray, freqs_mhz: np.ndarray, tsamp_s: float,
            dms: np.ndarray, width: int) -> np.ndarray:
    """Band-averaged S/N versus trial DM and time (the bowtie plot).

    ``data`` should already be normalised. Channel-averaging it first is a
    speed/fidelity trade, not a free one: the average happens BEFORE
    dedispersion, so a subband of BW_MHz smears the pulse by
    8.3e-3 ms x DM x BW_MHz / f_GHz^3 at every trial DM. Pick the factor with
    casm_t3.plotting.dmt_ffactor_for rather than reusing a display factor.
    """
    out = np.empty((len(dms), data.shape[1]), dtype=np.float32)
    for i, dm in enumerate(dms):
        out[i] = profile_snr(dedisperse(data, dm, freqs_mhz, tsamp_s).mean(axis=0), width)
    return out


# ------------------------------------------------------------- hella-like cleaning

#: What the deployed hella fork (casm_hella_vishnu, casm_hella_refactored) does
#: to every gulp before dedispersion, read from src/hella/src/flagger.cu and
#: timeseries.cu on 2026-10-12. The numbers are the live /tmp/hella_N.cfg plus
#: the fork defaults; see docs/architecture.md "Hella-like cleaning".
HELLA_GULP_SAMPLES = 8192            # GULP: every statistic is per gulp, no history
HELLA_MASK_CHANNELS = (0, 472)       # FLAG1/FLAG2: channels 484.375-470.0 MHz zeroed
HELLA_BASELINE_S = 0.2               # BASELINE_WIDTH: running-median window, seconds
HELLA_CLIP_SIGMA = 10.0              # U8_CLIP: symmetric rail before the uint8 pack
#: SCRUNCH stages as (tscrunch, fscrunch, threshold_sigma) in config order. The
#: deployed single stage has a 1e6 sigma threshold and never flags anything;
#: it exists only so the static mask kernel runs (casm-wiki hella-static-mask.md).
HELLA_SCRUNCHES = ((1, 1, 1.0e6),)


def running_median_clamped(x: np.ndarray, w: int) -> np.ndarray:
    """Running median with hella's edge rule: the window slides inside the
    series, so the first and last ``w // 2`` outputs repeat the first and last
    full-window medians (timeseries.cu running_median, lo clamped to [0, n-w])."""
    n = x.size
    w = min(int(w), n)
    if w < 3:
        return x.astype(np.float64, copy=True)
    win = np.lib.stride_tricks.sliding_window_view(x, w)
    med = np.median(win, axis=1)                     # n - w + 1 values, centre t = i + w//2
    out = np.empty(n, dtype=np.float64)
    half = w // 2
    out[half:half + med.size] = med
    out[:half] = med[0]
    out[half + med.size:] = med[-1]
    return out


def _standardise(d: np.ndarray) -> np.ndarray:
    """hella bandpass_correct: (x - mean) / std per channel over the gulp (plain
    moments, not median/MAD); a channel with zero or non-finite variance is 0."""
    mean = d.mean(axis=1, keepdims=True)
    std = d.std(axis=1, keepdims=True)
    ok = np.isfinite(std) & (std > 0) & np.isfinite(mean)
    return np.where(ok, (d - mean) / np.where(ok, std, 1.0), 0.0)


def _scrunch_flag(d: np.ndarray, tscrunch: int, fscrunch: int, thresh: float) -> np.ndarray:
    """One hella apply_scrunch(flag=1) stage on standardised data: box-average
    over (fscrunch channels x tscrunch samples), mask |value| above
    thresh / sqrt(tscrunch * fscrunch), grow the mask by the same box, and
    replace masked samples with the channel mean (0 here). Returns the mask."""
    from scipy.ndimage import uniform_filter

    size = (int(fscrunch), int(tscrunch))
    smooth = d if size == (1, 1) else uniform_filter(d, size=size, mode="nearest")
    mask = np.abs(smooth) > thresh / np.sqrt(size[0] * size[1])
    if size != (1, 1):
        mask = uniform_filter(mask.astype(np.float64), size=size, mode="nearest") > 0
    d[mask] = 0.0
    return mask


def _hella_clean_gulp(d: np.ndarray, tsamp_s: float, mask: tuple[int, int],
                      baseline_s: float, clip_sigma: float,
                      scrunches=HELLA_SCRUNCHES, stats: dict | None = None) -> np.ndarray:
    """One gulp through hella's flagging stage; ``d`` is (nchan, ntime) float64."""
    nchan, ntime = d.shape
    # 1. standardise every channel on its own gulp mean and std
    d = _standardise(d)
    # 2. the SCRUNCH stages, each on re-standardised data (apply_scrunch calls
    #    bandpass_correct first), then the static mask (replace_data: ch >= flag1
    #    and ch < flag2 -> 0) which lives in the same kernel
    flagged = np.zeros(d.shape, dtype=bool)
    for stage in scrunches:
        d = _standardise(d)
        flagged |= _scrunch_flag(d, *stage[:3])
    flagged[mask[0]:mask[1], :] = False          # the mask is not "flagging"
    if stats is not None:
        stats["flagged_frac"] = float(flagged[mask[1]:].mean()) if mask[1] < nchan else 0.0
    d[mask[0]:mask[1], :] = 0.0
    d = _standardise(d)                          # the final apply_scrunch(flag=0, ts=1)
    # 3. common-mode baseline: band mean over ALL channels, masked zeros included
    #    (measure_ts divides by NCHAN), running median over baseline_s, then the
    #    per-channel least-squares fit to it subtracted (subtract_channel_fit, mode 1).
    ts = d.mean(axis=0)
    w = int(baseline_s / tsamp_s)
    if w % 2 == 0:
        w += 1
    w = min(w, 1023)                                   # BASELINE_MAX_W
    if w > ntime:
        w = ntime if ntime % 2 else ntime - 1
    if w >= 3:
        s = running_median_clamped(ts, w)
        n = float(ntime)
        se, ss = s.sum(), (s * s).sum()
        xe = d.sum(axis=1)
        xs = d @ s
        den = se * se - ss * n
        if den != 0.0:
            alpha = (xe * se - xs * n) / den
            beta = (xs * se - xe * ss) / den
            alpha = np.where(np.isfinite(alpha), alpha, 1.0)
            beta = np.where(np.isfinite(beta), beta, 0.0)
        else:
            alpha, beta = np.ones(nchan), np.zeros(nchan)
        d = d - alpha[:, None] * s[None, :] - beta[:, None]
    else:                                              # window too short: divide_by_ts
        d = d / np.where(ts == 0, 1.0, ts)[None, :]
    # 4. one global scale for the gulp (normalize_data), then the symmetric rail
    gstd = d.std()
    if gstd > 0:
        d = (d - d.mean()) / gstd
    d[mask[0]:mask[1], :] = 0.0          # hella leaves them at -mean/std, ~1e-4; 0 reads better
    return np.clip(d, -clip_sigma, clip_sigma)


def hella_clean(data: np.ndarray, tsamp_s: float, sample0: int | None = None,
                gulp: int = HELLA_GULP_SAMPLES,
                mask: tuple[int, int] = HELLA_MASK_CHANNELS,
                baseline_s: float = HELLA_BASELINE_S,
                clip_sigma: float = HELLA_CLIP_SIGMA,
                scrunches=HELLA_SCRUNCHES) -> np.ndarray:
    """Approximate what hella searches: the deployed flagging stage applied offline.

    ``data`` is (nchan, ntime) in hella channel order (channel 0 = 484.375 MHz).
    ``sample0`` is the hella sample number of ``data[:, 0]`` (a trigger card's
    ``samp`` minus the event's offset into the cutout); it phases the pieces to
    hella's own gulp boundaries, which sit at multiples of ``gulp`` samples from
    the observation start. None: pieces start at the first sample and the log
    says so. A piece at either end of the cutout is shorter than a gulp and gets
    statistics from fewer samples than hella had; the returned array is float32
    in units of the gulp's global sigma, as in hella, and the mask channels are 0.

    ``scrunches`` are hella's SCRUNCH stages, (tscrunch, fscrunch, threshold
    sigma) each, run in order on re-standardised data before the static mask:
    box-average, mask above threshold / sqrt(t * f), grow by the box, replace
    with the channel mean. The default is the deployed single never-firing
    stage; e.g. ((1, 1, 5.0),) is a per-pixel 5 sigma cut, and the log reports
    the flagged fraction of the unmasked band per piece.

    Not reproduced: the uint8 quantisation (0.08 sigma steps) and the per-DM-row
    clipped renormalisation after dedispersion, neither visible by eye.
    """
    import logging

    data = np.asarray(data)
    nchan, ntime = data.shape
    phase = 0 if sample0 is None else int(sample0) % gulp
    edges = list(range((gulp - phase) % gulp, ntime, gulp))
    if not edges or edges[0] != 0:
        edges.insert(0, 0)
    edges.append(ntime)
    log = logging.getLogger(__name__)
    log.info("hella_clean: %d piece(s) of %d samples, boundaries at %s (sample0 %s), "
             "mask channels %d-%d, baseline %.2f s, clip %.0f sigma, scrunches %s",
             len(edges) - 1, gulp, edges[1:-1], "unknown" if sample0 is None else sample0,
             mask[0], mask[1] - 1, baseline_s, clip_sigma,
             [tuple(st[:3]) for st in scrunches])
    out = np.empty((nchan, ntime), dtype=np.float32)
    fracs = []
    for lo, hi in zip(edges, edges[1:]):
        stats = {}
        out[:, lo:hi] = _hella_clean_gulp(data[:, lo:hi].astype(np.float64), tsamp_s,
                                          mask, baseline_s, clip_sigma, scrunches, stats)
        fracs.append(stats.get("flagged_frac", 0.0))
    if any(f > 0 for f in fracs):
        log.info("hella_clean: flagged fraction of the unmasked band per piece: %s",
                 ["%.2e" % f for f in fracs])
    return out


def parse_scrunch(text: str) -> tuple[int, int, float]:
    """'t,f,thresh' (hella's SCRUNCH line order, nits ignored) -> (t, f, thresh)."""
    parts = [x for x in text.replace(" ", ",").split(",") if x]
    if len(parts) < 3:
        raise ValueError(f"scrunch needs 'tscrunch,fscrunch,threshold', got {text!r}")
    t, f, th = int(parts[0]), int(parts[1]), float(parts[2])
    if t < 1 or f < 1 or th <= 0:
        raise ValueError(f"scrunch factors must be >= 1 and the threshold > 0: {text!r}")
    return t, f, th
