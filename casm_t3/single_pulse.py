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

    Non-finite cells (NaN, Inf) are left out of the statistics and come out as 0;
    a channel with no finite cell is all 0. An all-finite array takes the plain
    path below, unchanged.
    """
    finite = np.isfinite(data)
    if not finite.all():
        return _normalise_finite_only(data, finite)
    med = np.median(data, axis=1, keepdims=True)
    mad = np.median(np.abs(data - med), axis=1, keepdims=True)
    std = 1.4826 * mad
    std[std == 0] = 1.0
    return np.ascontiguousarray((data - med) / std)


def _normalise_finite_only(data: np.ndarray, finite: np.ndarray) -> np.ndarray:
    """normalise() for an array with non-finite cells: median and 1.4826 x MAD
    per channel over its finite cells, 0 where the cell is not finite."""
    med = _row_median(data, finite)[:, None].astype(data.dtype)
    mad = _row_median(np.abs(data - med), finite)
    std = 1.4826 * mad
    std[std == 0] = 1.0
    out = (data - med) / std[:, None].astype(data.dtype)
    return np.ascontiguousarray(np.where(finite, out, 0.0).astype(data.dtype, copy=False))


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


# ------------------------------------------------------------ validity-weighted numerics
# Used only when a flag mask is supplied (plotting.draw_candidate_figure(mask=...)).
# The functions above are the raw path; on all-finite input they match the pre-mask
# code bit for bit (normalise only takes a different branch when a cell is NaN or
# Inf). A masked array is a pair
# (data, valid): ``valid`` is True where the cell is usable (not flagged, flag record
# exists, finite). Invalid cells carry 0 in ``data`` so plain sums are safe, and
# validity travels with the data through every step. Median/mean/convolve are not
# used on NaN-carrying arrays: a NaN spreads over the whole series in np.median and
# np.convolve, and a mean over a varying number of cells has varying noise.

#: A channel is usable when it keeps at least this fraction of the valid samples of
#: the best channel (and MIN_CHAN_VALID_SAMPLES of them): median/MAD from fewer
#: samples scatter by >10% (MAD efficiency ~37%, relative error ~1.2/sqrt(n)).
MIN_CHAN_VALID_FRAC = 0.10
MIN_CHAN_VALID_SAMPLES = 64
#: A time sample is available when at least this fraction of the live channels
#: (and MIN_NVALID_CHANNELS of them) are valid there. Below it the band profile is
#: NaN: the S/N of a pulse scales as sqrt(N_valid), so 10% of the band is already
#: a 3x loss and the profile would hang on a few channels of residual RFI.
MIN_NVALID_FRAC = 0.10
MIN_NVALID_CHANNELS = 8
#: A boxcar window is evaluated when at least this fraction of its samples is
#: available; a half-empty window has no meaningful S/N.
MIN_WINDOW_SUPPORT = 0.5
#: Fewer finite smoothed samples than this and the robust noise is not measurable.
MIN_NOISE_SAMPLES = 16
#: A display block (frequency x time average) is shown when at least this fraction
#: of its cells is valid, so a block of one surviving cell does not show as speckle.
MIN_BLOCK_VALID_FRAC = 0.5


def valid_cells(mask: np.ndarray, coverage: np.ndarray | None, data: np.ndarray) -> np.ndarray:
    """Usable-cell array from a flag mask (True = flagged), the per-sample
    ``coverage`` (False = flags unknown) and the data itself (non-finite cells
    are unusable). Shapes: mask and data (nchan, ntime), coverage (ntime,)."""
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != data.shape:
        raise ValueError(f"mask shape {mask.shape} does not match data {data.shape}")
    valid = ~mask & np.isfinite(data)
    if coverage is not None:
        cov = np.asarray(coverage, dtype=bool)
        if cov.shape != (data.shape[1],):
            raise ValueError(f"coverage shape {cov.shape} does not match ntime {data.shape[1]}")
        valid &= cov[None, :]
    return valid


def _row_median(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Median of the valid entries of every row, float64; 0 for a row with none."""
    n = valid.sum(axis=1)
    work = np.where(valid, x, np.inf).astype(x.dtype, copy=False)
    work.sort(axis=1)
    lo = np.take_along_axis(work, np.clip((n - 1) // 2, 0, None)[:, None], axis=1)[:, 0]
    hi = np.take_along_axis(work, np.clip(n // 2, 0, None)[:, None], axis=1)[:, 0]
    med = 0.5 * (lo.astype(np.float64) + hi.astype(np.float64))
    med[n == 0] = 0.0
    return med


def normalise_masked(data: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """normalise() over the valid cells only: (norm, valid_out).

    Median and 1.4826 x MAD per channel come from the valid samples of that
    channel; flagged and uncovered cells are excluded, not filled. A channel with
    fewer valid samples than MIN_CHAN_VALID_FRAC of the best channel's (and
    MIN_CHAN_VALID_SAMPLES) is invalid in every sample. ``norm`` is 0 where
    ``valid_out`` is False. C-contiguous, as normalise().
    """
    data = np.asarray(data)
    if not np.issubdtype(data.dtype, np.floating):
        data = data.astype(np.float32)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(data)
    n = valid.sum(axis=1)
    need = max(MIN_CHAN_VALID_SAMPLES, int(math.ceil(MIN_CHAN_VALID_FRAC * (n.max() if n.size else 0))))
    live = n >= need
    valid = valid & live[:, None]
    med = _row_median(data, valid)
    mad = _row_median(np.abs(data - med[:, None].astype(data.dtype)), valid)
    std = 1.4826 * mad
    std[std == 0] = 1.0
    out = (data - med[:, None].astype(data.dtype)) / std[:, None].astype(data.dtype)
    out = np.where(valid, out, 0.0).astype(data.dtype, copy=False)
    return np.ascontiguousarray(out), valid


def _shifts(dm: float, freqs_mhz: np.ndarray, tsamp_s: float) -> np.ndarray:
    """Per-channel dedispersion shift in samples (as dedisperse())."""
    f_ref = freqs_mhz.max()
    delays_s = 4.148808e3 * dm * (freqs_mhz ** -2 - f_ref ** -2)
    return np.round(delays_s / tsamp_s).astype(int)


def _shift_rows(x: np.ndarray, shifts: np.ndarray, fill) -> np.ndarray:
    """out[i, t] = x[i, t + shifts[i]] without wrap-around: samples that np.roll
    would bring in from the other end of the row are ``fill`` instead."""
    nchan, n = x.shape
    out = np.full((nchan, n), fill, dtype=x.dtype)
    for i, s in enumerate(shifts):
        s = int(s)
        if s >= 0:
            if s < n:
                out[i, :n - s] = x[i, s:]
        elif -s < n:
            out[i, -s:] = x[i, :n + s]
    return out


def dedisperse_masked(data: np.ndarray, valid: np.ndarray, dm: float, freqs_mhz: np.ndarray,
                      tsamp_s: float) -> tuple[np.ndarray, np.ndarray]:
    """dedisperse() with validity moving with the data: (out, valid_out).

    Each channel is shifted by its DM delay as in dedisperse(), but the samples
    np.roll would wrap round from the start of the row are invalid (0 in
    ``out``, False in ``valid_out``), not recycled.
    """
    shifts = _shifts(dm, freqs_mhz, tsamp_s)
    return _shift_rows(data, shifts, 0), _shift_rows(valid, shifts, False)


def uncovered_cells(coverage: np.ndarray, nchan: int, dm: float, freqs_mhz: np.ndarray,
                    tsamp_s: float) -> np.ndarray:
    """(nchan, ntime) bool: the cells of the dedispersed array whose flags are
    unknown (coverage False at their source sample). Wrapped cells are not
    uncovered, they are invalid for another reason."""
    unc = np.broadcast_to(~np.asarray(coverage, dtype=bool), (nchan, len(coverage)))
    return _shift_rows(unc, _shifts(dm, freqs_mhz, tsamp_s), False)


def uncovered_fraction(coverage: np.ndarray, freqs_mhz: np.ndarray, tsamp_s: float,
                       dms: np.ndarray) -> np.ndarray:
    """(ndm, ntime) float32: fraction of the channels whose source sample at each
    (trial DM, dedispersed time) lies in a coverage gap. Wrapped samples count as
    covered. This is what lets the DM-time panel tell "no flag record" from
    "flagged" without carrying a third array through the dedispersion."""
    unc = ~np.asarray(coverage, dtype=bool)
    n = unc.size
    out = np.zeros((len(dms), n), dtype=np.float32)
    if not unc.any():
        return out
    for j, dm in enumerate(dms):
        acc = np.zeros(n)
        values, counts = np.unique(_shifts(float(dm), freqs_mhz, tsamp_s), return_counts=True)
        for s, c in zip(values, counts):
            s = int(s)
            if s >= 0 and s < n:
                acc[:n - s] += c * unc[s:]
            elif s < 0 and -s < n:
                acc[-s:] += c * unc[:n + s]
        out[j] = acc / freqs_mhz.size
    return out


def min_nvalid_for(n_live_channels: int) -> int:
    """Smallest N_valid(t) at which a band-profile sample is available."""
    return max(MIN_NVALID_CHANNELS, int(math.ceil(MIN_NVALID_FRAC * n_live_channels)))


def band_profile_masked(data: np.ndarray, valid: np.ndarray,
                        min_nvalid: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Band-sum profile with unit noise whatever N_valid(t): (profile, nvalid).

    profile[t] = sum over valid channels of data / sqrt(N_valid(t)); for
    per-channel unit-noise data the noise is 1 at every t, where a mean over the
    valid channels has noise 1/sqrt(N_valid) that changes with the mask. Samples
    with N_valid below ``min_nvalid`` are NaN (unavailable).
    """
    nvalid = valid.sum(axis=0)
    total = data.sum(axis=0, dtype=np.float64, where=valid)
    prof = np.full(data.shape[1], np.nan)
    ok = nvalid >= max(1, int(min_nvalid))
    prof[ok] = total[ok] / np.sqrt(nvalid[ok])
    return prof, nvalid


def profile_snr_masked(profile: np.ndarray, width: int) -> np.ndarray:
    """profile_snr() for a unit-noise profile with NaN gaps.

    The boxcar sums the available samples of each window (same window alignment
    as profile_snr) and divides by sqrt(n_available), so a gap lowers the S/N of
    the windows that touch it and never spreads NaN beyond them. A window with
    fewer than MIN_WINDOW_SUPPORT of its samples available is NaN, as is the whole
    series when fewer than MIN_NOISE_SAMPLES smoothed samples exist. The result is
    then divided by its own robust noise, as profile_snr does.
    """
    width = max(1, int(width))
    prof = np.asarray(profile, dtype=np.float64)
    avail = np.isfinite(prof)
    out = np.full(prof.shape, np.nan)
    if avail.sum() < MIN_NOISE_SAMPLES:
        return out
    x = np.where(avail, prof - np.median(prof[avail]), 0.0)
    kernel = np.ones(width)
    total = np.convolve(x, kernel, mode="same")
    count = np.convolve(avail.astype(np.float64), kernel, mode="same")
    ok = count > max(1, math.ceil(MIN_WINDOW_SUPPORT * width)) - 0.5      # counts are integers
    if ok.sum() < MIN_NOISE_SAMPLES:
        return out
    z = np.full(prof.shape, np.nan)
    z[ok] = total[ok] / np.sqrt(count[ok])
    zz = z[ok]
    mad = np.median(np.abs(zz - np.median(zz)))
    sigma = 1.4826 * mad if mad > 0 else zz.std() or 1.0
    out[ok] = z[ok] / sigma
    return out


def block_sum_masked(data: np.ndarray, valid: np.ndarray, ffactor: int,
                     tfactor: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Block (sum over valid cells, number of valid cells) of a (nchan, ntime)
    pair in frequency and time; a trailing partial block is dropped."""
    nchan, ntime = data.shape
    nchan -= nchan % ffactor
    ntime -= ntime % tfactor
    shape = (nchan // ffactor, ffactor, ntime // tfactor, tfactor)
    v = valid[:nchan, :ntime].reshape(shape)
    d = np.where(v, data[:nchan, :ntime].reshape(shape), 0.0)
    return d.sum(axis=(1, 3), dtype=np.float32), v.sum(axis=(1, 3)).astype(np.float32)


def downsample_masked(data: np.ndarray, valid: np.ndarray, ffactor: int = 8,
                      tfactor: int = 1) -> np.ndarray:
    """downsample() averaging only the valid cells of each block; NaN where fewer
    than MIN_BLOCK_VALID_FRAC of the block is valid."""
    total, count = block_sum_masked(data, valid, ffactor, tfactor)
    out = np.full(total.shape, np.nan, dtype=np.float32)
    ok = count >= max(1.0, MIN_BLOCK_VALID_FRAC * ffactor * tfactor)
    out[ok] = total[ok] / count[ok]
    return out


def block_fraction(flag: np.ndarray, ffactor: int = 1, tfactor: int = 1) -> np.ndarray:
    """Fraction of True cells in every (ffactor x tfactor) block of a boolean
    (nrow, ntime) array; a trailing partial block is dropped."""
    nrow, ntime = flag.shape
    nrow -= nrow % ffactor
    ntime -= ntime % tfactor
    blocks = flag[:nrow, :ntime].reshape(nrow // ffactor, ffactor, ntime // tfactor, tfactor)
    return blocks.mean(axis=(1, 3))


def downsample_max_masked(data: np.ndarray, tfactor: int) -> np.ndarray:
    """downsample_max() skipping NaN; a block with no finite value stays NaN."""
    tfactor = max(1, int(tfactor))
    nrow, ntime = data.shape
    ntime -= ntime % tfactor
    blocks = data[:, :ntime].reshape(nrow, ntime // tfactor, tfactor)
    finite = np.isfinite(blocks)
    out = np.where(finite, blocks, -np.inf).max(axis=2)
    out[~finite.any(axis=2)] = np.nan
    return out.astype(data.dtype, copy=False)


def dm_time_masked(total: np.ndarray, count: np.ndarray, freqs_mhz: np.ndarray,
                   tsamp_s: float, dms: np.ndarray, width: int,
                   min_nvalid: int = 1) -> np.ndarray:
    """dm_time() on block sums: boxcar S/N vs (trial DM, time) with NaN where
    unavailable.

    ``total`` and ``count`` are the (nrow, ntime) block sums and valid counts of
    normalised data (block_sum_masked(norm, valid, f, 1)); ``freqs_mhz`` are the
    row centres. For every trial DM the rows are shifted without wrap-around
    (a wrapped sample is dropped from both sums), summed, divided by
    sqrt(N_valid) and passed through profile_snr_masked, i.e. the same
    weighting as band_profile_masked, on subband cells instead of channels.
    Rows with the same shift are added through a running sum over rows, so the
    cost per trial is the number of distinct shifts, not the number of rows.
    """
    nrow, n = total.shape
    order = np.argsort(-freqs_mhz, kind="stable")        # shifts are monotone in frequency
    freqs = freqs_mhz[order]
    csum = np.zeros((2, nrow + 1, n), dtype=np.float32)  # running sums over rows, [0] data, [1] counts
    np.cumsum(total[order], axis=0, dtype=np.float32, out=csum[0, 1:])
    np.cumsum(count[order], axis=0, dtype=np.float32, out=csum[1, 1:])
    out = np.empty((len(dms), n), dtype=np.float32)
    for j, dm in enumerate(dms):
        shifts = _shifts(float(dm), freqs, tsamp_s)
        starts = np.concatenate(([0], np.flatnonzero(np.diff(shifts)) + 1))
        stops = np.concatenate((starts[1:], [nrow]))
        acc_s, acc_n = np.zeros(n), np.zeros(n)
        for a, b in zip(starts, stops):
            s = int(shifts[a])
            if s >= 0 and s < n:
                acc_s[:n - s] += csum[0, b, s:] - csum[0, a, s:]
                acc_n[:n - s] += csum[1, b, s:] - csum[1, a, s:]
            elif s < 0 and -s < n:
                acc_s[-s:] += csum[0, b, :n + s] - csum[0, a, :n + s]
                acc_n[-s:] += csum[1, b, :n + s] - csum[1, a, :n + s]
        prof = np.full(n, np.nan)
        ok = acc_n > max(1, int(min_nvalid)) - 0.5
        prof[ok] = acc_s[ok] / np.sqrt(acc_n[ok])
        out[j] = profile_snr_masked(prof, width)
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
#: SumThreshold with a channel-neighbour reference (not in hella; offline trial).
SUMTHRESHOLD_DEFAULTS = {"chi1": 8.0, "rho": 1.5, "mmin": 16, "mmax": 512, "nb": 4, "grow_t": 0}


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


def _box_mean_clamped(x: np.ndarray, w: int) -> np.ndarray:
    """Running mean over ``w`` samples along the last axis, window clamped
    inside the series: output t uses the window starting at
    clip(t - w//2, 0, n - w), so an edge sample sees the first or last full
    window and the noise of the mean is the same everywhere."""
    n = x.shape[-1]
    w = min(int(w), n)
    cs = np.cumsum(x, axis=-1, dtype=np.float64)
    full = np.empty(x.shape[:-1] + (n - w + 1,), dtype=np.float64)
    full[..., 0] = cs[..., w - 1]
    full[..., 1:] = cs[..., w:] - cs[..., :n - w]
    full /= w
    idx = np.clip(np.arange(n) - w // 2, 0, n - w)
    return full[..., idx]


def _standardise(d: np.ndarray, flagged: np.ndarray | None = None) -> np.ndarray:
    """hella bandpass_correct: (x - mean) / std per channel over the gulp (plain
    moments, not median/MAD); a channel with zero or non-finite variance is 0.

    With ``flagged`` the moments come from the unflagged samples only and the
    flagged ones are set to 0, the channel mean. hella itself re-measures over
    every sample after a replacement, so a channel whose burst was replaced
    keeps a bias of (burst fraction x burst height) in its clean samples and a
    partly flagged burst is amplified by the shrunken std; the masked moments
    avoid both. The deployed (never-firing) configuration is unaffected.
    """
    if flagged is None or not flagged.any():
        mean = d.mean(axis=1, keepdims=True)
        std = d.std(axis=1, keepdims=True)
    else:
        n = (~flagged).sum(axis=1, keepdims=True).astype(float)
        good = np.where(flagged, 0.0, d)
        mean = good.sum(axis=1, keepdims=True) / np.maximum(n, 1)
        var = (np.where(flagged, 0.0, (d - mean) ** 2)).sum(axis=1, keepdims=True) / np.maximum(n, 1)
        std = np.sqrt(var)
        std[n[:, 0] < 2] = 0.0
    ok = np.isfinite(std) & (std > 0) & np.isfinite(mean)
    out = np.where(ok, (d - mean) / np.where(ok, std, 1.0), 0.0)
    if flagged is not None:
        out[flagged] = 0.0
    return out


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


def sumthreshold_flag(d: np.ndarray, chi1: float = 8.0, rho: float = 1.5, mmin: int = 16,
                      mmax: int = 512, nb: int = 4, grow_t: int = 0,
                      mask: tuple[int, int] | None = None) -> np.ndarray:
    """SumThreshold (Offringa et al. 2010) along time, against a channel-neighbour reference.

    ``d`` is a standardised (nchan, ntime) gulp, modified in place: flagged
    samples are set to the channel mean, 0. For each window length M in
    mmin, 2 mmin, ... mmax the data are box-averaged over M samples; the reference
    is the median over the ``nb`` channels on either side (self excluded by
    the odd-window median's symmetry) of that averaged data, recomputed at
    every M; a window whose |self - reference| exceeds chi1 / rho**log2(M) is
    flagged over its M samples. Flagged samples are replaced before the next
    M, Offringa's substitution, so a flagged burst does not re-trigger at
    larger M. Broadband emission is in the reference as much as in the
    channel and leaves no residual, whichever its brightness; that is the
    point of the reference. A dispersed pulse stays inside its neighbours'
    windows as long as the sweep across nb channels is shorter than M
    samples, which is why the reference is rebuilt from the M-averaged data
    and why M starts at ``mmin`` (16 samples covers +-4 channels up to DM
    ~1000 over most of the band; the bursts this targets are >= 100 samples).
    The thresholds keep Offringa's ladder from M = 1, so chi1 is still the
    per-sample value: at mmin = 16 the cut on the 16-sample mean is
    chi1 / rho**4.
    ``grow_t`` dilates the final mask by that many samples in time. Channels
    in ``mask`` (the static mask range) are excluded from the reference and
    never flagged. Returns the boolean mask.
    """
    from scipy.ndimage import maximum_filter1d, median_filter

    nchan, ntime = d.shape
    flagged = np.zeros(d.shape, dtype=bool)
    live = np.ones(nchan, dtype=bool)
    if mask is not None:
        live[mask[0]:mask[1]] = False
    work = d[live]
    m = max(1, int(mmin))
    while m <= mmax:
        thresh = chi1 / rho ** np.log2(m)
        # the window is clamped inside the gulp (hella's running-median rule):
        # "nearest" pads an edge window with copies of one sample, "reflect"
        # counts ~half the window twice; both double the edge noise and trip
        # the threshold on the first and last bins of every gulp
        avg = work if m == 1 else _box_mean_clamped(work, m)
        ref = median_filter(avg, size=(2 * nb + 1, 1), mode="nearest")
        hit = np.abs(avg - ref) > thresh
        if m > 1:
            # a hit at window centre t flags the whole window (running max is
            # O(N); a binary_dilation with a 512-wide structure is O(N M))
            hit = maximum_filter1d(hit.astype(np.uint8), m, axis=1, mode="constant") > 0
        new = hit & ~flagged[live]
        if new.any():
            work[new] = 0.0
            sub = flagged[live]
            sub |= hit
            flagged[live] = sub
        m *= 2
    if grow_t > 0:
        sub = maximum_filter1d(flagged[live].astype(np.uint8), 2 * grow_t + 1, axis=1,
                               mode="constant") > 0
        work[sub & ~flagged[live]] = 0.0
        flagged[live] = sub
    d[live] = work
    return flagged


def parse_sumthreshold(text: str) -> dict:
    """'chi1=6,rho=1.5,mmax=256,nb=4,grow_t=0' (any subset; 'default' = all defaults)."""
    cfg = dict(SUMTHRESHOLD_DEFAULTS)
    if text and text.strip().lower() not in ("default", "on", "yes", "1"):
        for part in text.replace(";", ",").split(","):
            if not part.strip():
                continue
            k, _, v = part.partition("=")
            k = k.strip()
            if k not in cfg:
                raise ValueError(f"unknown sumthreshold key {k!r}; keys: {sorted(cfg)}")
            cfg[k] = type(cfg[k])(float(v)) if k != "rho" else float(v)
    if cfg["mmax"] < cfg["mmin"] or cfg["mmin"] < 1 or cfg["nb"] < 1 or cfg["chi1"] <= 0 or cfg["rho"] <= 0:
        raise ValueError(f"bad sumthreshold parameters: {cfg}")
    return cfg


def _hella_clean_gulp(d: np.ndarray, tsamp_s: float, mask: tuple[int, int],
                      baseline_s: float, clip_sigma: float,
                      scrunches=HELLA_SCRUNCHES, stats: dict | None = None,
                      sumthreshold: dict | None = None) -> np.ndarray:
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
    if sumthreshold:
        d = _standardise(d)
        flagged |= sumthreshold_flag(d, mask=mask, **sumthreshold)
    flagged[mask[0]:mask[1], :] = False          # the mask is not "flagging"
    if stats is not None:
        stats["flagged_frac"] = float(flagged[mask[1]:].mean()) if mask[1] < nchan else 0.0
    d[mask[0]:mask[1], :] = 0.0
    d = _standardise(d, flagged)                 # the final apply_scrunch(flag=0, ts=1)
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
    # Flagged samples carry no data: back to exactly 0 after the baseline and
    # global steps moved them, and a channel more than half flagged is blanked.
    # Otherwise a downstream per-channel median/MAD (single_pulse.normalise)
    # sees a near-constant run, measures almost no scatter and inflates the
    # channel a hundredfold (seen on inj_20261012_0012, 2026-10-12).
    if flagged.any():
        d[flagged] = 0.0
        heavy = flagged.mean(axis=1) > 0.5
        d[heavy, :] = 0.0
        if stats is not None:
            stats["blanked_chans"] = int(heavy.sum())
    return np.clip(d, -clip_sigma, clip_sigma)


def hella_clean(data: np.ndarray, tsamp_s: float, sample0: int | None = None,
                gulp: int = HELLA_GULP_SAMPLES,
                mask: tuple[int, int] = HELLA_MASK_CHANNELS,
                baseline_s: float = HELLA_BASELINE_S,
                clip_sigma: float = HELLA_CLIP_SIGMA,
                scrunches=HELLA_SCRUNCHES,
                sumthreshold: dict | None = None) -> np.ndarray:
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

    ``sumthreshold`` (a dict of sumthreshold_flag keyword arguments, see
    SUMTHRESHOLD_DEFAULTS) adds, after the scrunch stages, a SumThreshold
    stage against a channel-neighbour reference. hella has no such stage; it
    is here for offline comparison.

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
    if sumthreshold:
        log.info("hella_clean: sumthreshold %s", sumthreshold)
    out = np.empty((nchan, ntime), dtype=np.float32)
    fracs, blanked = [], []
    for lo, hi in zip(edges, edges[1:]):
        stats = {}
        out[:, lo:hi] = _hella_clean_gulp(data[:, lo:hi].astype(np.float64), tsamp_s,
                                          mask, baseline_s, clip_sigma, scrunches, stats,
                                          sumthreshold)
        fracs.append(stats.get("flagged_frac", 0.0))
        blanked.append(stats.get("blanked_chans", 0))
    if any(f > 0 for f in fracs):
        log.info("hella_clean: flagged fraction of the unmasked band per piece: %s; "
                 "channels blanked (>50%% flagged): %s", ["%.2e" % f for f in fracs], blanked)
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
