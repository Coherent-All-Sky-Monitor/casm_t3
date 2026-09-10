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
