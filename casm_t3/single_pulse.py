"""Single-pulse processing for candidate plots: dedispersion and profiles."""

from __future__ import annotations

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
    """Subtract per-channel median and scale by a robust per-channel std."""
    med = np.median(data, axis=1, keepdims=True)
    mad = np.median(np.abs(data - med), axis=1, keepdims=True)
    std = 1.4826 * mad
    std[std == 0] = 1.0
    return (data - med) / std


def downsample(data: np.ndarray, ffactor: int = 8, tfactor: int = 1) -> np.ndarray:
    """Block-average a (nchan, ntime) array in frequency and time."""
    nchan, ntime = data.shape
    nchan -= nchan % ffactor
    ntime -= ntime % tfactor
    d = data[:nchan, :ntime].reshape(nchan // ffactor, ffactor, ntime // tfactor, tfactor)
    return d.mean(axis=(1, 3))


def profile_snr(profile: np.ndarray, width: int) -> np.ndarray:
    """Boxcar-smoothed profile in units of its own robust noise."""
    width = max(1, width)
    kernel = np.ones(width) / width
    smooth = np.convolve(profile - np.median(profile), kernel, mode="same")
    mad = np.median(np.abs(smooth - np.median(smooth)))
    sigma = 1.4826 * mad if mad > 0 else smooth.std() or 1.0
    return smooth / sigma
