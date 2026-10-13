"""Frozen copies of the single_pulse functions the raw figure path calls, as they were
before the mask-aware path was added. Tests compare the live functions and the raw
figure against these, so a change to the raw numerics fails loudly."""
import numpy as np


def old_dedisperse(data, dm, freqs_mhz, tsamp_s):
    f_ref = freqs_mhz.max()
    delays_s = 4.148808e3 * dm * (freqs_mhz**-2 - f_ref**-2)
    shifts = np.round(delays_s / tsamp_s).astype(int)
    out = np.empty_like(data)
    for i, shift in enumerate(shifts):
        out[i] = np.roll(data[i], -shift)
    return out


def old_normalise(data):
    med = np.median(data, axis=1, keepdims=True)
    mad = np.median(np.abs(data - med), axis=1, keepdims=True)
    std = 1.4826 * mad
    std[std == 0] = 1.0
    return np.ascontiguousarray((data - med) / std)


def old_profile_snr(profile, width):
    width = max(1, width)
    kernel = np.ones(width) / width
    smooth = np.convolve(profile - np.median(profile), kernel, mode="same")
    mad = np.median(np.abs(smooth - np.median(smooth)))
    sigma = 1.4826 * mad if mad > 0 else smooth.std() or 1.0
    return smooth / sigma


def old_downsample(data, ffactor=8, tfactor=1):
    nchan, ntime = data.shape
    nchan -= nchan % ffactor
    ntime -= ntime % tfactor
    d = data[:nchan, :ntime].reshape(nchan // ffactor, ffactor, ntime // tfactor, tfactor)
    return d.mean(axis=(1, 3))


def old_downsample_max(data, tfactor):
    tfactor = max(1, int(tfactor))
    nrow, ntime = data.shape
    ntime -= ntime % tfactor
    return data[:, :ntime].reshape(nrow, ntime // tfactor, tfactor).max(axis=2)


def old_dm_time(data, freqs_mhz, tsamp_s, dms, width):
    out = np.empty((len(dms), data.shape[1]), dtype=np.float32)
    for i, dm in enumerate(dms):
        out[i] = old_profile_snr(old_dedisperse(data, dm, freqs_mhz, tsamp_s).mean(axis=0), width)
    return out
