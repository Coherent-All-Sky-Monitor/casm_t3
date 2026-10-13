"""Validity-weighted numerics in single_pulse (the mask-aware path).

A masked array is (data, valid), valid True = usable. Expectations come from
the definitions (median over valid cells, unit noise per band-profile sample,
wrapped samples invalid), from brute-force recomputation, or from the raw
functions, which must not have changed.
"""
import math

import numpy as np
import pytest

from casm_t3 import single_pulse as sp
from legacy_numerics import old_dedisperse as _old_dedisperse
from legacy_numerics import old_normalise as _old_normalise
from legacy_numerics import old_profile_snr as _old_profile_snr

TSAMP = 1.048576e-3


def _freqs(nchan):
    return 484.375 - np.arange(nchan) * (0.030517578125 * 3072 / nchan)


# --- the raw path is untouched ----------------------------------------------------

def test_raw_numerics_are_bit_identical_to_the_pre_mask_code():
    """Frozen copies of the old normalise / dedisperse / profile_snr / dm_time /
    downsample: the functions the raw path calls must give identical arrays."""
    rng = np.random.default_rng(4)
    nchan, ntime = 96, 3000
    freqs = _freqs(nchan)
    data = (rng.normal(100, 10, (nchan, ntime)) + 5 * np.arange(nchan)[:, None]).astype(np.float32)
    data[:, 1000:1008] += 30.0
    norm = sp.normalise(data)
    assert np.array_equal(norm, _old_normalise(data))
    # a transposed (time-major .fil) view gives the same array, C-contiguous
    assert np.array_equal(sp.normalise(np.ascontiguousarray(data.T).T), norm)
    assert sp.normalise(np.ascontiguousarray(data.T).T).flags.c_contiguous
    dd = sp.dedisperse(norm, 60.0, freqs, TSAMP)
    assert np.array_equal(dd, _old_dedisperse(norm, 60.0, freqs, TSAMP))
    for width in (1, 4, 8):
        assert np.array_equal(sp.profile_snr(dd.mean(axis=0), width),
                              _old_profile_snr(dd.mean(axis=0), width))
    dms = sp.dm_grid(60.0, 9)
    small = sp.downsample(norm, 4, 1)
    f_small = freqs.reshape(-1, 4).mean(axis=1)
    ref = np.empty((dms.size, ntime), dtype=np.float32)
    for i, dm in enumerate(dms):
        ref[i] = _old_profile_snr(_old_dedisperse(small, dm, f_small, TSAMP).mean(axis=0), 4)
    assert np.array_equal(sp.dm_time(small, f_small, TSAMP, dms, 4), ref)
    assert np.array_equal(sp.downsample(dd, 4, 2), dd.reshape(24, 4, 1500, 2).mean(axis=(1, 3)))


# --- normalisation ------------------------------------------------------------------

def test_normalise_masked_takes_median_and_mad_from_valid_cells_only():
    rng = np.random.default_rng(1)
    nchan, ntime = 40, 2000
    data = rng.normal(50.0, 4.0, (nchan, ntime)).astype(np.float32)
    valid = np.ones(data.shape, dtype=bool)
    valid[:, 500:900] = False
    data[:, 500:900] += 5000.0                            # flagged RFI, must not leak in
    valid[7, ::2] = False                                 # a channel flagged half the time
    data[7, ::2] = -1e6
    norm, v = sp.normalise_masked(data, valid)
    assert v.sum() == valid.sum() and np.array_equal(v, valid)
    for c in (0, 7, 33):
        x = data[c, valid[c]].astype(np.float64)
        med = np.median(x)
        std = 1.4826 * np.median(np.abs(x - med))
        assert np.allclose(norm[c, valid[c]], (x - med) / std, atol=2e-5)
    assert np.all(norm[~v] == 0.0)                        # excluded, not filled with the median or NaN
    assert norm.flags.c_contiguous and norm.dtype == np.float32
    # nothing flagged: the same numbers as normalise()
    n_all, v_all = sp.normalise_masked(data[:, :400], np.ones((nchan, 400), bool))
    assert v_all.all() and np.allclose(n_all, sp.normalise(data[:, :400]), atol=2e-5)


def test_normalise_masked_drops_channels_with_too_few_valid_samples():
    rng = np.random.default_rng(2)
    data = rng.normal(0.0, 1.0, (20, 1000)).astype(np.float32)
    valid = np.ones(data.shape, dtype=bool)
    valid[3, 60:] = False                       # 60 valid samples: below MIN_CHAN_VALID_SAMPLES
    valid[4, 80:] = False                       # 80 valid but under 10% of the best channel's 1000
    valid[5, 150:] = False                      # 150: kept
    norm, v = sp.normalise_masked(data, valid)
    assert not v[3].any() and not v[4].any() and v[5, :150].all() and not v[5, 150:].any()
    assert np.all(norm[3] == 0.0) and np.all(norm[4] == 0.0)
    # the threshold follows the best channel, not the window: a heavily uncovered
    # window leaves every channel with its few valid samples, all kept
    short = np.zeros(data.shape, dtype=bool)
    short[:, :200] = True
    _, v2 = sp.normalise_masked(data, short)
    assert v2.sum() == short.sum()
    # nothing valid at all: everything invalid, no exception
    n0, v0 = sp.normalise_masked(data, np.zeros(data.shape, dtype=bool))
    assert not v0.any() and np.all(n0 == 0.0)


# --- dedispersion ---------------------------------------------------------------------

def test_dedisperse_masked_moves_validity_and_invalidates_wrapped_samples():
    nchan, ntime, dm = 64, 1200, 100.0
    freqs = _freqs(nchan)
    rng = np.random.default_rng(3)
    data = rng.normal(0.0, 1.0, (nchan, ntime)).astype(np.float32)
    valid = np.ones(data.shape, dtype=bool)
    valid[10, 400] = False                      # one flagged cell
    out, vout = sp.dedisperse_masked(data, valid, dm, freqs, TSAMP)
    shifts = np.round(4.148808e3 * dm * (freqs**-2 - freqs.max()**-2) / TSAMP).astype(int)
    assert 100 < shifts.max() < ntime            # a real sweep that fits in the series
    rolled = _old_dedisperse(data, dm, freqs, TSAMP)
    for c in (0, 10, 31, nchan - 1):
        s = shifts[c]
        # the np.roll result survives where no wrap happened; the wrapped tail is
        # invalid and 0, never the recycled start of the row
        assert np.array_equal(out[c, :ntime - s], rolled[c, :ntime - s])
        assert not vout[c, ntime - s:].any() and np.all(out[c, ntime - s:] == 0.0)
        assert vout[c, :ntime - s].sum() == ntime - s - (c == 10)
    assert not vout[10, 400 - shifts[10]]        # the flagged cell moved with its channel
    assert (~vout).sum() == shifts.sum() + 1
    # a shift longer than the series leaves nothing valid
    _, vbig = sp.dedisperse_masked(data[:2], valid[:2], 1e5, freqs[:2], TSAMP)
    assert not vbig[1].any()


# --- band profile and boxcar -----------------------------------------------------------

def test_band_profile_keeps_unit_noise_when_half_the_band_is_flagged_in_part_of_the_window():
    rng = np.random.default_rng(5)
    nchan, ntime = 512, 8000
    data = rng.normal(100.0, 10.0, (nchan, ntime)).astype(np.float32)
    valid = np.ones(data.shape, dtype=bool)
    valid[:256, 4000:] = False                  # half the band flagged for the second half
    data[:256, 4000:] += 700.0                  # whose values must not matter
    norm, v = sp.normalise_masked(data, valid)
    prof, nvalid = sp.band_profile_masked(norm, v, min_nvalid=1)
    assert nvalid[:4000].min() == 512 and nvalid[4000:].max() == 256
    first, second = prof[:4000], prof[4000:]
    assert first.std() == pytest.approx(1.0, rel=0.05)
    assert second.std() == pytest.approx(1.0, rel=0.05)        # where a plain mean has sqrt(2) more noise
    plain = np.where(v, norm, 0.0).sum(axis=0) / np.maximum(nvalid, 1)
    assert plain[4000:].std() / plain[:4000].std() == pytest.approx(math.sqrt(2.0), rel=0.06)


def test_band_profile_masked_makes_thin_samples_unavailable():
    nchan, ntime = 100, 50
    norm = np.ones((nchan, ntime), dtype=np.float32)
    valid = np.ones(norm.shape, dtype=bool)
    valid[:, 10:20] = False                     # nothing valid
    valid[5:, 20:30] = False                    # 5 valid channels
    prof, n = sp.band_profile_masked(norm, valid, min_nvalid=10)
    assert np.all(np.isnan(prof[10:30])) and np.all(np.isfinite(prof[:10])) and np.all(np.isfinite(prof[30:]))
    assert prof[0] == pytest.approx(100 / 10.0)                 # sum / sqrt(N)
    assert sp.min_nvalid_for(2600) == 260 and sp.min_nvalid_for(40) == sp.MIN_NVALID_CHANNELS


def test_profile_snr_masked_matches_profile_snr_without_gaps():
    rng = np.random.default_rng(6)
    prof = rng.normal(0.0, 1.0, 6000)
    prof[3000:3008] += 3.0
    for width in (1, 4, 8, 16):
        new = sp.profile_snr_masked(prof, width)
        old = sp.profile_snr(prof, width)
        # same boxcar, same robust noise; only the half-empty windows at the two ends
        # differ (the old code zero-pads them, the new one divides by sqrt(n_available))
        inner = slice(2 * width, -2 * width)
        assert np.allclose(new[inner], old[inner], rtol=1e-3, atol=1e-3)
        assert np.nanargmax(new) == np.argmax(old)
        assert np.isfinite(new).all()


def test_profile_snr_masked_skips_gaps_and_does_not_spread_nan():
    rng = np.random.default_rng(7)
    n, width = 4000, 8
    prof = rng.normal(0.0, 1.0, n)
    prof[2000:2008] += 4.0                      # S/N 4 * sqrt(8) = 11 pulse
    full = sp.profile_snr_masked(prof, width)
    gapped_in = prof.copy()
    gapped_in[1000:1060] = np.nan               # a 60-sample hole elsewhere
    gap = sp.profile_snr_masked(gapped_in, width)
    hole = np.isnan(gap)
    assert hole[1000 + width:1060 - width].all()                  # inside the hole: unavailable
    assert hole.sum() < 60 + 2 * width                             # NaN stays near the hole
    assert not hole[:1000 - width].any() and not hole[1060 + width:].any()
    # away from the hole the series is what it was, up to the noise estimate
    ratio = gap[1500:3000] / full[1500:3000]
    assert np.nanstd(ratio[np.abs(full[1500:3000]) > 1]) < 0.01
    assert np.nanmax(gap[1990:2030]) == pytest.approx(np.nanmax(full[1990:2030]), rel=0.03)
    # a window half-covered by the hole is evaluated; one with less than half is not
    n_avail = np.convolve((~np.isnan(gapped_in)).astype(float), np.ones(width), "same")
    assert np.all(np.isnan(gap[n_avail < width / 2]))
    assert np.isfinite(gap[1000 - 1]) and np.isfinite(gap[1060 + width])
    # the pulse next to a hole keeps its S/N (the window loses samples, S/N goes with sqrt(n))
    near = prof.copy()
    near[2004:2012] = np.nan                    # the second half of the pulse is missing
    half = sp.profile_snr_masked(near, width)
    assert np.nanmax(half[1990:2030]) < np.nanmax(full[1990:2030])
    assert np.nanmax(half[1990:2030]) > 0.6 * np.nanmax(full[1990:2030])
    # too little to measure the noise: all NaN, not zeros
    sparse = np.full(100, np.nan)
    sparse[:10] = 1.0
    assert np.all(np.isnan(sp.profile_snr_masked(sparse, 4)))


# --- display averaging ------------------------------------------------------------------

def test_downsample_masked_averages_valid_cells_only():
    data = np.arange(8 * 6, dtype=np.float32).reshape(8, 6)
    valid = np.ones(data.shape, dtype=bool)
    valid[0, 0] = False                         # 1 of 4 cells in the first 2x2 block
    valid[2:4, 2:4] = False                     # block (1, 1) fully flagged
    valid[4, 4:6] = False                       # block (2, 2): two of four, kept (>= 50%)
    valid[6, 4:6] = False                       # block (3, 2): three of four flagged, dropped
    valid[7, 4] = False
    out = sp.downsample_masked(data, valid, 2, 2)
    assert out.shape == (4, 3)
    assert out[0, 0] == pytest.approx(np.mean([data[0, 1], data[1, 0], data[1, 1]]))
    assert np.isnan(out[1, 1]) and np.isnan(out[3, 2])
    assert out[2, 2] == pytest.approx(np.mean([data[5, 4], data[5, 5]]))
    assert out[0, 1] == pytest.approx(data[0:2, 2:4].mean())
    # all valid: the plain block mean
    assert np.allclose(sp.downsample_masked(data, np.ones(data.shape, bool), 2, 2), sp.downsample(data, 2, 2))


def test_downsample_max_masked_skips_nan():
    x = np.array([[1.0, np.nan, 3.0, 2.0], [np.nan, np.nan, 5.0, np.nan]], dtype=np.float32)
    out = sp.downsample_max_masked(x, 2)
    assert out[0, 0] == 1.0 and out[0, 1] == 3.0 and np.isnan(out[1, 0]) and out[1, 1] == 5.0
    y = np.random.default_rng(0).normal(size=(3, 12)).astype(np.float32)
    assert np.array_equal(sp.downsample_max_masked(y, 4), sp.downsample_max(y, 4))


# --- DM-time ------------------------------------------------------------------------------

def test_dm_time_masked_equals_a_per_trial_brute_force():
    """The run-sum shortcut in dm_time_masked gives what dedisperse_masked +
    band_profile_masked + profile_snr_masked give trial by trial."""
    rng = np.random.default_rng(8)
    nchan, ntime, f = 96, 2500, 4
    freqs = _freqs(nchan)
    data = rng.normal(0.0, 1.0, (nchan, ntime)).astype(np.float32)
    valid = rng.random(data.shape) > 0.3
    valid[:, 1200:1400] &= rng.random(1) > 2                    # 200 uncovered samples
    norm, v = sp.normalise_masked(data, valid)
    total, count = sp.block_sum_masked(norm, v, f, 1)
    f_small = freqs.reshape(-1, f).mean(axis=1)
    dms = sp.dm_grid(120.0, 7)
    got = sp.dm_time_masked(total, count, f_small, TSAMP, dms, 4, min_nvalid=20)
    for j, dm in enumerate(dms):
        dd, vd = sp.dedisperse_masked(total, count > 0, dm, f_small, TSAMP)
        cd, _ = sp.dedisperse_masked(count, count > 0, dm, f_small, TSAMP)
        s = dd.sum(axis=0, dtype=np.float64)
        nn = cd.sum(axis=0, dtype=np.float64)
        prof = np.where(nn >= 20, s / np.sqrt(np.maximum(nn, 1)), np.nan)
        want = sp.profile_snr_masked(prof, 4)
        assert np.allclose(got[j], want, atol=2e-4, equal_nan=True), j
        assert np.array_equal(np.isnan(got[j]), np.isnan(want))


def test_dm_time_masked_recovers_a_pulse_through_flagged_channels():
    rng = np.random.default_rng(9)
    nchan, ntime, dm, width, f = 384, 6000, 90.0, 4, 2
    freqs = _freqs(nchan)
    data = rng.normal(100.0, 10.0, (nchan, ntime)).astype(np.float32)
    delays = np.round(4.148808e3 * dm * (freqs**-2 - freqs.max()**-2) / TSAMP).astype(int)
    for c in range(nchan):
        data[c, 2000 + delays[c]: 2000 + delays[c] + width] += 4.0
    flagged = np.zeros(data.shape, dtype=bool)
    flagged[::3] = True                         # a third of the channels, whole time
    data[flagged] += 3000.0
    dms = sp.dm_grid(dm, 33)
    j = int(np.argmin(np.abs(dms - dm)))

    def peak(d, fl):
        norm, v = sp.normalise_masked(d, ~fl)
        tot, cnt = sp.block_sum_masked(norm, v, f, 1)
        f_small = freqs[: tot.shape[0] * f].reshape(-1, f).mean(axis=1)
        return np.nanmax(sp.dm_time_masked(tot, cnt, f_small, TSAMP, dms, width, 10)[j])

    clean = data.copy()
    clean[flagged] -= 3000.0
    expected = peak(clean, np.zeros_like(flagged)) * math.sqrt(2.0 / 3.0)
    assert peak(data, flagged) == pytest.approx(expected, rel=0.08)


# --- coverage maps ---------------------------------------------------------------------------

def test_uncovered_fraction_and_cells_follow_the_dispersive_shift():
    nchan, ntime, dm = 64, 1500, 100.0
    freqs = _freqs(nchan)
    cov = np.ones(ntime, dtype=bool)
    cov[600:700] = False
    frac0 = sp.uncovered_fraction(cov, freqs, TSAMP, [0.0, dm])
    assert np.array_equal(frac0[0] > 0, ~cov) and set(np.unique(frac0[0])) == {0.0, 1.0}
    # brute force: dedisperse the "unknown" flag per channel
    cells = sp.uncovered_cells(cov, nchan, dm, freqs, TSAMP)
    assert np.allclose(frac0[1], cells.mean(axis=0), atol=1e-6)
    shifts = np.round(4.148808e3 * dm * (freqs**-2 - freqs.max()**-2) / TSAMP).astype(int)
    for c in (0, 20, 63):
        assert np.array_equal(np.flatnonzero(cells[c]),
                              np.arange(max(0, 600 - shifts[c]), max(0, 700 - shifts[c])))
    # a full-coverage array has none, and the wrapped samples are not "uncovered"
    assert not sp.uncovered_fraction(np.ones(ntime, bool), freqs, TSAMP, [dm]).any()
    assert not sp.uncovered_cells(np.ones(ntime, bool), nchan, dm, freqs, TSAMP).any()
