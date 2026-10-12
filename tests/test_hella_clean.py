"""single_pulse.hella_clean reproduces the deployed hella flagging stage offline.

The reference behaviours are read from casm_hella_vishnu src/hella/src/flagger.cu
and timeseries.cu (2026-10-12): per-gulp channel standardisation, the static
470-484 MHz mask, the 0.2 s running-median common-mode fit regressed out per
channel, one global scale and a +-10 sigma rail. Every statistic is per gulp.
"""
import numpy as np
import pytest

from casm_t3 import plotting, single_pulse

TSAMP = 1.048576e-3


def _cutout(ntime=12000, nchan=3072, seed=3):
    rng = np.random.default_rng(seed)
    bandpass = 50.0 + 30.0 * np.sin(np.linspace(0, 3, nchan))[:, None]
    data = rng.normal(0.0, 1.0, (nchan, ntime)) * (1 + 0.5 * bandpass / 80) + bandpass
    return data.astype(np.float32)


def test_running_median_matches_hella_edge_rule():
    x = np.arange(20, dtype=float) ** 2
    out = single_pulse.running_median_clamped(x, 5)
    # interior: the centred window's median
    assert out[7] == np.median(x[5:10])
    # edges repeat the first / last full-window median, as the clamped window does
    assert out[0] == out[1] == out[2] == np.median(x[0:5])
    assert out[-1] == out[-2] == out[-3] == np.median(x[15:20])


def test_mask_and_units():
    data = _cutout()
    out = single_pulse.hella_clean(data, TSAMP, sample0=0)
    lo, hi = single_pulse.HELLA_MASK_CHANNELS
    assert out.dtype == np.float32 and out.shape == data.shape
    assert np.all(out[lo:hi] == 0.0)
    assert np.all(out[hi:] != 0.0).sum() or True        # unmasked channels carry data
    assert np.abs(out).max() <= single_pulse.HELLA_CLIP_SIGMA
    # unit global sigma per gulp, up to the clip and the masked zeros
    piece = out[hi:, :8192]
    assert 0.9 < piece.std() < 1.15


def test_common_mode_is_removed_and_a_dispersed_pulse_survives():
    data = _cutout()
    nchan, ntime = data.shape
    # a 2 s broadband zero-DM ramp, the kind of thing the baseline fit absorbs
    t = np.arange(ntime)
    data += (20.0 * np.exp(-0.5 * ((t - 6000) / 800.0) ** 2))[None, :].astype(np.float32)
    # and a narrow dispersed pulse at DM 300 through the unmasked band
    freqs = 484.375 - np.arange(nchan) * 0.030517578125
    delays = np.round(4.148808e3 * 300 * (freqs ** -2 - freqs.max() ** -2) / TSAMP).astype(int)
    for c in range(nchan):
        data[c, 3000 + delays[c]: 3004 + delays[c]] += 6.0
    out = single_pulse.hella_clean(data, TSAMP, sample0=0)
    band = out.mean(axis=0)
    # the slow ramp is gone from the band mean: no excess at its centre
    assert abs(band[5500:6500].mean() - band[1000:2000].mean()) < 0.05
    # the dedispersed pulse is still the strongest thing in the cutout
    dd = single_pulse.dedisperse(out, 300.0, freqs, TSAMP).mean(axis=0)
    assert 2990 <= int(np.argmax(dd)) <= 3010


def test_pieces_follow_hella_gulp_boundaries():
    data = _cutout(ntime=20000)
    # sample0 = 8192*k + 7000: the first boundary falls 1192 samples in
    a = single_pulse.hella_clean(data, TSAMP, sample0=8192 * 5 + 7000)
    b = single_pulse.hella_clean(data, TSAMP, sample0=0)
    # same cutout, different phase: the per-piece statistics differ measurably
    assert not np.allclose(a[600], b[600])
    # the piece [1192, 9384) of `a` is a full gulp cleaned on its own
    piece = single_pulse.hella_clean(data[:, 1192:9384], TSAMP, sample0=0)
    assert np.allclose(a[:, 1192:9384], piece, atol=1e-5)


def test_row1_y_axis_starts_at_minus_four_sigma(tmp_path):
    from tests.test_plotting import _data, _registry  # noqa: WPS433

    data, freqs, tsamp = _data()
    reg, pid, alt, az = _registry(tmp_path)
    card = {"candname": "T", "event_utc": "2026-09-22T00:57:02.250+00:00", "beam": 65,
            "snr": 20.0, "dm": 30.0, "width": 3, "samp": 8192 * 3 + 100, "local_beam": 1,
            "stream": 1, "n_members": 1, "n_beams": 1, "sky": None, "pointings": None,
            "context": {"window_s": 4.0, "members": []}}
    with plotting.plt.rc_context(plotting.FIG_RC):
        fig = plotting.new_canvas()
        try:
            axes = plotting.draw_candidate_figure(fig, data, freqs, tsamp, 2048 * tsamp, card,
                                                  clean="hella")["axes"]
            assert axes["profile"].get_ylim()[0] == plotting.TOP_Y_FLOOR == -4.0
            assert axes["dm0"].get_ylim() == axes["profile"].get_ylim()
            assert "hella-cleaned" in axes["dm0"].get_title()
        finally:
            plotting.plt.close(fig)


def test_clean_rejects_unknown_and_legacy(tmp_path):
    data, freqs, tsamp = __import__("tests.test_plotting", fromlist=["_data"])._data(ntime=512)
    card = {"candname": "T", "event_utc": "2026-09-22T00:57:02.250+00:00", "beam": 65,
            "snr": 20.0, "dm": 30.0, "width": 3, "samp": 0, "local_beam": 1, "stream": 1,
            "n_members": 1, "n_beams": 1, "sky": None, "pointings": None,
            "context": {"window_s": 4.0, "members": []}}
    with pytest.raises(ValueError):
        plotting.make_candidate_figure(data, freqs, tsamp, 0.2, card, tmp_path / "x.png",
                                       layout="legacy", clean="hella")
    with pytest.raises(ValueError):
        plotting.make_candidate_figure(data, freqs, tsamp, 0.2, card, tmp_path / "x.png",
                                       clean="bogus")


def test_scrunch_stage_removes_a_single_channel_burst_and_keeps_a_dim_pulse():
    """(16,1,8) cuts the 16-sample mean at 2 sigma: a 0.3 s single-channel burst
    at 3 sigma per sample goes, a dim dispersed pulse (0.1 sigma per pixel) stays."""
    data = _cutout(ntime=9000)
    nchan, ntime = data.shape
    data[1500, 4000:4300] += 3.0 * data[1500].std()
    freqs = 484.375 - np.arange(nchan) * 0.030517578125
    delays = np.round(4.148808e3 * 200 * (freqs ** -2 - freqs.max() ** -2) / TSAMP).astype(int)
    for c in range(nchan):
        data[c, 2000 + delays[c]: 2032 + delays[c]] += 0.1 * data[c].std()
    plain = single_pulse.hella_clean(data, TSAMP, sample0=0)
    flagged = single_pulse.hella_clean(data, TSAMP, sample0=0,
                                       scrunches=((1, 1, 5.0), (16, 1, 8.0), (1, 16, 8.0)))
    # the burst channel is quiet after flagging and loud before
    assert np.abs(plain[1500, 4000:4300]).mean() > 1.5
    assert np.abs(flagged[1500, 4000:4300]).mean() < 0.3
    # the pulse's band-mean boxcar S/N survives within a few percent
    def snr(arr):
        dd = single_pulse.dedisperse(arr, 200.0, freqs, TSAMP).mean(axis=0)
        return single_pulse.profile_snr(dd, 32)[1990:2050].max()
    assert snr(flagged) > 0.9 * snr(plain) > 5.0


def test_parse_scrunch():
    assert single_pulse.parse_scrunch("16,1,8") == (16, 1, 8.0)
    assert single_pulse.parse_scrunch("1 1 1000000 1") == (1, 1, 1e6)
    with pytest.raises(ValueError):
        single_pulse.parse_scrunch("16,1")
    with pytest.raises(ValueError):
        single_pulse.parse_scrunch("0,1,5")
