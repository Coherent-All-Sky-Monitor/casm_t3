"""Smoke test: full candidate figure from synthetic data, coordinates from the card's sky block."""
import json
from pathlib import Path

import numpy as np

from casm_t2 import weights_registry as wr
from casm_t3 import plotting


def _registry(tmp_path):
    reg = wr.Registry(tmp_path / "reg")
    alt = [20.0 + (i % 70) for i in range(512)]
    az = [float(i * 360 / 512) for i in range(512)]
    pid = reg.record_product(h5_path="/x.h5", stream_md5={s: f"m{s}" for s in range(6)}, alt_deg=alt, az_deg=az)
    return reg, pid, alt, az


def _data(nchan=3072, ntime=4096, dm=30.0, tsamp=1.048576e-3):
    rng = np.random.default_rng(1)
    freqs = 484.375 - np.arange(nchan) * 0.030517578125
    data = rng.normal(100, 10, (nchan, ntime)).astype(np.float32)
    delays = (4.148808e3 * dm * (freqs ** -2 - freqs.max() ** -2) / tsamp).astype(int)
    for c in range(nchan):
        data[c, 2048 + delays[c]: 2056 + delays[c]] += 6.0
    return data, freqs, tsamp


def test_figure_renders_with_sky_and_footprint(tmp_path):
    reg, pid, alt, az = _registry(tmp_path)
    data, freqs, tsamp = _data()
    members = [[0.0, 336, 30.0, 20.0, 3], [0.01, 337, 31.0, 14.0, 3], [-2.0, 10, 200.0, 13.0, 5]]
    card = {"candname": "TESTcand", "source": "blind", "event_utc": "2026-09-02T20:00:00.000+00:00",
            "beam": 336, "snr": 20.0, "dm": 30.0, "width": 3,
            "sky": {"weights_id": pid, "alt_deg": alt[336], "az_deg": az[336], "ra_deg": 150.0, "dec_deg": -5.0,
                    "sun_alt_deg": 40.0, "sun_az_deg": 200.0},
            "context": {"window_s": 4.0, "members": members}}
    out = plotting.make_candidate_figure(data, freqs, tsamp, 2048 * tsamp, card, tmp_path / "c.png", registry=reg, layout="v2")
    assert Path(out).stat().st_size > 50_000
    out2 = plotting.make_candidate_figure(data, freqs, tsamp, 2048 * tsamp, card, tmp_path / "legacy.png")
    assert Path(out2).stat().st_size > 50_000


def test_figure_renders_without_sky(tmp_path):
    reg, pid, alt, az = _registry(tmp_path)
    data, freqs, tsamp = _data()
    card = {"candname": "TESTnosky", "source": "blind", "event_utc": "2026-09-02T20:00:00.000+00:00",
            "beam": 5, "snr": 20.0, "dm": 30.0, "width": 3, "context": {"window_s": 4.0, "members": []}}
    out = plotting.make_candidate_figure(data, freqs, tsamp, 2048 * tsamp, card, tmp_path / "n.png", registry=reg)
    assert Path(out).exists()


def test_subbands_for_matches_target_per_pixel_sigma():
    # nrows chosen so snr / sqrt(nrows) lands near SUBBAND_TARGET_SIGMA per pixel
    assert plotting.subbands_for(23) == 32          # 23 / 5.7 = 4.1 sigma
    assert plotting.subbands_for(43) == 128         # 43 / 11.3 = 3.8 sigma
    assert plotting.subbands_for(12) == 16          # clamped at the coarse end
    assert plotting.subbands_for(200) == 256        # clamped at the fine end
    for snr in (8.0, 12.5, 17.0, 23.0, 30.0, 43.0, 60.0, 120.0, 500.0):
        n = plotting.subbands_for(snr)
        assert 16 <= n <= 256
        assert n & (n - 1) == 0                      # power of two
        assert 3072 % n == 0
    # no usable S/N falls back to the historical fixed value
    assert plotting.subbands_for(None) == plotting.DEFAULT_SUBBANDS
    assert plotting.subbands_for(float("nan")) == plotting.DEFAULT_SUBBANDS


def test_waterfall_title_reports_adaptive_subbands(tmp_path, monkeypatch):
    """The waterfall title names the row count actually used, and it tracks S/N."""
    from matplotlib.axes import Axes

    data, freqs, tsamp = _data()
    seen: list[str] = []
    real_set_title = Axes.set_title

    def record(self, label, *a, **kw):
        seen.append(str(label))
        return real_set_title(self, label, *a, **kw)

    monkeypatch.setattr(Axes, "set_title", record)

    for snr in (23.0, 43.0):
        seen.clear()
        card = {"candname": f"TESTsub{int(snr)}", "source": "blind",
                "event_utc": "2026-09-02T20:00:00.000+00:00",
                "beam": 5, "snr": snr, "dm": 30.0, "width": 3,
                "context": {"window_s": 4.0, "members": []}}
        plotting.make_candidate_figure(data, freqs, tsamp, 2048 * tsamp, card,
                                       tmp_path / f"s{int(snr)}.png", layout="v2")
        wf = [t for t in seen if t.startswith("waterfall")]
        assert len(wf) == 1 and not wf[0].endswith("subbands)"), wf

    # explicit override wins over the S/N rule, and only the override shows the count
    seen.clear()
    card = {"candname": "TESToverride", "source": "blind",
            "event_utc": "2026-09-02T20:00:00.000+00:00",
            "beam": 5, "snr": 23.0, "dm": 30.0, "width": 3,
            "context": {"window_s": 4.0, "members": []}}
    plotting.make_candidate_figure(data, freqs, tsamp, 2048 * tsamp, card,
                                   tmp_path / "ovr.png", layout="v2", subbands=256)
    assert any(t.endswith("(256 subbands)") for t in seen)


def test_dmt_ffactor_keeps_intra_subband_smear_under_the_boxcar():
    chan = 0.030517578125
    tsamp = 1.048576e-3
    # DM 300, 4-sample boxcar (4.2 ms): even 2 channels smear past half the boxcar
    assert plotting.dmt_ffactor_for(300.0, 4, tsamp, chan) == 1
    # DM 300, 16-sample boxcar (16.8 ms): 6 channels fit, 8 do not
    assert plotting.dmt_ffactor_for(300.0, 16, tsamp, chan) == 6
    # low DM, 8-sample boxcar: the coarsest choice is fine
    assert plotting.dmt_ffactor_for(42.5, 8, tsamp, chan) == 16
    for dm, ibox in ((0.0, 1), (42.5, 3), (300.0, 2), (900.0, 2), (2000.0, 5)):
        f = plotting.dmt_ffactor_for(dm, 2 ** ibox, tsamp, chan)
        assert f in plotting.DMT_FFACTOR_CHOICES and 3072 % f == 0
        smear = plotting.DMT_SMEAR_K_MS * dm * f * chan / plotting.DMT_SMEAR_F_GHZ ** 3
        assert f == 1 or smear <= 0.5 * 2 ** ibox * tsamp * 1e3


def test_dm_time_recovers_a_narrow_high_dm_pulse(tmp_path):
    """A 3 ms pulse at DM 900: the DM-time panel must keep the profile S/N.

    The trial-DM grid is the production one from single_pulse.dm_grid, which
    contains the true DM exactly, so this measures the intra-subband smearing
    alone.
    """
    from casm_t3 import single_pulse

    dm, ibox = 900.0, 2
    width = 2 ** ibox
    nchan, ntime, tsamp = 3072, 16384, 1.048576e-3
    freqs = 484.375 - np.arange(nchan) * 0.030517578125
    rng = np.random.default_rng(3)
    data = rng.normal(100, 10, (nchan, ntime)).astype(np.float32)
    delays = (4.148808e3 * dm * (freqs ** -2 - freqs.max() ** -2) / tsamp).astype(int)
    for c in range(nchan):
        data[c, 2000 + delays[c]: 2003 + delays[c]] += 8.0     # ~3 ms pulse

    norm = single_pulse.normalise(data)
    prof_peak = single_pulse.profile_snr(
        single_pulse.dedisperse(norm, dm, freqs, tsamp).mean(axis=0), width).max()
    dms = single_pulse.dm_grid(dm)
    j = int(np.argmin(np.abs(dms - dm)))
    assert np.isclose(dms[j], dm)

    def peak_at_true_dm(f):
        small = single_pulse.downsample(norm, f, 1)
        f_small = plotting._block_mean_freqs(freqs, f)[: small.shape[0]]
        return single_pulse.dm_time(small, f_small, tsamp, dms, width)[j].max()

    f_new = plotting.dmt_ffactor_for(dm, width, tsamp, 0.030517578125, nchan)
    assert f_new == 1
    assert peak_at_true_dm(f_new) >= 0.9 * prof_peak         # within 10%
    assert peak_at_true_dm(8) < 0.8 * prof_peak              # the old fixed factor loses it


def test_dm_grid_contains_the_candidate_dm():
    """Every trial grid must sample the candidate DM itself, at any DM."""
    from casm_t3 import single_pulse

    for dm in (0.0, 5.0, 42.5, 300.0, 900.0, 2000.0):
        dms = single_pulse.dm_grid(dm)
        assert dms.size == 65
        assert np.isclose(dms, dm).any()
        if dm >= max(0.4 * dm, 15.0):       # grid not clipped at DM 0
            assert np.isclose(dms[dms.size // 2], dm)   # dm sits at the centre trial
        assert np.all(np.diff(dms) > 0)
        assert dms[0] >= 0.0
        # uniform step of max(0.4 dm, 15) / 32, and the full span above dm
        step = max(0.4 * dm, 15.0) / 32
        assert np.allclose(np.diff(dms), step)
        assert dms[-1] >= dm + max(0.4 * dm, 15.0) - 1e-9


def test_dmt_display_keeps_a_narrow_pulse_and_fits_the_axis():
    """Narrow pulse: the displayed DM-time array must keep the native peak.

    Mean-pooling at half a boxcar width plus imshow's nearest-neighbour column
    dropping turned an S/N 24.7 candidate into a faint dot. Max-pooling at one
    boxcar width, capped at DMT_MAX_COLUMNS drawn columns, keeps it.
    """
    from casm_t3 import single_pulse

    dm, ibox = 300.0, 2
    width = 2 ** ibox
    nchan, ntime, tsamp = 3072, 4096, 1.048576e-3
    freqs = 484.375 - np.arange(nchan) * 0.030517578125
    rng = np.random.default_rng(7)
    data = rng.normal(100, 10, (nchan, ntime)).astype(np.float32)
    delays = (4.148808e3 * dm * (freqs ** -2 - freqs.max() ** -2) / tsamp).astype(int)
    for c in range(nchan):
        data[c, 1500 + delays[c]: 1500 + width + delays[c]] += 8.0

    norm = single_pulse.normalise(data)
    f_dmt = plotting._dmt_plan(dm, width, freqs, tsamp)
    small = single_pulse.downsample(norm, f_dmt, 1)
    f_small = plotting._block_mean_freqs(freqs, f_dmt)[: small.shape[0]]
    dms = single_pulse.dm_grid(dm)
    j = int(np.argmin(np.abs(dms - dm)))
    dmt = single_pulse.dm_time(small, f_small, tsamp, dms, width)

    xlim = (-1.0, 1.0)
    tfactor = plotting._dmt_tfactor(width, xlim, tsamp)
    disp = single_pulse.downsample_max(dmt, tfactor)
    assert disp[j].max() >= 0.95 * dmt[j].max()

    # a 1 s window at 1.048576 ms must not be drawn with more columns than the
    # panel has pixels to show
    ncol = int(np.ceil((xlim[1] - xlim[0]) / tsamp / tfactor))
    assert ncol <= plotting.DMT_MAX_COLUMNS
    assert plotting._dmt_tfactor(2, xlim, tsamp) >= 2
    assert int(np.ceil((xlim[1] - xlim[0]) / tsamp / plotting._dmt_tfactor(2, xlim, tsamp))) <= 400
    # wide candidates keep one-boxcar binning, the column cap does not bite
    assert plotting._dmt_tfactor(16, xlim, tsamp) == 16
