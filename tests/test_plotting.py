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
        wf = [t for t in seen if t.lower().startswith("waterfall")]
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


# --- layout v3 (approved 2026-09-22) -----------------------------------------
# Expectations below come from the plan (figure fractions, canvas size), from
# the card itself (trials, clusters, members, sources) or from what matplotlib
# rendered, never from the plotting module's own helpers.

import contextlib
import re

import matplotlib.image as mimage
import matplotlib.pyplot as plt
import pytest
from matplotlib.legend import Legend

import synthetic_gulp as sg

W_PX, H_PX = 1510, 1517
LIVE_CARD = Path(__file__).resolve().parent / "data" / "260922qlsiym_live_card.json"


def _live_card() -> dict:
    """The live t2d card (casm_t2 branch card-gulp-block, 260922qlsiym): event
    at samp_hi, one red beam, truncation fields present."""
    return json.loads(LIVE_CARD.read_text())


@contextlib.contextmanager
def _drawn(card, data=None, **kw):
    """Draw the v3 figure for ``card`` on synthetic data; yields (fig, axes)."""
    if data is None:
        data = sg.synthetic_data(card["dm"], width_samp=2 ** int(card["width"]))
    arr, freqs, tsamp, t_rel = data
    with plt.rc_context(plotting.FIG_RC):
        fig = plotting.new_canvas()
        try:
            info = plotting.draw_candidate_figure(fig, arr, freqs, tsamp, t_rel, card, **kw)
            yield fig, info["axes"]
        finally:
            plt.close(fig)


def _boxes_px(fig) -> dict[str, tuple]:
    """Every axes (inset colour bars included) by label, as a pixel bbox."""
    r = fig.canvas.get_renderer()
    out = {}
    for ax in fig.axes:
        for a in [ax, *getattr(ax, "child_axes", [])]:
            b = a.get_window_extent(r)
            out[a.get_label()] = (b.x0, b.y0, b.x1, b.y1)
    return out


def _max_diff(a: dict, b: dict, keys=None) -> float:
    keys = sorted(keys if keys is not None else a)
    return float(np.max(np.abs(np.array([a[k] for k in keys]) - np.array([b[k] for k in keys]))))


def _geometry_cards() -> list[dict]:
    cards = [sg.build_card(case) for case in sg.CASES]
    cards[1] = dict(cards[1], width=5)              # a different boxcar and DM range too
    cards.append(_live_card())
    return cards


def test_v3_geometry_is_identical_across_cards():
    """Every axes rectangle is the same to the pixel whatever the card holds."""
    boxes = []
    for card in _geometry_cards():
        with _drawn(card) as (fig, axes):
            boxes.append(_boxes_px(fig))
            r = fig.canvas.get_renderer()
            # job labels, then the corr column at least 6 px to their right
            texts = axes["beams"].texts
            job_x1 = max(t.get_window_extent(r).x1 for t in texts if t.get_gid() == "job_label")
            corr_x0 = min(t.get_window_extent(r).x0 for t in texts if t.get_gid() == "corr_label")
            assert corr_x0 - job_x1 >= 6.0
    assert len(boxes) >= 3
    for b in boxes[1:]:
        assert set(b) == set(boxes[0])
        assert _max_diff(boxes[0], b) == 0.0

    # and the rectangles are the approved ones (plan section 1), in pixels
    b = boxes[0]
    fx = lambda f: f * W_PX                          # noqa: E731
    fy = lambda f: f * H_PX                          # noqa: E731
    for label, (x0, x1), (y0, y1) in (
            ("profile", (0.060, 0.455), (0.7412, 0.8965)),
            ("dm0", (0.5232, 0.8850), (0.7412, 0.8965)),
            ("waterfall", (0.060, 0.455), (0.3795, 0.6744)),
            ("dm_time", (0.5232, 0.8850), (0.3795, 0.6744))):
        assert b[label] == pytest.approx((fx(x0), fy(y0), fx(x1), fy(y1)), abs=0.1), label
    assert b["beams"][0] == pytest.approx(fx(0.0735), abs=0.1)
    assert b["beams"][2] == pytest.approx(fx(0.455), abs=0.1)
    assert b["beams"][1] == pytest.approx(fy(0.052089), abs=0.1)
    assert b["hist"][3] == pytest.approx(fy(0.302812), abs=0.1)
    x0, y0, x1, y1 = b["sky"]
    assert x1 - x0 == pytest.approx(y1 - y0, abs=0.5)                # square
    assert (x0 + x1) / 2 == pytest.approx(fx(0.7041), abs=0.5)       # on the column centre
    # gutters: >= 60 px left of the left column, >= 90 px between the columns
    assert b["profile"][0] >= 60 and b["dm0"][0] - b["profile"][2] >= 90
    cb = b["dm_time_cbar"]
    dmt_w = b["dm_time"][2] - b["dm_time"][0]
    assert cb[0] == pytest.approx(b["dm_time"][0] + 1.02 * dmt_w, abs=0.1)


def test_corner_text_stays_left_of_the_event_line():
    """Every corner line ends before 0.46 of the axes (t = 0 is at 0.5), with
    a two-line block when the DM = 0 panel has an off-scale spike."""
    card = sg.build_card("default")
    arr, freqs, tsamp, t_rel = sg.synthetic_data(card["dm"])
    spiky = arr.copy()
    spiky[:, 2048 + 300: 2048 + 304] += 400.0       # undispersed, 0.3 s after the event
    for data in ((arr, freqs, tsamp, t_rel), (spiky, freqs, tsamp, t_rel)):
        with _drawn(card, data=data) as (fig, axes):
            r = fig.canvas.get_renderer()
            n_lines = {}
            for name in ("profile", "dm0"):
                ax = axes[name]
                box = ax.get_window_extent(r)
                corner = [t for t in ax.texts if t.get_gid() == "corner"]
                assert corner
                n_lines[name] = len(corner)
                for t in corner:
                    right = (t.get_window_extent(r).x1 - box.x0) / box.width
                    patch_right = (t.get_bbox_patch().get_window_extent(r).x1 - box.x0) / box.width
                    assert right < 0.46 and patch_right < 0.47, (name, t.get_text(), right)
    # the spike case says it is off scale, on the DM = 0 panel
    texts = " ".join(t.get_text() for t in axes["dm0"].texts if t.get_gid() == "corner")
    assert "off scale" in texts and n_lines["dm0"] >= 3


def test_corner_text_overrun_aborts_and_closes_the_figure(tmp_path, monkeypatch):
    card = sg.build_card("single_beam")
    arr, freqs, tsamp, t_rel = sg.synthetic_data(card["dm"])
    monkeypatch.setattr(plotting, "CORNER_MAX_FRAC", 0.95)     # let the text run long
    before = set(plt.get_fignums())
    with pytest.raises(plotting.CornerTextError):
        plotting.make_candidate_figure(arr, freqs, tsamp, t_rel, card, tmp_path / "x.png")
    assert set(plt.get_fignums()) == before
    assert not (tmp_path / "x.png").exists()


def test_sky_source_legend_survives_the_size_key():
    """Four source entries from the card, plus the S/N key, and no legend
    marker within 6 px of the rim."""
    card = sg.build_card("default")
    expected = []
    for name in ("Sun", "Cas A", "Cyg A", "Tau A"):
        alt = card["sky"]["sources"][name][0]
        expected.append(name if alt >= 18 else
                        f"{name} (alt {alt:.0f}\N{DEGREE SIGN}, no beam)" if alt > 0 else
                        f"{name} (set)")
    with _drawn(card) as (fig, axes):
        ax = axes["sky"]
        legends = [a for a in ax.get_children() if isinstance(a, Legend)]
        labels = [[t.get_text() for t in lg.get_texts()] for lg in legends]
        assert expected in labels
        assert any(all(s.startswith("S/N ") for s in lab) for lab in labels)
        centre = ax.transData.transform([[0.0, 0.0]])[0]
        rim = np.hypot(*(ax.transData.transform([[0.0, ax.get_rmax()]])[0] - centre))
        for lg in legends:
            for h in lg.legend_handles:
                p = h.get_offset_transform().transform(h.get_offsets())[0]
                radius = np.sqrt(h.get_sizes()[0]) / 2 * fig.dpi / 72
                assert np.hypot(*(p - centre)) - radius - rim >= 6.0


def _drawn_beams(ax, gid, pointings) -> set[int]:
    """Beams under the scatter markers with this gid, matched on (az, alt)."""
    theta = np.radians(np.asarray(pointings["az_deg"]))
    r = 90.0 - np.asarray(pointings["alt_deg"])
    beams = set()
    for coll in ax.collections:
        if coll.get_gid() != gid:
            continue
        for t, rr in coll.get_offsets():
            hit = np.nonzero(np.isclose(theta, t) & np.isclose(r, rr))[0]
            assert hit.size == 1
            beams.add(int(hit[0]))
    return beams


@pytest.mark.parametrize("case", ["default", "sky_wide_triggered", "single_beam", "storm"])
def test_sky_title_counts_equal_the_drawn_beams(case):
    card = sg.build_card(case)
    gulp = card["gulp"]
    trig = [c for c in gulp["clusters"] if c["outcome"] == "triggered"]
    red = {b for c in trig for b, _s in c["beams"]}
    grey = {b for c in gulp["clusters"] if c not in trig for b, _s in c["beams"]} - red
    with _drawn(card) as (fig, axes):
        drawn_red = _drawn_beams(axes["sky"], "sky_red", card["pointings"])
        drawn_grey = _drawn_beams(axes["sky"], "sky_grey", card["pointings"])
        title = [t.get_text() for t in fig.texts if t.get_gid() == "sky_title"]
    assert drawn_red == red and drawn_grey == grey
    assert len(title) == 1
    line1, line2 = title[0].split("\n")
    assert line1 == "Beams with a T1 candidate in this 8.6 s gulp:"
    assert int(re.match(r"(\d+) in the triggered group \(red\)", line2).group(1)) == len(red)
    if grey:
        n = int(re.search(r"(\d+) others? \(grey\)", line2).group(1))
        assert n == len(grey)
        assert ("1 other (grey)" in line2) == (len(grey) == 1)
    else:
        assert "grey" not in line2


@pytest.mark.parametrize("case", list(sg.CASES) + ["live"])
def test_histogram_bars_sum_to_the_trials(case):
    card = _live_card() if case == "live" else sg.build_card(case)
    gulp = card["gulp"]
    n = len(gulp["trials"])
    with _drawn(card) as (fig, axes):
        bars = [p for p in axes["hist"].patches if p.get_gid() == "hist"]
        title = axes["hist"].get_title()
    assert sum(p.get_height() for p in bars) == n
    # 96-sample bins on the gulp's own grid, starting at samp_lo
    tsamp = gulp["tsamp_s"]
    assert bars[0].get_width() == pytest.approx(96 * tsamp)
    assert bars[0].get_x() == pytest.approx((gulp["samp_lo"] - card["samp"]) * tsamp)
    head = title.split("\n")[0]
    if gulp.get("trials_truncated"):
        assert head.startswith(f"{n} of {gulp['n_trials_total']} T1 candidates clustered in gulp")
    else:
        noun = "candidate" if n == 1 else "candidates"
        assert head.startswith(f"{n} T1 {noun} clustered in gulp {gulp['gulp']},")
    assert title.endswith("(storm gulp)") == bool(gulp["storm"].get("stormy"))


def test_no_gulp_card_draws_the_live_bottom_row():
    base = sg.build_card("default")
    tsamp = sg.TSAMP_S
    members = [[0.0, 336, 29.5, 24.1, 3], [0.01, 337, 31.0, 17.0, 3],
               [0.2, 12, 150.0, 14.0, 4], [0.2, 12, 150.0, 16.0, 4],   # one beam, twice
               [1.5, 400, 200.0, 13.0, 5]]                              # outside the window
    card = {k: v for k, v in base.items() if k != "gulp"}
    card["context"] = {"window_s": 4.0, "members": members}
    n_lit = len({int(m[1]) for m in members if abs(m[0]) <= 256 * tsamp})
    with _drawn(base) as (fig, _axes):
        gulp_boxes = _boxes_px(fig)
    with _drawn(card) as (fig, axes):
        boxes = _boxes_px(fig)
        assert {"members", "sky"} <= set(boxes) and not {"hist", "beams"} & set(boxes)
        assert axes["members"].get_title() == "T1 candidates within \N{PLUS-MINUS SIGN}4 s of the event"
        caption = [t.get_text() for t in axes["sky"].texts if "beams with a T1" in t.get_text()]
        assert caption == [f"{n_lit} of 512 beams with a T1 candidate within 0.27 s"]
        labels = [t.get_text() for lg in axes["sky"].get_children() if isinstance(lg, Legend)
                  for t in lg.get_texts()]
        assert "Sun" in labels and "Tau A" in labels
    # rows 1 and 2 do not move
    top = ["profile", "dm0", "waterfall", "dm_time", "dm_time_cbar"]
    assert _max_diff(gulp_boxes, boxes, top) == 0.0


def test_wrapped_titles_never_move_axes():
    base = sg.build_card("default")
    with _drawn(base) as (fig, _axes):
        ref = _boxes_px(fig)
    long_card = sg.build_card("storm", candname="INJECTION: inj_20260913_0017 "
                                                "(replay truncated below 420 MHz)")
    long_card["gulp"]["gulp"] = 123456789012
    with _drawn(long_card, subbands=256) as (fig, axes):
        boxes = _boxes_px(fig)
        r = fig.canvas.get_renderer()
        wrapped = 0
        for name in ("hist", "waterfall"):
            title = axes[name].title
            assert title.get_window_extent(r).width <= axes[name].get_window_extent(r).width
            wrapped += "\n" in title.get_text()
        assert wrapped == 2
    assert set(boxes) == set(ref)
    assert _max_diff(ref, boxes) == 0.0


@pytest.mark.parametrize("gulp", [True, False])
def test_no_ink_in_the_canvas_edge_strip(tmp_path, gulp):
    card = sg.build_card("default")
    if not gulp:
        card = {k: v for k, v in card.items() if k != "gulp"}
        card["context"] = {"window_s": 4.0, "members": [[0.0, 336, 29.5, 24.1, 3]]}
    arr, freqs, tsamp, t_rel = sg.synthetic_data(card["dm"])
    png = plotting.make_candidate_figure(arr, freqs, tsamp, t_rel, card, tmp_path / "e.png")
    img = mimage.imread(png)
    assert img.shape[:2] == (H_PX, W_PX)
    rgb = img[..., :3]
    for strip in (rgb[:3], rgb[-3:], rgb[:, :3], rgb[:, -3:]):
        assert np.all(strip == 1.0)


def test_title_lines():
    card = sg.build_card("default")
    with _drawn(card) as (fig, _axes):
        lines = fig._suptitle.get_text().split("\n")
    assert lines[0] == "SYNTHdefault   2026-09-18 07:51:03.208 UTC"
    assert lines[1].startswith("hella S/N = 24.1   DM = 29.50 pc cm")
    assert lines[1].endswith(f"width = {8 * sg.TSAMP_S * 1e3:.1f} ms")
    assert lines[2].startswith("beam 336: alt ")
    assert len(lines) == 3


def _sigproc_header(fields: list[tuple[str, str, object]]) -> bytes:
    """A SIGPROC header built by hand: (key, kind, value), kind in d/i/b/s."""
    import struct

    def s(text):
        return struct.pack("<i", len(text)) + text.encode()

    out = s("HEADER_START")
    for key, kind, value in fields:
        out += s(key)
        out += {"d": lambda v: struct.pack("<d", v), "i": lambda v: struct.pack("<i", v),
                "b": lambda v: struct.pack("<b", v), "s": s}[kind](value)
    return out + s("HEADER_END")


def test_fil_round_trip(tmp_path):
    """write_event_fil (casm_io) -> fil_reader returns the same beam and times;
    a hand-built header with the 1-byte ``signed`` key parses."""
    from datetime import datetime, timezone
    from types import SimpleNamespace

    from casm_t3 import event_archive, fil_reader

    nchan, ntime, tsamp = 64, 500, 1.048576e-3
    freqs = 484.375 - np.arange(nchan) * 0.030517578125
    data = np.random.default_rng(5).normal(100, 10, (nchan, ntime)).astype(np.float32)
    t0 = datetime(2026, 9, 18, 5, 27, 45, 123456, tzinfo=timezone.utc)
    header = SimpleNamespace(freqs_mhz=freqs, tsamp_s=tsamp, t0=t0)
    path = event_archive.write_event_fil(data, header, {"candname": "RT", "beam": 336},
                                         tmp_path / "rt.fil")
    back, hdr = fil_reader.read_fil(path)
    assert back.dtype == np.float32 and np.array_equal(back, data)
    assert np.allclose(hdr.freqs_mhz, freqs) and hdr.tsamp_s == pytest.approx(tsamp)
    assert abs((hdr.t0 - t0).total_seconds()) < 1e-3
    assert hdr.keys["source_name"] == "RT" and hdr.keys["ibeam"] == 336

    raw = np.arange(4 * 3, dtype=np.float32)            # 3 samples x 4 channels, time-major
    hand = tmp_path / "hand.fil"
    hand.write_bytes(_sigproc_header([
        ("source_name", "s", "HAND"), ("nchans", "i", 4), ("nbits", "i", 32),
        ("signed", "b", 1), ("tsamp", "d", 1e-3), ("fch1", "d", 480.0),
        ("foff", "d", -1.0), ("nifs", "i", 1), ("tstart", "d", 60000.5)]) + raw.tobytes())
    arr, h = fil_reader.read_fil(hand)
    assert h.keys["signed"] == 1 and h.nchans == 4 and h.foff_mhz == -1.0
    assert h.t0 == datetime(2023, 2, 25, 12, 0, tzinfo=timezone.utc)   # MJD 60000.5
    assert np.array_equal(arr, raw.reshape(3, 4).T)

    bad = tmp_path / "bad.fil"
    bad.write_bytes(b"\x08\x00\x00\x00NOTAFILE")
    with pytest.raises(ValueError):
        fil_reader.read_header(bad)


def _archived_event(tmp_path, dump_dir=None):
    """An events tree with <cand>/<cand>.fil and a card whose dump is gone."""
    from datetime import datetime, timezone
    from types import SimpleNamespace

    from casm_t3 import event_archive

    card = sg.build_card("default", candname="REPLOTcand")
    arr, freqs, tsamp, t_rel = sg.synthetic_data(card["dm"])
    t0 = datetime.fromisoformat(card["event_utc"]) - __import__("datetime").timedelta(seconds=t_rel)
    root = tmp_path / "EVENTS"
    event_archive.write_event_fil(arr, SimpleNamespace(freqs_mhz=freqs, tsamp_s=tsamp,
                                                       t0=t0.astimezone(timezone.utc)),
                                  card, root / "REPLOTcand" / "REPLOTcand.fil")
    if dump_dir is not None:
        card["dump_dir"] = str(dump_dir)
    card_path = tmp_path / "REPLOTcand.json.done"
    card_path.write_text(json.dumps(card))
    return card_path, root


def test_replot_falls_back_to_the_archived_fil(tmp_path, monkeypatch):
    from casm_t3.apps import replot

    # the working directory holds a .dada, which must never be picked up
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / "2026-09-18-05:27:45_0.000000.dada").write_bytes(b"not a dump")
    monkeypatch.chdir(cwd)
    for dump_dir in (tmp_path / "gone", None, ""):
        card_path, root = _archived_event(tmp_path, dump_dir)
        out = tmp_path / f"replot_{dump_dir!s:.4}.png"
        replot.main([str(card_path), "--out", str(out), "--events-root", str(root)])
        assert mimage.imread(out).shape[:2] == (H_PX, W_PX)

    # --events-root '' turns the fallback off
    with pytest.raises(SystemExit, match="fallback is off"):
        replot.main([str(card_path), "--out", str(tmp_path / "off.png"), "--events-root", ""])
    assert not (tmp_path / "off.png").exists()
    # an explicit --fil works with the fallback off
    replot.main([str(card_path), "--out", str(tmp_path / "fil.png"), "--events-root", "",
                 "--fil", str(root / "REPLOTcand" / "REPLOTcand.fil")])
    assert (tmp_path / "fil.png").exists()
