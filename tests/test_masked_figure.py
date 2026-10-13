"""Candidate figure with hella's flag mask: drawn from the unflagged cells only.

Truth comes from the synthetic dump (tests/synthetic_mask.py: a known pulse, a
strong narrow-band burster, a time-frequency blob, a mask that flags exactly the
RFI), from the card, or from what matplotlib drew. The raw figure is checked
against frozen copies of the pre-mask numerics (tests/legacy_numerics.py).
"""
import contextlib
import re

import matplotlib.image as mimage
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib import colors as mcolors
from matplotlib.text import Text

import legacy_numerics as old
import synthetic_gulp as sg
import synthetic_mask as sm
from casm_t3 import plotting, single_pulse

W_PX, H_PX = 1510, 1517
NCHAN = 1024                       # a third of the band: the same layout, a third of the time


def _dump(**kw):
    kw.setdefault("nchan", NCHAN)
    kw.setdefault("pulse_amp", 2.5)
    return sm.make_dump(**kw)


@contextlib.contextmanager
def _drawn(d, card=None, **kw):
    """Draw the v3 figure for a synthetic dump; yields (fig, axes, info)."""
    card = card or sm.card_for(d)
    with plt.rc_context(plotting.FIG_RC):
        fig = plotting.new_canvas()
        try:
            info = plotting.draw_candidate_figure(fig, d["data"], d["freqs"], d["tsamp"], d["t_rel"],
                                                  card, **kw)
            yield fig, info["axes"], info
        finally:
            plt.close(fig)


def _trace(ax):
    """(time, S/N) of the first line on a profile axes."""
    line = ax.lines[0]
    return np.asarray(line.get_xdata(), dtype=float), np.asarray(line.get_ydata(), dtype=float)


def _event_peak(ax, width, tsamp):
    t, y = _trace(ax)
    near = np.abs(t) <= 2 * width * tsamp
    return float(np.nanmax(y[near]))


def _corner_text(ax):
    return " ".join(t.get_text() for t in ax.texts if t.get_gid() == "corner")


def _boxes_px(fig):
    r = fig.canvas.get_renderer()
    out = {}
    for ax in fig.axes:
        for a in [ax, *getattr(ax, "child_axes", [])]:
            b = a.get_window_extent(r)
            out[a.get_label()] = (b.x0, b.y0, b.x1, b.y1)
    return out


# --- the raw path ---------------------------------------------------------------------------

def test_raw_figure_arrays_equal_the_pre_mask_numerics():
    """Without a mask every drawn array is what the old functions give."""
    d = _dump()
    card = sm.card_for(d)
    width, dm, tsamp, freqs = 2 ** card["width"], card["dm"], d["tsamp"], d["freqs"]
    norm = old.old_normalise(d["data"])
    dedis = old.old_dedisperse(norm, dm, freqs, tsamp)
    ntime = dedis.shape[1]
    n_ok = max(width * 4, ntime - plotting.wrap_samples(dm, freqs, tsamp))
    prof = old.old_profile_snr(dedis.mean(axis=0), width)
    with _drawn(d) as (fig, axes, info):
        assert "mask" not in info
        assert np.array_equal(np.asarray(axes["profile"].lines[0].get_ydata()), prof[:n_ok])
        assert np.array_equal(np.asarray(axes["dm0"].lines[0].get_ydata()),
                              old.old_profile_snr(norm.mean(axis=0), width))
        # the waterfall: block means of the dedispersed array on the grid phased to the profile peak
        i_event = int(round(d["t_rel"] / tsamp))
        lo, hi = max(0, i_event - 2 * width), min(ntime, i_event + 2 * width + 1)
        i_peak = lo + int(np.argmax(prof[lo:hi]))
        i_on = int(np.clip(i_peak - width + 1 + (width - 1) // 2, 0, max(0, n_ok - width)))
        wf = np.asarray(axes["waterfall"].images[0].get_array())
        ff = d["data"].shape[0] // wf.shape[0]
        assert np.array_equal(wf, old.old_downsample(dedis[:, i_on % width:], ff, width))
        # DM-time: the old per-trial transform, max-pooled to the drawn column width
        f_dmt = plotting._dmt_plan(dm, width, freqs, tsamp)
        small = old.old_downsample(norm, f_dmt, 1)
        f_small = plotting._block_mean_freqs(freqs, f_dmt)[: small.shape[0]]
        dmt = old.old_dm_time(small, f_small, tsamp, single_pulse.dm_grid(dm), width)
        tfac = plotting._dmt_tfactor(width, axes["dm_time"].get_xlim(), tsamp)
        assert np.array_equal(np.asarray(axes["dm_time"].images[0].get_array()),
                              old.old_downsample_max(dmt, tfac))


def test_raw_figure_with_only_a_label_is_the_raw_figure_plus_a_title_note():
    d = _dump()
    label = "before hella RFI flagging; IB subtraction off"
    with _drawn(d) as (fig_a, axes_a, _):
        text_a = fig_a._suptitle.get_text()
        y_a = np.asarray(axes_a["dm0"].lines[0].get_ydata())
        box_a = _boxes_px(fig_a)
    with _drawn(d, mask_label=label) as (fig_b, axes_b, info_b):
        text_b = fig_b._suptitle.get_text()
        assert np.array_equal(np.asarray(axes_b["dm0"].lines[0].get_ydata()), y_a)
        assert "mask" not in info_b and _boxes_px(fig_b) == box_a
    la, lb = text_a.split("\n"), text_b.split("\n")
    assert lb[0] == la[0] and lb[2] == la[2] and lb[1] == la[1] + "   " + label


# --- RFI excluded, pulse recovered ---------------------------------------------------------

def test_mask_removes_rfi_from_the_dm0_series_and_keeps_the_pulse():
    d = _dump()
    card = sm.card_for(d)
    width, tsamp = 2 ** card["width"], d["tsamp"]
    with _drawn(d) as (_f, axes, _i):
        raw_dm0 = np.asarray(axes["dm0"].lines[0].get_ydata())
        raw_peak = _event_peak(axes["profile"], width, tsamp)
    assert np.nanmax(raw_dm0) > 50.0                       # the RFI dominates the raw DM = 0 series
    assert raw_peak > 2 * 22.0 * 0.9                       # and inflates the raw event window
    with _drawn(d, mask=d["mask"]) as (_f, axes, info):
        dm0 = np.asarray(axes["dm0"].lines[0].get_ydata())
        peak = _event_peak(axes["profile"], width, tsamp)
        text = _corner_text(axes["dm0"])
    assert np.isfinite(dm0).all()
    assert np.nanmax(dm0) < 6.0                            # noise only: no RFI, no pulse (dispersed away)
    assert np.nanmax(dm0) < 0.1 * np.nanmax(raw_dm0)
    assert "off scale" not in text
    assert peak > 15.0                                     # the pulse is there

    # the same dump without RFI, flagged only in the static band, is the reference: the masked
    # figure keeps every pulse cell the blob did not take, so S/N falls by sqrt(valid fraction)
    clean = _dump(rfi=False)
    assert np.array_equal(clean["data"][:, :50], d["data"][:, :50])          # same noise draw
    with _drawn(clean, mask=clean["mask"]) as (_f, axes, _i):
        ref = _event_peak(axes["profile"], width, tsamp)
    delays = np.round(4.148808e3 * d["dm"] * (d["freqs"] ** -2 - d["freqs"].max() ** -2) / tsamp).astype(int)
    pulse_ok = lambda m: sum(int((~m[c, sm.EVENT_SAMP + delays[c]:sm.EVENT_SAMP + delays[c] + 8]).sum())
                             for c in range(len(delays)))                      # noqa: E731
    expect = ref * np.sqrt(pulse_ok(d["mask"]) / pulse_ok(clean["mask"]))
    assert expect < 0.97 * ref                              # the blob does take pulse cells
    assert peak == pytest.approx(expect, rel=0.06)
    assert peak == pytest.approx(ref, rel=0.12)


def test_masked_figure_reports_what_it_used():
    d = _dump()
    cov = np.ones(d["data"].shape[1], dtype=bool)
    cov[1300:1700] = False
    with _drawn(d, mask=d["mask"], coverage=cov) as (_f, _a, info):
        m = info["mask"]
    live = int((~d["mask"][:, cov]).all(axis=1).sum())
    assert m["coverage_frac"] == pytest.approx(cov.mean())
    assert m["n_live_channels"] >= live and m["n_live_channels"] <= NCHAN - int(472 * NCHAN / 3072)
    assert m["min_nvalid"] == single_pulse.min_nvalid_for(m["n_live_channels"])
    assert 0.15 < m["flagged_frac"] < 0.25                  # the static band (15%) plus the RFI cells
    assert m["tag"] == "hella RFI flags applied (partial coverage)"


def test_all_flagged_band_profile_is_unavailable_not_zero():
    d = _dump()
    mask = d["mask"].copy()
    mask[:, 3000:3100] = True                               # every channel flagged for 100 samples
    with _drawn(d, mask=mask) as (_f, axes, _i):
        _, y = _trace(axes["dm0"])
    # a boxcar window with half or more of its samples available is evaluated, so the NaN
    # run is the hole minus one window's reach at each end
    assert np.isnan(y[3001:3100]).all()
    assert np.isfinite(y[:2990]).all() and np.isfinite(y[3110:]).all()


# --- no flag record -------------------------------------------------------------------------

def test_coverage_gap_gives_nan_and_a_hatched_span_not_zeros():
    d = _dump()
    cov = np.ones(d["data"].shape[1], dtype=bool)
    cov[1300:1700] = False                                  # -0.78 to -0.37 s, inside the plotted window
    with _drawn(d, mask=d["mask"], coverage=cov) as (fig, axes, _i):
        t, y = _trace(axes["dm0"])
        # NaN over the gap minus the half-window reach at each end; never a zero
        assert np.isnan(y[1301:1700]).all()
        assert np.isfinite(y[:1290]).all() and np.isfinite(y[1710:]).all()
        assert not np.any(y[1300:1700] == 0.0)
        for name in ("dm0", "waterfall"):
            assert any(c.get_gid() == "uncovered" for c in axes[name].get_children()), name
        # in the waterfall the gap is slanted by the dispersion sweep and lighter than flagged cells
        wf = np.asarray(axes["waterfall"].images[0].get_array())
        assert np.isnan(wf).any()
        keys = [t.get_text() for lg in axes["waterfall"].get_children()
                if lg.__class__.__name__ == "Legend" for t in lg.get_texts()]
        assert keys == ["flagged by hella", "no flag record"]
        # the dedispersed profile only has a gap where (nearly) the whole band is unknown
        _, y_dd = _trace(axes["profile"])
        assert np.isfinite(y_dd[1000:]).all()


def test_flagged_cells_are_neutral_and_distinct_from_no_record():
    d = _dump()
    cov = np.ones(d["data"].shape[1], dtype=bool)
    with _drawn(d, mask=d["mask"], coverage=cov) as (fig, axes, info):
        im = axes["waterfall"].images[0]
        assert np.allclose(mcolors.to_rgba(plotting.FLAGGED_COLOR), im.get_cmap().get_bad())
        assert np.isnan(np.asarray(im.get_array())).any()                       # flagged cells are NaN
        assert not any(c.get_gid() == "uncovered" for c in axes["waterfall"].get_children())
        keys = [t.get_text() for lg in axes["waterfall"].get_children()
                if lg.__class__.__name__ == "Legend" for t in lg.get_texts()]
        assert keys == ["flagged by hella"]
        assert info["mask"]["tag"] == "hella RFI flags applied"
    assert plotting.FLAGGED_COLOR != plotting.UNCOVERED_COLOR


def test_event_inside_a_coverage_gap_still_renders():
    d = _dump()
    cov = np.ones(d["data"].shape[1], dtype=bool)
    cov[1800:2400] = False                                  # the event and its sweep
    with _drawn(d, mask=d["mask"], coverage=cov) as (fig, axes, _i):
        text = _corner_text(axes["profile"])
    assert "peak S/N near event" in text


def test_a_fully_flagged_dump_renders_and_says_n_a():
    d = _dump()
    with _drawn(d, mask=np.ones(d["data"].shape, dtype=bool)) as (fig, axes, info):
        assert "n/a" in _corner_text(axes["profile"]) and "n/a" in _corner_text(axes["dm0"])
        assert np.isnan(np.asarray(axes["waterfall"].images[0].get_array())).all()
        assert info["mask"]["n_live_channels"] == 0 and info["mask"]["flagged_frac"] == 1.0
    cov = np.zeros(d["data"].shape[1], dtype=bool)          # nothing covered at all
    with _drawn(d, mask=d["mask"], coverage=cov) as (fig, axes, info):
        assert "n/a" in _corner_text(axes["profile"])
        assert info["mask"]["coverage_frac"] == 0.0 and np.isnan(info["mask"]["flagged_frac"])


# --- title, geometry, bottom row ---------------------------------------------------------------

def test_titles_say_what_was_applied_and_nothing_stronger():
    d = _dump()
    cov = np.ones(d["data"].shape[1], dtype=bool)
    part = cov.copy()
    part[1300:1700] = False
    raw_label = "before hella RFI flagging; IB subtraction off"
    cases = [(dict(), None),
             (dict(mask=d["mask"]), "hella RFI flags applied"),
             (dict(mask=d["mask"], coverage=cov), "hella RFI flags applied"),
             (dict(mask=d["mask"], coverage=part), "hella RFI flags applied (partial coverage)"),
             (dict(mask=d["mask"], mask_label="hella RFI flags applied (replay)"),
              "hella RFI flags applied (replay)"),
             (dict(mask_label=raw_label), raw_label)]
    for kw, suffix in cases:
        with _drawn(d, **kw) as (fig, axes, _i):
            lines = fig._suptitle.get_text().split("\n")
            assert len(lines) == 3
            assert lines[1].endswith(suffix) if suffix else lines[1].endswith("ms")
            b = fig._suptitle.get_window_extent(fig.canvas.get_renderer())
            assert b.x0 >= 0 and b.x1 <= W_PX
            everything = [t.get_text() for t in fig.findobj(Text)]
            assert not any("exactly what hella saw" in t for t in everything)
    with _drawn(d, mask=d["mask"], coverage=part, mask_label="custom") as (fig, _a, _i):
        assert fig._suptitle.get_text().split("\n")[1].endswith("custom (partial coverage)")


def test_mask_does_not_move_a_panel_and_leaves_the_bottom_row_alone(tmp_path):
    card = sg.build_card("default")
    arr, freqs, tsamp, t_rel = sg.synthetic_data(card["dm"], nchan=512)
    mask = np.zeros(arr.shape, dtype=bool)
    mask[:80] = True
    mask[200:230, 2100:2400] = True
    cov = np.ones(arr.shape[1], dtype=bool)
    cov[500:800] = False
    d = {"data": arr, "freqs": freqs, "tsamp": tsamp, "t_rel": t_rel}
    with _drawn(d, card=card) as (fig, _a, _i):
        ref = _boxes_px(fig)
    with _drawn(d, card=card, mask=mask, coverage=cov) as (fig, _a, _i):
        boxes = _boxes_px(fig)
    assert set(boxes) == set(ref)
    assert max(abs(a - b) for k in ref for a, b in zip(ref[k], boxes[k])) == 0.0

    raw = plotting.make_candidate_figure(arr, freqs, tsamp, t_rel, card, tmp_path / "raw.png",
                                         mask_label="before hella RFI flagging; IB subtraction off")
    masked = plotting.make_candidate_figure(arr, freqs, tsamp, t_rel, card, tmp_path / "masked.png",
                                            mask=mask, coverage=cov)
    a, b = mimage.imread(raw), mimage.imread(masked)
    assert a.shape[:2] == b.shape[:2] == (H_PX, W_PX)
    row3_top = int((1.0 - plotting.ROW3[1]) * H_PX) - 70       # the histogram's two-line title
    assert np.array_equal(a[row3_top:], b[row3_top:])           # T1 trials and sky: unchanged
    assert not np.array_equal(a[:row3_top], b[:row3_top])


def test_png_renders_for_the_live_bottom_row_too(tmp_path):
    d = _dump()
    card = sm.card_for(d)                                       # no gulp block
    for name, kw in (("masked", dict(mask=d["mask"])), ("raw", dict(mask_label="raw")), ("plain", {})):
        png = plotting.make_candidate_figure(d["data"], d["freqs"], d["tsamp"], d["t_rel"], card,
                                             tmp_path / f"{name}.png", **kw)
        assert mimage.imread(png).shape[:2] == (H_PX, W_PX)


# --- arguments ------------------------------------------------------------------------------------

def test_mask_arguments_are_validated(tmp_path):
    d = _dump()
    card = sm.card_for(d)
    args = (d["data"], d["freqs"], d["tsamp"], d["t_rel"], card, tmp_path / "x.png")
    with pytest.raises(ValueError, match="mask shape"):
        plotting.make_candidate_figure(*args, mask=d["mask"][:, :-1])
    with pytest.raises(ValueError, match="coverage shape"):
        plotting.make_candidate_figure(*args, mask=d["mask"], coverage=np.ones(10, bool))
    with pytest.raises(ValueError, match="coverage needs a mask"):
        plotting.make_candidate_figure(*args, coverage=np.ones(d["data"].shape[1], bool))
    with pytest.raises(ValueError, match="exclusive"):
        plotting.make_candidate_figure(*args, mask=d["mask"], clean="hella")
    with pytest.raises(ValueError, match="v2 layout"):
        plotting.make_candidate_figure(*args, mask=d["mask"], layout="legacy")
    with pytest.raises(ValueError, match="v2 layout"):
        plotting.make_candidate_figure(*args, mask_label="raw", layout="legacy")
    assert not (tmp_path / "x.png").exists()
    assert not plt.get_fignums()                                # a refused render leaves no figure open


def test_clean_hella_replica_still_draws_without_a_mask(tmp_path):
    """The offline replica path (t3-replay-injection) is separate from the recorded mask."""
    d = _dump(nchan=256, rfi=False)
    card = sm.card_for(d)
    png = plotting.make_candidate_figure(d["data"], d["freqs"], d["tsamp"], d["t_rel"], card,
                                         tmp_path / "clean.png", clean="hella")
    assert mimage.imread(png).shape[:2] == (H_PX, W_PX)
