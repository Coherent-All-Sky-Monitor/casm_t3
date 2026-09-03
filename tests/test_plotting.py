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
