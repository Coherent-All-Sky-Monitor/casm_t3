"""Synthetic dump with a known pulse, RFI and a hella-style flag mask.

noise (sigma 10 counts around a per-channel bandpass) + one dispersed pulse +
a strong narrow-band burster + a time-frequency blob. ``mask`` flags exactly
the RFI cells (plus the always-flagged top band, as hella does), so a figure
drawn with it should lose only the flagged fraction of the band.

    python tests/synthetic_mask.py /tmp/outdir     # renders a masked / raw pair
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TSAMP_S = 1.048576e-3
CHAN_MHZ = 0.030517578125
STATIC_BAND = (0, 472)             # channels hella always masks (484.4 to 470.0 MHz)
BURST_CHAN = 1500
BLOB_CHANS = (800, 1100)
BLOB_SAMPLES = (2100, 2400)
EVENT_SAMP = 2048


def freqs_mhz(nchan: int = 3072) -> np.ndarray:
    return 484.375 - np.arange(nchan) * (CHAN_MHZ * 3072 / nchan)


def make_dump(dm: float = 100.0, nchan: int = 3072, ntime: int = 8192, width_samp: int = 8,
              pulse_amp: float = 1.7, rfi: bool = True, seed: int = 5):
    """dict: data (nchan, ntime) float32, freqs, tsamp, t_rel, mask (True = flagged), card fields."""
    rng = np.random.default_rng(seed)
    freqs = freqs_mhz(nchan)
    bandpass = 100.0 + 20.0 * np.sin(np.linspace(0.0, 3.0, nchan))[:, None]
    data = rng.normal(0.0, 10.0, (nchan, ntime)) + bandpass
    delays = np.round(4.148808e3 * dm * (freqs ** -2 - freqs.max() ** -2) / TSAMP_S).astype(int)
    for c in range(nchan):
        lo = EVENT_SAMP + delays[c]
        data[c, lo: lo + width_samp] += pulse_amp
    flagged = np.zeros(data.shape, dtype=bool)
    scale = nchan / 3072.0
    flagged[int(STATIC_BAND[0] * scale):int(STATIC_BAND[1] * scale), :] = True
    if rfi:
        c = int(BURST_CHAN * scale)
        for t0 in rng.choice(np.arange(200, ntime - 200), size=20, replace=False):
            data[c, t0:t0 + 30] += 20000.0                     # ~2000 sigma bursts
            flagged[c, max(0, t0 - 4):t0 + 34] = True          # hella flags them, grown a little
        b0, b1 = int(BLOB_CHANS[0] * scale), int(BLOB_CHANS[1] * scale)
        data[b0:b1, BLOB_SAMPLES[0]:BLOB_SAMPLES[1]] += 80.0   # 8 sigma, 300 channels
        flagged[b0:b1, BLOB_SAMPLES[0] - 16:BLOB_SAMPLES[1] + 16] = True
    return {"data": data.astype(np.float32), "freqs": freqs, "tsamp": TSAMP_S,
            "t_rel": EVENT_SAMP * TSAMP_S, "mask": flagged, "dm": dm, "width_samp": width_samp}


def card_for(dump: dict, candname: str = "SYNTHmask") -> dict:
    """A minimal trigger card (no gulp block: the live bottom row) for ``dump``."""
    from datetime import datetime, timedelta, timezone

    t = datetime(2026, 10, 13, 20, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=dump["t_rel"])
    return {"candname": candname, "source": "blind", "event_utc": t.isoformat(timespec="milliseconds"),
            "beam": 336, "snr": 24.0, "dm": float(dump["dm"]),
            "width": int(np.log2(dump["width_samp"])),
            "sky": {"alt_deg": 40.0, "az_deg": 111.0, "ra_deg": 150.0, "dec_deg": -5.0},
            "context": {"window_s": 4.0, "members": []}}


def main() -> None:
    from casm_t3 import plotting

    out = Path(sys.argv[1] if len(sys.argv) > 1 else "synthetic_mask")
    out.mkdir(parents=True, exist_ok=True)
    d = make_dump()
    card = card_for(d)
    cov = np.ones(d["data"].shape[1], dtype=bool)
    cov[1000:1900] = False                                     # 0.9 s with no flag record, before the pulse
    for name, kw in (("masked", dict(mask=d["mask"], coverage=cov)),
                     ("raw", dict(mask_label="before hella RFI flagging; IB subtraction off"))):
        print(plotting.make_candidate_figure(d["data"], d["freqs"], d["tsamp"], d["t_rel"], card,
                                             out / f"{name}.png", **kw))


if __name__ == "__main__":
    main()
