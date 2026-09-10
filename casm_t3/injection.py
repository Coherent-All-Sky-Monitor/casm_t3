"""Rebuild the synthetic pulse an injection added to hella's searched stream.

Injected pulses never appear in a beam dump: the dump daemons read the d0
ring (post-corner-turn beamformer output) while injections are merged into
the assembled stream at d2, downstream of it (casm-wiki
``dump-stream-content.md``). To see an injection the way hella saw it, the
same pulse has to be put back into the dump offline.

The pulse is a faithful re-implementation of ``gen_filterbank`` in
``/home/casm/software/dev/make_noise_fil_with_frb_snr.py``, which is what
the injection bot runs with ``--no_noise``:

* per-channel delay ``4.148808 * DM * (1/f_GHz^2 - 1/fref_GHz^2)`` ms
  referenced to the top channel, ROUNDED to whole samples;
* a Gaussian in time, ``amp * exp(-0.5*((t-c0)/sigma_samp)^2)`` with
  ``sigma_samp = max(1, sigma_ms/(1000*tsamp_s))``, painted only within
  +-ceil(4*sigma_samp) samples of the channel's arrival;
* on a zero floor, then ``np.clip(block, 0, 255).astype(np.uint8)``, which
  TRUNCATES: the counts that reach the stream are ``floor`` of the Gaussian.

``convert_fil_to_dada.py`` casts those uint8 counts to float16 with no
rescaling, and the merge is additive, so the array returned here is exactly
what is added to the beam.

``tests/test_injection.py`` checks this against the generator itself.
"""

from __future__ import annotations

import numpy as np

# Constants as hard-coded in make_noise_fil_with_frb_snr.py. The channel
# width there is 0.03051757812 MHz, very slightly truncated relative to the
# exact 0.030517578125 the dump header carries; the difference is 1e-8 MHz
# and moves no channel by a whole sample, but the generator's own grid is
# used by default so the delays are bit-identical to the injected pulse.
K_MS = 4.148808
GEN_FCH1_MHZ = 484.375
GEN_CHAN_BW_MHZ = 0.03051757812
GEN_NCHAN = 3072
GEN_TSAMP_S = 0.001048576
GEN_CLIP_MAX = 255.0


def generator_freqs_mhz(nchan: int = GEN_NCHAN) -> np.ndarray:
    """The channel centre frequencies the pulse generator uses (descending)."""
    return GEN_FCH1_MHZ - np.arange(nchan) * GEN_CHAN_BW_MHZ


def channel_delay_samples(freqs_mhz: np.ndarray, dm: float, tsamp_s: float,
                          fref_mhz: float | None = None) -> np.ndarray:
    """Per-channel dispersion delay in whole samples, generator-style."""
    f_ghz = np.asarray(freqs_mhz, dtype=np.float64) / 1000.0
    fref_ghz = (GEN_FCH1_MHZ if fref_mhz is None else fref_mhz) / 1000.0
    delay_ms = K_MS * dm * (1.0 / f_ghz**2 - 1.0 / fref_ghz**2)
    return np.round(delay_ms / (tsamp_s * 1000.0)).astype(np.int64)


def sigma_samples(sigma_ms: float, tsamp_s: float) -> float:
    """Gaussian sigma in samples, with the generator's one-sample floor."""
    return max(1.0, sigma_ms / (1000.0 * tsamp_s))


def injected_pulse(nchan: int, ntime: int, dm: float, amp: float, sigma_ms: float,
                   c0_samp: int, tsamp_s: float = GEN_TSAMP_S,
                   freqs_mhz: np.ndarray | None = None,
                   quantise: bool = True) -> np.ndarray:
    """The (nchan, ntime) float32 pulse an injection adds to the stream.

    ``c0_samp`` is the arrival sample of the top channel (the time the
    cluster's ``event_utc`` refers to). Channels whose delayed arrival falls
    outside the array simply contribute nothing.
    """
    if freqs_mhz is None:
        freqs_mhz = generator_freqs_mhz(nchan)
    freqs_mhz = np.asarray(freqs_mhz, dtype=np.float64)
    if freqs_mhz.size != nchan:
        raise ValueError(f"freqs_mhz has {freqs_mhz.size} channels, expected {nchan}")

    delays = channel_delay_samples(freqs_mhz, dm, tsamp_s)
    sigma = sigma_samples(sigma_ms, tsamp_s)
    halfwin = int(np.ceil(4.0 * sigma))
    out = np.zeros((nchan, ntime), dtype=np.float32)

    for ch in range(nchan):
        c0 = int(c0_samp) + int(delays[ch])
        lo, hi = max(0, c0 - halfwin), min(ntime - 1, c0 + halfwin)
        if lo > hi:
            continue
        t = np.arange(lo, hi + 1)
        g = amp * np.exp(-0.5 * ((t - c0) / sigma) ** 2, dtype=np.float64)
        out[ch, lo:hi + 1] = g.astype(np.float32)

    if quantise:
        # exactly the generator's write step: clip, then uint8 truncation
        out = np.clip(out, 0.0, GEN_CLIP_MAX).astype(np.uint8).astype(np.float32)
    return out


def fwhm_ms(sigma_ms: float) -> float:
    """FWHM of a Gaussian of the given sigma."""
    return float(2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma_ms)
