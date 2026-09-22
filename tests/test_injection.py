"""The replayed pulse must be the pulse the injection bot generated.

The reference is the generator itself: ``gen_filterbank`` from
/home/casm/software/dev/make_noise_fil_with_frb_snr.py run with no noise,
read back with casm_io's FilterbankFile and compared sample for sample.
"""
import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from casm_t3 import injection
from casm_t3.apps import replay_injection

GENERATOR = Path("/home/casm/software/dev/make_noise_fil_with_frb_snr.py")


def _load_generator():
    spec = importlib.util.spec_from_file_location("mnfwfs", GENERATOR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not GENERATOR.exists(), reason="pulse generator not on this node")
def test_pulse_matches_generator(tmp_path):
    from casm_io.filterbank import FilterbankFile

    gen = _load_generator()
    nsamp, dm, amp, sigma_ms = 2048, 100.0, 8.0, 2.0
    fil = gen.gen_filterbank(duration_samps=nsamp, DM=dm, pulse_amp_counts=amp,
                             pulse_sigma_ms=sigma_ms, with_noise=False,
                             output_fname_base=str(tmp_path / "ref.fil"))
    ref = np.asarray(FilterbankFile(str(fil), verbose=False).data, dtype=np.float32)
    ref = ref.reshape(nsamp, injection.GEN_NCHAN).T          # (nchan, ntime)

    mine = injection.injected_pulse(injection.GEN_NCHAN, nsamp, dm, amp, sigma_ms,
                                    c0_samp=nsamp // 2)
    assert mine.shape == ref.shape
    assert np.array_equal(mine, ref)
    assert mine.max() == np.floor(amp)


def test_pulse_geometry():
    nchan, ntime = 3072, 4096
    p = injection.injected_pulse(nchan, ntime, 300.0, 8.0, 5.0, c0_samp=1000)
    # top channel peaks at c0, every channel later, monotonically
    peaks = p.argmax(axis=1)
    assert peaks[0] == 1000
    assert np.all(np.diff(peaks) >= 0)
    delays = injection.channel_delay_samples(injection.generator_freqs_mhz(nchan),
                                             300.0, injection.GEN_TSAMP_S)
    assert peaks[-1] == 1000 + delays[-1]
    # unquantised is a Gaussian of the requested sigma
    raw = injection.injected_pulse(8, 200, 0.0, 8.0, 5.0, c0_samp=100, quantise=False)
    sigma = injection.sigma_samples(5.0, injection.GEN_TSAMP_S)
    assert raw[0, 100] == pytest.approx(8.0)
    assert raw[0, 100 + int(round(sigma))] == pytest.approx(8.0 * np.exp(-0.5), rel=0.05)


def test_replay_renders_and_recovers_snr(tmp_path):
    """End to end on a small synthetic dump: the added pulse must be visible."""
    nchan, ntime, tsamp = 3072, 3072, injection.GEN_TSAMP_S
    freqs = 484.375 - np.arange(nchan) * 0.030517578125
    rng = np.random.default_rng(3)
    data = rng.normal(100.0, 20.0, (nchan, ntime)).astype(np.float32)
    dm, c0 = 100.0, 700
    data += injection.injected_pulse(nchan, ntime, dm, 8.0, 2.0, c0)

    inj = {"id": 1, "dm": dm, "amp": 8.0, "sigma_ms": 2.0, "est_snr": 19.7,
           "rec_snr": None, "rec_dm": None, "stream": 1, "beam": 90}
    card = replay_injection.build_card(inj, None, datetime(2026, 9, 3, tzinfo=timezone.utc),
                                       beam=90, local_beam=26, stream=1, dm=dm,
                                       width=2, snr=19.7, members=[], window_s=4.0,
                                       registry=None)
    assert card["candname"] == "inj1"
    # no replay parameters in the source (it would reach the figure title);
    # they are in the one-line summary for the Slack thread
    assert card["source"] == "injection"
    summary = card["injection"]["summary"]
    assert summary.startswith("injected DM 100.0, FWHM 4.7 ms, amp 8 counts, S/N 19.7")
    assert summary.endswith("; not recovered by hella")
    out = replay_injection.plotting.make_candidate_figure(
        data, freqs, tsamp, c0 * tsamp, card, tmp_path / "inj.png", layout="v2")
    assert Path(out).stat().st_size > 50_000

    snr = replay_injection.measure_snr(data, freqs, tsamp, dm, 2, c0 * tsamp)
    assert snr > 8.0


def test_select_dump_files_needs_a_dada(tmp_path):
    with pytest.raises(SystemExit):
        replay_injection.select_dump_files(tmp_path, None)
    f = tmp_path / "x.dada"
    f.write_bytes(b"")
    assert replay_injection.select_dump_files(f, None) == [f]


def test_width_index_from_sigma():
    # 5 ms sigma -> 11.8 ms FWHM -> 11.2 samples -> nearest boxcar 2^3
    # (hella clustered id 661 at width 4; the fallback is only used when no
    # matched cluster exists, and one boxcar step costs little S/N)
    assert replay_injection._width_index(5.0, injection.GEN_TSAMP_S) == 3
    assert replay_injection._width_index(0.5, injection.GEN_TSAMP_S) == 0
