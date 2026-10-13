"""Synthetic dump files, hella record files and a trigger card for the event-folder tests.

Reduced dimensions keep them quick: 4 beams x 512 channels in the dump, hella
gulps of 4096 samples. The dump is two files of 4096 samples whose OBS_OFFSETs
straddle 10**10, so the unpadded file names sort the wrong way by name. It
starts 3880 samples before the end of gulp 595 and so spans gulps 595 (216
samples), 596 and 597. Global beam 5 is local beam 1 of the dump and the
second beam of the hella stream that starts at beam 4.

    python tests/synthetic_event.py /tmp/outdir     # one masked / raw figure pair
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

import synthetic_gulp as sg
from casm_io.candidates import hella_mask as hm

NCHAN, NBEAM, NINNER = 512, 4, 64
TSAMP_S = 1.048576e-3
BYTES_PER_SAMPLE = NBEAM * NCHAN * 2                  # fp16
UTC_START = "2026-10-13-18:22:38"
FILE_SAMP = 4096                                      # samples per dump file
SAMPLE0 = 2_441_000                                   # OBS_OFFSET = 9_998_336_000 bytes
GULP, MMIN, NBIN = 4096, 16, 256                      # hella gulp, bin, bins per gulp
BEAM0 = 4                                             # first global beam of the hella stream
GLOBAL_BEAM, LOCAL_BEAM = 5, 1
FIRST_GULP = SAMPLE0 // GULP                          # 595
STATIC_CHANS = 60                                     # flag2 of the records
RFI_CHAN, WHOLE_CHAN = 300, 200
BLOB_CHANS, BLOB_BINS = (100, 140), (10, 30)
GULP_SAMP = 8192                                      # T2's gulp, for the card's gulp block
EVENT_S = 3.0                                         # event time after the dump start
CHANBW = 93.75 / NCHAN
CANDNAME = "261013tsta01"
WEIGHTS_ID = "a7658dba9b34376a"


def dump_samples() -> int:
    return 2 * FILE_SAMP


def beam_data(dm: float, seed: int = 3) -> np.ndarray:
    """(nchan, ntime) float32: noise, a dispersed pulse at EVENT_S, RFI where hella flags."""
    rng = np.random.default_rng(seed)
    ntime = dump_samples()
    freqs = 484.375 - np.arange(NCHAN) * CHANBW
    data = rng.normal(100.0, 10.0, (NCHAN, ntime)) + 20.0 * np.sin(np.linspace(0, 3, NCHAN))[:, None]
    ev = int(round(EVENT_S / TSAMP_S))
    delays = np.round(4.148808e3 * dm * (freqs ** -2 - freqs.max() ** -2) / TSAMP_S).astype(int)
    for c in range(NCHAN):
        data[c, ev + delays[c]: ev + delays[c] + 8] += 8.0
    data[:STATIC_CHANS, :] = 3000.0                                    # the always-masked band
    g596 = slice(596 * GULP - SAMPLE0, 597 * GULP - SAMPLE0)
    burst = lambda p: rng.choice([0.0, 3000.0], size=GULP, p=[1 - p, p])   # noqa: E731
    data[RFI_CHAN, g596] += burst(0.10)                                # blanked for gulp 596
    data[WHOLE_CHAN, g596] += burst(0.05)
    b0 = 597 * GULP - SAMPLE0 + BLOB_BINS[0] * MMIN
    b1 = 597 * GULP - SAMPLE0 + BLOB_BINS[1] * MMIN
    data[BLOB_CHANS[0]:BLOB_CHANS[1], b0:b1] += 60.0                   # SumThreshold blob, gulp 597
    return data.astype(np.float32)


def write_dump_file(path: Path, obs_offset: int, data: np.ndarray, **extra) -> Path:
    """One .dada file; ``data`` is (NBEAM, NCHAN, nsamp) with nsamp a multiple of NINNER."""
    nsamp = data.shape[2]
    keys = {"HDR_SIZE": 4096, "UTC_START": UTC_START, "PICOSECONDS": 0,
            "OBS_OFFSET": obs_offset, "BYTES_PER_SECOND": f"{BYTES_PER_SAMPLE / TSAMP_S:.1f}",
            "NCHAN": NCHAN, "NBEAM": NBEAM, "NBIT": 16, "TSAMP": f"{TSAMP_S * 1e6:.6f}",
            "RESOLUTION": NBEAM * NCHAN * NINNER * 2, "FREQ_START": 484.375,
            "CHANBW": -CHANBW, "BFCORR_SUB_INCOH": 0}
    keys.update(extra)
    text = "\n".join(f"{k} {v}" for k, v in keys.items() if v is not None) + "\n"
    frames = data.reshape(NBEAM, NCHAN, nsamp // NINNER, NINNER).transpose(2, 0, 1, 3)
    with open(path, "wb") as f:
        f.write(text.encode().ljust(4096, b"\0"))
        frames.astype(np.float16).tofile(f)
    return path


def write_dump(dump_dir: Path, dm: float = 29.5, gap: bool = False, **extra) -> list[Path]:
    """Two contiguous dump files (or with a one-frame ``gap`` between them), named as real ones."""
    dump_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    full = rng.normal(100.0, 10.0, (NBEAM, NCHAN, dump_samples())).astype(np.float32)
    full[LOCAL_BEAM] = beam_data(dm)
    paths = []
    for i in range(2):
        off = (SAMPLE0 + i * FILE_SAMP + (NINNER if gap and i else 0)) * BYTES_PER_SAMPLE
        paths.append(write_dump_file(
            dump_dir / f"{UTC_START}_{off}.000000.dada", off,
            full[:, :, i * FILE_SAMP:(i + 1) * FILE_SAMP], **extra))
    return paths


def record_header(**kw) -> hm.FileHeader:
    base = dict(utc_start=UTC_START, obs_offset=0, nbeam=NBEAM, beam0=BEAM0, nchan=NCHAN,
                gulp_len=GULP, st_mmin=MMIN, st_nbin=NBIN, st_mmax=512, flag1=0,
                flag2=STATIC_CHANS, process_start_unix=1.0e9, pid=4242,
                binary_md5="ab" * 16, cfg_text="INPUT DADA\nFLAG2 60\n")
    base.update(kw)
    return hm.FileHeader(**base)


def _gulp_flags(h: hm.FileHeader, k: int) -> hm.BeamFlags:
    """Flags of the target beam in the k-th gulp of the dump (see the module docstring)."""
    mask = np.zeros((h.nchan, h.st_nbin), bool)
    reasons, whole = {}, []
    if k == 1:
        reasons, whole = {RFI_CHAN: 1}, [WHOLE_CHAN]
    elif k == 2:
        mask[BLOB_CHANS[0]:BLOB_CHANS[1], BLOB_BINS[0]:BLOB_BINS[1]] = True
    return hm.BeamFlags.from_dense(GLOBAL_BEAM, st_mask=mask, reasons=reasons, whole=whole)


def write_records(root: Path, ks=(0, 1, 2), name: str | None = None, part: bool = False,
                  **header_kw) -> Path:
    """Record file with the gulps FIRST_GULP + k for k in ``ks``; returns its path."""
    h = record_header(**header_kw)
    records = []
    for k in ks:
        beams = [hm.BeamFlags(h.beam0 + i) for i in range(h.nbeam)]
        beams[GLOBAL_BEAM - h.beam0] = _gulp_flags(h, k)
        records.append(dict(gulp_index=FIRST_GULP + k, first_sample=(FIRST_GULP + k) * GULP,
                            beams=beams))
    d = Path(root) / UTC_START
    d.mkdir(parents=True, exist_ok=True)
    first = FIRST_GULP + min(ks)
    # the writer's name: gulp is first_sample // gulp_len, pid the writing process
    name = name or f"hmask_b{BEAM0:03d}_g{first:07d}_p{h.pid}.hmr" + (".part" if part else "")
    return hm.write_record_file(d / name, h, records)


def event_utc() -> str:
    """event_utc of the card: EVENT_S after the first dump sample."""
    t0 = (datetime.strptime(UTC_START, "%Y-%m-%d-%H:%M:%S").replace(tzinfo=timezone.utc)
          + timedelta(seconds=SAMPLE0 * TSAMP_S))
    return (t0 + timedelta(seconds=EVENT_S)).isoformat(timespec="milliseconds")


def card(dump_dir: Path, candname: str = CANDNAME, **kw) -> dict:
    """A schema-1 style trigger card (with a gulp block) for the synthetic dump."""
    c = sg.build_card("default", candname=candname)
    c.update(beam=GLOBAL_BEAM, local_beam=LOCAL_BEAM, stream=1, event_utc=event_utc(),
             dm=29.5, snr=24.1, width=3, samp=ev_samp(),
             dump_dir=str(dump_dir), dump_utc_start="2026-10-13-19:05:17.600",
             dump_utc_stop="2026-10-13-19:05:26.200", t2d_commit="0123456789ab",
             trigger_reason="tier_B")
    c["pointings"].update(weights_id=WEIGHTS_ID, beam_fwhm_x_deg=19.326, beam_fwhm_y_deg=4.131)
    g = c["gulp"]
    g["weights_id"] = WEIGHTS_ID
    shift = ev_samp() - sg.EVENT_SAMP                  # move the block's gulp onto the dump
    g.update(utc_start=UTC_START, samp_lo=g["samp_lo"] + shift, samp_hi=g["samp_hi"] + shift)
    g["gulp"] = g["samp_lo"] // GULP_SAMP
    c.update(kw)
    return c


def ev_samp() -> int:
    return SAMPLE0 + int(round(EVENT_S / TSAMP_S))


def main() -> None:
    import tempfile

    from casm_t3 import plotting

    out = Path(sys.argv[1] if len(sys.argv) > 1 else "synthetic_event")
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        write_dump(tmp / "dump")
        write_records(tmp / "records")
        from casm_t3 import dump_reader
        header, data = dump_reader.read_beams(sorted((tmp / "dump").glob("*.dada")), [LOCAL_BEAM])
        win = hm.window_mask(tmp / "records", UTC_START, GLOBAL_BEAM, header.sample0,
                             data.shape[2], fch1=484.375, foff=-CHANBW)
        c = card(tmp / "dump")
        t_rel = (datetime.fromisoformat(c["event_utc"]) - header.t0).total_seconds()
        for name, kw in (("masked", dict(mask=win.combined, coverage=win.coverage)),
                         ("raw", dict(mask_label="before hella RFI flagging; IB subtraction off"))):
            print(plotting.make_candidate_figure(data[0], header.freqs_mhz, header.tsamp_s, t_rel, c,
                                                 out / f"{name}.png", **kw))


if __name__ == "__main__":
    main()
