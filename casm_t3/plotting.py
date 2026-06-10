"""Candidate diagnostic figure.

Layout follows the DSA-110 candidate plotter and transientX conventions:

    dedispersed profile      | DM=0 timestream
    dedispersed freq-time    | DM-time bowtie
    beam vs time (T1 members)| per-beam S/N near the event

The two beam panels are driven by the T1 "context" candidates that
casm_t2's source watcher embeds in the trigger card: every candidate from
every beam within a few seconds of the event. RFI lights up many beams at
unrelated DMs; a real pulse stays compact in beam and DM.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from . import single_pulse

NBEAM_TOTAL = 512


def _block_mean_freqs(freqs_mhz: np.ndarray, ffactor: int) -> np.ndarray:
    n = (freqs_mhz.size // ffactor) * ffactor
    return freqs_mhz[:n].reshape(-1, ffactor).mean(axis=1)


def _beam_panels(ax_bt, ax_bs, card: dict, fig) -> None:
    """Beam-time scatter coloured by DM, and per-beam S/N around the event."""
    ctx = card.get("context") or {}
    members = ctx.get("members") or []
    beam = int(card["beam"])

    if not members:
        for ax in (ax_bt, ax_bs):
            ax.text(0.5, 0.5, "no context candidates", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="0.4")

    if members:
        m = np.asarray(members, dtype=float)  # columns: dt, beam, dm, snr, width
        dt, mbeam, mdm, msnr = m[:, 0], m[:, 1], m[:, 2], m[:, 3]

        sc = ax_bt.scatter(dt, mbeam, c=mdm, s=4 + 2 * np.clip(msnr - 10, 0, 20),
                           cmap="plasma", vmin=0, vmax=max(50.0, 1.5 * card["dm"]),
                           alpha=0.7, linewidths=0)
        fig.colorbar(sc, ax=ax_bt, pad=0.01, label=r"DM (pc cm$^{-3}$)")
        for edge in range(64, NBEAM_TOTAL, 64):
            ax_bt.axhline(edge, color="0.85", lw=0.4, zorder=0)

        # Localization profile: best S/N per beam close to the event time.
        near = m[np.abs(dt) <= 0.5]
        if near.size:
            beams_near = near[:, 1].astype(int)
            best = {}
            for b, s in zip(beams_near, near[:, 3]):
                best[b] = max(best.get(b, 0.0), s)
            bb = np.array(sorted(best))
            ss = np.array([best[b] for b in bb])
            markerline, stemlines, _ = ax_bs.stem(bb, ss)
            plt.setp(markerline, markersize=3)
            plt.setp(stemlines, linewidth=0.8)
            ax_bs.set_xlim(beam - 32.5, beam + 32.5)

    ax_bt.scatter([0], [beam], marker="s", s=90, facecolors="none",
                  edgecolors="red", linewidths=1.2, zorder=5)
    ax_bt.set_ylim(-5, NBEAM_TOTAL + 4)
    ax_bt.set_xlabel("Time - event (s)")
    ax_bt.set_ylabel("Beam")

    ax_bs.axvline(beam, color="red", lw=0.8, alpha=0.5)
    ax_bs.set_xlabel("Beam")
    ax_bs.set_ylabel("max S/N (|t| < 0.5 s)")


def make_candidate_figure(data: np.ndarray, freqs_mhz: np.ndarray, tsamp_s: float,
                          t_rel_event_s: float, card: dict, out_png: str | Path,
                          ffactor: int = 8) -> Path:
    """Render the candidate plot.

    Parameters
    ----------
    data : (nchan, ntime) float32, raw (dispersed) cutout for the detection beam.
    t_rel_event_s : candidate time (top-of-band arrival) relative to the
        first sample of ``data``.
    card : trigger-card dict (candname/source/snr/dm/width/beam/event_utc,
        optional context member list).
    """
    dm, width = float(card["dm"]), int(card["width"])
    norm = single_pulse.normalise(data)
    dedis = single_pulse.dedisperse(norm, dm, freqs_mhz, tsamp_s)

    tfactor = max(1, width // 2)
    wf_dd = single_pulse.downsample(dedis, ffactor, tfactor)
    prof_dd = single_pulse.profile_snr(dedis.mean(axis=0), width)
    prof_dm0 = single_pulse.profile_snr(norm.mean(axis=0), width)

    small = single_pulse.downsample(norm, ffactor, 1)
    f_small = _block_mean_freqs(freqs_mhz, ffactor)[: small.shape[0]]
    dms = single_pulse.dm_grid(dm)
    dmt = single_pulse.dm_time(small, f_small, tsamp_s, dms, width)
    dmt_disp = single_pulse.downsample(dmt, 1, tfactor)

    t_prof = np.arange(prof_dd.size) * tsamp_s - t_rel_event_s
    t_wf = (np.arange(wf_dd.shape[1]) * tfactor + tfactor / 2) * tsamp_s - t_rel_event_s

    fig, axes = plt.subplots(3, 2, figsize=(12, 11))
    (ax_prof, ax_dm0), (ax_wf, ax_dmt), (ax_bt, ax_bs) = axes

    ax_prof.plot(t_prof, prof_dd, "k-", lw=0.7)
    ax_prof.axvline(0, color="red", alpha=0.4, lw=1)
    ax_prof.set_ylabel(r"S/N ($\sigma$)")
    ax_prof.set_title(f"dedispersed at DM={dm:.2f}", fontsize=9)

    ax_dm0.plot(t_prof, prof_dm0, "-", color="0.3", lw=0.7)
    ax_dm0.axvline(0, color="red", alpha=0.4, lw=1)
    ax_dm0.set_ylabel(r"S/N ($\sigma$)")
    ax_dm0.set_title("DM = 0 timestream", fontsize=9)

    vmin, vmax = np.percentile(wf_dd, [2, 98])
    ax_wf.imshow(wf_dd, aspect="auto", interpolation="nearest",
                 extent=[t_wf[0], t_wf[-1], freqs_mhz[-1], freqs_mhz[0]],
                 vmin=vmin, vmax=vmax, cmap="viridis")
    ax_wf.set_ylabel("Freq (MHz)")
    ax_wf.set_xlabel("Time - event (s)")

    ax_dmt.imshow(dmt_disp, aspect="auto", origin="lower", interpolation="nearest",
                  extent=[t_wf[0], t_wf[-1], dms[0], dms[-1]], cmap="magma")
    ax_dmt.axhline(dm, color="cyan", lw=0.6, alpha=0.6)
    ax_dmt.set_ylabel(r"DM (pc cm$^{-3}$)")
    ax_dmt.set_xlabel("Time - event (s)")

    for ax in (ax_prof, ax_dm0):
        ax.set_xlim(t_wf[0], t_wf[-1])

    _beam_panels(ax_bt, ax_bs, card, fig)

    fig.suptitle(
        f"{card['candname']}   {card['source']}   S/N={card['snr']:.1f}   "
        f"DM={dm:.2f}   width={width * tsamp_s * 1e3:.1f} ms   "
        f"beam {card['beam']}   {card['event_utc']}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png
