"""Aggregated T1 -> T2 stats chart for the web monitor.

gulp_stats already collapses each ~8.6 s gulp into scalar counters, so a
full-day chart reads ~10k rows -- the raw T1 stream (millions of candidates
an hour) never gets near the browser. Ad-hoc per-candidate exploration stays
in hiplot fed by t2-replay's CSV export; this page is the always-on view.

The window is the current observing day: from the most recent 05:00 at OVRO
to now, so the chart grows through the day and resets each morning.

The PNG is re-rendered at most once a minute (the route checks mtime) and
written atomically so a refresh never sees a half-drawn file.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, time as dtime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

GULP_S = 8192 * 1.048576e-3  # gulp duration: 8192 samples at 1.048576 ms

# Chart axis only -- the DB and all pipeline timestamps stay UTC. zoneinfo
# applies the real PST/PDT rules, so the axis is daylight-saving safe.
OVRO_TZ = ZoneInfo("America/Los_Angeles")
DAY_START_HOUR = 5  # observing day rolls over at 05:00 local

# Intensity ring look-back: CAND_DUMP_READ_DELAY in medusa_bf_proc.cfg.
# Dumps requested later than this after the event miss the ring.
RING_LOOKBACK_S = 28.0


def utc_cut(hours: float) -> str:
    """ISO cutoff comparable to the DB's 'T'-separated UTC strings.

    sqlite's datetime('now', ...) renders with a space separator, and
    ' ' < 'T' makes string comparisons against ISO timestamps silently
    select the wrong window -- always build cutoffs here instead.
    """
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime(
        "%Y-%m-%dT%H:%M:%S")


def day_start() -> datetime:
    """Most recent 05:00 at OVRO, as aware UTC. DST-safe: the boundary is
    constructed on the local calendar, not by subtracting fixed hours."""
    now_l = datetime.now(OVRO_TZ)
    start = datetime.combine(now_l.date(), dtime(DAY_START_HOUR), OVRO_TZ)
    if start > now_l:
        start = datetime.combine(now_l.date() - timedelta(days=1),
                                 dtime(DAY_START_HOUR), OVRO_TZ)
    return start.astimezone(timezone.utc)


def day_cut() -> str:
    return day_start().strftime("%Y-%m-%dT%H:%M:%S")


def _fetch(db_path: str | Path):
    cut = day_cut()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10)
    try:
        gulps = conn.execute(
            "SELECT gulp_utc, n_cands, n_clusters, n_stored, n_would,"
            " clustering_ms FROM gulp_stats WHERE gulp_utc > ?"
            " ORDER BY gulp_utc", (cut,)).fetchall()
        # Dump attempts with their event->command lag: the strict-mode
        # margin against the ring look-back.
        attempts = conn.execute(
            "SELECT c.event_utc, t.action,"
            " (julianday(t.created_utc) - julianday(c.event_utc)) * 86400"
            " FROM triggers t JOIN clusters c ON c.name = t.candname"
            " WHERE t.action IN ('triggered', 'refused_daemon', 'failed')"
            " AND c.event_utc > ?", (cut,)).fetchall()
        events = conn.execute(
            "SELECT event_utc, beam, dm FROM clusters"
            " WHERE name IS NOT NULL AND event_utc > ?", (cut,)).fetchall()
    finally:
        conn.close()
    return gulps, attempts, events


def render(db_path: str | Path, out_png: Path) -> Path:
    gulps, attempts, events = _fetch(db_path)
    dumps = [a for a in attempts if a[1] == "triggered"]
    misses = [a for a in attempts if a[1] != "triggered"]

    fig, axes = plt.subplots(6, 1, figsize=(11, 13), sharex=True)
    ax_rate, ax_clus, ax_ev, ax_lag, ax_ms, ax_beam = axes
    t0 = day_start()
    now = datetime.now(timezone.utc)

    if gulps:
        t = np.array([datetime.fromisoformat(g[0]).timestamp() for g in gulps])
        cands, clusters, stored, would, ms = (
            np.array([g[i] for g in gulps], dtype=float) for i in range(1, 6))

        span = max(t.max() - t.min(), 1.0)
        bw = max(60.0, span / 240)  # ~6 min bins over a full day
        edges = np.arange(t.min(), t.max() + bw, bw)
        idx = np.clip(np.digitize(t, edges) - 1, 0, len(edges) - 2)
        n = np.bincount(idx, minlength=len(edges) - 1).astype(float)
        live = n > 0

        def per_bin(x):
            s = np.bincount(idx, weights=x, minlength=len(edges) - 1)
            return np.where(live, s, np.nan)

        centers = [datetime.fromtimestamp(e + bw / 2, tz=timezone.utc)
                   for e in edges[:-1]]
        nn = np.where(live, n, np.nan)

        ax_rate.semilogy(centers, per_bin(cands) / (nn * GULP_S), "-",
                         color="#06c", lw=1)
        ax_rate.set_ylabel("T1 cands / s")

        ax_clus.plot(centers, per_bin(clusters) / nn, "-", color="#393", lw=1)
        ax_clus.set_ylabel("clusters / gulp")
        ax_clus.set_ylim(bottom=0)

        ax_ev.plot(centers, per_bin(stored) / (nn * GULP_S) * 3600, "-",
                   color="0.4", lw=1, label="stored / h")
        ax_ev.plot(centers, per_bin(would) / (nn * GULP_S) * 3600, "-",
                   color="#06c", lw=1, label="trigger-worthy / h")
        for d in dumps:
            ax_ev.axvline(datetime.fromisoformat(d[0]), color="red",
                          alpha=0.7, lw=1.2)
        if dumps:  # one labelled proxy line so the legend mentions dumps
            ax_ev.plot([], [], color="red", lw=1.2, label="dump fired")
        ax_ev.legend(loc="upper left", fontsize=8, ncol=3)
        ax_ev.set_ylabel("events / h")
        ax_ev.set_ylim(bottom=0)

        mx = np.zeros(len(edges) - 1)
        np.maximum.at(mx, idx, ms)
        mx = np.where(live, mx, np.nan)
        ax_ms.plot(centers, per_bin(ms) / nn, "-", color="#639", lw=1,
                   label="avg")
        ax_ms.plot(centers, mx, "-", color="#639", lw=0.6, alpha=0.4,
                   label="max")
        ax_ms.legend(loc="upper left", fontsize=8, ncol=2)
        ax_ms.set_ylabel("DBSCAN compute time (ms)")
        ax_ms.set_ylim(bottom=0)

        duty = 100 * n.sum() * GULP_S / (edges[-1] - edges[0])
        local0 = t0.astimezone(OVRO_TZ)
        ax_rate.set_title(
            f"observing day since {local0:%Y-%m-%d %H:%M} local - "
            f"{len(gulps)} gulps, duty cycle {duty:.0f}%, "
            f"{len(dumps)} dumps, {len(misses)} ring misses", fontsize=10)
    else:
        ax_rate.text(0.5, 0.5, "no gulp_stats rows in window", ha="center",
                     va="center", transform=ax_rate.transAxes, color="0.4")

    # Event -> dump-command lag against the ring look-back: the margin
    # strict cluster-first T2 lives or dies by.
    ax_lag.axhline(RING_LOOKBACK_S, color="red", ls="--", lw=0.8, alpha=0.6)
    ax_lag.text(0.998, RING_LOOKBACK_S, f" ring look-back {RING_LOOKBACK_S:.0f} s ",
                transform=ax_lag.get_yaxis_transform(), ha="right", va="bottom",
                fontsize=7, color="red", alpha=0.8)
    if dumps:
        ax_lag.plot([datetime.fromisoformat(d[0]) for d in dumps],
                    [d[2] for d in dumps], "o", ms=5, color="#393",
                    label="dumped")
    if misses:
        ax_lag.plot([datetime.fromisoformat(m[0]) for m in misses],
                    [m[2] for m in misses], "x", ms=7, mew=1.5, color="red",
                    label="missed ring")
    if dumps or misses:
        ax_lag.legend(loc="upper left", fontsize=8, ncol=2)
    ax_lag.set_ylabel("dump lag (s)")
    ax_lag.set_ylim(0, max([RING_LOOKBACK_S * 1.3]
                           + [a[2] * 1.15 for a in attempts if a[2]]))

    # Beam occupancy of T2 stored events (the clusters table), coloured
    # by DM: RFI storms light up beam blocks at low DM, a transiting
    # source draws a compact track.
    if events:
        et = [datetime.fromisoformat(e[0]) for e in events]
        eb = np.array([e[1] for e in events], dtype=float)
        ed = np.array([e[2] for e in events], dtype=float)
        sc = ax_beam.scatter(et, eb, c=ed, s=6, cmap="plasma", vmin=0,
                             vmax=max(50.0, float(np.percentile(ed, 98))),
                             alpha=0.8, linewidths=0)
        cax = ax_beam.inset_axes((1.008, 0.0, 0.012, 1.0))
        fig.colorbar(sc, cax=cax, label=r"DM (pc cm$^{-3}$)")
        for edge in range(64, 512, 64):
            ax_beam.axhline(edge, color="0.85", lw=0.4, zorder=0)
    else:
        ax_beam.text(0.5, 0.5, "no stored events in window", ha="center",
                     va="center", transform=ax_beam.transAxes, color="0.4",
                     fontsize=9)
    ax_beam.set_ylabel("beam (T2 stored events)")
    ax_beam.set_ylim(-5, 516)

    for ax in axes:
        ax.grid(alpha=0.25)
    ax_beam.set_xlim(t0, now)
    ax_beam.set_xlabel("OVRO local time (America/Los_Angeles)")
    ax_beam.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=OVRO_TZ))
    fig.tight_layout()

    out_png = Path(out_png)
    tmp = out_png.with_suffix(".tmp.png")
    fig.savefig(tmp, dpi=110)
    plt.close(fig)
    tmp.replace(out_png)
    return out_png


def render_injections(db_path: str | Path, out_png: Path) -> Path:
    """Injected vs recovered S/N for every gate-checked injection.

    The 1:1 line is the target; the systematic offset from it is the
    est_snr calibration error (injector predicts from bf_proc stats,
    recovery is hella's matched-filter S/N).
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10)
    try:
        rows = conn.execute(
            "SELECT est_snr, rec_snr, gate_t1, gate_t2, fail_reason"
            " FROM injections WHERE gate_t1 IS NOT NULL").fetchall()
    finally:
        conn.close()

    fig, ax = plt.subplots(figsize=(11, 4.6))
    rec = [(e, r) for e, r, g1, g2, _ in rows if r]
    lost = [e for e, r, g1, g2, _ in rows if not r]
    n_t2 = sum(1 for _, _, _, g2, _ in rows if g2)
    top = 1.1 * max([max(e, r) for e, r in rec] + lost + [100])
    ax.plot([0, top], [0, top], "--", color="0.75", lw=1, label="1:1",
            zorder=1)
    if rec:
        ax.scatter([e for e, _ in rec], [r for _, r in rec], s=55,
                   facecolors="#2a9d8f", edgecolors="white", linewidths=0.8,
                   alpha=0.9, label="recovered", zorder=3)
    if lost:
        ax.scatter(lost, [0] * len(lost), s=60, marker="x", color="#e63946",
                   linewidths=2, label="lost", zorder=3)
    ax.set_xlim(0, top), ax.set_ylim(0, top)
    ax.set_xlabel("injected S/N (est_snr)")
    ax.set_ylabel("recovered S/N (hella)")
    ax.set_title(f"{len(rows)} injected / {n_t2} recovered at T2",
                 fontsize=11)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()

    out_png = Path(out_png)
    tmp = out_png.with_suffix(".tmp.png")
    fig.savefig(tmp, dpi=110)
    plt.close(fig)
    tmp.replace(out_png)
    return out_png
