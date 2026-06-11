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
        injections = conn.execute(
            "SELECT inject_utc, est_snr, gate_t1, gate_t2, gate_trigger"
            " FROM injections WHERE inject_utc > ?", (cut,)).fetchall()
    finally:
        conn.close()
    return gulps, attempts, injections


def render(db_path: str | Path, out_png: Path) -> Path:
    gulps, attempts, injections = _fetch(db_path)
    dumps = [a for a in attempts if a[1] == "triggered"]
    misses = [a for a in attempts if a[1] != "triggered"]

    fig, axes = plt.subplots(6, 1, figsize=(11, 13), sharex=True)
    ax_rate, ax_clus, ax_ev, ax_lag, ax_ms, ax_inj = axes
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
        ax_ms.set_ylabel("DBSCAN ms")
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

    # Injections: one marker per shot at its estimated S/N, coloured by the
    # deepest gate it cleared once reconciled (~3 min after injection).
    if injections:
        groups = {  # (label, colour, marker, y-values, x-values)
            "recovered (T2)": ("#393", "o"), "lost": ("red", "x"),
            "pending": ("0.6", "o")}
        pts = {k: ([], []) for k in groups}
        for utc, snr, g1, g2, gt in injections:
            k = ("pending" if g2 is None else
                 "recovered (T2)" if g2 else "lost")
            pts[k][0].append(datetime.fromisoformat(utc))
            pts[k][1].append(snr)
        for k, (c, m) in groups.items():
            if pts[k][0]:
                ax_inj.plot(pts[k][0], pts[k][1], m, ms=6, color=c,
                            mew=1.5, label=k,
                            mfc="none" if m == "o" else c)
        ax_inj.legend(loc="upper left", fontsize=8, ncol=3)
    else:
        ax_inj.text(0.5, 0.5, "no injections in window", ha="center",
                    va="center", transform=ax_inj.transAxes, color="0.4",
                    fontsize=9)
    ax_inj.set_ylabel("injected S/N")
    ax_inj.set_ylim(bottom=0)

    for ax in axes:
        ax.grid(alpha=0.25)
    ax_inj.set_xlim(t0, now)
    ax_inj.set_xlabel("OVRO local time (America/Los_Angeles)")
    ax_inj.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=OVRO_TZ))
    fig.tight_layout()

    out_png = Path(out_png)
    tmp = out_png.with_suffix(".tmp.png")
    fig.savefig(tmp, dpi=110)
    plt.close(fig)
    tmp.replace(out_png)
    return out_png
