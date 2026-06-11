"""Aggregated T1 -> T2 stats chart for the web monitor.

gulp_stats already collapses each ~8.6 s gulp into scalar counters, so a
24 h chart reads ~10k rows -- the raw T1 stream (millions of candidates an
hour) never gets near the browser. Ad-hoc per-candidate exploration stays
in hiplot fed by t2-replay's CSV export; this page is the always-on view.

The PNG is re-rendered at most once a minute (the route checks mtime) and
written atomically so a refresh never sees a half-drawn file.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

GULP_S = 8192 * 1.048576e-3  # gulp duration: 8192 samples at 1.048576 ms


def utc_cut(hours: float) -> str:
    """ISO cutoff comparable to the DB's 'T'-separated UTC strings.

    sqlite's datetime('now', ...) renders with a space separator, and
    ' ' < 'T' makes string comparisons against ISO timestamps silently
    select the wrong window -- always build cutoffs here instead.
    """
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime(
        "%Y-%m-%dT%H:%M:%S")


def _fetch(db_path: str | Path, hours: float):
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10)
    try:
        gulps = conn.execute(
            "SELECT gulp_utc, n_cands, n_clusters, n_stored, n_would,"
            " clustering_ms FROM gulp_stats WHERE gulp_utc > ?"
            " ORDER BY gulp_utc", (utc_cut(hours),)).fetchall()
        dumps = [r[0] for r in conn.execute(
            "SELECT c.event_utc FROM triggers t JOIN clusters c"
            " ON c.name = t.candname WHERE t.action = 'triggered'"
            " AND c.event_utc > ?", (utc_cut(hours),))]
    finally:
        conn.close()
    return gulps, dumps


def render(db_path: str | Path, out_png: Path, hours: float = 4) -> Path:
    gulps, dumps = _fetch(db_path, hours)

    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    ax_rate, ax_clus, ax_ev, ax_ms = axes

    if gulps:
        t = np.array([datetime.fromisoformat(g[0]).timestamp() for g in gulps])
        cands, clusters, stored, would, ms = (
            np.array([g[i] for g in gulps], dtype=float) for i in range(1, 6))

        bw = max(60.0, hours * 3600 / 240)  # 1 min bins at the 4 h default
        edges = np.arange(t.min(), t.max() + bw, bw)
        idx = np.clip(np.digitize(t, edges) - 1, 0, len(edges) - 2)
        n = np.bincount(idx, minlength=len(edges) - 1).astype(float)
        live = n > 0

        def per_bin(x, weights=True):
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
                   color="#c60", lw=1, label="trigger-worthy / h")
        for d in dumps:
            ax_ev.axvline(datetime.fromisoformat(d), color="red", alpha=0.6,
                          lw=1)
        if dumps:  # one labelled proxy line so the legend mentions dumps
            ax_ev.plot([], [], color="red", lw=1, label="dump fired")
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
        ax_rate.set_title(
            f"last {hours:g} h - {len(gulps)} gulps, duty cycle "
            f"{duty:.0f}%, {len(dumps)} dumps", fontsize=10)
    else:
        ax_rate.text(0.5, 0.5, "no gulp_stats rows in window", ha="center",
                     va="center", transform=ax_rate.transAxes, color="0.4")

    for ax in axes:
        ax.grid(alpha=0.25)
    ax_ms.set_xlabel("UTC")
    ax_ms.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=timezone.utc))
    fig.tight_layout()

    out_png = Path(out_png)
    tmp = out_png.with_suffix(".tmp.png")
    fig.savefig(tmp, dpi=110)
    plt.close(fig)
    tmp.replace(out_png)
    return out_png
