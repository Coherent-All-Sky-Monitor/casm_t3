"""CASM monitor web UI.

Server-rendered FastAPI app on corr1 (LAN only, no auth) reading the T2
SQLite database and the candidates artifact tree. Pages:

    /                events table (filterable)
    /event/<name>    plots + trigger card + label buttons
    /funnel          per-gulp T1->T2 funnel statistics
    /injections      injection gate table + recovery fractions
    /frbs            the FRB catalog

Labelling is the only write path: POST /event/<name>/label inserts into
labels, and an 'frb' label promotes the event into the frbs table.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from fastapi import FastAPI, Form, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from casm_t2 import db as t2db

from . import nowpanel, statsplot

DB_PATH = t2db.DEFAULT_PATH
T2D_CONFIG = Path("/home/casm/software/dev/casm_t2/config/t2d.yaml")
CANDIDATES_DIR = Path("/mnt/nvme5/casm_pipeline/candidates")
SPOOL_DIRS = [Path("/mnt/nvme4/data/casm/t2_spool")]
LABELS = ("frb", "pulsar", "rfi", "unsure")

app = FastAPI(title="CASM Monitor")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def ovro_local(utc_iso: str) -> str:
    """UTC ISO string -> OVRO wall clock. zoneinfo applies the real
    PST/PDT rules, so the column is daylight-saving safe."""
    try:
        t = datetime.fromisoformat(utc_iso)
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return t.astimezone(statsplot.OVRO_TZ).strftime("%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return ""


templates.env.filters["ovro_local"] = ovro_local


def q(sql: str, args: tuple = ()) -> list[sqlite3.Row]:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(sql, args).fetchall()
    finally:
        conn.close()


def q_write(sql: str, args: tuple = ()) -> None:
    conn = t2db.connect(DB_PATH)
    try:
        with conn:
            conn.execute(sql, args)
    finally:
        conn.close()


def trigger_cfg() -> dict:
    """Live tier/DM-floor values from the t2d config, so the legend and
    held-reasons can never drift from what the daemon enforces."""
    out = {"A": 30.0, "B": 15.0, "C": 12.0, "dm_floor": 20.0}
    try:
        cfg = yaml.safe_load(T2D_CONFIG.read_text())
        out.update({k: float(v) for k, v in cfg.get("tiers", {}).items()})
        out["dm_floor"] = float(cfg.get("filters", {}).get("dm_floor", out["dm_floor"]))
    except Exception:
        pass
    return out


def held_reason(tier: str, tags: str, dm: float, tcfg: dict) -> str:
    """Why a stored event never reached a dump attempt (no triggers row).

    Mirrors t2d._wants_trigger: tag exclusions first, then the tier/DM
    gate (blind triggers need tier A/B and DM >= the floor).
    """
    if "injection" in tags:
        return "injection: never dumped (policy)"
    if "veto" in tags:
        return "vetoed beam"
    if "rfi_wide" in tags:
        return "RFI: too many beams"
    if tier not in ("A", "B"):
        return f"S/N below tier B ({tcfg['B']:g})"
    if dm < tcfg["dm_floor"]:
        return f"DM below {tcfg['dm_floor']:g} floor"
    return "held by filters"


def friendly_outcome(action: str, detail: str) -> str:
    """One memorable phrase per trigger outcome for the events table."""
    if action == "triggered":
        return "dumped"
    tail = detail.rsplit(";", 1)[-1]
    if "gulp_dup" in tail:
        return "storm: 1 dump/gulp"
    if tail == "spacing":
        return "storm: 60 s gate"
    if tail == "daily_cap":
        return "cap: daily max"
    if "Bad command" in tail:
        return "missed gulp window (T2 too slow for ring)"
    if tail.startswith("disk"):
        return "low disk"
    if action == "shadow":
        return "shadow"
    return f"{action}: {tail}"


@app.get("/")
def index(request: Request, tier: str = "", tag: str = "", limit: int = 500,
          view: str = "candidates"):
    where, args = ["name IS NOT NULL"], []
    if view != "all" and not tag:
        # Default: dump attempts only — events T2 shortlisted AND tried to
        # dump. Each has a plot or a red miss reason. Storm-gate and 60 s
        # suppressions live in the all-stored-events view. A label search
        # spans every stored event instead, so labels on held/suppressed
        # events still surface.
        where.append("name IN (SELECT candname FROM triggers WHERE"
                     " action IN ('triggered', 'refused_daemon', 'failed')"
                     " OR (action = 'refused' AND detail LIKE '%disk%'))")
    if tier:
        where.append("tier = ?")
        args.append(tier)
    if tag:
        # Matches the human-assigned label (frb/pulsar/rfi/unsure), newest
        # label per event — NOT the pipeline's automatic tags column.
        where.append("name IN (SELECT name FROM labels WHERE label LIKE ?"
                     " AND id IN (SELECT MAX(id) FROM labels GROUP BY name))")
        args.append(f"%{tag}%")
    rows = q(f"SELECT name, event_utc, tier, tags, snr, dm, width, beam, n_beams,"
             f" n_members FROM clusters WHERE {' AND '.join(where)}"
             f" ORDER BY id DESC LIMIT ?", (*args, limit))
    labels = {r["name"]: r["label"] for r in
              q("SELECT name, label FROM labels WHERE id IN"
                " (SELECT MAX(id) FROM labels GROUP BY name)")}
    # Best trigger outcome per event, translated into a why-no-plot phrase;
    # the raw audit detail rides along as a hover tooltip.
    actions = {r["candname"]: (friendly_outcome(r["action"], r["detail"]),
                               f"{r['action']}: {r['detail']}")
               for r in q("SELECT candname, action, detail FROM triggers"
                          " ORDER BY action = 'triggered', id")}
    plots = {p.name for p in CANDIDATES_DIR.iterdir()} if CANDIDATES_DIR.exists() else set()
    tcfg = trigger_cfg()
    held = {r["name"]: held_reason(r["tier"], r["tags"], r["dm"], tcfg)
            for r in rows if r["name"] not in actions}
    return templates.TemplateResponse(request, "index.html", dict(
        rows=rows, labels=labels, plots=plots, actions=actions, held=held,
        tier=tier, tag=tag, limit=limit, view=view, tiers=tcfg))


@app.get("/event/{name}")
def event(request: Request, name: str):
    rows = q("SELECT * FROM clusters WHERE name = ?", (name,))
    if not rows:
        return RedirectResponse("/")
    triggers = q("SELECT * FROM triggers WHERE candname = ? ORDER BY id", (name,))
    labels = q("SELECT * FROM labels WHERE name = ? ORDER BY id DESC", (name,))
    art_dir = CANDIDATES_DIR / name
    pngs = sorted(p.name for p in art_dir.glob("*.png")) if art_dir.exists() else []
    meta = {}
    for j in ([art_dir / f"{name}.json"] if art_dir.exists() else []) + \
             [d / f"{name}.json.done" for d in SPOOL_DIRS] + \
             [d / f"{name}.json" for d in SPOOL_DIRS]:
        if j.exists():
            meta = json.loads(j.read_text())
            meta.get("context", {}).pop("members", None)  # too big for a page
            break
    if meta.get("data_available") is True:
        data_status = "raw dump on disk"
    elif meta.get("data_available") is False:
        data_status = "raw dump deleted after plotting"
    elif any(t["cleaned_utc"] for t in triggers):
        data_status = "raw dump deleted by janitor"
    elif any(t["action"] in ("triggered", "partial") for t in triggers):
        data_status = "unknown (pre-tracking dump)"
    elif triggers:
        last = triggers[-1]
        data_status = f"no dump — {friendly_outcome(last['action'], last['detail'])}"
    else:
        e = rows[0]
        data_status = f"no dump attempt — {held_reason(e['tier'], e['tags'], e['dm'], trigger_cfg())}"
    # auto src: tags hide on the web -- DM-overlap matching mislabels
    # RFI; humans assign source labels instead.
    tags_display = ",".join(t for t in rows[0]["tags"].split(",")
                            if t and not t.startswith("src:"))
    return templates.TemplateResponse(request, "event.html", dict(
        ev=dict(rows[0]), triggers=triggers, labels=labels, pngs=pngs,
        tags_display=tags_display,
        meta=json.dumps(meta, indent=2), label_choices=LABELS,
        data_status=data_status))


@app.get("/event/{name}/plot/{fname}")
def plot(name: str, fname: str):
    f = (CANDIDATES_DIR / name / Path(fname).name).resolve()
    if not f.is_file() or CANDIDATES_DIR.resolve() not in f.parents:
        return RedirectResponse(f"/event/{name}")
    # candidate PNGs are ~0.8 MB and effectively immutable once rendered;
    # let browsers keep them so remote (ssh-tunnelled) viewing only pays
    # the transfer once.
    return FileResponse(f, headers={"Cache-Control": "public, max-age=86400"})


@app.post("/event/{name}/label")
def label(name: str, label: str = Form(...), notes: str = Form("")):
    if label in LABELS:
        now = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        q_write("INSERT INTO labels (name, label, who, notes, created_utc)"
                " VALUES (?,?,?,?,?)", (name, label, "web", notes, now))
        if label == "frb":
            ev = q("SELECT * FROM clusters WHERE name = ?", (name,))
            if ev:
                e = ev[0]
                q_write("INSERT OR IGNORE INTO frbs (name, event_utc, snr, dm,"
                        " width, beam, notes, created_utc) VALUES (?,?,?,?,?,?,?,?)",
                        (name, e["event_utc"], e["snr"], e["dm"], e["width"],
                         e["beam"], notes, now))
    return RedirectResponse(f"/event/{name}", status_code=303)


@app.get("/funnel")
def funnel_redirect():
    return RedirectResponse("/stats")


STATS_TTL_S = 60


def _stats_png(hours: int) -> Path:
    """Per-window cache file so switching ranges doesn't thrash one PNG."""
    return Path(tempfile.gettempdir()) / f"casm_t3_stats_{hours}h.png"


def _window_stats(cut: str) -> dict:
    r = q("SELECT count(*) gulps, sum(n_cands) cands, sum(n_clusters) clusters,"
          " sum(n_stored) stored, sum(n_would) would,"
          " round(avg(clustering_ms),1) ms FROM gulp_stats WHERE gulp_utc > ?",
          (cut,))[0]
    d = dict(r)
    span_s = (r["gulps"] or 0) * statsplot.GULP_S
    d["cands_s"] = round((r["cands"] or 0) / span_s, 1) if span_s else 0.0
    return d


@app.get("/stats")
def stats(request: Request, limit: int = 60, hours: int = 24):
    if hours not in {h for h, _ in statsplot.WINDOW_PRESETS}:
        hours = 24
    png = _stats_png(hours)
    try:
        if (not png.exists()
                or time.time() - png.stat().st_mtime > STATS_TTL_S):
            statsplot.render(DB_PATH, png, hours=hours)
    except Exception:  # the page must render even if charting breaks
        pass
    now = nowpanel.snapshot()
    rows = q("SELECT gulp_utc, n_jobs, n_cands, n_clusters, n_stored, n_would,"
             " clustering_ms FROM gulp_stats ORDER BY id DESC LIMIT ?", (limit,))
    return templates.TemplateResponse(request, "funnel.html", dict(
        rows=rows, hour=_window_stats(statsplot.utc_cut(1)),
        win=_window_stats(statsplot.utc_cut(hours)), now=now,
        hours=hours, win_label=statsplot.window_label(hours),
        presets=statsplot.WINDOW_PRESETS, ts=int(time.time())))


@app.get("/stats/plot.png")
def stats_plot(hours: int = 24):
    if hours not in {h for h, _ in statsplot.WINDOW_PRESETS}:
        hours = 24
    png = _stats_png(hours)
    if not png.exists():
        return RedirectResponse(f"/stats?hours={hours}")
    return FileResponse(png, headers={"Cache-Control": "no-cache"})


INJ_PLOT_DIR = CANDIDATES_DIR / "injections"
INJ_SNR_PNG = Path(tempfile.gettempdir()) / "casm_t3_inj_snr.png"


@app.get("/injections/plot/{file_id}")
def injection_plot(file_id: str):
    f = (INJ_PLOT_DIR / f"{Path(file_id).name}.png").resolve()
    if not f.is_file() or INJ_PLOT_DIR.resolve() not in f.parents:
        return RedirectResponse("/injections")
    return FileResponse(f)


@app.get("/injections/snr.png")
def injections_snr_plot():
    if not INJ_SNR_PNG.exists():
        return RedirectResponse("/injections")
    return FileResponse(INJ_SNR_PNG, headers={"Cache-Control": "no-cache"})


@app.get("/injections")
def injections(request: Request, limit: int = 200):
    try:
        if (not INJ_SNR_PNG.exists()
                or time.time() - INJ_SNR_PNG.stat().st_mtime > STATS_TTL_S):
            statsplot.render_injections(DB_PATH, INJ_SNR_PNG)
    except Exception:  # the page must render even if charting breaks
        pass
    rows = q("SELECT i.*, c.name AS event_name FROM injections i"
             " LEFT JOIN clusters c ON c.id = i.matched_cluster"
             " ORDER BY i.id DESC LIMIT ?", (limit,))
    # NB datetime('now',...) renders a space-separated string that mis-sorts
    # against the DB's ISO 'T' timestamps (' ' < 'T'), silently pulling in
    # stale rows -- always build the cutoff with statsplot.utc_cut().
    day = q("SELECT count(*) n, sum(gate_t1) t1, sum(gate_t2) t2,"
            " sum(gate_trigger) tr, count(gate_t1) done FROM injections"
            " WHERE inject_utc > ?", (statsplot.utc_cut(24),))
    inj_plots = ({p.stem for p in INJ_PLOT_DIR.glob("*.png")}
                 if INJ_PLOT_DIR.exists() else set())
    return templates.TemplateResponse(request, "injections.html",
                                      dict(rows=rows, day=day[0],
                                           inj_plots=inj_plots,
                                           ts=int(time.time())))


@app.get("/frbs")
def frbs(request: Request):
    rows = q("SELECT * FROM frbs ORDER BY id DESC")
    return templates.TemplateResponse(request, "frbs.html", dict(rows=rows))


def main() -> None:
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8050, log_level="warning")


if __name__ == "__main__":
    main()
