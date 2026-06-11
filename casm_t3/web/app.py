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

from fastapi import FastAPI, Form, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from casm_t2 import db as t2db

from . import statsplot

DB_PATH = t2db.DEFAULT_PATH
CANDIDATES_DIR = Path("/mnt/nvme5/casm_pipeline/candidates")
SPOOL_DIRS = [Path("/mnt/nvme4/data/casm/t2_spool")]
LABELS = ("frb", "pulsar", "rfi", "unsure")

app = FastAPI(title="CASM Monitor")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


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
def index(request: Request, tier: str = "", tag: str = "", limit: int = 200,
          view: str = "candidates"):
    where, args = ["name IS NOT NULL"], []
    if view != "all":
        # Default: dump attempts only — events T2 shortlisted AND tried to
        # dump. Each has a plot or a red miss reason. Storm-gate and 60 s
        # suppressions live in the all-stored-events view.
        where.append("name IN (SELECT candname FROM triggers WHERE"
                     " action IN ('triggered', 'refused_daemon', 'failed')"
                     " OR (action = 'refused' AND detail LIKE '%disk%'))")
    if tier:
        where.append("tier = ?")
        args.append(tier)
    if tag:
        where.append("tags LIKE ?")
        args.append(f"%{tag}%")
    rows = q(f"SELECT name, event_utc, tier, tags, snr, dm, width, beam, n_beams,"
             f" n_members FROM clusters WHERE {' AND '.join(where)}"
             f" ORDER BY id DESC LIMIT ?", (*args, limit))
    labels = {r["name"]: r["label"] for r in
              q("SELECT name, label FROM labels GROUP BY name HAVING max(id)")}
    # Best trigger outcome per event, translated into a why-no-plot phrase;
    # the raw audit detail rides along as a hover tooltip.
    actions = {r["candname"]: (friendly_outcome(r["action"], r["detail"]),
                               f"{r['action']}: {r['detail']}")
               for r in q("SELECT candname, action, detail FROM triggers"
                          " ORDER BY action = 'triggered', id")}
    plots = {p.name for p in CANDIDATES_DIR.iterdir()} if CANDIDATES_DIR.exists() else set()
    return templates.TemplateResponse(request, "index.html", dict(
        rows=rows, labels=labels, plots=plots, actions=actions, tier=tier,
        tag=tag, limit=limit, view=view))


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
    else:
        data_status = "no dump (trigger refused/failed)"
    return templates.TemplateResponse(request, "event.html", dict(
        ev=dict(rows[0]), triggers=triggers, labels=labels, pngs=pngs,
        meta=json.dumps(meta, indent=2), label_choices=LABELS,
        data_status=data_status))


@app.get("/event/{name}/plot/{fname}")
def plot(name: str, fname: str):
    f = (CANDIDATES_DIR / name / Path(fname).name).resolve()
    if not f.is_file() or CANDIDATES_DIR.resolve() not in f.parents:
        return RedirectResponse(f"/event/{name}")
    return FileResponse(f)


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


STATS_PNG = Path(tempfile.gettempdir()) / "casm_t3_stats.png"
STATS_TTL_S = 60


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
def stats(request: Request, limit: int = 60):
    try:
        if (not STATS_PNG.exists()
                or time.time() - STATS_PNG.stat().st_mtime > STATS_TTL_S):
            statsplot.render(DB_PATH, STATS_PNG)
    except Exception:  # the page must render even if charting breaks
        pass
    rows = q("SELECT gulp_utc, n_jobs, n_cands, n_clusters, n_stored, n_would,"
             " clustering_ms FROM gulp_stats ORDER BY id DESC LIMIT ?", (limit,))
    day0 = statsplot.day_start().astimezone(statsplot.OVRO_TZ)
    return templates.TemplateResponse(request, "funnel.html", dict(
        rows=rows, hour=_window_stats(statsplot.utc_cut(1)),
        win=_window_stats(statsplot.day_cut()),
        day_label=day0.strftime("%H:%M local %b %d"), ts=int(time.time())))


@app.get("/stats/plot.png")
def stats_plot():
    if not STATS_PNG.exists():
        return RedirectResponse("/stats")
    return FileResponse(STATS_PNG, headers={"Cache-Control": "no-cache"})


INJ_PLOT_DIR = CANDIDATES_DIR / "injections"


@app.get("/injections/plot/{file_id}")
def injection_plot(file_id: str):
    f = (INJ_PLOT_DIR / f"{Path(file_id).name}.png").resolve()
    if not f.is_file() or INJ_PLOT_DIR.resolve() not in f.parents:
        return RedirectResponse("/injections")
    return FileResponse(f)


@app.get("/injections")
def injections(request: Request, limit: int = 200):
    rows = q("SELECT i.*, c.name AS event_name FROM injections i"
             " LEFT JOIN clusters c ON c.id = i.matched_cluster"
             " ORDER BY i.id DESC LIMIT ?", (limit,))
    day = q("SELECT count(*) n, sum(gate_t1) t1, sum(gate_t2) t2,"
            " sum(gate_trigger) tr, count(gate_t1) done FROM injections"
            " WHERE inject_utc > datetime('now', '-1 day')")
    inj_plots = ({p.stem for p in INJ_PLOT_DIR.glob("*.png")}
                 if INJ_PLOT_DIR.exists() else set())
    return templates.TemplateResponse(request, "injections.html",
                                      dict(rows=rows, day=day[0], inj_plots=inj_plots))


@app.get("/frbs")
def frbs(request: Request):
    rows = q("SELECT * FROM frbs ORDER BY id DESC")
    return templates.TemplateResponse(request, "frbs.html", dict(rows=rows))


def main() -> None:
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8050, log_level="warning")


if __name__ == "__main__":
    main()
