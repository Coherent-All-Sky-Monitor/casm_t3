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
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from casm_t2 import db as t2db

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


@app.get("/")
def index(request: Request, tier: str = "", tag: str = "", limit: int = 200,
          view: str = "candidates"):
    where, args = ["name IS NOT NULL"], []
    if view != "all":
        # Default: candidates (events that reached the trigger stage). Under
        # the everything-gets-plotted policy this is also the plot list; the
        # plot column flags the rare failed render or refused dump.
        where.append("name IN (SELECT candname FROM triggers)")
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
    plots = {p.name for p in CANDIDATES_DIR.iterdir()} if CANDIDATES_DIR.exists() else set()
    return templates.TemplateResponse(request, "index.html", dict(
        rows=rows, labels=labels, plots=plots, tier=tier, tag=tag, limit=limit,
        view=view))


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


@app.get("/stats")
def stats(request: Request, limit: int = 120):
    rows = q("SELECT gulp_utc, n_jobs, n_cands, n_clusters, n_stored, n_would,"
             " clustering_ms FROM gulp_stats ORDER BY id DESC LIMIT ?", (limit,))
    day = q("SELECT count(*) gulps, sum(n_cands) cands, sum(n_stored) stored,"
            " sum(n_would) would, round(avg(clustering_ms),1) ms FROM gulp_stats"
            " WHERE created_utc > datetime('now', '-1 day')")
    return templates.TemplateResponse(request, "funnel.html",
                                      dict(rows=rows, day=day[0]))


@app.get("/injections")
def injections(request: Request, limit: int = 200):
    rows = q("SELECT * FROM injections ORDER BY id DESC LIMIT ?", (limit,))
    day = q("SELECT count(*) n, sum(gate_t1) t1, sum(gate_t2) t2,"
            " sum(gate_trigger) tr, count(gate_t1) done FROM injections"
            " WHERE inject_utc > datetime('now', '-1 day')")
    return templates.TemplateResponse(request, "injections.html",
                                      dict(rows=rows, day=day[0]))


@app.get("/frbs")
def frbs(request: Request):
    rows = q("SELECT * FROM frbs ORDER BY id DESC")
    return templates.TemplateResponse(request, "frbs.html", dict(rows=rows))


def main() -> None:
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8050, log_level="warning")


if __name__ == "__main__":
    main()
