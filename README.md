# casm_t3

T3 stage of the CASM fast-transient search: turns the beam intensity dumps
triggered by [casm_t2](../casm_t2) into candidate diagnostic plots, serves
the monitoring web UI, and keeps the dump disks within quota.

Pipeline position:

    T1  casm-hella GPU single-pulse search
    T2  casm_t2: clustering, filtering, trigger policy
    T3  this repo: plotting, web monitor, disk janitor

## What it does

- Reads casm_cand_dump `.dada` beam dumps (float16, 64 beams x 3072
  channels) and renders a five-panel candidate figure: dedispersed profile,
  raw DM=0 timeseries, dedispersed waterfall, DM-time bowtie, and a
  beam-vs-time scatter of T1 context candidates. Titles carry the beam's
  RA/Dec at the event time.
- One plotter instance runs per backend node and only touches that node's
  local dumps; only small PNG/JSON artifacts cross the network.
- Serves a server-rendered FastAPI monitor: events table with per-event
  trigger/miss reasons, event pages with labelling (an `frb` label promotes
  into the FRB catalog), live pipeline stats, injection recovery, and an
  "now at OVRO" clock/source panel.
- A janitor enforces dump-directory quotas and ages out plotted dumps,
  never touching events labelled frb/pulsar.

## Install

    python -m venv env && source env/bin/activate
    pip install -e .

Python >= 3.10. Depends on casm_t2 (schema, timing, and beam maps are
owned there), numpy, matplotlib, astropy, fastapi, jinja2, uvicorn.

## Run

    t3-dump-plotter <args>      # per-node dump -> plot daemon
    t3-web                      # monitor UI on :8050
    t3-janitor                  # disk quota sweep
    t3-replot card.json.done    # offline re-render of one candidate

## Documentation

- [docs/architecture.md](docs/architecture.md) — dump format, plotting
  pipeline, sky positions, web app.
- [docs/operations.md](docs/operations.md) — per-node deployment, runbooks,
  janitor rules.

## License

MIT — see [LICENSE](LICENSE).
