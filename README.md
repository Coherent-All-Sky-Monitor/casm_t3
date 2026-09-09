# casm_t3

T3 stage of the CASM fast-transient search: turns the beam dumps that
casm_t2 triggers into candidate plots, serves the monitoring web UI, and
keeps the dump disks within quota.

The design constraint that shapes everything here is that bulk data never
crosses the network. One plotter instance runs on each backend node and
only reads that node's dumps; the only things that travel are small PNG
and JSON artifacts. The web app reads the T2 SQLite database read-only —
human labels are its single write path.

## What's here

`dump_reader` parses casm_cand_dump `.dada` files (4096-byte ASCII
header, float16, 64 beams x 3072 channels). `single_pulse` has the
numerics: per-channel normalisation, incoherent dedispersion, boxcar S/N,
a DM-time transform in true S/N units. `plotting` renders the five-panel
candidate figure (dedispersed profile, raw DM=0 timeseries, waterfall,
DM-time bowtie, beam-vs-time context scatter), framed so the pulse always
sits at t=0 and plots are comparable across events. `skypos` converts
beam number plus event time to RA/Dec; the pointing table is an 8 KB JSON
extracted from the beamforming weights so offline nodes don't need the
400 MB HDF5.

The daemons: `t3-dump-plotter` (polls the T2 spool, waits for the dump to
settle, plots, ships artifacts, keeps a per-event archive), `t3-web`
(FastAPI monitor on :8050 — events with per-event trigger/miss reasons,
labelling, day stats, an OVRO clock/source panel, injection recovery),
`t3-janitor` (size and age quotas on the dump trees; never deletes
anything labelled frb or pulsar), and `t3-replot` for offline
re-rendering — use that for all plotter iteration, never card requeue.

## Event archive

Each event gets a directory under `--events-root` (default
`/mnt/nvme3/T3/EVENTS`) on the node that made the dump:

    <events-root>/<candname>/
        <candname>.png     the figure that goes to Slack
        <candname>.json    the result record
        <candname>.fil     the detection beam
        .slack             written by t3-collect once the PNG is posted

The `.fil` is a single-beam float32 SIGPROC filterbank of the detection
beam only, roughly 55 MB. The other 63 beams go away with the `.dada`,
which is still deleted right after plotting — the archive is written
first. Writing it never blocks the figure or the artifact shipping: a
failure is logged and the event goes on.

`t3-collect` rsyncs corr2's event tree to corr1 alongside the artifact
pull, so both nodes' events land in one place, and mirrors the `.slack`
marker into the event directory after a successful post. Pass
`--events-root ''` to the plotter to turn the whole thing off.

`t3-janitor` does not touch the event tree; it only sweeps the `.dada`
dump directories.

## Install

    pip install -e .

Python >= 3.10. Depends on casm_t2 (schema, timing, and beam maps live
there), numpy, matplotlib, astropy, fastapi, jinja2, uvicorn.

See `docs/architecture.md` and `docs/operations.md`.

GPL-3.0 license.
