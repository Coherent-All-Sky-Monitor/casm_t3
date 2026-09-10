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

## Weights watch and the beam ellipse

`t3-weights-watch` tails the medusa weights log and records in
`casm_t2.weights_registry` every weights load that was not an upload, so T2 and
T3 always know which product was live at an event time.

Since 2026-09-09 it also gives each product its **beam ellipse**: the E-W and
N-S FWHM of the synthesised beam, computed with
`bf_weights_generator.config.compute_beam_fwhm` from the product's own h5
(`array_config/positions_enu` masked by `array_config/active_mask`, i.e. only
the antennas included in beamforming) at that h5's band centre. T2 clusters
candidates on the sky with those FWHMs as its link scales, so the clustering
ellipse now follows the deployed weights instead of a constant in
`config/t2d.yaml` — those config values are the fallback for a product with no
ellipse. Everything is fail-soft: an unreadable h5 or a missing generator logs
a warning and leaves the product without an ellipse.

Products registered before that date carry no ellipse. Fill them in with

    t3-weights-watch --backfill-fwhm --dry-run     # prints what it would write
    t3-weights-watch --backfill-fwhm               # writes it

which walks every product in the registry, reads each h5 still on disk, and
stores the pair on the product record. Products whose h5 is gone are reported
as `unavailable` and left alone.

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

## Injection replay

Injections are merged into the assembled stream (d2) that hella searches,
while the dump daemons read d0, upstream of that merge — so the pulse that
fired the trigger is by construction absent from every dump (casm-wiki
`dump-stream-content.md`). `t3-replay-injection` puts it back:

    t3-replay-injection --inject-id 661 --dump <file or dir> --out inj661.png

It reads the injection's row from the T2 ledger (read-only) and its matched
cluster, pulls the detection beam out of the dump you point it at, adds the
same synthetic pulse the injection bot generated, and renders the ordinary
v2 figure. `--no-pulse` renders the dump untouched for comparison,
`--card-json` writes the synthetic card, `--fil` writes the beam plus pulse
as a float32 single-beam filterbank. When there is no matched cluster (or
you are replaying into a dump from a different observation), give
`--event-utc` or `--event-offset-s` and, if the beam differs,
`--local-beam`; `--dm/--amp/--sigma-ms` override the injected parameters.

The pulse itself lives in `casm_t3.injection` and is a re-implementation of
`gen_filterbank` in `/home/casm/software/dev/make_noise_fil_with_frb_snr.py`,
down to the integer-sample per-channel delays and the uint8 truncation of
the Gaussian; `tests/test_injection.py` compares it against the generator's
own output sample for sample.

The dump has to be long enough to hold the dispersion sweep after the event
(2.85 s at DM 300): channels whose arrival falls past the end of the dump
contribute nothing and the tool warns that the replayed S/N is then a lower
limit.

## Install

    pip install -e .

Python >= 3.10. Depends on casm_t2 (schema, timing, and beam maps live
there), numpy, matplotlib, astropy, fastapi, jinja2, uvicorn.

See `docs/architecture.md` and `docs/operations.md`.

GPL-3.0 license.
