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
header, float16, 64 beams x 3072 channels), orders the files of one dump by
`OBS_OFFSET`, checks they are contiguous and gives the absolute first sample
(`sample0`) that aligns the dump with hella's flag records. `single_pulse` has the
numerics: per-channel normalisation, incoherent dedispersion, boxcar S/N,
a DM-time transform in true S/N units, and validity-weighted twins of these
for masked data. `plotting` renders the five-panel
candidate figure (dedispersed profile, raw DM=0 timeseries, waterfall,
DM-time bowtie, beam-vs-time context scatter), framed so the pulse always
sits at t=0 and plots are comparable across events. Given hella's recorded
flag mask (`mask=`, `coverage=`, `mask_label=` on `make_candidate_figure`,
v2 layout) every panel is computed from the unflagged cells only, with flagged
cells drawn grey and time with no flag record hatched; see
docs/architecture.md "Masked figures". `skypos` converts
beam number plus event time to RA/Dec; the pointing table is an 8 KB JSON
extracted from the beamforming weights so offline nodes don't need the
400 MB HDF5.

The daemons: `t3-dump-plotter` (polls the T2 spool, waits for the dump to
settle, plots the figure pair, ships artifacts, keeps a schema-2 per-event
folder; `--mask-records DIR` applies hella's own RFI flags),
`t3-janitor` (size and age quotas on the dump trees; never deletes
anything labelled frb or pulsar; `--targets hella_masks` runs the same
kind of sweep over hella's flag-record chunks, run hourly by
`t3-mask-janitor.timer`), and `t3-replot` for offline
re-rendering — use that for all plotter iteration, never card requeue.

The legacy `t3-web` service on 8050 was retired September 25, 2026. Use the
Candidates tab at `http://localhost:8061/cands`. T2/T3 processing, collection,
plotters and weights-watch remain running; saved event data were not removed.

## Weights watch and the beam ellipse

`t3-weights-watch` tails the medusa weights log and records in
`casm_t2.weights_registry` identified non-upload loads. Unknown or mixed loads
leave coordinates unavailable rather than inheriting an unrelated beam table.

Live-tail transfers wait up to 60 seconds (`--upload-grace-s`) for the uploader's
per-stream receipt. Remote log times are corrected using a read-only SSH clock
probe, cached five minutes; the original time and uncertainty remain in evidence.
A failed probe uses the local observation upper bound. No host clock is changed.
START matching uses two raw times from the same node. An unmatched transfer near
an upload, without a START, stays unknown instead of being assigned disk defaults.
There is a bounded attribution delay while a transfer waits; this is not fresh
GPU identity. `--from-start` / `--once` retain historical log clocks and refuse
the live registry path; use a separate registry copy for review. A missing START
near an upload remains ambiguous even when disk defaults match a known product:
the observed defaults hash is retained in evidence, not asserted as FIFO identity.
The pending queue is in memory, so do not restart the watcher during a transfer's
grace period. Historical clock steps cannot be reconstructed from a current probe.

The October 6 timing-race repair is narrowly guarded in
`scripts/repair_20261006_upload_events.py`, dry-run unless `--apply` is supplied.
It backs up the event file, preserves corrected rows under `original`, and never
uploads or posts to Slack. See `casm-wiki/evidence/2026-10-06/upload-registry-race.md`.

Since 2026-09-09 it also gives each product its **beam ellipse**: the E-W and
N-S FWHM of the synthesised beam, computed with
`bf_weights_generator.beam_metadata.beam_fwhm_from_h5` from the product's own h5.
It uses positions of nonzero CB slots across every channel, polarization and
beam, at the saved mean frequency. Metadata-only legacy files use `active_mask`.
The canonical uploader now uses this same lambda/D helper before its live events;
the watcher remains a fallback for missing widths. This union ellipse is not an
exact per-beam response. T2 clusters
candidates on the sky with those FWHMs as its link scales, so the clustering
ellipse now follows the deployed weights instead of a constant in
`config/t2d.yaml` — those config values are the fallback for a product with no
ellipse. T2's product and width caches now refresh changed metadata. Missing or
mixed six-stream evidence is partial, not a trustworthy single pointing; watcher
alerts describe that distinction. Everything is fail-soft: an unreadable h5 or a missing generator logs
a warning and leaves the product without an ellipse.

Products registered before that date carry no ellipse. Fill them in with

    t3-weights-watch --backfill-fwhm --dry-run     # prints what it would write
    t3-weights-watch --backfill-fwhm               # writes it

which walks every product in the registry, reads each h5 still on disk, and
stores the pair on the product record. Products whose h5 is gone are reported
as `unavailable` and left alone.

## Event archive

Each event gets a schema-2 folder (casm_io `docs/t3_events.md`) under
`--events-root` (default `/mnt/nvme3/T3/EVENTS`) on the node that made the dump:

    <events-root>/<candname>/
        <candname>.png            the figure that goes to Slack
        <candname>_raw.png        the same beam before hella flagging
        <candname>.fil            the detection beam
        <candname>.mask.npz       hella's mask aligned to the fil (when records covered the dump)
        <candname>.json           summary: candidate, trigger, provenance, mask statistics
        <candname>.t1.csv  .clusters.csv  .beams.csv
        <candname>.manifest.json  size and sha256 of every file above, written last
        .slack                    written by t3-collect once the PNG is posted

With `--mask-records DIR` (hella's per-gulp flag records, casm_io
`docs/hella_masks.md`) the Slack figure has hella's recorded flags applied:
every panel is computed from the unflagged cells and the title says `hella RFI
flags applied`. The records are found by absolute sample index (the dump
header's `OBS_OFFSET`), so a dump that is not contiguous gets no mask. Without
usable records the Slack figure is the raw beam titled `hella RFI flags
unavailable (<reason>)`. The raw figure's title states the IB subtraction state
read from the dump header (`BFCORR_SUB_INCOH`), `on`, `off` or `unknown`. The
flag is off by default. See docs/architecture.md "Event folder and hella's flags"
and docs/operations.md for the reasons, the deploy order and the delete rule.

The `.fil` is a single-beam float32 SIGPROC filterbank of the detection
beam only, roughly 55 MB. The other 63 beams go away with the `.dada`,
which is deleted right after plotting, but only once the fil, both PNGs, the
summary, the manifest and, for a masked Slack figure, its mask.npz are written;
otherwise the dump stays and the log says what is missing. Writing the fil, the mask or the CSVs never blocks the
figure or the artifact shipping: a failure is logged and the event goes on.

The plotter ships the two PNGs and the summary JSON to the archive tree, then
the manifest. `t3-collect` rsyncs corr2's event tree to corr1 alongside the
artifact pull, so both nodes' events land in one place, and mirrors the `.slack`
marker into the event directory after a successful post. Pass
`--events-root ''` to the plotter to turn the archive off.

Schema-2 folders are posted differently. `t3-collect` posts only
once the manifest exists, parses, and agrees with the PNG sizes on disk; a folder
without it is incomplete and is retried every cycle with no marker. The main figure
goes first (marker line `kind=main`), then `<candname>_raw.png` as a thread
reply to it (`kind=raw`, or `kind=raw_unthreaded` when the main post's message
ts did not resolve). The two posts are tracked separately: a failed reply is
retried on later cycles without reposting the main figure. Schema-1 folders
keep the single-line marker and get no raw post. `t3-collect` never rewrites a
candidate's JSON, which `rsync -a` mirrors from corr2.

`t3-replot --event-dir <folder> --view masked|raw|both` re-renders a stored
folder (either schema) from its fil and mask.

`t3-janitor` does not touch the event tree. It sweeps the `.dada` dump
directories and, with `--targets hella_masks`, hella's mask-record chunks
(docs/operations.md, "Hella mask records").

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
`--clean hella` passes the beam (pulse included) through the deployed hella
flagging stage first (`single_pulse.hella_clean`; also on `t3-replot`),
`--card-json` writes the synthetic card, `--fil` writes the beam plus pulse
as a float32 single-beam filterbank. `--mask-records DIR` with `--raw-out PATH`
draws the replay with hella's recorded flags applied at `--out` and the beam
before flagging at `--raw-out`, both titled `synthetic replay; ...` (the pulse
is added here, the records describe the live stream); without coverage
`--clean hella` keeps the offline replica, labelled as one. The card JSON gains
`kind: "synthetic_replay"` and `hella_mask`. When there is no matched cluster (or
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
