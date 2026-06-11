# casm_t3 architecture

## Dump reader (`dump_reader.py`)

casm_cand_dump writes `.dada` files: a 4096-byte ASCII header, then
float16 frames ordered (outer_time, beam, channel, inner_time) with
64 beams x 3072 channels x 64 inner samples per frame. The absolute time
of the first sample comes from `UTC_START` + `OBS_OFFSET` /
`BYTES_PER_SECOND`. The reader returns the (nchan, ntime) cutout for one
beam plus its start time.

## Single-pulse toolbox (`single_pulse.py`)

Robust per-channel normalisation, roll-based incoherent dedispersion,
block downsampling, matched boxcar profile S/N, and a DM-time transform
whose output is in true S/N units.

## Candidate figure (`plotting.py`)

Five panels, transientX/DSA-110 conventions, viridis for both image
panels:

    dedispersed profile     | DM=0 raw timeseries
    dedispersed waterfall   | DM-time bowtie (S/N colorbar)
    beam vs time of T1 context candidates (full width)

Framing is fixed relative to the event (pulse at t=0) so plots are
directly comparable: the dedispersed panels show ~1 s of context (more
for wide pulses), the DM-time window scales with the bowtie wings, and
the DM=0 panel keeps the full dump span as RFI context. The suptitle
carries the event name, S/N/DM/width, and the beam's RA/Dec.

## Sky positions (`skypos.py`)

Beams are fixed in (alt, az); RA/Dec follows from the event time via
astropy at the OVRO site. The 512-row pointing table is extracted from
the deployed beamforming-weights HDF5 into
`casm_t3/data/beam_pointings.json` (8 KB) so it travels with the repo;
regenerate with `python -m casm_t3.skypos <weights.h5>` when new weights
are deployed. IERS auto-download is disabled — offline nodes extrapolate
the bundled tables, a milliarcsecond-level error against degree-scale
beams.

## Plotter daemon (`apps/dump_plotter.py`)

Polls the local T2 spool for trigger cards, waits for the dump file to
land and settle, renders the figure, ships PNG+JSON artifacts, and
renames the card `.done`/`.failed`. One instance per backend node; bulk
dump data never crosses the network (nodes that cannot reach the archive
host write artifacts locally for the collector to pull). The
card-to-figure core is the side-effect-free `render_card()`, which
`apps/replot.py` reuses for offline re-rendering — use `t3-replot` for
all plotter iteration, never card requeue.

## Web monitor (`web/`)

Server-rendered FastAPI + Jinja, no JS build chain, LAN only. Reads the
T2 SQLite database read-only; labels are the only write path.

- `/` — events table. Default view shows dump attempts (each has a plot
  or a red miss reason); `?view=all` shows every stored event with the
  reason it was held (injection / veto / wide-beam RFI / tier / DM floor).
- `/event/<name>` — plots, trigger audit, data status, label buttons;
  an `frb` label promotes the event into the `frbs` catalog.
- `/stats` — observing-day funnel charts rendered server-side from the
  per-gulp counters (never the raw T1 stream), plus a live OVRO
  local/UTC/LST clock and bright-source transit table.
- `/injections` — ledger with recovery gates, injected-vs-recovered S/N,
  and per-shot recovery vs DM.
- `/frbs` — the catalog.

Chart PNGs are cached for 60 s and written atomically. SQLite timestamp
comparisons must use Python-built ISO cutoffs (see `statsplot.utc_cut`).

## Janitor (`apps/janitor.py`)

Sweeps the dump trees on both nodes (over ssh for the remote one):
per-tree size quota and maximum age, oldest plotted-and-unlabelled dumps
deleted first. Dumps of events labelled `frb` or `pulsar` are never
deleted. Deletions are recorded in the trigger audit.
