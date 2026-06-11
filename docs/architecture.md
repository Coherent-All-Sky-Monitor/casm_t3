# casm_t3 architecture

## Reading dumps

casm_cand_dump writes `.dada` files: a 4096-byte ASCII header, then
float16 frames ordered (outer_time, beam, channel, inner_time), 64 beams
x 3072 channels x 64 inner samples per frame. The absolute time of the
first sample is `UTC_START` + `OBS_OFFSET` / `BYTES_PER_SECOND`.
dump_reader returns the (nchan, ntime) cutout for one beam plus its start
time; single_pulse has the numerics on top of it — robust per-channel
normalisation, roll-based incoherent dedispersion, block downsampling,
matched boxcar S/N, and a DM-time transform whose output is in true S/N
units (which is why the bowtie panel can carry an honest colorbar).

## The candidate figure

Five panels, following transientX and the DSA-110 plotter, viridis for
both image panels:

    dedispersed profile     | DM=0 raw timeseries
    dedispersed waterfall   | DM-time bowtie
    beam vs time of T1 context candidates (full width)

Framing is fixed relative to the event — the pulse sits at t=0 in every
plot, the dedispersed panels show about a second of context (more for
wide pulses), and the DM-time window scales with the bowtie wings. The
one exception is the DM=0 panel, which keeps the full dump span: its job
is showing the RFI weather around the event, not the pulse. The suptitle
carries name, S/N, DM, width, and the beam's RA/Dec at the event time.

The beam panel comes from the T1 context candidates embedded in the
trigger card: every candidate from every beam within a few seconds of the
event. RFI lights up many beams at unrelated DMs; a real pulse stays
compact in beam and DM. It is usually the fastest tell in the figure.

## Sky positions

Beams are fixed in (alt, az) — the array doesn't track — so a beam's
RA/Dec depends only on the event time: AltAz at OVRO to ICRS via astropy
(skypos.py). The 512-row pointing table is extracted from the deployed
beamforming-weights HDF5 into `casm_t3/data/beam_pointings.json` (8 KB),
so the 400 MB weights file never needs to leave the node it lives on.
Regenerate with `python -m casm_t3.skypos <weights.h5>` when new weights
are deployed. IERS auto-download is off: offline nodes extrapolate the
bundled tables, a milliarcsecond error against degree-scale beams.

## Daemons

t3-dump-plotter polls the local T2 spool for trigger cards, waits for the
dump file to land and settle, renders the figure, ships PNG+JSON
artifacts, and renames the card `.done` or `.failed`. One instance per
backend node; bulk dump data never crosses the network. A node that
cannot reach the archive host writes artifacts locally for t3-collect to
pull. The card-to-figure core is the side-effect-free `render_card()`,
which t3-replot reuses — iterate on plotting with t3-replot, never by
re-queueing cards, because the live plotter may delete a dump after
rendering it.

t3-janitor sweeps the dump trees on both nodes (ssh for the remote one):
per-tree size quota and a maximum age, oldest plotted-and-unlabelled
first. Dumps of events labelled frb or pulsar are never deleted.
Deletions go into the trigger audit.

## Web monitor

Server-rendered FastAPI + Jinja, no JS build chain, LAN only, no auth.
It reads the T2 database read-only; the label buttons are the single
write path (an `frb` label also promotes the event into the catalog).

The events table defaults to dump attempts — every row has either a plot
or a red reason for the miss — with `?view=all` showing every stored
event and why it was held (injection, veto, wide-beam RFI, tier, DM
floor). `/stats` renders the observing-day funnel charts server-side from
the per-gulp counters, never the raw T1 stream, plus a live OVRO
local/UTC/LST clock and a bright-source transit table. `/injections`
shows the ledger with its recovery gates and the injected-vs-recovered
S/N scatter.

Chart PNGs are cached for 60 s and written atomically so a refresh never
sees a half-drawn file. SQLite timestamp comparisons use Python-built ISO
cutoffs (`statsplot.utc_cut`) — sqlite's space-separated `datetime('now')`
strings compare wrongly against the DB's `T`-separated timestamps.
