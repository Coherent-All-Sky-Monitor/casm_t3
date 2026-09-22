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

Layout v3, approved by Vishnu on 2026-09-22 and drawn for `layout="v2"`
(the default the daemons and CLIs pass). Six panels on a fixed 1510 x 1517
px canvas, under a three-line title:

    <candname>   2026-09-22 00:57:02.250 UTC
    hella S/N = 20.7   DM = 247.60 pc cm^-3   width = 33.6 ms
    beam 65: alt 27.0°  az 145.8°   RA 19h13m15.4s  Dec -18d13m31s

    pulse profile at the card DM       | DM = 0 band-mean timeseries
    waterfall at the card DM           | boxcar S/N vs trial DM
    T1 candidates in the gulp          | sky: beams with a T1 candidate

"hella S/N" is hella's reported score. The figure's own S/N values are the
corner text of the two top panels.

Row 1. Both panels plot `single_pulse.profile_snr` at the card width: the
band mean of the per-channel normalised data is boxcar-smoothed and divided
by its robust noise. Left is the band mean dedispersed at the card DM, right
the same array unshifted (DM 0). The noise is measured over the whole series,
wrap tail included. The dedispersed trace stops where the dedispersion wrap
begins. Both panels share the x window (about 1 s either side of the event),
the ticks and one y range from 0: 1.1 x min(max(drawn maxima, 5), 3 x the
event's local peak). A spike above that gets a corner line "max in window:
N (off scale)". Corner text: "peak S/N near event" (maximum within two
boxcars of t = 0) and "peak S/N at DM 0 near event". Every corner line ends
before 0.46 of the axes width; the red t = 0 line is at 0.5. A line that is
too wide wraps, and a render whose corner text still reaches 0.47 is aborted
(`CornerTextError`), so the card is marked failed.

Row 2. The waterfall averages channels into S/N-adaptive subbands (16 to 256
rows, `subbands_for`) and time into one boxcar per column. The column grid is
phased so that the boxcar around the profile peak is exactly one column. Both
image panels take their extent from the column edges, not the centres. The red
curve is where a DM 0 burst at the event time lands after dedispersion. The DM-time panel has its
own finer channel averaging (`dmt_ffactor_for`), since averaging before
dedispersion smears the pulse, and max-pools time bins so a narrow pulse
keeps its peak.

Row 3, with a `gulp` block on the card (casm_t2 CARD_SCHEMA; t2d writes it
from its own clustering of the gulp). Left: two stacked axes sharing x over
the whole gulp span, samp_lo to samp_hi relative to the event. Upper: T1
candidates per 96-sample bin (100.66 ms, bins cut on the gulp's sample grid),
log y. Its title counts len(gulp.trials), or "N of M" when t2d capped the list
(`trials_truncated`). Lower: beam index against time, one marker per
candidate sized by S/N. The triggered cluster's candidates are red, all
others grey, and the beams carry job and correlator labels on the right.
Right: the alt/az sky with one marker per beam that had a candidate: red
for member beams of the triggered cluster (ring on its peak beam), grey
for member beams of any other cluster. A beam in both sets is red only. The
two-line title counts exactly those markers. The sun, Cas A, Cyg A and
Tau A markers keep their legend next to the S/N size key.

Row 3 without a `gulp` block (every card t2d wrote before the gulp block
existed): the pre-v2 panels in their old rectangles. Left, T1 context
members against beam and time, coloured by DM. Right, beams with a member
within the 0.27 s coincidence window. The sky panel is 6% smaller than it
was, to clear the DM-time x label.

Geometry is constants only: no tight_layout, nothing measured from the data,
and every panel reserves a two-line title band. Every candidate therefore has
the same axes to the pixel, and a wrapped title never moves a panel
(tests/test_plotting.py checks both). `layout="legacy"` keeps the
pre-2026-09-03 figure. The approved prototype and reference renders are in
/mnt/nvme3/vishnu/candidate_plot_v3/.

The beam and sky panels show the fastest tell in the figure: RFI lights
many beams at once and a real pulse stays compact on the sky.

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
dump file to land and settle, renders the figure, writes the per-event
archive, ships PNG+JSON artifacts, and renames the card `.done` or
`.failed`. One instance per backend node; bulk dump data never crosses
the network. A node that cannot reach the archive host writes artifacts
locally for t3-collect to pull. The card-to-figure core is the
side-effect-free `render_card()`, which t3-replot reuses — iterate on
plotting with t3-replot, never by re-queueing cards, because the live
plotter may delete a dump after rendering it. When the dump is gone,
t3-replot renders from the archived `.fil` (`fil_reader`,
`render_card_from_fil`).

The per-event archive lives at `<events-root>/<candname>/` (default root
`/mnt/nvme3/T3/EVENTS`) and holds the PNG, the result JSON, a single-beam
float32 SIGPROC `.fil` of the detection beam, and the `.slack` marker
t3-collect writes after posting. The `.fil` is written straight into the
event directory before the `.dada` is deleted; the other 63 beams are not
kept. Every step of it is fail-soft — a full or missing events disk costs
the archive, never the figure or the Slack post. t3-collect pulls corr2's
event tree to corr1 the same way it pulls artifacts.

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
