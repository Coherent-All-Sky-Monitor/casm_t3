# casm_t3 architecture

## Reading dumps

casm_cand_dump writes `.dada` files: a 4096-byte ASCII header, then
float16 frames ordered (outer_time, beam, channel, inner_time), 64 beams
x 3072 channels x 64 inner samples per frame. The absolute time of the
first sample is `UTC_START` + `OBS_OFFSET` / `BYTES_PER_SECOND`.
dump_reader returns the (nchan, ntime) cutout for one beam plus its start
time. Several files of one dump are ordered by each header's `OBS_OFFSET`,
numerically (the byte offsets in the names are not zero-padded, so a name sort
puts 10000000 before 9000000), and must butt together: each file starts at the
previous file's `OBS_OFFSET` plus its data bytes, the previous file ends on a
whole frame, and all are one observation. The returned header carries
`sample0` (`OBS_OFFSET` / `bytes_per_sample`, the absolute sample index from
`UTC_START`; `bytes_per_sample` = nbeam x nchan x nbit / 8), `bytes_per_sample`,
`contiguous` and `contiguity_note` (the first failed check). A dump with a gap,
an overlap, or an `OBS_OFFSET` that is not a whole number of samples is still
read in order, but its sample indices are not trusted, so no hella record is
looked up for it. `read_beams(..., sort_by_name=False)` keeps the caller's order.
single_pulse has the numerics on top of it — robust per-channel
normalisation, roll-based incoherent dedispersion, block downsampling,
matched boxcar S/N, and a DM-time transform whose output is in true S/N
units (which is why the bowtie panel can carry an honest colorbar). Each step
has a validity-weighted twin (`normalise_masked`, `dedisperse_masked`,
`band_profile_masked`, `profile_snr_masked`, `downsample_masked`,
`dm_time_masked`) that the figure uses only when it is given a flag mask; see
"Masked figures". The raw functions are unchanged.

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
the ticks and one y range from -4 (`TOP_Y_FLOOR`, so the noise is visible;
since 2026-10-12, it was 0) to 1.1 x min(max(drawn maxima, 5), 3 x the
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

## Hella-like cleaning (`clean="hella"`)

`single_pulse.hella_clean` applies, offline, what the deployed hella fork
(casm_hella_vishnu `casm_hella_refactored`, `src/hella/src/flagger.cu` and
`timeseries.cu`, live `/tmp/hella_N.cfg`, read 2026-10-12) does to every
gulp before dedispersion, so the figure shows roughly what the search saw:

1. per channel, (x - mean) / std over the gulp (plain moments, not
   median/MAD); a zero-variance channel is blanked;
2. the static mask: channels 0-471 (484.375-470.0 MHz, the DTV block,
   `FLAG1 0 / FLAG2 472`) set to 0;
3. the common-mode baseline: band mean over all 3072 channels (masked zeros
   included), running median over 0.2 s (191 samples, window clamped inside
   the series), fitted to each channel by least squares and subtracted;
4. one global scale for the gulp and a symmetric +-10 sigma clip.

Every statistic is per 8192-sample gulp with no history, as in hella. The
pieces are phased to hella's gulp boundaries (multiples of 8192 hella
samples) through the card's `samp` and the event's offset into the cutout;
a card without `samp` starts the pieces at the first sample and the log
says so. The pieces at either end of a cutout are shorter than a gulp. Not
reproduced: the uint8 quantisation and the per-DM-row clipped
renormalisation after dedispersion. The deployed configuration has no
time-frequency flagger (its one SCRUNCH line has a threshold of 1e6 sigma
and exists only to make the static mask run), so there is none to
reproduce. The per-channel median/MAD normalisation still runs afterwards;
on cleaned data it is close to the identity.

`make_candidate_figure(..., clean="hella")` (v2 layout only), `t3-replot
--clean hella` and `t3-replay-injection --clean hella` switch it on; the
DM = 0 panel title then ends in ", hella-cleaned" and the result record
carries `clean`. The default is the data as recorded.

The SCRUNCH stages are configurable (`scrunches=`, CLI `--scrunch T,F,SIGMA`,
repeatable, in config order) with hella's semantics: each stage
re-standardises, box-averages over T samples x F channels, masks
|value| > SIGMA / sqrt(T F), grows the mask by the same box and replaces with
the channel mean. The default is the deployed never-firing stage. The
DM = 0 title and the result record name the stages used. Measured on
inj_20261012_0012 (2026-10-12, docs/operations.md): the single-channel
0.1-1 s bursts that survive the deployed chain are removed by a (16,1)
stage, and SIGMA 8 (a 2 sigma cut on the 17 ms mean) keeps every synthetic
pulse detectable, whereas SIGMA 4 removes pulses above S/N ~250.

`single_pulse.sumthreshold_flag` (`sumthreshold=`, CLI `--sumthreshold
default` or `KEY=VAL,...`) is a stage hella does not have, for offline
comparison: SumThreshold (Offringa et al. 2010) along time against a
channel-neighbour reference. For each window M = mmin, 2 mmin, ... mmax
(defaults 16 to 512 samples) the standardised gulp is box-averaged over M
samples (window clamped inside the gulp, hella's running-median rule, so
the edge bins see the first or last full window and no extra noise), the reference is the median over the nb (default 4) channels on
either side of that averaged data, and a window whose |self - reference|
exceeds chi1 / rho**log2(M) (defaults 8 and 1.5, so 1.58 sigma on the
16-sample mean down to 0.21 sigma on the 512-sample mean) is flagged over
its M samples and replaced by the channel mean before the next M. Broadband
emission is in the reference as much as in the channel and leaves no
residual whatever its brightness; a dispersed pulse stays inside its
neighbours' windows as long as the sweep across nb channels is shorter than
M, which is why M starts at 16 rather than 1 (at DM 300 the sweep across
+-4 channels is +-3 samples). Masked channels are excluded from the
reference and never flagged. On pure noise the stage flags about 0.1 % of
samples. It runs after the scrunch stages, in numpy and scipy, about 35 s
per gulp.

After any flagging (scrunch or SumThreshold) the final per-channel moments
are taken over the unflagged samples only, flagged samples are returned to
exactly 0 after the baseline and global steps, and a channel more than half
flagged is blanked. hella re-measures over every sample after its
replacement; the difference matters only when a lot is flagged. The figure's
own per-channel median/MAD normalisation runs on the cleaned array, and a
channel with a long near-constant replaced run would otherwise be inflated
by it (channel 2728 of inj_20261012_0012 went to a std of 653 and the pulse
S/N from 24 to 9 before this was handled).

Injection replay cards (`t3-replay-injection`) carry t2d's injection gulp
block when one exists, with the injection cluster set to "triggered" in the
replay's copy, so the injection is the red group and the sky title reads
"triggered group"; see operations.md.

Row 3 without a `gulp` block (every card t2d wrote before the gulp block
existed, and injection replays with no block): the pre-v2 panels in their old rectangles. Left, T1 context
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

## Masked figures (`mask=`, `coverage=`, `mask_label=`)

`make_candidate_figure` and `draw_candidate_figure` (v2 layout only) take
hella's recorded flag decisions and draw every panel from the unflagged cells.

| argument | shape | meaning |
|---|---|---|
| `mask` | (nchan, ntime) bool, aligned with `data` | True = flagged by hella. Same channel order as `data` (channel 0 = 484.375 MHz). |
| `coverage` | (ntime,) bool | False where hella has no record, so the flags there are unknown. None = every sample covered. Needs `mask`. |
| `mask_label` | str | Replaces the title suffix. With no `mask` it marks a raw figure. |

Three states: no arguments draws the dump as recorded, exactly as before (the
PNG matched the pre-mask code byte for byte on seven synthetic cards when the
mask path was added, and `tests/test_masked_figure.py` compares the drawn
arrays with frozen copies of the old numerics);
`mask_label` alone draws the same raw figure with the label on the title line;
`mask` draws the masked figure. `mask` cannot be combined with
`clean="hella"` (that is the offline replica, the mask is hella's own
record) and the legacy layout rejects all three. `draw_candidate_figure`
returns a `"mask"` entry with `flagged_frac` (of the covered cells, static
band included), `coverage_frac`, `n_live_channels`, `min_nvalid` and the
title `tag`.

Numerics. A masked array is a pair (data, valid). A cell is valid when it is
not flagged, its sample has a flag record and it is finite. Invalid cells hold
0 and validity travels with the data through each step, because np.median,
np.mean and np.convolve spread a NaN over the whole series and a mean over a
varying number of cells has varying noise.

- `normalise_masked`: median and 1.4826 x MAD per channel from that channel's
  valid samples. Flagged cells are excluded, not zero-filled. A channel with
  fewer valid samples than `MIN_CHAN_VALID_FRAC` (10%) of the best channel's,
  or than `MIN_CHAN_VALID_SAMPLES` (64), is invalid everywhere: the median/MAD
  of fewer samples scatter by more than 10%.
- `dedisperse_masked`: each channel is shifted as in `dedisperse`, validity
  with it. The samples np.roll would wrap round are invalid, not recycled.
- `band_profile_masked`: sum over valid channels divided by sqrt(N_valid(t)),
  so a unit-noise channel set gives a profile with unit noise at every t
  whatever the mask. N_valid below `MIN_NVALID_FRAC` (10%) of the live
  channels (channels valid at all after normalisation; at least
  `MIN_NVALID_CHANNELS`, 8) makes the sample NaN. At 10% a pulse already
  loses a factor 3 in S/N, and the profile would rest on a few channels of
  residual RFI.
- `profile_snr_masked`: the boxcar sums the available samples of each window
  (same alignment as `profile_snr`) and divides by sqrt(n_available). A window
  with fewer than `MIN_WINDOW_SUPPORT` (half) of its samples available is NaN,
  so a gap lowers the S/N of the windows that touch it and never spreads
  further. The series is then divided by its own robust noise, as before. Fewer
  than `MIN_NOISE_SAMPLES` (16) finite values gives all NaN.
- `downsample_masked`: block mean over the valid cells; a display block with
  fewer than `MIN_BLOCK_VALID_FRAC` (half) valid is NaN.
- `dm_time_masked`: the DM-time panel averages channels first, so it works on
  block sums and valid counts, shifts both without wrap-around, and applies the
  same sum / sqrt(N_valid) and `profile_snr_masked` per trial DM. With nothing
  flagged it reproduces `dm_time` up to the noise estimate.

The raw path (no `mask`) has its own guard: `normalise` leaves non-finite cells
(NaN, Inf) out of the per-channel median and MAD and returns 0 there, so a
channel with no finite cell is all 0. One NaN channel would otherwise turn every
dedispersed profile to NaN and blank the figure. An all-finite array takes the
original code path and gives the same bits as before.

Uncovered samples are unavailable, not unflagged data: they are excluded from
the normalisation statistics and every profile, and the profile is NaN where
no channel has a record.

Drawing. Flagged cells are NaN in the image colour maps and show in the colour
map's lowest colour (`FLAGGED_COLOR`), as hella's zeroed cells look in the
injection replay. Cells whose source samples have no record show as a light
grey under a hatch (`UNCOVERED_COLOR`), so "unknown" differs from
"flagged"; the waterfall classifies them per cell in the dedispersed frame
(a gap is slanted by the sweep), the DM-time panel per trial DM. A small key on
each image panel lists the two kinds actually drawn in its window. It sits under
the panel, right-aligned with the axes edge on the x-label line (`KEY_DROP_PT`
below the axes), so it covers no data cell, title or tick label. In the
profile panels a gap is a break in the line, and a hatched band marks where it
is missing because the flags are unknown. The bottom row, the T1 trials of the
card's gulp block, is not touched, and no axes moves (the geometry is constants).

Title. The suffix goes at the end of line 2, after the width:

| call | suffix |
|---|---|
| `mask` | `hella RFI flags applied` |
| `mask` and `coverage` with any False | `hella RFI flags applied (partial coverage)` |
| `mask_label` alone (raw figure) | the label, for example `raw intensity, before hella RFI flagging; IB subtraction off` |
| `mask` and `mask_label` | the label, plus ` (partial coverage)` when coverage is partial |

"Partial coverage" refers to the whole data window, which is what the
normalisation statistics use, not only the plotted second either side of the
event. The wording is "applied": hella's standardisation, baseline pass, clip
and uint8 quantisation are not reproduced, so the figure is not what hella saw.

What this does not do. The 1/sqrt(N_valid) weighting makes the noise uniform
only to the extent that normalised channels are independent; the reported S/N
is still measured from the series, so correlated channels are absorbed in the
overall scale but not in its variation with N_valid. S/N is the figure's own,
not hella's. Tests: `tests/test_masked_numerics.py`, `tests/test_masked_figure.py`;
`python tests/synthetic_mask.py <outdir>` renders a masked and a raw figure of a
synthetic dump with a known pulse, a narrow-band burster and a blob.

## Event folder and hella's flags

`t3-dump-plotter` builds one schema-2 folder per card (casm_io `docs/t3_events.md`)
in `<events-root>/<candname>/` (with `--events-root ''` the same files are built
flat in `--plots-dir`, without the fil). Per card:

1. `dump_reader.read_beams` reads the detection beam (`card["local_beam"]`) and
   the header's `sample0`, `contiguous` and `BFCORR_SUB_INCOH`; the beam goes to
   `<event>.fil`.
2. `lookup_mask` asks casm_io `hella_mask.window_mask(--mask-records, UTC_START,
   card["beam"], sample0, nsamp)` for hella's flags over exactly the samples
   read. It never raises; it returns a window and a short reason.
3. With coverage, `<event>.mask.npz` is written (casm_io `save_event_mask`) and
   `<event>.png` is drawn with `mask=win.combined, coverage=win.coverage`.
   Without it, or if that render raises, `<event>.png` is the raw beam labelled
   `hella RFI flags unavailable (<reason>)` and the mask file is removed.
4. `<event>_raw.png` is always the raw beam, labelled `before hella RFI
   flagging; IB subtraction on|off|unknown`. The state is `BFCORR_SUB_INCOH` of
   the dump header ("1"/"true" on, "0"/"false" off; the summary records
   `sub_incoh` true, false or null and `sub_incoh_source` `dump_header` or
   `unknown`). It is never assumed.
5. `t1.csv`, `clusters.csv` and `beams.csv` come from the card (casm_io
   `write_sidecars`), then the summary JSON (casm_io `summary_from_card`), then
   `<event>.manifest.json` last (`write_manifest`).
6. The PNGs and JSON are copied to `--plots-dir` and shipped to the archive tree,
   and the manifest ships after them, so a reader that waits for it never sees a
   half-shipped event.
7. The dump is deleted only when `--delete-dumps` is set, the trigger is not a
   known source, and the fil (when the archive is on), both PNGs, the summary and
   the manifest were all written, plus `mask.npz` when the Slack figure has hella's
   flags applied (the figure cannot be redrawn without it). The CSVs are optional,
   and so is `mask.npz` for a raw Slack figure. Otherwise the dump stays and the
   log names what is missing. A failed summary or
   manifest write raises, so nothing ships and the card is renamed `.failed`.

Reasons a figure has no hella flags (the label and `hella_mask.reason`):

| reason | cause |
|---|---|
| `mask records off` | `--mask-records ''` |
| `dump files not contiguous` | gap, overlap, partial frame or mixed observations (the log has the detail) |
| `dump has no OBS_OFFSET` / `dump header has no UTC_START` | alignment unknown |
| `mask record dir missing` | the records root does not exist on this node |
| `no hella record covers the dump` | root exists, nothing aligned and checksummed overlaps the window |
| `mask lookup failed: <Error>` | any other exception from casm_io |
| `mask render failed: <Error>` | the masked figure raised, for example a mask whose channel count differs from the dump |

Partial coverage still draws the masked figure; the title says `(partial
coverage)` and uncovered time is hatched ("Masked figures"). The flags apply
to the samples of the dump by index. The records and the dump both count
samples from `UTC_START`, so the corr1 clock offset that shifts `tstart` and
`event_utc` does not enter.

The summary's `provenance` holds the observation `UTC_START` (dump header),
`weights_id` and beam FWHMs (card), `t2d_commit` (card), `t3_commit` (git of the
casm_t3 source, computed once at startup: `<sha>`, `<sha>+dirty` or `unknown`),
`hella_binary_md5`, `hella_cfg_sha256` and `hella_flag` (flag1, flag2 and the
SumThreshold settings, from the first record header used; null without a mask),
`sub_incoh`, `sub_incoh_source` and `utc_clock`. `dump` has `utc_start`,
`utc_stop` (from the data read), `sample0`, `nsamp` and `host`. `data_available`
keeps its meaning (false once the dump is deleted). The `hella_mask` block is
casm_io's, plus `reason` when the mask was computed but could not be stored
(`mask.npz could not be written`). Its `searched` is the fraction of window samples
whose gulp record's `n_beams_searched` reached the detection beam, that is whether
hella's search step that ran right after flagging that gulp covered the beam. It is
not per-sample search coverage: hella searches a gulp's samples over several
consecutive steps (its rewind buffer is longer than a gulp) and skips the first steps
of an observation, so an event hella reported can sit in a gulp with `searched` 0.

`t3-replay-injection` gives the replay the same pair. With `--mask-records` and
records that cover the replay window for the injected global beam, `--out` is
drawn with hella's flags applied (titled `synthetic replay; hella RFI flags
applied`) and `--raw-out` is the beam before flagging (`synthetic replay; before
hella RFI flagging`). The pulse is added by the tool and the records describe
the live stream that carried the real injection, so the title says so. The mask
replaces `--clean hella`. Without coverage the old path stays: `--clean hella`
draws the offline replica (titled `synthetic replay; offline replica of hella
flagging`), otherwise the dump as recorded, labelled unavailable when records
were asked for. The card JSON gains `kind: synthetic_replay`, `hella_mask`
(casm_io block) and `replay.raw_plot`; the mask is stored beside `--fil`.

`t3-replot --event-dir DIR` re-renders a stored folder of either schema from its
fil: `casm_io.load_event` reads it, `event_archive.card_from_event` rebuilds the
card (`gulp` block, `context.members` and `pointings` come back from the CSVs;
a schema-1 folder's JSON is the card), `--view masked|raw|both` draws the stored
mask, the raw beam or both. A mask whose shape differs from the fil is an error.
A folder written by the plotter re-renders to the same PNG bytes.

Tests: `tests/test_event_folder.py` (end to end on a synthetic dump and records),
`tests/test_dump_reader.py`, `tests/test_replay_masked.py`,
`tests/test_replot_event_dir.py`; `python tests/synthetic_event.py <outdir>`
renders a masked and a raw figure of the synthetic event.

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
dump file to land and settle, renders the figure pair, writes the per-event
folder, ships the PNGs, summary JSON and manifest, and renames the card `.done` or
`.failed`. One instance per backend node; bulk dump data never crosses
the network. A node that cannot reach the archive host writes artifacts
locally for t3-collect to pull. The card-to-figure core is the
side-effect-free `render_card()` (raw figure only), which t3-replot reuses;
`render_event()` is the daemon's version that adds the mask and the raw twin.
Iterate on plotting with t3-replot, never by re-queueing cards, because the
live plotter may delete a dump after rendering it. When the dump is gone,
t3-replot renders from the archived `.fil` (`fil_reader`,
`render_card_from_fil`).

The per-event archive lives at `<events-root>/<candname>/` (default root
`/mnt/nvme3/T3/EVENTS`) and is a schema-2 folder (casm_io `docs/t3_events.md`):
the two PNGs, the `.fil` of the detection beam, `<candname>.mask.npz` when hella
records cover the dump, the summary JSON, three CSVs, the manifest written last
and the `.slack` marker t3-collect writes after posting ("Event folder and
hella's flags" below). The `.fil` is written straight into the event directory
before the `.dada` is deleted; the other 63 beams are not kept. Writing the fil,
the mask or the CSVs is fail-soft: a full or missing events disk costs those
products, never the figure or the Slack post. t3-collect pulls corr2's
event tree to corr1 the same way it pulls artifacts. For a schema-2 folder it
posts only after the manifest the plotter writes last, then sends the raw
figure as a thread reply; the two posts are separate `.slack` marker lines so a
failed reply is retried without reposting the main figure (operations.md,
Runbooks).

t3-janitor sweeps the dump trees on both nodes (ssh for the remote one):
per-tree size quota and a maximum age, oldest plotted-and-unlabelled
first. Dumps of events labelled frb or pulsar are never deleted.
Deletions go into the trigger audit.

The same janitor has a second target, `--targets hella_masks`, for hella's
flag-record chunks (casm_io `docs/hella_masks.md`). It deletes closed chunks
past an age limit or over a per-node quota, never an open chunk, and keeps the
chunks a pending trigger card in the T2 spool could still need; an open chunk
(`.hmr.part`) untouched for twice the age limit has no live writer and is deleted
as an orphan, unless pinned. Chunk names are
`hmask_b<beam0>_g<gulp>_p<pid>.hmr`. It runs hourly from `t3-mask-janitor.timer`
and does not open the trigger database. Rules and pins: operations.md, "Hella
mask records".

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
