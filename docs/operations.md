# casm_t3 operations

## Deployment

systemd user units, `deploy/systemd/`. Run `loginctl enable-linger` once
per node or they die at logout.

| unit | host | role |
|---|---|---|
| t3-dump-plotter-corr1 | corr1 | plot local dumps |
| t3-dump-plotter-corr2 | corr2 | plot local dumps, stage artifacts |
| t3-collect | corr1 | pull corr2 artifacts and events (corr2 cannot ssh back); post candidates to Slack, clickable candidate ID -> casm_monitor `http://localhost:8061/cands/<name>` (`--web-base`) |
| t3-web | corr1 | Retired September 25, 2026; stopped/disabled. Use casm_monitor on 8061. |
| t3-janitor | corr1 | disk quota sweep of the dump trees across both nodes |
| t3-mask-janitor (timer, hourly) | corr1 | `t3-janitor --targets hella_masks --once`: age and quota sweep of hella's mask records on both nodes |

Both nodes run the same venv. The second node has no internet — install
there from wheels copied across, and keep the venvs in sync.

## Runbooks

Iterating on the candidate figure:

    t3-replot /path/to/card.json.done --out /tmp/test.png

re-renders offline from the on-disk dump, on the node that holds it.
When the dump is gone, or the card never had one (T2 backtest cards), it
renders from the archived single-beam filterbank instead: the card's own
`fil` if that file exists, else `<events-root>/<candname>/<candname>.fil`
(default root `/mnt/nvme3/T3/EVENTS`). `--fil PATH` names the file
explicitly. `--events-root ''` turns the fallback off. A card with no
`dump_dir` is never looked up in the current directory.

    t3-replot /mnt/nvme3/T3/EVENTS/260922lruuie/260922lruuie.json --out /tmp/x.png

`--clean hella` (also on `t3-replay-injection`) first passes the cutout
through the deployed hella flagging stage (static 470-484 MHz mask,
per-gulp channel standardisation, 0.2 s common-mode fit, +-10 sigma clip;
architecture.md "Hella-like cleaning") so the panels show roughly what the
search saw. First used on inj_20261012_0009 (2026-10-12): the 442 MHz line
that dominates the recorded waterfall is down-weighted away, the masked
block is blank, the replayed peak S/N moved from 23.7 to 25.4 against
hella's 23.8.

`--scrunch T,F,SIGMA` (repeatable) replaces the deployed never-firing SCRUNCH
stage by real ones, hella semantics. Offline trial on inj_20261012_0012,
2026-10-12 (scratch scripts `bright_pulse.py`, `width_test.py`, `full_tf.py`;
figures in `/mnt/nvme5/casm_pipeline/scratchpad/hella_clean_20261012/`):
the RFI left by the deployed chain is one channel wide, 0.1-1 s long, 2-5
sigma per sample, at 401-437 MHz. `(1,1,5)` alone barely touches it.
`(1,1,5) (16,1,4) (1,16,4)` removes it but makes pulses brighter than
S/N ~250 undetectable (S/N 258 -> 14, 775 -> 6). `(1,1,5) (16,1,8) (1,16,8)`
removes 8 of the 9 hot channels, flags 0.03% of the band, and every pulse
from 1 to 60 ms FWHM and 6 to 250 counts stays above S/N 30 (bright ones are
reduced, e.g. 648 -> 33 at 26.6 ms, 100 counts). The `(1,16,8)` stage adds
nothing on this RFI; `(16,1,8)` does the work. Nothing of this is live.

`--sumthreshold default` (or `KEY=VAL,...`) adds the channel-neighbour
SumThreshold stage (architecture.md). Same trial, same day
(`st_compare.py`): zero hot channels left, 0.3-0.5 % of the band flagged,
6-10 channels blanked per gulp; dim pulses unchanged (6-15 counts within a
few percent of the deployed chain at every width 1-60 ms) and bright ones
13-15 % *above* the deployed chain (120 counts: 776 -> 898; 26.6 ms, 250
counts: 1447 -> 1650), because the RFI noise is gone and nothing broadband
is flagged. Combining it with `(1,1,5)` brings the amplitude cliff back
(400 counts: 2033 -> 248); do not. Report with figures:
`/mnt/nvme5/casm_pipeline/scratchpad/hella_clean_20261012/hella_cleaning_replay.html`.

Never re-queue cards to test plotting: the live plotter may delete the
dump after rendering, and a re-queued card can fire side effects.

The figure's bottom row depends on the card. Cards carrying a `gulp` block
(written by t2d from the card-gulp-block change on) get the gulp candidate
histogram, beam scatter and the red/grey sky. Older cards get the pre-v2
context-member panels under the new top rows. Both are expected.

Injection replays (`t3-replay-injection`) name the figure
`INJECTION: <file_id>`. Line 2 of the title is hella's recovered S/N at the
ledger DM. The injected parameters are written to the card JSON as
`injection.summary`, one line for the Slack thread, and never reach the title.

A replay gets the v3 bottom row when t2d wrote a gulp block for the injection
gulp (casm_t2 `injection.gulp_dir`, CARD_SCHEMA.md "Injection gulp blocks").
`--gulp-dir` (default `/mnt/nvme5/casm_pipeline/injection_gulps`, on corr1
where the replay runs) is searched for the matched cluster's gulp; the block
whose `reference` is the matched cluster (by id, else by observation, peak
sample and beam) is attached as `card["gulp"]` with the injection as the red
("triggered") group. A real event that triggered in the same gulp is drawn
grey as `also_triggerable`. dt_s is re-referenced when `--event-utc` differs
from the cluster peak. No block (unrecovered shot, shots before 2026-09-22, or
`--gulp-dir ''`) gives the legacy bottom row, as before.

Retroactivity: anything baked into the PNG (layout, titles, framing)
applies from the next event onward, because dumps may already be gone.
Anything computed from the DB at page-render time — reasons, tables,
stats — is retroactive for free. Known-source dumps are kept on disk and
can always be re-rendered.

New beamforming weights:

    python -m casm_t3.skypos /path/to/new_weights.h5

regenerates the pointing sidecar; commit the JSON.

Janitor dry run before trusting a config change:

    t3-janitor --dry-run

prints would-deletes and touches nothing. Whatever the config says, the
janitor refuses to delete dumps of events labelled frb or pulsar.

Hella mask records (`--targets`, default `dumps`; `hella_masks` or
`dumps,hella_masks`). The `hella_masks` target holds the record chunks
hella writes under `--mask-root` (default `/mnt/nvme4/data/casm/hella_masks`,
casm_io `docs/hella_masks.md`) to an age limit and a quota on each node, corr2
through the same ssh as the dump sweep, and never opens the T2 database:

    t3-janitor --targets hella_masks --dry-run     # prints would-deletes, touches nothing
    t3-janitor --targets hella_masks --once

| flag | default | meaning |
|---|---|---|
| `--mask-root` | `/mnt/nvme4/data/casm/hella_masks` | `<root>/<UTC_START>/hmask_b<beam0>_g<first_gulp>.hmr` on each node |
| `--mask-max-age-h` | 24 | delete closed chunks older than this (file mtime) |
| `--mask-max-gb` | 100 | per-node quota; counts open and closed chunks, deletes closed ones oldest first |
| `--mask-pin-max-age-h` | 72 | a pending trigger card older than this stops pinning chunks |
| `--spool` | `/mnt/nvme4/data/casm/t2_spool` | T2 spool whose pending cards pin chunks |

Rules. Only a closed `.hmr` is ever deleted; an open `.hmr.part` never is, even
when it is old (the log warns about an open chunk untouched for the age limit).
The quota deletion credits what the age rule already removes. Pinned and open
chunks stay under both rules, so a node can sit over quota while they alone hold
it; the log says so. An observation directory is removed once it holds no chunk
files and has not changed for 10 minutes (`rmdir` refuses a directory with
anything else in it). When the chunk listing or the spool scan for a node fails
(ssh down), nothing is deleted on that node in that pass.

Pins. A chunk is kept whatever its age or the quota when a pending trigger card
could still need it. Pending cards are the bare `*.json` files in the same
node's T2 spool; the plotter renames a processed card to `.json.done` or
`.json.failed`, which never pin. For each pending card the janitor finds the
observation (`gulp.utc_start`; for a card without a gulp block, the latest
observation started before its `dump_utc_start`) and the hella gulp
(`gulp.gulp`; else `(samp + 960) // 8192`). For every stream in that
observation it pins the chunk with the largest first gulp not above the card's
gulp, plus the chunk before and the chunk after it, since a dump spans one to
two gulps and can cross a chunk boundary. A card with no gulp number pins its
whole observation. A card whose observation cannot be found, or that cannot be
read, pins every chunk on that node. The rule assumes the card's gulp number
equals the gulp index in the records; the neighbour chunks absorb an offset of
up to one chunk (about 1 h). Confirm this against the first deployed records.

Install on corr1 (a deploy step, with the hella recording):

    cp ~/software/dev/casm_t3/deploy/systemd/t3-mask-janitor.{service,timer} ~/.config/systemd/user/
    systemctl --user daemon-reload
    systemctl --user enable --now t3-mask-janitor.timer

The timer runs `t3-mask-janitor.service`, a one-shot with
`--targets hella_masks --once --mask-max-age-h 24 --mask-max-gb 100`, log
`/mnt/nvme5/casm_pipeline/logs/t3_mask_janitor.log`. It is independent of
`t3-janitor.service`, which keeps the default `dumps` target.

Per-event archive: `/mnt/nvme3/T3/EVENTS/<candname>/` holds the PNG, the
result JSON, a single-beam float32 `.fil` of the detection beam and the
`.slack` posting marker, on both nodes, with corr2's copy rsynced to
corr1 by t3-collect. It is not under the janitor's quota, so watch
/mnt/nvme3 by hand. To turn it off, run the plotter with
`--events-root ''`.

Slack alerts no-op silently until a bot token exists at
`~/.config/slack_api`.

Schema-2 events (casm_io `docs/t3_events.md`) post in two steps. `t3-collect`
waits for `<candname>.manifest.json` in the candidate dir (the plotter writes it
last, and it must parse and agree with the PNG sizes on disk), posts
`<candname>.png` with the usual caption, then posts `<candname>_raw.png` as a
thread reply with the caption "raw view, before hella RFI flagging". The
`.slack` marker carries one line per post:

    posted <iso> ts=<ts> channel=<id> kind=main
    posted <iso> ts=<ts> channel=<id> kind=raw

A failed reply leaves only the `kind=main` line and is retried every cycle
without reposting the main figure. When the main post's ts did not resolve,
the raw figure is posted unthreaded with a caption naming the candidate and the
line says `kind=raw_unthreaded`. A raw figure older than `--max-age-h` gets a
`skipped stale` line instead. Schema-1 folders keep today's single
`posted <iso> ts=<ts>` line, no manifest wait and no raw post. The marker is
mirrored into the event dir with its full content after each post. A missing
manifest on a folder older than `--max-age-h` ends as `skipped stale`.
`alerts.post_candidate` gained `thread_ts` (passed to
`files.completeUploadExternal`); it still returns the message ts, `True` when
the ts did not resolve, `False` on failure.

New candidate captions link the candidate ID directly to its 8061 event page
and show the search S/N, DM, UTC event time and Pacific event time on one text
line. Pacific time uses `America/Los_Angeles`, choosing PST/PDT for the event
date. If the local date differs from UTC, both dates are shown. The separate
"Open in dashboard" label is omitted to save space; Slack may still wrap on a
narrow display. Readers need their usual SSH tunnel to localhost:8061.

After a caption or posting change, reload only `t3-collect` between posting
passes. Keep all `.slack` markers so already-posted events remain untouched. T2, plotters,
the monitor, weights and injection routing need no restart or change.

## Failure isolation

One bad candidate must never stop a polling loop. The plotter wraps each
card; a failure renames it `.failed` and the loop moves on. The web app
renders every page even when charting or astropy breaks — panels degrade
to absent rather than to a 500.
