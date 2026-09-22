# casm_t3 operations

## Deployment

systemd user units, `deploy/systemd/`. Run `loginctl enable-linger` once
per node or they die at logout.

| unit | host | role |
|---|---|---|
| t3-dump-plotter-corr1 | corr1 | plot local dumps |
| t3-dump-plotter-corr2 | corr2 | plot local dumps, stage artifacts |
| t3-collect | corr1 | pull corr2 artifacts and events (corr2 cannot ssh back) |
| t3-web | corr1 | monitor UI on :8050 |
| t3-janitor | corr1 | disk quota sweep across both nodes |

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

Per-event archive: `/mnt/nvme3/T3/EVENTS/<candname>/` holds the PNG, the
result JSON, a single-beam float32 `.fil` of the detection beam and the
`.slack` posting marker, on both nodes, with corr2's copy rsynced to
corr1 by t3-collect. It is not under the janitor's quota, so watch
/mnt/nvme3 by hand. To turn it off, run the plotter with
`--events-root ''`.

Slack alerts no-op silently until a bot token exists at
`~/.config/slack_api`.

## Failure isolation

One bad candidate must never stop a polling loop. The plotter wraps each
card; a failure renames it `.failed` and the loop moves on. The web app
renders every page even when charting or astropy breaks — panels degrade
to absent rather than to a 500.
