# casm_t3 operations

## Deployment

systemd user units, `deploy/systemd/`. Run `loginctl enable-linger` once
per node or they die at logout.

| unit | host | role |
|---|---|---|
| t3-dump-plotter-corr1 | corr1 | plot local dumps |
| t3-dump-plotter-corr2 | corr2 | plot local dumps, stage artifacts |
| t3-collect | corr1 | pull corr2 artifacts (corr2 cannot ssh back) |
| t3-web | corr1 | monitor UI on :8050 |
| t3-janitor | corr1 | disk quota sweep across both nodes |

Both nodes run the same venv. The second node has no internet — install
there from wheels copied across, and keep the venvs in sync.

## Runbooks

Iterating on the candidate figure:

    t3-replot /path/to/card.json.done --out /tmp/test.png

re-renders offline from the on-disk dump, on the node that holds it.
Never re-queue cards to test plotting — the live plotter may delete the
dump after rendering, and a re-queued card can fire side effects.

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

Slack alerts no-op silently until a bot token exists at
`~/.config/slack_api`.

## Failure isolation

One bad candidate must never stop a polling loop. The plotter wraps each
card; a failure renames it `.failed` and the loop moves on. The web app
renders every page even when charting or astropy breaks — panels degrade
to absent rather than to a 500.
