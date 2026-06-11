# casm_t3 operations

## Deployment

systemd user units (`deploy/systemd/`, requires `loginctl
enable-linger`):

| unit | host | role |
|---|---|---|
| `t3-dump-plotter-corr1` | corr1 | plot local dumps |
| `t3-dump-plotter-corr2` | corr2 | plot local dumps, stage artifacts |
| `t3-collect` | corr1 | pull corr2 artifacts (corr2 cannot ssh back) |
| `t3-web` | corr1 | monitor UI on :8050 |
| `t3-janitor` | corr1 | disk quota sweep on both nodes |

Both nodes run the same venv; keep them in sync (the second node has no
internet — install from wheels copied across).

## Runbooks

**Iterate on the candidate figure**

    t3-replot /path/to/card.json.done --out /tmp/test.png

Re-renders offline from the on-disk dump on the node that holds it.
Never re-queue cards to test plotting: the live plotter may delete the
dump after rendering.

**Changes and retroactivity** — anything baked into the PNG (layout,
titles, framing) applies from the next event onward, because dumps may be
deleted after plotting. Anything computed from the DB at page-render time
(reasons, tables, stats) is retroactive automatically. Known-source dumps
are kept on disk and can be re-rendered.

**New beamforming weights deployed**

    python -m casm_t3.skypos /path/to/new_weights.h5

regenerates `casm_t3/data/beam_pointings.json`; commit it.

**Janitor dry run**

    t3-janitor --dry-run

prints would-deletes without touching anything. The janitor never deletes
dumps of events labelled frb/pulsar.

**Slack alerts** are a silent no-op until a bot token exists at
`~/.config/slack_api`.

## Failure isolation

One bad candidate must never stop a polling loop: the plotter wraps each
card, failures rename the card `.failed` and move on. The web app renders
every page even when charting or astropy fails (panels degrade to absent
rather than erroring).
