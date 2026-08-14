# CarbonTeq AI Trackio fork

This repository is an additive fork of
[`gradio-app/trackio`](https://github.com/gradio-app/trackio). It preserves the
Trackio Python package, HTTP API, SQLite-compatible and Parquet persistence,
artifacts, and the existing UI while adding platform-specific observability
integrations.

## Python distribution

CarbonTeq publishes the fork as `carbonteq-trackio` while preserving the
`trackio` import package and `trackio` console command. The current published
fork release is `0.31.5.post13`, derived from upstream Trackio `0.31.5`; the
working candidate is `0.31.5.post14.dev10`.
Post-release numbers advance when CarbonTeq publishes additional fork changes
without moving the upstream base.

Framework packages must depend on `carbonteq-trackio`, not the upstream-owned
`trackio` distribution and not a transitive Git URL. Self-deployed Trackio
Spaces use the same CarbonTeq distribution identity so the deployed runtime
retains the fork's storage, trace, and query behavior.

## Current extension

`trackio.VerifiersTrace` stores a queryable display projection alongside the
complete JSON-safe Verifiers trace record. Native Verifiers `traces.jsonl`
remains authoritative. Trackio is an idempotent, query-optimized copy keyed by
`(run_id, trace_type, external_id)`.

The existing `trackio.Trace` type and trace UI are unchanged. Verifiers traces
have a separate `/verifiers` workspace because a rollout is a graph with
branches, rewards, model calls, tools, phase timing, and environment errors—not
just a conversation. The Verifiers UI renders structured projections and links
each rollout to its producing experiment; it never displays the raw payload.
For an evaluation run, it also joins the selected rollout to the run's aggregate
`eval/*` metrics so per-rollout evidence and overall results remain distinct but
visible together.

The bundled dashboard uses the secure Vega 6 dependency line (`vega` 6.3.1,
`vega-lite` 6.4.3, and `vega-embed` 7.1.0). Trackio emits Vega-Lite v6 schemas,
keeps the canvas renderer explicit across the Embed 7 default-renderer change,
and escapes dashboard-generated axis-label lookup expressions before Vega
parses them. Frontend release gates include a production dependency audit,
representative hostile-label compilation/execution, the complete frontend test
suite, and a reproducible wheel build containing the generated dashboard. This
upgrade closes `GHSA-7f2v-3qq3-vvjf` in Vega expressions and
`GHSA-m9rg-mr6g-75gm` in `vega-functions`; the lock also carries patched
Svelte, Vite, and Vitest releases and both full and production-only audits must
remain at zero.

The self-hosted artifact API supports bounded, resumable uploads for model-sized
blobs. When `TRACKIO_ARTIFACT_STORAGE_BACKEND=s3`, the client negotiates a
multipart session and uploads each part directly to short-lived presigned URLs
for the configured S3-compatible endpoint; Trackio receives only part ETags,
completes the provider upload, streams the finished object once to verify its
size and SHA-256, and commits an artifact version only after that verification.
The configured endpoint must be reachable by the producing client (it may be
AWS S3, RustFS, MinIO, or another S3-compatible service). Deployments where
the server uses a private provider address can set
`TRACKIO_ARTIFACT_S3_PRESIGN_ENDPOINT` to the worker-reachable address used
only for signed URLs. With the default
`local` backend, or against an older server, the client uses the bounded
server-side compatibility path; legacy whole-file uploads remain capped at
32 MiB instead of buffering larger files in server memory.

Incomplete upload sessions are project-scoped staging state. Cleanup reports
reclaimable bytes in dry-run mode and expires only incomplete, aged sessions.
Completed content-addressed blobs and artifact versions are not eligible for
session cleanup.

The unreleased compatibility repair also routes the older server-side
resumable completion endpoint through the configured artifact store. This is
required while post8 clients coexist with an S3-backed post10 server: the
legacy endpoint must not write a verified blob only to the historical local
CAS and then let the manifest authority look for it in S3. Completed legacy
sessions recover an existing verified local-CAS blob into the configured store
before returning success. Publication and deployment remain open gates.

The same release repairs stale direct-upload receipts. A completed multipart
session is no longer treated as proof that its content-addressed object still
exists: Trackio checks the configured store before returning the completed
receipt and restarts the upload when retention or migration removed the blob.
This prevents an artifact producer from skipping the upload and then failing
manifest commit with `Manifest references blobs not on server`.

The intended distribution for these compatibility and importer changes is
`carbonteq-trackio==0.31.5.post11`. It remains a candidate until the exact
commit, wheel hashes, deployment, direct-upload canary, and Doris backlog
readback are recorded.

`0.31.5.post12` adds an opt-in bounded background artifact publication path to
the client. `Run.log_artifact(..., background=True)` queues work on a per-run
executor, returns an artifact with a stable submission id and explicit state,
and `Run.flush_artifacts()` drains committed identities before run finalization.
The synchronous API remains the default and old callers continue to work.
The Posttrain adapter uses this option only when the installed client exposes
it, preserving a short compatibility window for older images.

`0.31.5.post13` also keeps bounded trace listings analytically useful without
returning transcript or tool-call bodies. For native Verifiers records,
`include_payload=false` derives latency, prompt/completion/reasoning usage,
model-call count, and tool-call count from the complete stored record, then
returns only those safe scalar summaries. This repairs historical Observatory
rows whose full detail had timing and token evidence while their paged summary
showed it as missing.

## Trace-facts candidate (`0.31.5.post14.dev10`)

This candidate adds a generic, typed trace-facts projection for native
Verifiers traces. The full native record remains in `traces.payload` as replay
authority. The current projection is stored as nullable scalar columns on its
own `traces` row, while unbounded named reward components live in
`trace_reward_components`. It provides a narrow aggregation API over approved
dimensions and numeric measures, without JSON-path queries or Posttrain/model
semantics in Trackio.

Its aggregation contract also supports long-form reward-component contribution,
score, and weight. Callers may filter by an exact component name or group an
all-component query by component name and source kind; component coverage is
reported independently from the matching trace count. Scalar and component
aggregates are intentionally separate requests so a multi-component trace
cannot silently multiply scalar coverage.

The producer-facing `trackio.Run` exposes the same bounded aggregation method
as the read-side API. A remote run can therefore write its native trace and
then verify aggregate availability without constructing an internal read
object or reaching into storage.

`0.31.5.post14.dev8` establishes the client-side causal write boundary for a source
trace followed immediately by fact enrichment: it synchronously sends queued
native log records before the dependent upsert, under the same client lock.
This retains asynchronous logging for ordinary observations while preventing a
fact update from arriving before its parent trace.

`0.31.5.post14.dev9` completes that boundary for the Doris service. Native
trace batches bypass the asynchronous inbox acknowledgement and are committed
before the request returns; scalar-only metric batches retain the durable inbox
throughput path. This gives trace facts the read-after-write guarantee their
foreign-key-like parent relation requires.

`0.31.5.post14.dev10` completes the client half of that boundary when a caller
explicitly flushes before enrichment. Remote `Run.flush()` now delivers queued
metric records to the service instead of only checkpointing them in the local
retry buffer, so an immediate trace-fact upsert cannot overtake a source trace
that was flushed by a bridge.

The contract is implemented in `trackio/trace_facts.py`, accepted on an
initial `VerifiersTrace` write or through `Run.upsert_trace_facts`, persisted by
both `SQLiteStorage` and `DorisStorage`, and served through
`/upsert_trace_facts` and `/get_trace_facts`. Projection IDs are verified
SHA-256 identities, so retries are idempotent and a replacement component set
cannot leave stale component rows visible. Trackio validates generic shapes
and accounting invariants; Posttrain's Verifiers projector remains responsible
for tokens, tools, truncation, reward semantics, and model/template rules.

Apache Doris moves directly from global schema version 1 to version 2. The
candidate includes an explicit backup-gated `trackio storage migrate-doris`
command and requires coordinated server deployment; an old server is not
compatible with a migrated database. This deliberately replaces a
per-capability compatibility layer.

The migration importer defaults to exact logical-project reconciliation for a
new target. A coordinated recovery cutover may explicitly choose
`--verification-mode source-inclusion`: it proves every stopped SQLite logical
record is present while allowing an existing Doris project to retain additional
valid history. Run-bound tables are queried only for source run IDs in bounded
chunks, so this verification does not scan unrelated retained runs.

Remote trace-fact enrichment has a bounded causal-readiness retry. A native
trace is sent by Trackio's normal asynchronous log batch, while a later
trace-keyed enrichment may immediately follow. The SDK retries only the
specific missing-parent response long enough for that queued trace to arrive;
other errors are not retried or hidden.

No fork workflow builds or publishes releases. A maintainer builds and checks
the candidate locally, commits and pushes the exact source, creates an
immutable prerelease with its wheel, sdist, and SHA-256 receipt, then manually
dispatches the Posttrain-owned retained-asset publisher to `carbonteq/dev`.
Only after clean dev-index install and real-Doris qualification may the same
bytes be promoted by the repository-owned workflow and selected as a stable
Posttrain dependency.

## Storage engine

Turso is the default SQL metadata engine through the `pyturso` embedded driver.
It retains Trackio's SQLite-compatible database-per-project model and local-first
operation. Set `TRACKIO_DATABASE_ENGINE=sqlite` for the stdlib SQLite fallback.

Apache Doris is implemented as a first-class engine selected with
`TRACKIO_DATABASE_ENGINE=doris`. Its native provider and migration path have
passed real-Doris SDK/HTTP, clean-schema bootstrap, content-reconciled
migration, artifact, restart, concurrency, SQLite/Turso regression,
backup/restore, retained-project migration, and deployed Trackio/Observatory
checks. Metric time windows, canonical run listing, tab classification,
artifact relogging, legacy-link folding, and stable ordering match the logical
SQLite provider behavior. Doris is authoritative for run, metric,
system-metric, trace, alert, artifact-metadata, and lineage records; it is not
a downstream analytical projection. Model and media bytes remain behind
Trackio's existing server-managed artifact boundary.

This implementation is published on the CarbonTeq fork and must be consumed by
immutable commit. The current deployment is still bound to an older
source-diff and wheel-digest receipt until the corrected Trackio/Observatory
services are rebuilt and promoted. Operate one Trackio server replica until
artifact-version allocation is made safe across replicas.

The unreleased inbox-throughput repair keeps `TRACKIO_ASYNC_DORIS_WRITES` as a
durable-fragment mode, not an asyncio writer. HTTP requests append JSONL
fragments; a single scanner claims bounded batches and a bounded thread pool
groups synchronous Doris writes. Scalar metric/event fragments have a dedicated
lane ahead of large rollout-trace fragments, and server startup no longer waits
for an unbounded inbox replay. The native Verifiers trace artifact remains the
complete replay authority. This repair is not deployed or consumable until its
tests, immutable commit, wheel, and real-Doris backlog replay pass.

Raw project SQL remains deliberately unavailable with Doris because its tables
are shared across projects rather than stored in one project-local database.
The first release remains single-server: the process lock and idempotent
artifact operations do not provide cross-process version allocation or a
multi-table transaction. HA requires a staged or optimistic artifact protocol
and schema-level coordination before additional Trackio writers are allowed.

Turso stores run, metric, trace, artifact-manifest, and lineage metadata. It is
not the object store: media, model bytes, and native evaluation bundles remain
under Trackio's artifact storage boundary. The server may use the local CAS or
any S3-compatible backend. With the S3 backend, clients receive short-lived
presigned multipart URLs and upload bytes directly; Trackio verifies the
completed object's SHA-256 before committing metadata. A hosted Hugging Face
Space is optional and is not required for local operation. Remote sync can be
added later without changing the Trackio SDK contract.

## Upstream baseline

| CarbonTeq release | Upstream repository | Upstream commit |
| --- | --- | --- |
| `0.31.5.post1` | `gradio-app/trackio` | `438cb28d2c82c7b7d42431e45d5677a8cc90eb77` |
| `0.31.5.post2` | `gradio-app/trackio` | `438cb28d2c82c7b7d42431e45d5677a8cc90eb77` |
| `0.31.5.post3` | `gradio-app/trackio` | `438cb28d2c82c7b7d42431e45d5677a8cc90eb77` |
| `0.31.5.post4` | `gradio-app/trackio` | `438cb28d2c82c7b7d42431e45d5677a8cc90eb77` |
| `0.31.5.post5` | `gradio-app/trackio` | `438cb28d2c82c7b7d42431e45d5677a8cc90eb77` |
| `0.31.5.post6` | `gradio-app/trackio` | `438cb28d2c82c7b7d42431e45d5677a8cc90eb77` |
| `0.31.5.post7` | `gradio-app/trackio` | `438cb28d2c82c7b7d42431e45d5677a8cc90eb77` |
| `0.31.5.post8` | `gradio-app/trackio` | `438cb28d2c82c7b7d42431e45d5677a8cc90eb77` |
| `0.31.5.post12` | `gradio-app/trackio` | `438cb28d2c82c7b7d42431e45d5677a8cc90eb77` |
| `0.31.5.post13` | `gradio-app/trackio` | `438cb28d2c82c7b7d42431e45d5677a8cc90eb77` |

`0.31.5.post4` adds project-scoped bulk read APIs so a client can describe every
run without one configuration request and one history request per run:

- `SQLiteStorage.get_run_lifecycles` / `DorisStorage.get_run_lifecycles` and
  shared shaping in `trackio/lifecycle.py`
- server `/get_run_lifecycles`
- client `Api.run_lifecycles` and `Api.run_configs`

`0.31.5.post4` on `carbonteq/stable` was published with distribution metadata
`post4` while `trackio._version.__version__` still said `post3`. That index is
non-volatile, so `0.31.5.post5` is the corrected release: same APIs, matching
import and distribution versions. Do not install `post4` from the index.

`0.31.5.post6` adds an authenticated, preview-first project purge API. Its
summary names the project-owned runs, artifact versions, and local artifact or
media bytes before deletion; the apply endpoint removes only that project's
server-owned storage. It is deliberately not a lineage cascade: a caller that
wants to remove jobs and their downstream consumers must construct and confirm
that closure in its job orchestration layer.

`0.31.5.post7` adds exact run purge endpoints. A caller first
previews a list of provider run ids; Trackio reports artifact consumers and
blocks the preview when an unselected run would retain a dependency. Apply is
bound to the preview's SHA-256 digest, deletes run rows and unreferenced
artifact versions, and removes only CAS blobs no longer present in a retained
manifest. The API is intentionally provider-scoped: the framework remains
responsible for cross-plane dependency closure and the user-facing dry-run
plan. The same digest binding now covers the existing project-boundary delete
endpoint, so project apply cannot race a changed inventory while retaining
backward compatibility for older callers that only inspect the post6 summary.

`0.31.5.post8` preserves the post7 purge contract and makes rejected applies
actionable over the HTTP transport. Stale run/project digests now return a
client error, and `RemoteClient` surfaces the server's obtain-a-new-preview
message instead of replacing it with a generic HTTP status exception.

Every CarbonTeq release must add a row before it is tagged. Platform consumers
should prefer the published `carbonteq-trackio` distribution once it is on the
configured index; until then they may pin an immutable CarbonTeq commit.

## Updating from upstream

```bash
git fetch upstream
git switch main
git pull --ff-only origin main
git switch -c maintenance/upstream-YYYY-MM-DD
git merge --no-ff upstream/main
uv sync --extra dev --extra spaces
uv run pytest tests/unit -q
git push -u origin maintenance/upstream-YYYY-MM-DD
```

Open a pull request into `carbonteq-ai/trackio:main`. Resolve conflicts by
keeping upstream behavior intact and reapplying only the additive Verifiers
fields. Before merging, verify both Turso and SQLite modes, standard traces,
metrics, artifacts, API queries, SQLite-compatible migrations, Parquet round
trips, and both trace UIs as well as the Verifiers tests.

The repository should keep these remotes:

```text
origin    git@github.com:carbonteq-ai/trackio.git
upstream  https://github.com/gradio-app/trackio.git
```

Rust, Tokio, and direct worker access to object-storage credentials remain
deferred. Native Doris support is additive; Trackio's existing server-managed
artifact store remains the byte-storage boundary.
