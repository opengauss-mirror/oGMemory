# Optional semantic ContextNode graph

## dev port (2026-10-23)

`og-v3-dev` starts from official `dev` commit `fc13d829` and ports the graph-only
changes from PR !266, including its reviewed fixes at `bd5dd5e7`. The sections
below record earlier AntTrail checkpoints; their benchmark numbers are not
measurements of this dev port.

The implementation reuses dev's GraphService, ContextReader, authorized SQL
storage, vector records, shared pools, extraction pipeline, outbox and native
evidence renderer. It does not replace dev with the older AntTrail runtime.
Graph support remains default-off and does not add conflict, duplicate or
validity management. No provider, model, temperature, output cap, keyword-routing
or cache default is changed.

Port-specific compatibility:

- Preserve dev's identity namespaces, read-bound updates, incremental
  `edit_memory` / `append_memory`, source occurrence reconciliation, durable
  checkpoints, deadline protections and Agent-memory handlers. Graph-on edits
  carry the same validated connection descriptor as complete extractions.
- Empty annotation metadata does not alter pre-graph replay/receipt hashes.
  Transient native relation proposals do not change a durable node-write digest
  when consumed after checkpoint acceptance; validated descriptors still do.
- Code RESOURCE index payloads remain revision-independent and deterministic.
- Retain scene per-query reserves and native global result ranking. Append graph
  results after native selection; apply caller categories and memory directories
  to graph hydration, traversal bridges and supplemental delivery.
- Keep native evidence's 15-candidate quota and source-preserving rendering.
  Graph memories use the existing renderer separately and are appended whole
  only if the actual budget permits. Supplemental probes use a bounded full-body
  allocation because dev treats a zero allocation as empty, not unlimited.
  Native evidence is not displaced or re-ranked by those additions.
- Preserve dev's public read metadata and canonical Agent-body rendering, and
  reuse that hydration for fail-closed graph reads.

Verification: **3,339 extraction/retrieval/service/index/provider/commit/filesystem
tests passed, 40 skipped**, plus **8 bounded directory-handler tests** and **11
real PostgreSQL/pgvector graph integration tests** using a disposable database
and fake model/embedding providers. **1,977 server/MCP, scene and native graph
tests passed** with the repository-declared MCP/build dependencies installed.
Together these selections passed 5,335 tests. The broad unit selection excludes the two
legacy directory-handler demonstration tests; optional backend tests retain
their existing skip conditions. No live openGauss or fresh BEAM/LoCoMo accuracy
claim is made for this dev port.

## og-v3 integration checkpoint (2026-10-12)

The `og-v3` branch is based on remote `anttrail` at
`da754639c81ab4d8618c71cb525c7179dae4062d`. It incorporates the existing
upstream-adapted reuse-v3 graph port (`1e9940f3`) and the blank auxiliary
session-summary fix (`c30847e5`) by fast-forward, not by replacing the
upstream runtime with the older server branch.

Graph ingestion, native one-hop retrieval and protected two-hop additions are
implemented. Existing service, storage, indexing and provider implementations
remain authoritative. Conflict/duplication and validity algorithms are not
included. Graph support remains opt-in; use the configuration below to enable
it. Ordinary upstream model settings, keyword routing and cache defaults are
unchanged.

Previous PR266 checkpoint verification: **432 passed, 5 skipped**. Four skips require an
explicit disposable PostgreSQL database (`GRAPH_TEST_DSN`); one requires
Chroma. The selection covers graph on/off boundaries, descriptors, relation
generation, projections, retrieval, prefix/budget protection, the summary fix,
and upstream code-resource, lexical-index, provider and threading regressions.
`git diff --check anttrail..HEAD` is clean. No model API, live-database or fresh
BEAM/LoCoMo benchmark was run for this checkpoint; it does not establish
historical accuracy parity.

### Review fixes verified on 2026-10-22

- One-hop and two-hop graph results obey explicit or planner-effective category
  filters. Canonical categories are checked again after hydration and before
  supplemental delivery; excluded categories cannot become traversal bridges.
- The graph invalidation trigger uses `EXECUTE PROCEDURE`, supported by both
  dialects. Node changes enqueue durable `PROJECT_GRAPH` events in the same
  transaction, rather than relying on process-local remembered accounts.
- Relation writes, projection completion and node moves acquire canonical-node
  locks before job locks and relation locks. Node and job locks use stable order.
  The old edge/node lock inversion was reproduced as `DeadlockDetected` on the
  unchanged PR266 commit; the same scenario completes with these fixes.
- Projection claims carry a private lease token. Both success and failure reject
  expired/replaced claims, including retries of the same node revisions. The
  token is not included in searchable relation metadata.
- L3 descriptors are excluded from lexical projection/backfill and ordinary
  memory BM25 queries. Node embeddings persist their actual model and dimension;
  supplemental traversal rejects unidentified or incompatible stored vectors.
- Node moves migrate projection endpoint URIs, edge IDs and relation projection
  IDs atomically, invalidate old worker leases and enqueue replacement jobs.
  Moved-node index events reuse `build_index_records` to rebuild L3 and L0/L2,
  retaining the move path's existing L1 records.
- Endpoint indexes `(account_id, source_uri)` and `(account_id, target_uri)`
  support invalidation without scanning all completed tenant jobs.

Validation: **559 selected unit tests passed**, plus **11 real PostgreSQL/pgvector
integration tests passed** using a dedicated disposable database and fake
providers. Includes stale failure/success, same-revision lease replacement,
restart after ordinary-event acknowledgement, source/target moves, descriptor
rebuilding, subsequent re-projection and a deterministic concurrent update.
This is not a full-suite or live-openGauss verification claim; no new paid
benchmark was run for these review fixes.

Upgrade: stop old indexing workers before applying the idempotent schema update
and restart them afterwards. The update adds `claim_token` and endpoint indexes;
it preserves canonical memories/relations. Legacy node vectors without model
identity must be re-indexed before they can qualify incoming supplemental edges.
Do not guess their model from dimensions or relabel them as the current model.
The feature remains default-off and contains no conflict-management changes.

This feature ports the accepted graph-control ingestion and retrieval design
from `reuse-v3-good-20261009` (`65784e86111fa7258f3c1484182c1955274767b9`)
onto the current `anttrail` branch (`da754639`). It is not a merge of the old
server runtime. It does not include conflict/duplicate classification, Scent,
validity management, benchmark scripts, frozen corpora or experiment artifacts.

## Enable explicitly

Add this section to your existing configuration:

```yaml
storage:
  backend: sql
graph:
  backend: sql                 # default: none (fully disabled)
  enable_expansion: true
  expand_top_k: 5              # 1..5 ordinary anchors for native one-hop
  expand_limit: 3              # 1..3 outgoing edges per anchor
  min_weight: 0.0
  edge_similarity_threshold: 0.35
  edge_embedding_profile: shared-node-embedding-v1
  supplement_enabled: true
```

The existing SQL connection, shared vector index, embedder, LLM and indexing
worker configuration remain authoritative. Model, provider, temperature,
output limits, cache, query routing, skill sedimentation and code/M3 settings
are **not changed** by this feature. There is no new graph database dependency.
`graph.backend=none` disables graph extraction, recommendation, relation LLM
calls, edge projection workers and retrieval. `enable_expansion=false` disables
graph retrieval while still building the graph when the backend is enabled.

Environment equivalents are `GRAPH_BACKEND`, `ENABLE_GRAPH_EXPANSION`,
`GRAPH_EXPAND_TOP_K`, `GRAPH_EXPAND_LIMIT`, `GRAPH_MIN_WEIGHT`,
`GRAPH_EDGE_SIMILARITY_THRESHOLD`, `GRAPH_EDGE_EMBEDDING_PROFILE` and
`GRAPH_SUPPLEMENT_ENABLED`. Existing YAML/environment precedence applies.

## Ingestion

1. The current upstream extractor still generates normal memory fields and
   provenance. When graph-enabled, each extraction tool additionally requires
   a single-line `update_activation` connection descriptor. The normal prompt,
   examples, skill behavior, speaker attribution and validation are retained.
   The descriptor instruction is appended, not a replacement prompt.
2. `ArchiveBuilder`, `ContextWriter` and the existing outbox persist/index the
   memory. The descriptor is a separate L3 index record; ordinary searches
   continue to search only L0/L1/L2. Its version/content digest and embedding
   model/dimension must match the current authorized memory.
3. After successful writes, `RelationCandidateService` searches historical
   descriptors using `category | routing key | abstract`, excluding the
   current batch. At most five historical candidates per new memory and
   twenty overall are supplied for relation generation (similarity >= 0.40).
4. One additional ingestion LLM call proposes directed relations. Exact
   endpoint membership, source evidence, no self links and no historical-only
   links are validated. Cross-prompt relations are capped at three per new
   memory and twenty-four per batch. Failed writes cannot shift the endpoint
   mapping; alignment failure abstains from linking.
5. `SQLRelationStore` owns canonical `relation_edges`. Canonical edges,
   projection jobs and their `PROJECT_GRAPH` outbox wakeups commit together.
   The existing worker embeds **relation + target memory**, not the source
   or evidence reason. Evidence remains on the canonical edge.

Existing nodes without descriptors are not silently regenerated. New nodes
can link to each other; historical recommendation requires an indexed,
compatible descriptor. Graph diagnostics in the write result distinguish
model/lookup abstention from persistence failure.

## Retrieval: no generative LLM

- The upstream planner, seed retriever, hierarchical search, ranker and
  formatter remain in use. Resource/skill-only and session-summary rounds do
  not run graph expansion. Existing routing is respected.
- Native one-hop uses the first five ordinary candidates and up to three
  outgoing edges each. It scores a query embedding against stored compatible
  edge vectors (cosine >= 0.35; score = cosine x weight). Strict canonical
  hydration authorizes every added memory. Ordinary Top-K results are kept
  before the bounded graph stream, not displaced by it.
- Compose's primary memory search also collects a request-local supplement:
  select up to five graph-connected anchors from the first fifteen pregraph
  candidates, traverse incoming/outgoing edges for at most two hops, expand
  at most thirty nodes, and inspect at most 128 unique destinations per
  adjacency. Local selection interleaves semantic Top3 and positive-IDF BM25
  Top3. Incoming edges can additionally qualify using stored node vectors;
  this does not boost their ranking score. Second-hop paths decay by 0.8.
- Delivery preserves the ordinary/native-one-hop formatted prefix and only
  appends complete, previously absent memories that fit unused existing
  budget. Total graph-origin deliveries are capped at fifteen. No replacement,
  reordering, new clipping or increased budget is allowed. Scope, query,
  canonical revision and body hash are checked again at injection.
- The seed query vector is reused by the supplemental lane when the query
  matches; native one-hop retains its query embedding call. Stored node/edge
  vectors are read, not re-embedded. Batched stored-node vector reads use the
  existing SQL vector index. No retrieval-time LLM call is introduced.

## Reused and added modules

| Responsibility | Existing implementation reused | Graph extension |
|---|---|---|
| Extraction | Extractor, dynamic schemas, ReAct loop, provenance | Descriptor argument/instruction; relation generation validator |
| Persistence | ArchiveBuilder, ContextWriter, SQLContextFS | Validated descriptor metadata |
| Index lifecycle | IndexRecordBuilder, OutboxWorker, SQL outbox | L3 descriptor; durable edge projection jobs/wakeups |
| Vector storage | Shared embedder and vector index | Metadata filters; batched stored-vector reads |
| Graph storage | SQLRelationStore / relation_edges | Versioned edge projections; bidirectional reads |
| Graph API | GraphService | ContextNode relation-reader adapter (M3 methods unchanged) |
| Retrieval | Planner, seed/hierarchical retrievers, ranker, ContextReader | OneHopRetriever, GraphSupplement, strict authorized hydration |
| Injection | ReadAPI, MemoryService, normal formatter/budgets | Request-bound append-only delivery |

The SQL schema migration is additive and idempotent: nullable projection
columns, durable job table with tenant RLS, a memory-update invalidation trigger,
and widening existing relation labels to at least 256 characters. Revision
changes invalidate projections; an old worker result cannot complete a newer
revision's job. Queue SQL supports the existing PostgreSQL/openGauss dialect
seam; openGauss avoids `SKIP LOCKED`. Live integration coverage currently uses
PostgreSQL/pgvector, not a live openGauss deployment.

The vector adapter also guards a legacy `node_id` backfill: current upstream
`context_nodes` is URI-keyed and has no `node_id` column. No legacy node-id
schema is reintroduced.

Changing the embedding model, dimension or profile does not make old vectors
compatible automatically. Re-index/rebuild projections before expecting graph
recall with a new identity. Disabling the feature does not delete stored graph
data. Existing SQL worker retry/dead-letter/operator retry behavior applies.

## Verification

Graph unit coverage is in:

- `tests/unit/service/test_semantic_graph_feature.py`: unchanged off path,
  additive extraction schema/prompt, graph write hook, shared runtime wiring.
- `tests/unit/index/test_activation_projection.py`: separate L3 indexing,
  content binding, malformed/obsolete descriptors, delete/archive cleanup.
- `tests/unit/extraction/test_integrated_cross_prompt.py`: historical exclusion,
  authorization, source evidence, endpoint validation and failed-write alignment.
- `tests/unit/retrieval/test_graph_supplement.py`: bidirectional traversal,
  revision/profile gates, request isolation, prefix/budget/delivery caps.

SQL integration tests use a disposable database only:

```bash
GRAPH_TEST_DSN='<disposable PostgreSQL connection string>' \
  python -m pytest tests/integration/test_semantic_context_graph.py -q
```

They exercise real schema application, canonical edges, durable outbox processing,
stored vector reads, live one-/two-hop retrieval, authorization, long relation
labels, memory-update invalidation and rejection of stale worker results. Without
`GRAPH_TEST_DSN`, these tests skip. No external LLM/embedding API is needed.

These tests establish implementation contracts, not BEAM/LoCoMo accuracy parity:
the current upstream extractor/prompt is intentionally retained. A fresh corpus
benchmark is still required before claiming historical accuracy numbers.

### Local verification (2026-10-09)

- Retrieval, service, extraction, index, server and commit regression selection:
  **1,920 passed, 1 skipped, 1 deselected**.
- SQL relation/vector/config/dialect regression selection:
  **177 passed, 12 skipped, 6 deselected**.
- Disposable PostgreSQL/pgvector graph integration: **4 passed**, including
  embedding failure followed by worker-restart recovery using durable wakeups,
  and the real upstream extraction/validation/writer/index pipeline with fake
  provider responses (span identification and dual extraction retained).
- `git diff --check`: clean.

These are selected runs, not a claim that the entire upstream suite passes.
The MCP SDK is absent from the local test environment; MCP-server/deployment
files and one SDK-dependent client test were excluded. The directory-event
handler file was excluded after its existing DAG test stalled. Six stale config
tests (removed code-mode/TextRank attributes and undefined `uc`) and two old
vector-mock tests fail identically on a detached untouched `da754639` checkout.
Two deployment config tests also reproduce upstream placeholder-integer errors
on that untouched checkout. They were not fixed in this graph feature.
