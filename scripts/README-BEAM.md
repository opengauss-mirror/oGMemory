# BEAM on PR314: native baseline, graph-only and official grading

These are the portable versions of the external BEAM source-ingestion,
native OpenClaw QA and pinned official-grading tools used by this project.
They are experiment tools, not an alternative implementation of graph or
conflict management. Production memory code is unchanged by this bundle.

This package runs the **current checkout**: reviewed PR314 graph functionality
integrated with Dev, not a copied historical memory implementation. It offers
`--profile baseline` (default, graph off) and `--profile graph` (graph on).
Neither includes conflict recognition or conflict-aware delivery.

The harness was adapted from the native Dev BEAM package; the passive capture,
strict snapshot checks and official scorer are shared with earlier experiments.
That reuse does **not** establish historical-score equivalence. In particular,
current Dev's 400-token after-turn buffer, 0.10 recall score-gap cutoff, native
occurrence qualification and native keyword routing are retained. The later
wwllwwll graph-on qualification bypass and keyword-disabling hooks are not
copied. Porting those would be a separate production/policy change.

## Entry points

| File | Purpose |
| --- | --- |
| `run_beam_eval.py` | One setup file; launch each evaluation stage |
| `create_beam_fixture.py` | Generate private native configs and a disposable Compose fixture; never starts Docker |
| `beam_sql_entrypoint.sh` | BEAM-only SQL deployment with one native indexing owner; no production entrypoint edits |
| `ingest_beam_fresh.py` | Native `after_turn(wait=true)` / `dispose` source ingestion |
| `run_stage8r_beam_clean_eval.py` | Isolated preflight, full SQL snapshot and native QA |
| `beam_dataset.py` | Read source pairs separately from held-out probes |
| `grade_beam_answers.py` | Pinned official BEAM prompts, grading and reporting |
| `beam_official.py`, `vendor/beam/` | Hash-verified upstream scoring source and MIT license |
| `beam_capture/`, `experiment_compose_capture.py` | Capture the first actual compose, without another retrieval |
| `beam_eval_safety.py` | SQL/content/source identity checks and zero-tools audit |
| `prepare_beam_agent.py` | Preserve the intended stopped-agent config across image startup |
| `beam_snapshot.py` | Audited PostgreSQL schema round-trip qualification; strict restore checks remain enabled |

The old experiment-33 reference wrapper is intentionally not copied: it pins
a particular workstation path, container image, model and diagnostic artifacts.
This package removes those deployment-specific bindings, not the native QA
question prefix, first-answer rule or official scoring functions. It does not
promise reproduction of historical scores against a changing provider model.

Dev's native occurrence qualification, extraction passes and error handling
are retained. This port adds no qualification retries, qualification bypass,
prompt replacement, final-fallback budget override, source replay or output
salvage. Native span extraction normally makes a configured-temperature pass
and a second temperature-zero pass; temperature 0 means **two native passes
at 0**, not a forced single pass. Other native calls (prefetch, relation
extraction, merge, etc.) retain their implementation's settings.

## 1. Install and obtain data

From this checkout, with Python 3.11+:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m pip install -r scripts/requirements-beam.txt
python scripts/run_beam_eval.py --help
python scripts/grade_beam_answers.py --help
```

Obtain the official parquet for the required split from
[Mohammadta/BEAM](https://huggingface.co/datasets/Mohammadta/BEAM).
Keep the dataset and results outside the checkout and outside the answering
agent's mounts. `conversation_index` is **zero-based**: 0 means task 1,
1 means task 2. No fixed 94-pair or 50-rubric assumption is imposed on all tasks.
Each selected conversation must contain all 10 categories / 20 questions.
Dataset download and use remain subject to the upstream dataset terms.

## 2. Generate a complete isolated fixture

Requirements: Linux Docker Engine with Compose v2+, free ports and sufficient
disk for the two native images, PostgreSQL and snapshots. Do not build on an
almost-full disk. The memory image uses `docker/Dockerfile.standalone` and the
Agent image uses `docker/Dockerfile.openclaw`, both from **this checkout**.
Use a pinned image/digest when a fixed infrastructure version is required.
The generated PostgreSQL fixture listens inside its container on port 5432
and publishes only the chosen loopback host port; memory/Agent use host networking.

Obtain credentials from your environment/secret manager. The generator requires:

| Environment variables | Purpose |
| --- | --- |
| `BEAM_LLM_API_KEY`, `BEAM_LLM_BASE_URL`, `BEAM_LLM_MODEL` | Native memory extraction provider |
| `BEAM_EMBEDDING_API_KEY`, `BEAM_EMBEDDING_BASE_URL`, `BEAM_EMBEDDING_MODEL` | Native embedding provider; dimension must match |
| `BEAM_QA_API_KEY`, `BEAM_QA_BASE_URL`, `BEAM_QA_MODEL` | Native OpenClaw answering provider |
| `BEAM_AGENT_TOKEN` | Dedicated answering gateway authentication |
| `BEAM_MEMORY_API_KEY` | Dedicated memory root API key, used by ingestion and the plugin |
| `BEAM_POSTGRES_PASSWORD` | Dedicated disposable PostgreSQL password |
| `BEAM_JUDGE_API_KEY` | Official-grader provider; needed only for paid grading |

All three model endpoints are OpenAI-compatible. Models are explicitly chosen
by the operator, not inferred from previous workstation experiments. No keys
are stored in setup JSON or printed to stdout. Optionally set `BEAM_JUDGE_MODEL`,
`BEAM_JUDGE_BASE_URL` and `BEAM_POSTGRES_IMAGE` before generation.

```bash
export BEAM_FIXTURE=/absolute/path/outside-checkout/beam-dev-t1-unique-1234
export BEAM_PARQUET=/absolute/path/to/BEAM-100K.parquet

# Inspect the file-generation plan: no secrets read, files written or services started.
python scripts/create_beam_fixture.py \
  --directory "$BEAM_FIXTURE" --parquet "$BEAM_PARQUET" \
  --run-id beam-dev-t1-unique-1234 --conversation-index 0 \
  --embedding-dimension 1536

# Generate NEW private files; still no Docker/model calls.
python scripts/create_beam_fixture.py \
  --directory "$BEAM_FIXTURE" --parquet "$BEAM_PARQUET" \
  --run-id beam-dev-t1-unique-1234 --conversation-index 0 \
  --embedding-dimension 1536 --profile baseline --execute

export BEAM_SETUP="$BEAM_FIXTURE/setup.json"
```

For **graph only**, generate a separate fixture with a new directory/run ID and
free ports, using the same command with `--profile graph`. Do not edit an
existing completed corpus to pretend it was generated under another profile.
For example (after choosing the model environment above):

```bash
python scripts/create_beam_fixture.py \
  --directory /absolute/path/outside-checkout/beam-graph-t1-unique-1234 \
  --parquet "$BEAM_PARQUET" --run-id beam-graph-t1-unique-1234 \
  --conversation-index 0 --embedding-dimension 1536 \
  --sql-port 25433 --memory-port 18801 --agent-port 20801 \
  --profile graph --execute
```

| Setting | Baseline profile | Graph profile |
| --- | --- | --- |
| Graph backend | `none` | `sql` |
| Graph expansion / supplementation | Off | On, current PR314 defaults |
| Graph extraction prompt | Native non-graph selection | Native `extraction_graph.yaml` selection |
| Scene generation / retrieval | Off | Off |
| Keyword routing | Native keywords, unchanged | Native keywords, unchanged |
| Occurrence qualification | Native Dev | Native Dev; **not bypassed** |
| After-turn buffer / recall gap | Native 400 / 0.10 | Native 400 / 0.10 |
| SQL outbox indexing owner | MemoryService | MemoryService, including graph projection |

Both profiles use a dedicated BEAM-only SQL entrypoint to avoid running an
independent index service without the API's graph-worker binding. This changes
only the generated fixture's process topology; it does not edit the production
deployment or replace the native indexing algorithm. The API uses one worker.
Native keyword routing can still choose non-memory/resource searches, which can
bypass graph expansion; this is deliberate here, not silently disabled to match
older scored runs. A profile selects actual fixture configuration; a setup
`variant` is only a reporting label and alone cannot activate the feature.

Use a new run ID/directory for every corpus. For concurrent fixtures, also choose
different `--sql-port`, `--memory-port` and `--agent-port` values. Defaults are
25432, 18800 and 20800. `--embedding-dimension 1536` is an example for an actual
1536-dimensional provider, not an instruction to alter another model's output.

Both generated BEAM profiles keep Dev's **0.7 temperature**, but use a
**32768-token extraction output cap for this run only**. A live task-1 run hit
the native 4096 cap on source pair 7's second extraction pass. This larger cap
is written only to the NEW private `memory.private.yaml`; production defaults,
repository deployment configs and existing services remain unchanged at 4096.
It is a ceiling, not a forced output length, and does not guarantee identical
model output or eliminate every possible truncation. Record the cap when
comparing accuracy. To test the exact native cap, pass `--max-tokens 4096`.
The stage launcher does not rewrite an existing fixture's settings.

For a different experimental temperature, add `--temperature 0` during
generation. For DeepSeek endpoints that reject forced tool selection while
thinking is enabled, also pass `--thinking-disabled`; this is explicit provider
compatibility configuration, not a production prompt or algorithm change.
QA uses the current native
Agent template (8192 max tokens / 131072 context); QA temperature is not forced.
The judge defaults to `deepseek-chat`, temperature zero. Verify model names with
your provider and retain its resolved response model/run date.

The generated `setup.json` records model names, embedding dimension, explicit
sampling/output settings, the two native temperature arguments `[null, 0]`,
effective span temperatures, graph/Scene mode, routing policy and fixture
isolation settings. Preflight additionally reads the **actual running** config,
dual-pass temperatures, recall score-gap default and keyword-list sizes; it
hashes the complete deployed source and full resolved config without printing
credentials. Each restart/QA question must match that receipt. Keep the raw
provider response's resolved model and date as well if your provider offers
them; an alias such as `deepseek-chat` is not a permanent model-version pin.

Review the generated `memory.private.yaml`, `agent-state/openclaw.json`,
`compose.private.yaml` and `setup.json` before the next commands. The first three
contain credentials: keep the entire fixture private. Native `decision`,
routing, hierarchy, ranker, 400-token buffer and score-gap policy are not replaced.
Scene is explicitly disabled to isolate this comparison.
Cache is explicitly off for isolation; MCP is off and memory HTTP access is
loopback-allowlisted with authentication. These are documented fixture settings,
not a claim that every service setting equals Dev's defaults.

```bash
# Parse the complete Compose configuration without printing secrets.
docker compose -f "$BEAM_FIXTURE/compose.private.yaml" config --quiet

# These are explicit infrastructure operations, not performed by the generator.
docker compose -f "$BEAM_FIXTURE/compose.private.yaml" build memory agent
docker compose -f "$BEAM_FIXTURE/compose.private.yaml" up -d --wait postgres
docker compose -f "$BEAM_FIXTURE/compose.private.yaml" create memory agent

# Initialize native canonical/vector schema without source ingestion or retrieval.
docker compose -f "$BEAM_FIXTURE/compose.private.yaml" run --rm --no-deps \
  --entrypoint python3 memory -c \
  'import core; from server.memory_service import MemoryService; MemoryService().get_read_api(start_outbox_worker=False); print("Native schema ready")'

docker compose -f "$BEAM_FIXTURE/compose.private.yaml" start memory
python -c 'import json,os; from scripts.run_stage8r_beam_clean_eval import wait_health; wait_health(json.load(open(os.environ["BEAM_SETUP"]))["memory_url"])'
```

Leave the Agent **created but stopped**, then follow sections 4–5. The generator
does not start a paid run and refuses to overwrite an existing directory.
Do not run `docker compose down --volumes` or delete a fixture you still need
for recovery. Stop only these dedicated services after retaining the artifacts.

## 3. Existing dedicated services (alternative to generation)

The stage launcher **does not provision a database, image, provider account or agent**.
The preceding generator supplies configs/Compose; the explicit Docker commands
still perform deployment. If using your own fixtures instead, ensure:
Use a fresh SQL-backed memory fixture and a dedicated OpenClaw fixture, with:

- A dedicated disposable database named `beam_eval_` followed by 8–48 lowercase
  letters, digits or underscores. The snapshot runner drops/recreates **that
  database only** between questions. Its PostgreSQL instance must support
  `psql`, `pg_dump`, `pg_restore` and `pg_control_system()` (administrative
  evaluation access is required). Never point this at a production database.
- Memory and answering containers using Linux host networking and these labels:
  `anttrail.eval.run=<run_id>` and `anttrail.eval.role=memory` / `agent`.
  Labels must be set when creating the containers, not guessed from names.
- Both memory SQL storage and vector storage pointing at the same dedicated
  database. Use the `vector_index` table, `storage.backend=sql`, and disable
  runtime caching. Initialize the schema but keep corpus tables empty before
  preflight. Configure the account/user in the memory fixture to match setup.
- Memory source under `/opt/ogmem`, built from this exact checkout. Its effective
  YAML must be discoverable through `/tmp/ogmem.resolved.yaml`, `OGMEM_CONFIG`
  or the explicit `BEAM_RUNTIME_CONFIG` container environment variable.
- For this SQL fixture, use `docker/Dockerfile.standalone` **with**
  `entrypoint: ["/bin/sh", "/opt/ogmem/scripts/beam_sql_entrypoint.sh"]`, as the
  generator does, or an equivalent dedicated single-worker SQL command.
  The regular standalone supervisor also starts an independent index service;
  that is not this experiment's single-owner SQL topology. Do not change the
  normal deployment to run the experiment.
- OpenClaw's native `/v1/responses` endpoint, `main` agent and enabled
  `og-memory-context-engine` plugin connected to the same account/user and
  memory URL. Start with a fresh session directory and an empty
  `/home/node/.openclaw/workspace`. Set `agents.defaults.skipBootstrap=true`,
  `tools.deny=["*"]`, no per-agent model overrides, and no other agents.
  Set plugin `autoCapture=false` during answering. Create the agent container
  **without starting it** until the `prepare-agent` stage below has completed.
  The agent must **not** mount the checkout, evaluator, dataset, snapshots,
  answers or results. Existing setup guidance:
  [OpenClaw integration](../docs/integrations/openclaw.md).

For example, when creating the dedicated memory fixture, add these options:

```bash
--network host \
--label anttrail.eval.run=beam-t1-unique-1234 \
--label anttrail.eval.role=memory \
-v "$PWD/scripts/beam_capture:/opt/beam-capture:ro" \
-e PYTHONPATH=/opt/beam-capture:/opt/ogmem \
-e BEAM_COMPOSE_CAPTURE=1
```

The capture hook is **memory-container-only**, explicitly opt-in. It saves the
first actual compose to `/tmp/ogmem-compose/<session-uuid>.json` with restrictive
permissions and returns the same result. It does not change routing, prompts,
sampling, token limits, graph construction, retrieval or truncation handling.
If a deployment already captures the actual compose in the agent's native
`/tmp/ogmem_ws` files, that is also supported.

Choose the graph profile, extraction provider/model, temperature,
output limit, embedding model/dimension/profile, budgets and routing explicitly
in the memory fixture. The launcher **does not silently modify any of them**.
Conflict recognition and conflict-aware delivery are not part of this bundle.
`variant` in the setup is a result label, **not a feature toggle**. Pin and
record the answering image/model as well. Do not assume a label proves that
two runs used the same settings or corpus.

## 4. Setup file and credentials

Copy and edit [beam.setup.example.json](beam.setup.example.json) outside the
checkout. All paths must be absolute and the output root must be new. Match
the run ID, labels, database, ports and identities to your dedicated fixtures.
`source_root` is automatically this checkout; no `/home/luzh/...` paths remain.

Credentials are **environment variables**, never setup JSON or command flags:

```bash
export BEAM_AGENT_TOKEN='your-isolated-openclaw-gateway-token'
export BEAM_MEMORY_API_KEY='your-authorized-memory-api-key'
export BEAM_JUDGE_API_KEY='your-judge-provider-key'
export BEAM_SETUP=/absolute/path/to/beam-setup.json
```

The judge also accepts `DEEPSEEK_API_KEY` for compatibility. Its model/base URL
can be set in setup or via `BEAM_JUDGE_MODEL` / `BEAM_JUDGE_BASE_URL`
(`DEEPSEEK_BASE_URL` is a fallback). The extraction/embedding credentials belong
only in the memory fixture's normal environment/config. The answering model's
credentials belong only in its normal provider config. Avoid retaining shell
history containing real keys; use your environment/secret manager where possible.

## 5. Run the stages

First inspect the plan; no provider, Docker, SQL or output writes occur:

```bash
python scripts/run_beam_eval.py preflight --setup "$BEAM_SETUP"
```

Then explicitly execute one stage at a time:

```bash
python scripts/run_beam_eval.py prepare-agent --setup "$BEAM_SETUP" --execute
python scripts/run_beam_eval.py preflight --setup "$BEAM_SETUP" --execute
python scripts/run_beam_eval.py ingest    --setup "$BEAM_SETUP" --execute
python scripts/run_beam_eval.py freeze    --setup "$BEAM_SETUP" --execute
python scripts/run_beam_eval.py qa        --setup "$BEAM_SETUP" --execute
python scripts/run_beam_eval.py grade     --setup "$BEAM_SETUP" --execute
```

The first stage prepares the **stopped**, explicitly configured agent. Like
other stages, it can be inspected without `--execute` and makes no live calls.

Some OpenClaw images replace the entire config with a default template if the
plugin lacks `authApiKey` or `authAccountId`. Preparation adds only missing
empty values for these schema-supported keys; existing credentials, model,
sampling, tool policy, identity, prompts and memory URL remain unchanged. The
original file is backed up privately in `prepared-agent/`. Invalid or already
overwritten configs fail validation: restore the **intended** config yourself
before preparation; the runner never guesses a backup or selects a model.
QA starts the agent and verifies its actual config hash against the preparation
receipt before sending any question. It rejects missing/mismatched receipts.

Preflight inspects the initialized empty fixture and leaves memory stopped.
Ingestion starts it, sends **source user/assistant pairs only**, waits for
extraction/indexing and retains pair receipts. Freeze requires a complete,
hash-bound ingestion manifest. It leaves memory stopped and retains a full
database dump. Freeze also tests the native PostgreSQL restore. A known first-
restore deparser rewrite distributes literal `varchar[]` to `text[]` casts in
CHECK constraints/index predicates. Only this exact, value/order-preserving
rewrite can qualify: every table row hash, sequence and all other schema bytes
must match. The original snapshot is retained at `freeze/`; a separately bound
snapshot at `freeze/qualified/` must pass **two complete exact native restores**.
Its receipts and any schema rewrite proof are retained. Other schema/data
differences fail closed; schema/constraint/index checks are never disabled.
QA automatically selects this verified snapshot; historical snapshots without
qualification metadata remain supported. Incomplete qualification is rejected.
QA restores and fingerprints the whole corpus before **each**
question, uses a fresh native answering session with all tools denied, and
retains the first HTTP answer, compose, session log and restore receipt.
Both containers are stopped and the frozen corpus is restored at the end.

This conservative full-snapshot runner prioritizes isolation over throughput;
it is not the historical optimized read-only/frozen-snapshot campaign launcher.
Never parallelize QA workers against the same database/containers. Independent
complete fixtures may run in parallel, with separate identities, ports and files.

Failed ingestion/QA is retained and is **not** automatically replayed. Inspect
the failure and start a fresh isolated run rather than accidentally duplicating
source pairs or replacing first answers. No experimental output salvage,
splitting, prompt rewrite or token-limit increase is silently injected.
Official grading alone can resume retained requests:

The grader retries transport failures (timeouts, connection errors, HTTP 429
or 5xx) at most five times with the **same request**. It never retries a
completed judgment to select a better score. A malformed/truncated completed
response remains retained and fails the run instead of being silently replaced.

```bash
python scripts/run_beam_eval.py grade --setup "$BEAM_SETUP" --execute --resume
# Recompute from fully retained judge responses, with no provider calls:
python scripts/run_beam_eval.py grade --setup "$BEAM_SETUP" --resume --verify-offline
```

## 6. Outputs and metrics

The output root contains `prepared-agent/`, `preflight/`, `ingest/`, `freeze/`,
`qa/`, `grading/`. Agent preparation backups contain credentials and must stay
private, just like the fixture's normal config.
Raw source text, injected memories, answers and provider responses are sensitive
experiment artifacts; do not commit or expose them publicly.
Pair `.completed.json` receipts include ingestion elapsed seconds and node/edge
counts. Actual compose responses retain native statistics where available; QA
HTTP elapsed time includes answering and is **not** retrieval latency. These
tools do not automatically intercept every raw provider request/response.

`grading/results.json` reports `official_category_macro_mean` and the ten
`official_category_scores`. Event ordering uses the official report's
`tau_norm` aggregation, not an invented binary-accuracy rule. Historical
`question_mean` / `rubric_mean` are also retained but are different metrics.
Compare runs only after matching the dataset, corpus, source version,
provider/model, answer/judge settings and metric.

The upstream BEAM scorer and judge prompt are pinned to commit
`b2da22eac88bb0874c64665f13457eb99835774a`; exact file hashes, source URLs and
MIT license are in [vendor/beam/provenance.json](vendor/beam/provenance.json)
and [vendor/beam/LICENSE](vendor/beam/LICENSE). No downloads or heavyweight
BLEU/embedding model initialization are needed for this official LLM judge path.

The scripts do not start an MLflow server or silently attach to a local
experiment. Save/upload these artifacts to your tracking service explicitly.
This packaging has offline unit coverage; a new paid benchmark run is a
separate operation, not performed by copying or testing these scripts.

## 7. Offline regression tests

```bash
python -m pytest -q tests/unit/scripts/test_beam_*.py tests/unit/scripts/test_run_beam_eval.py
```

These cover source-only column projection, generic task denominators, command
planning without side effects, full snapshot safety, passive compose capture,
mocked native QA, and exact official-grader replay. They are not fresh accuracy
validation of this port. An additional opt-in test verifies native Dev SQL
schema initialization and complete snapshot restoration on one temporary,
labelled tmpfs PostgreSQL container (cached image required, no model calls):

```bash
BEAM_SQL_SMOKE=1 python -m pytest -q tests/integration/test_beam_native_dev_snapshot.py
```

It removes only its own newly created container. It does not build/start the
memory/Agent images or validate live answering/model availability.
