# Code Review — branch `rychkov/m3-parallel` vs spec `specs/parallel_dockerize.md`

**Base:** `anttrail` (d139c5fb) → **HEAD** 32a84e80, 29 commits, 52 files, +4962/−720.
**Verdict:** Workstreams 2, 3, 4, 5, 6 are implemented faithfully and well.
Workstream 1 (Docker) is only partially done and diverges from the spec in
operator-facing ways. A few real defects found (P1–P3 below). Overall: mergeable
after the P1/P2 items are addressed; WS 1.3/1.4 gaps are deliberate-looking but
must be either implemented or explicitly documented as out of scope.

## Test status (verified in container `ogmemory-m3-runner:rychkov`)

- `pytest tests/unit -q`: **5200 passed, 21 failed, 40 skipped**.
- 20 of the 21 failures reproduce identically on base `anttrail` (pre-existing:
  `cli/test_onboard`, `resource/test_code_workspace`, `scene/test_packaging`,
  `server/test_mcp_*`, `service/test_code_workspace_sync`).
- The 21st (`tests/unit/m3/test_eval_harness.py::test_build_judge_llm_defaults_to_minimax`)
  fails **only when `OGMEM_CONFIG` points at a rendered deploy YAML** that
  configures `m3.judge_llm` (then `build_judge_llm()` correctly returns
  `OpenAILLM`, not MiniMax). Without the env it passes (38/38). Not a code bug,
  but the test is env-sensitive — see P3-4.
- All spec-mandated new tests exist and pass: `test_schema_shortcircuit.py`,
  `test_wipe_stages.py` (354 lines), `test_bench_video_parallel.py`,
  `test_parallel.py`, extended `test_eval_harness.py` / `test_qa_runtime_wiring.py` /
  `test_resume.py`, new embedder and seed-retriever degrade tests.

---

## Findings

### P1 — bugs / correctness

**P1-1. `ThreadLocalModels.warmup()` is useless as designed** — `m3/_parallel.py:149-172`.
Warmup threads are *fresh daemon threads*, but instances are stored in
`threading.local()`, so the built models die with the warmup threads and are
never seen by the `ThreadPoolExecutor` worker threads. The pool workers still
pay serial first-call construction. Currently harmless because **no caller
invokes `warmup()`** (grep confirms zero call sites). Fix: either delete
`warmup()` or make it actually pre-build into the pool (e.g. submit
`factory.get` as the first N tasks to the executor and block on them, or
replace `threading.local` with an instance pool keyed by worker checkout).

**P1-2. `SeedRetriever` now swallows configuration errors** — `retrieval/seed_retriever.py:96-117`.
The new BM25 `try/except` wraps the pre-existing
`raise RuntimeError("hybrid retrieval requires a lexical backfill coordinator")`
and `raise RuntimeError("required lexical backfill is not ready")`. Those are
**misconfiguration** signals, not transient outages; they now degrade to a
warning + empty lexical results, i.e. a misconfigured deployment silently
returns vector-only results forever. Fix: catch only operational exceptions
around `_bm25_search`/`ensure_ready` *calls* (or re-raise `RuntimeError`
unchanged and catch e.g. `Exception` minus `RuntimeError`), so config errors
still fail fast while genuine BM25 outages degrade.

### P2 — spec-conformance gaps (Workstream 1)

WS 1 is the weak spot. 1.2 (`scripts/cv_service.py`) and 1.5
(`docker/entrypoint-standalone.sh` env resolution/DSN patching) already existed
on the base branch and satisfy the spec. The branch's own WS-1 work diverges:

**P2-1. `docker/Dockerfile.m3-runner` does not follow spec 1.1.**
- `FROM ogmemory:ogmem-rychkov-local` — a **local, rychkov-specific base image**.
  Not buildable by anyone else; the spec requires a self-contained
  `python:3.11-slim-bookworm` image with the full layer list (apt
  `libpq-dev curl ffmpeg procps gcc g++`, the serving+retrieval pip set
  incl. `rank-bm25 nltk networkx`, `faster-whisper==1.2.1` pinned, the
  insightface/onnxruntime CV layer with the opencv headless force-reinstall,
  source COPY, `PYTHONPATH`/`PYTHONPYCACHEPREFIX`, **no forced USER**).
- `faster-whisper>=1.0.0` unpinned (spec: `==1.2.1`).
- Ends with `USER ogmem` (spec: *no* `USER` forced; the runner is launched
  with `--user $(id -u):$(id -g)` so host-mounted work dirs are owned by the
  invoking user). Also note this repo's AGENTS stage-3 recipe runs it as
  `--user 1003:1003`, which means the image-baked `HF_HOME=/home/ogmem/.cache`
  (chowned to `ogmem`) is not writable by the invoking uid — first-use model
  download will fail unless the operator bind-mounts a writable HF cache.
  The Dockerfile comment mentions the mount but the default is a trap.
- No insightface/onnxruntime/pyannote layers (cv_service in this repo runs as
  a separate `Dockerfile.cv` image — acceptable deviation, but say so in the
  Dockerfile header/AGENTS).
- Good: NLTK corpora are baked with a build-time verification line (spec 1.1.3).

**P2-2. Spec 1.3 not implemented — no `with-m3runner` compose profile.**
`deploy/docker-compose.yml` has no `cv_service` or `runner` services and no
`with-m3runner` profile. The acceptance criterion (`docker compose --profile
with-db --profile with-m3runner up -d`) cannot be run. If the team
intentionally runs the runner via ad-hoc `docker run` (as in AGENTS.md stage
3), add the compose services anyway or record the deviation explicitly.

**P2-3. Spec 1.4 not implemented — no `deploy/ogmemory.m3.yaml`.**
The M3 runtime config exists only as the gitignored
`deploy/ogmemory-rendered.yaml`. Provide a committed, placeholder-parameterised
`deploy/ogmemory.m3.yaml` per the spec (m3 block with `episodic_llm`,
`qa_llm`, `embedding`, `vector_index.table_name: m3_episodic_vectors`).

### P3 — minor / polish

**P3-1. Embedder semaphore frozen at import time** — `providers/embedder/openai_embedder.py:54`.
`_EMBED_SEMAPHORE` is constructed once at module import from
`OGMEM_EMBED_MAX_CONCURRENCY`; changing the env after import (e.g. in tests or
long-lived processes) has no effect. Acceptable, but document it, or lazily
initialise on first use.

**P3-2. Standalone vs shared pool timeout semantics differ** —
`fs/sql_adapter/pool.py`. Standalone `_get_connection(timeout=None)` blocks
forever; `SharedConnectionPool.get_connection` defaults to `timeout=30.0`.
Same call site can therefore hang forever or raise after 30 s depending on
pool wiring. Spec-conformant, but worth a docstring note on the mixin.

**P3-3. `voice_asr._unit`: `zip(materialized, segments)` silently truncates**
on length mismatch (`m3/voice_asr.py`, unchanged from serial version but now
harder to notice). Consider logging when `len(materialized) != len(segments)`.

**P3-4. `test_build_judge_llm_defaults_to_minimax` is env-sensitive** —
passes only when no `m3.judge_llm` YAML is in scope. Add
`monkeypatch.delenv("OGMEM_CONFIG", raising=False)` (and/or point
`config_path` at a nonexistent file) so the test is hermetic.

**P3-5. `m3/qa.py` UnitError fallback row loses `question_index` etc. only in
the impossible path** — fine, but `_run_one_question`'s row dict and the
fallback dict drifted (fallback omits `search_stats`/`temporal_scope`… no, it
includes them). Actually verified consistent — no action; just noting the
duplicated row-shape is a maintenance hazard worth a helper.

**P3-6. `deploy/check.sh` exists but the verification checklist item
"`cd deploy && bash check.sh` covers the new services" can't hold** while
`with-m3runner` doesn't exist (see P2-2). Re-run check.sh after adding the
profile.

---

## Per-workstream conformance detail

- **WS 3.1 (pool)** — ✅ Faithful. Semaphore bounds total live conns, lock
  guards idle queue, `__new__`-constructed objects fall back via `getattr`
  guards, subclasses (`sql_graph_store`, `opengauss_index`, outbox,
  control-plane, archive, relation, trace, context_fs) all delegate and keep
  module-level psycopg2 refs via `_create_connection`. openGauss override
  rolls back dirty transactions and always releases the slot via the mixin.
  `ON CONFLICT (node_id) DO NOTHING` added to the `graph_nodes` INSERT.
- **WS 3.2 (schema)** — ✅ Advisory lock + `to_regclass('public.graph_nodes')`
  short-circuit, `_backend_is_postgres` mock guard, unlock in `finally`,
  autocommit restored. The partially-applied-DDL trade-off is documented in a
  comment (good).
- **WS 3.3 (embedder)** — ✅ Module semaphore, SDK retries disabled
  (`max_retries=0`), jittered backoff, transient classification covers both
  the openai SDK and the raw-httpx multimodal path. See P3-1.
- **WS 3.4 (retrieval degrade)** — ✅ SeedRetriever keeps root URIs on embed
  failure, BM25-only fusion path preserved; pipeline falls back to
  root-URI-only `SeedResult` instead of re-raising. See P1-2.
- **WS 3.5 (LLM)** — ✅ `OGMEM_LLM_TIMEOUT` honored only when the caller
  didn't pass an explicit timeout (config still wins); `_track_usage` moved to
  the accepted response in both branches of the tool-call retry loop.
- **WS 3.6 (pool knobs)** — ✅ `OGMEM_GRAPH_POOL_SIZE` (32) in
  `load_clips`+`resume`; `OGMEM_VECTOR_POOL_SIZE` env-wins-over-YAML in
  `m3/config.py`.
- **WS 3.7 (face clustering)** — ✅ Blocked GEMM, upper-triangle masking,
  within-clip masking semantics exactly match the legacy per-row check
  (incl. the None-coordinate → never-masked rule and zero-norm → distance
  1.0 rule). Dim-mismatch raises, as before.
- **WS 3.8 (env table)** — ✅ `docs/OGMEMORY_ENV.md` documents all 15 knobs.
- **WS 2.1 (_parallel)** — ✅ `env_max_workers` clamp `[1,128]`,
  `face_max_workers` default 4, `UnitError`, deterministic input-order
  results, `workers==1` sync fallback, thread-safety contract documented in
  the module docstring. See P1-1 (warmup).
- **WS 2.2 (drivers)** — ✅ All seven drivers flattened + `_unit` +
  per-clip `dataclasses.replace(ctx, trace_id=...)`; `detect_faces` fans out
  per frame with `face_max_workers()`; `voice_asr` + `qa` + `eval` use
  `ThreadLocalModels`; main thread serializes JSONL writes; QA failure
  classification (error AND no predicted answer) matches spec.
- **WS 2.3 (ASR tolerance)** — ✅ Driver returns 2 over ratio; stage layer
  rechecks from the summary's own `failed_audio`/`attempted_audio`.
- **WS 2.4 (account scoping)** — ✅ `--account-id` added to voice_asr,
  speaker_diarization, speaker_embedding stages.
- **WS 2.5 (video-parallel bench)** — ✅ `--video-workers`, live per-video
  judge via `live_judge_video`, merge-of-per-video-judges with
  `merge_of_per_video_judges: true`, `Lock`-guarded aggregate state, resume
  by per-video JSONL coverage.
- **WS 4 (span extractor)** — ✅ In `extraction/tools.py` (not
  `span_extractor.py` — target-structure adaptation per Global constraint 4).
  Phase-1 degrades to `[]` (fallback span), Phase-2 `wait(FIRST_COMPLETED)`
  loop drops stragglers, one future per span, retry budget with capped
  backoff, temp=0 failure keeps pass-1 output, single global
  `_merge_dual_run` preserved (commit 1675dc5a fixed a real
  merge-equivalence bug here — good).
- **WS 5 (wipe_stages)** — ✅ Dependency ORDERED_STAGES (with an extra
  `m3.voice_clustering` stage vs the spec list — verified consistent with
  STAGE_SPECS and tests), recursive CTE over `source_raw_id`, dry-run
  default, account mandatory, post-ops semantics per spec.
- **WS 6** — 6.2 ✅ (`ogmemory.example.yaml` synced), 6.3 ✅ (all tests
  present), 6.4 ✅ (`qa_params_argv` last-wins). 6.1: repo-root `AGENTS.md`
  is **untracked** (`git status`: `?? AGENTS.md`) — the mind-map index doc
  was never committed. Commit it (or confirm it's intentionally local-only).
- **WS 1** — ⚠️ see P2-1/P2-2/P2-3. 1.2/1.5 pre-existed on base.
- **Beyond-spec additions** (fine, tested): episodic video mode
  (`--use-video`, commits 1448cc05/97cb623c), `scripts/m3_srt_asr_import.py`,
  per-driver tests.

## Suggested fix order for the next agent

1. P1-1 (delete or repair `ThreadLocalModels.warmup`).
2. P1-2 (re-raise config `RuntimeError`s in `SeedRetriever`; add a regression
   test asserting misconfiguration still fails).
3. P2-1 (rewrite `Dockerfile.m3-runner` self-contained per spec 1.1, or at
   minimum: document the local base, pin faster-whisper, drop forced USER).
4. P2-2 + P2-3 (compose `with-m3runner` profile + committed
   `deploy/ogmemory.m3.yaml`), then re-run the WS-1 acceptance commands.
5. P3-4 (hermetic minimax test), P3-3 (zip length log), commit `AGENTS.md`.
6. Full suite: `pytest tests/unit -q` must stay at ≤ the 20 pre-existing
   failures (run without `OGMEM_CONFIG` set).
