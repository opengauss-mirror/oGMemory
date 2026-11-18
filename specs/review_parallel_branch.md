# Code Review — branch `rychkov/m3-parallel` vs `specs/parallel_dockerize.md`

Reviewed: 17 commits on `rychkov/m3-parallel` (base `d139c5fb`), plus uncommitted
working-tree changes in `extraction/operators/builtin/episodic_memory_generator.py`
and its test (video-mode `include_audio` fallback fix — looks correct, keep).

Overall: **WS 2.1–2.4, 3.1–3.8, and 4 are implemented and match the spec's contracts
well.** WS 1 (Docker), WS 2.5, WS 5, and most of WS 6 are **not implemented**. No
correctness blockers found in the implemented code; a handful of minor issues and
spec deviations below. Unit tests could not be executed in this shell (host Python
lacks `pyagfs`); re-run the suite in the runner container after fixes.

---

## A. Missing scope (implement or consciously defer)

| Spec item | Status | Notes |
|---|---|---|
| **WS 1.1** `docker/Dockerfile.m3runner` (full image) | ⚠️ Partial | `docker/Dockerfile.m3-runner` is a 13-line variant on top of the local serving image: only ffmpeg + opencv. Missing: rank-bm25, nltk + baked corpora (`punkt_tab`, `stopwords`), faster-whisper, insightface/onnxruntime (+ opencv non-headless uninstall dance), optional pyannote/speechbrain, `PYTHONPATH`/`PYTHONPYCACHEPREFIX` env, no-forced-USER contract, `CMD` default. **Verify the base serving image already ships rank-bm25/nltk** — spec calls out that without them every retrieval returns empty and QA degrades to "cannot determine". |
| **WS 1.2** `scripts/cv_service.py` | ✅ Present | Pre-existing on the base branch; check it meets the ORT thread-cap + dual content-type contract if not already done there. |
| **WS 1.3** compose `cv_service` + `runner` services / `with-m3runner` profile | ❌ Missing | `deploy/docker-compose.yml` untouched by this branch. |
| **WS 1.4** `deploy/ogmemory.m3.yaml` | ❌ Missing | |
| **WS 1.5** `entrypoint-standalone.sh` env resolution + DSN patching | ❌ Missing | No `${VAR}` resolution, no `VECTOR_DB_*`/`STORAGE_DB_*` patching, no resolved-YAML handoff. (Note: AGENTS.md documents a local workaround `deploy/ogmemory-rendered.yaml`; that is a workaround, not the spec.) |
| **WS 2.5** `--video-workers` in `scripts/run_m3_bench_subset.py` + live per-video judge | ❌ Missing | Script has no `video_workers` flag. Blocks the 100-video run path. Test `tests/unit/m3/test_bench_video_parallel.py` also missing. |
| **WS 5** `m3/wipe_stages.py` | ❌ Missing | No stage-bounded wipe; `tests/unit/m3/test_wipe_stages.py` missing. |
| **WS 6.1** AGENTS.md mind-map index update | ⚠️ Partial | Only the env-knob table landed (in `docs/OGMEMORY_ENV.md` — good). Docker-first quick start, profiles, M3 run commands, wipe usage not documented. |
| **WS 6.2** config sync (`deploy/ogmemory.example.yaml` embedding block, `config/openclaw/ogmem.template.json`) | ❌ Missing | |
| **WS 6.3** `tests/unit/m3/test_qa_runtime_wiring.py` updates | ❌ Missing | Depends on 6.4. |
| **WS 6.4** QA-only resume retrieval knobs (`qa_params_argv`, `--set qa.params.*` passthrough in `m3/resume.py` + `m3/run_video_pipeline.py`) | ❌ Missing | `run_qa_only` has no `qa_params` param; no `qa_params_argv`. |
| Env `OGMEM_TENANT_SCOPE` | ⚠️ Documented only | In the 3.8 table (docs) but nothing implements/sets it (compose change lives in WS 1.3). Either implement with WS 1.3 or drop from the table until then. |

---

## B. Issues in implemented code (fix these)

### B1. `retrieval/seed_retriever.py` — degrade path drops `root_uris`
When the embed call fails, the code sets `root_uris = []` together with
`global_results = []`. Root URIs do not depend on the embedder — they are the
hierarchical seeds `_merge_starting_points` walks — so a saturated embedder now
loses both vector search *and* the root-URI starting points, weakening the
degradation the spec intends (BM25-only retrieval "so downstream behaves
consistently"). Compute `root_uris = self._get_root_uris(...)` unconditionally
(before the embed try/except) and only gate `_global_vector_search` on
`query_vector` being non-empty.

### B2. `extraction/tools.py` — double `_merge_dual_run` in multi-span path
`_structure_spans` collects per-span results that `_structure_prepared_span_dual`
has *already* merged (line ~744), then merges the concatenation again at the end
(line ~675). `_merge_dual_run` merges by `(category, routing_key)` with
confidence/richness heuristics; merging already-merged candidates is not
identical to one global merge of the raw candidates (the `len(content) > 1.5×`
upgrade branch and confidence comparison behave differently on merged content),
so this is a subtle behavior change vs the old single-merge path. Decide
deliberately: either have `_structure_prepared_span_dual` return the resolved
(unmerged) candidate list and keep the single global merge, or drop the outer
merge and document per-span-merge semantics. Add a test pinning whichever is chosen.

### B3. `fs/sql_adapter/pool.py` — closed-connection accounting in legacy fallback
`_standalone_return_connection` legacy path (no semaphore, `__new__`-built test
objects): a connection with `conn.closed != 0` hits neither branch and is silently
dropped — fine. But `_standalone_get_connection` legacy path pops one idle conn
and if it's closed, falls through to `_create_connection` *without closing the
stale conn object*. Harmless for psycopg (closed already) but inconsistent with
the semaphore path which calls `conn.close()` defensively. Low priority; align
for symmetry.

### B4. `fs/sql_adapter/schema.py` — short-circuit assumes full-DDL success history
`to_regclass('public.graph_nodes') IS NOT NULL` → skip all DDL. Because the DDL
runs in autocommit, a historical crash mid-script leaves `graph_nodes` present but
later statements (indexes/triggers/tables created after it) unapplied — and now
never repaired. The spec accepts this trade-off, but add a code comment + log
line spelling out the assumption (partially-applied schema is not detectable),
and note it in AGENTS.md ops notes.

### B5. `providers/graph_store/sql_graph_store.py` / `fs/sql_adapter/sql_trace_store.py` — no `_create_connection` override
Both modules `import psycopg2` at module level but now rely on the mixin's
`_create_connection` (which does a fresh `import psycopg2` inside the method).
Functionally identical today, and no current test patches those module symbols —
but this silently breaks the spec's own "subclasses that patch psycopg2 at module
level must override `_create_connection`" contract for exactly the modules most
likely to gain such tests. Add the trivial override (as done in
`sql_context_fs.py`, `sql_outbox_store.py`, etc.) for consistency.

### B6. `m3/_parallel.py` — `parallel_clips` docstring vs `max_workers=0`
`max(1, max_workers)` clamps 0→1 silently; fine, but `env_max_workers` warns on
garbage while an explicit `max_workers=0` is quiet. Trivial; document or warn.
(Not a bug — flag for polish only.)

### B7. `extraction/tools.py` — span timeout is per-*completion-gap*, not per-span
`wait(..., timeout=self._span_timeout_seconds, return_when=FIRST_COMPLETED)`
resets the budget every time any span completes, so one pathological span can
extend the chunk far beyond `OGMEM_EXTRACT_SPAN_TIMEOUT` as long as siblings keep
finishing. This matches the spec text ("drop stragglers when nothing completes
within the budget"), but the env-var doc (`docs/OGMEMORY_ENV.md`) describes it as
"per-span budget". Fix the doc wording, or implement a true per-span deadline.

---

## C. Verified good (no action)

- **WS 3.1 pool**: semaphore + lock structure is correct — acquire before pop,
  release on create-failure and always in `_standalone_return_connection`'s
  `finally`; `getattr(self, "_semaphore", None)` fallback preserves `__new__`
  test-object behavior; all six subclasses delegate correctly;
  `opengauss_index._return_connection` rollback/drop logic is right; graph-store
  `ON CONFLICT (node_id) DO NOTHING` present.
- **WS 3.2 schema**: advisory lock is session-scoped under autocommit, unlock in
  `finally`, non-postgres mock bypass via `server_version` int check, autocommit
  restored.
- **WS 3.3 embedder**: semaphore is module-level (correct — endpoint saturates,
  not instance); SDK `max_retries=0`; transient classification (timeout /
  connection / rate-limit / 5xx, httpx included) matches spec; multimodal path
  includes `raise_for_status` inside the retry.
- **WS 3.5 LLM**: `OGMEM_LLM_TIMEOUT` only applies when no explicit timeout is
  passed (config still wins); `_track_usage` moved to accepted-response-only in
  both the auto-tool-choice and required branches.
- **WS 2.1 `_parallel`**: deterministic order via index-keyed futures, sync
  fallback at 1 worker, `UnitError` contract, `ThreadLocalModels` + barrier
  warmup — all per spec.
- **WS 2.2 drivers**: all seven drivers (audio, frames, faces, asr, episodic,
  qa, eval) follow the flatten → `_unit` with `dataclasses.replace(ctx,
  trace_id=...)` → ordered results pattern; `detect_faces` uses
  `face_max_workers()` and per-frame units; `voice_asr`/`qa`/`eval` use
  `ThreadLocalModels` correctly; main-thread serialization of JSONL writes
  preserves order.
- **WS 2.3/2.4**: ASR ratio tolerance in both driver (`rc=2` over ratio) and
  `_audio_driver_result` (belt-and-suspenders recompute); `--account-id` added
  to voice_asr / speaker_diarization / speaker_embedding stages.
- **WS 3.7 face clustering**: blocked GEMM masking (strict upper triangle,
  within-clip exclusion with -1 semantics) is an exact match of the legacy
  per-row check; zero-norm rows → distance 1.0 contract preserved.
- **WS 4 extractor**: Phase-1 degrade-to-`[]`, one-future-per-span,
  cancel-futures straggler drop, retry budget `1 + retries` with capped backoff,
  pass-2 failure keeps pass-1 output — all per spec.
- Uncommitted `episodic_memory_generator.py` fix (`include_audio` skipped in
  video-mode frames-fallback, `_ensure_video_stream`, `_MIN_LLM_VIDEO_SECONDS`)
  is sound and covered by new tests.

---

## D. Recommended fix order for the follow-up agent

1. B1, B2 (behavior-affecting, small).
2. B5 (contract consistency, trivial).
3. B3, B4, B7 (polish/docs).
4. Missing scope in spec's own dependency order: WS 2.5 + its test → WS 5 +
   test → WS 6.4/6.3 → WS 1.1–1.5 (verify rank-bm25/nltk in the base image
   first) → WS 6.1/6.2 docs/config sync → resolve `OGMEM_TENANT_SCOPE`.
5. Re-run the full unit suite **inside the runner container** (host shell lacks
   `pyagfs`): expected 5120 passing / 17 pre-existing failures — new failures =
   regressions from the fixes.
