# M3 Multimodal Memory Pipeline — Replication Spec

> **Purpose.** This document specifies, in enough detail to reproduce, the
> bundle of work shipped on the `m3-multimodal-dockerize` branch of the
> `ant-trail` fork of oGMemory. The target codebase is **upstream ogmem by
> openGauss** — a sibling repo with the same core abstractions (`RawNode` →
> `ExtractedNode` → `MemoryNode`, `GraphService`, `RequestContext`,
> `PoolAdapterMixin`, `SQLGraphStore`, `OpenGaussVectorIndex`,
> `OpenAIEmbedder`, `OpenAILLM`, `Extractor` span-structuring, M3-Bench
> `m3/` evaluation suite). Replicate each unit below into that codebase.
>
> **Scope is the whole bundle**, organized into 6 workstreams. Each
> workstream lists: *what to add/change*, *the exact contract*, *the env
> knobs*, and *the test that proves it*. Files are referenced by their
> path in the source repo; map them to the equivalent path in the target.

## Global constraints (read first)

1. **No China/Huawei specifics.** The target is the standard upstream
   build. Drop **all** Huawei mirror configuration:
   - No `sed -i 's/deb.debian.org/mirrors.tools.huawei.com/...'` in any
     Dockerfile. Use the default Debian/PyPI indexes.
   - No `pip config set global.index-url http://mirrors.tools.huawei.com/...`.
   - No offline-vendored-wheel assumption (`docker/py_wheels` + `--no-index
     --find-links`). Install torch / pyannote / speechbrain from the
     public index. (If the target host is offline, vendoring is a *deploy*
     concern, not a *code* concern — keep the Dockerfile using the public
     index and let the operator mirror it.)
   - The `OG_IMAGE` default `swr.cn-north-4.myhuaweicloud.com/...` in
     compose must become a public openGauss image tag (or be left to the
     operator via env).
2. **No gated-model / offline-cache gotchas baked into code.** The
   `huggingface_hub==0.26.3` pin, the `refs/main` no-trailing-newline
   workaround, and `sitecustomize.py`'s `torch.load(weights_only=False)`
   patch exist *only* because the source repo ran fully offline with a
   vendored pyannote cache. In the target, use current `huggingface_hub`
   and let models download normally; only keep `sitecustomize.py` if the
   target also pins `torch>=2.6` and runs vendored pyannote checkpoints
   (document it as optional, not required).
3. **Env-var names are part of the contract.** Keep the `OGMEM_*` names
   verbatim — tests, compose env, and AGENTS docs reference them. The only
   ones that may be renamed are the `OPENGAUSS_*` legacy fallbacks in
   `entrypoint-standalone.sh` (already deprecated in favor of `VECTOR_DB_*`);
   you can drop the legacy fallbacks entirely in a clean target.
4. **Behavior, not line numbers.** The diffs below describe behavior and
   contracts. Reproduce the behavior; do not copy byte-for-byte if the
   target file structure differs.
5. **Python 3.11+.** The image base is `python:3.11-slim-bookworm`.

---

## Workstream 1 — Dockerize the M3 multimodal runner

**Goal:** a standalone image + compose profiles that bring up the full
multimodal pipeline (face recognition CV service + reusable pipeline
runner) alongside the existing ogmem/openGauss/postgres stack.

### 1.1 `docker/Dockerfile.m3runner` (NEW)

A second image, distinct from the slim serving image
(`Dockerfile.standalone`). It adds the multimodal runtimes the pipeline
needs on top of the same base packages the serving image uses (so every
`core/` `providers/` `m3/` import resolves identically).

Layers, in order:

1. **Base** — `python:3.11-slim-bookworm`; apt install
   `libpq-dev curl ffmpeg procps gcc g++` (ffmpeg is required by
   `m3/extract_audio.py`; gcc/g++ for any C-extension wheels).
2. **Base Python packages** — the serving image's set **plus** the M3
   retrieval deps:
   `flask gunicorn openai psycopg2-binary rich requests apscheduler httpx
   pyyaml jinja2 json-repair chromadb opencv-python-headless numpy uvicorn
   fastapi rank-bm25 nltk networkx`.
   - `rank-bm25` + `nltk` power `retrieval/bm25_index._tokenize` (the
     `[Search]` BM25 rerank). Without them, every retrieval returns
     nothing and the model answers "cannot determine" for everything.
   - `networkx` for session/textrank rolling compression.
3. **NLTK corpora** — `punkt_tab` (tokenizers) + `stopwords` (corpora).
   NLTK downloads on first use; bake them into the image so no runtime
   network is needed. In the source repo they were vendored as zips in
   `docker/nltk_data/` and extracted at build time. In the target, the
   clean approach is `python -c "import nltk; nltk.download('punkt_tab');
   nltk.download('stopwords')"` at build time (requires build-time
   network — acceptable for upstream). Extract to
   `/usr/local/share/nltk_data`.
4. **ASR** — `faster-whisper==1.2.1` (CTranslate2 backend, no torch).
   Used by `m3/voice_asr.py`.
5. **Speaker diarization + speaker embeddings** (optional, only if the
   target wants the diarization stages) —
   `torch==2.8.0 torchaudio==2.8.0 pyannote.audio==3.3.2
   speechbrain==1.0.1 matplotlib==3.9.4`. **From the public index** (not
   vendored wheels). Keep `huggingface_hub` at whatever the dependency
   closure pulls (do NOT pin to 0.26.3 unless you also vendor the cache —
   see Global constraint 2).
6. **CV** — `onnxruntime==1.29.0 insightface==1.0.1 Pillow scipy filelock
   tqdm requests six cython`, then **uninstall** the non-headless
   `opencv-python` that insightface drags in and **reinstall**
   `opencv-python-headless` (`--force-reinstall --no-deps`). This is
   critical: insightface hard-depends on the libGL build of opencv, which
   conflicts with the headless build the rest of the image uses.
7. **`sitecustomize.py`** (COPY into
   `/usr/local/lib/python3.11/site-packages/`) — *optional*. Only needed
   if you keep `torch>=2.6` AND run vendored pyannote checkpoints. It
   monkeypatches `torch.load` to force `weights_only=False` so
   pyannote/pytorch-lightning checkpoints load. If models download from
   HF normally, **omit this file**.
8. **Source COPY** — copy `core/ retrieval/ extraction/ commit/ index/
   providers/ service/ fs/ server/ session/ scripts/ perf/ m3/ config/`
   into `/opt/ogmem/`. `chmod -R 755 /opt/ogmem`.
9. **Env** — `PYTHONPATH=/opt/ogmem`,
   `PYTHONDONTWRITEBYTECODE=1`, `PYTHONPYCACHEPREFIX=/tmp/ogmem-pycache`
   (so the read-only `/opt/ogmem` sources never need to be writable).
10. **No `USER` forced** — the runner is launched with
    `--user $(id -u):$(id -g)` so host-mounted work dirs are owned by the
    invoking user. `WORKDIR /opt/ogmem`,
    `CMD ["python3", "-m", "m3.run_video_pipeline", "--help"]`.

### 1.2 `scripts/cv_service.py` (NEW)

A FastAPI face-recognition HTTP service that wraps face detection and
returns responses in the format
`extraction.operators.backends.http_backend.HTTPBackend._parse_response`
consumes, so the built-in `face_detection` operator can be wired against
it via `HTTPBackend(base_url=...)`.

Contract:
- **Backends:** `HaarBackend` (OpenCV Haar cascade, bundled, no download,
  confidence fixed 0.9) is the default/always-available; `InsightFaceBackend`
  (insightface `buffalo_l`, full detection + 512-d embeddings) is opt-in
  via `--backend insightface`.
- **`POST /detect/faces`** accepts either raw image bytes
  (`Content-Type: application/octet-stream` — the path `HTTPBackend` takes)
  or JSON `{"image_b64": "..."}`. Query params: `threshold` (default 0.0),
  `max_items` (default 20). Response: `{"detections": [{label, confidence,
  bbox:[x1,y1,x2,y2] normalized 0-1, metadata:{face_id, detector},
  embedding:[...] | null}]}`.
- **`/detect/objects`** and **`/detect/pose`** are stubs returning empty
  `detections` (placeholder for a future YOLOv8 backend).
- **`/health`** reports `backend`, `insightface_available`,
  `face_embeddings`, `model_name`.
- **ONNXRuntime thread cap (important):** insightface's `FaceAnalysis`
  creates ORT sessions without `SessionOptions`, so ORT defaults to ALL
  cores → thread thrash. Monkeypatch `onnxruntime.InferenceSession.__init__`
  at import time to inject `intra_op_num_threads` from env
  `ORTSESSION_INTRA_OP_THREADS` (default 8). This is the single biggest
  cv_service stability fix.
- CLI: `--host 0.0.0.0 --port 8081 --backend {haar,insightface} -v`.
  Run via `uvicorn`.

### 1.3 `deploy/docker-compose.yml` — add two services + profiles

Add to the existing compose file:

- **`cv_service`** (profile `with-m3runner`): builds from
  `Dockerfile.m3runner`, `network_mode: host`, runs
  `python3 scripts/cv_service.py --backend insightface --port ${CV_PORT:-8083}`.
  Mount insightface weights read-only (`buffalo_l` pack) + a small writable
  HOME. `user: "${PUID:-0}:${PGID:-0}"`.
- **`runner`** (profile `with-m3runner`): builds from `Dockerfile.m3runner`,
  `network_mode: host`, `command: ["sleep","infinity"]` (so
  `compose exec`/`run --rm` work), auto-wires env so pipeline commands need
  almost no flags:
  - `OGMEM_CONFIG=/etc/ogmem/config.yaml` (mount `ogmemory.m3.yaml` ro)
  - `OGMEM_GRAPH_CONNECTION_STRING` = built from `STORAGE_DB_*` (the same
    Postgres the ogmem service uses)
  - `OGMEM_CV_SERVICE_URL=http://127.0.0.1:${CV_PORT:-8083}`
  - `NLTK_DATA` (overridable corpora source)
  - HF env (`HF_HOME`, `PYANNOTE_CACHE`, `HF_HUB_OFFLINE`, `HF_TOKEN`) —
    only if diarization stages are enabled; point at a host HF cache mount
  - `OMP_NUM_THREADS=8` (pyannote 23× faster at 8 intra-op threads than
    the default 192 — keep this even on upstream)
  - volumes: dataset (`/data_m3` ro), HF cache (`/data/hf` ro), work dir
    (`/work` rw), config (ro)
- The existing `opengauss` / `postgres` / `ogmem` / `openclaw` services
  stay as-is. Document the two DB connection families
  (`STORAGE_DB_*` for the SQL store, `VECTOR_DB_*` for the vector DB) and
  the `with-db` / `with-openclaw` / `with-m3runner` / `with-m3` profiles.

### 1.4 `deploy/ogmemory.m3.yaml` (NEW)

M3 runtime config consumed by the runner. Blocks: `llm` (the multimodal
model), `embedding` (the embedder — note openGauss HNSW caps at 2000 dims,
so the embedder must be ≤2000-dim), `storage` (SQL backend, DSN),
`vector_db` (`type: opengauss`, `connection_string`, `table_name`,
`dimension` matching the embedder), `identity` (`account_id`/`user_id`/
`agent_id` — the M3 bench account, e.g. `m3bench`), and an `m3:` block with
`episodic_llm`, `qa_llm`, `embedding`, `vector_index`
(`table_name: m3_episodic_vectors`, `pool_size`). Keep the YAML verbatim
except for model names / endpoints / passwords (operator-supplied).

### 1.5 `docker/entrypoint-standalone.sh` (MODIFY — serving image only)

Add env-var resolution + DSN patching at container start (the serving
image, not the runner — the runner reads the YAML directly):
- Resolve `${ENV_VAR}` references in the mounted config YAML via a small
  inline Python `re.sub`.
- Patch `vector_db.connection_string` from `VECTOR_DB_*` env
  (`VECTOR_DB_HOST`, `VECTOR_DB_HOST_PORT`, `VECTOR_DB_NAME`,
  `VECTOR_DB_USER`, `VECTOR_DB_PASSWORD`). Legacy `OPENGAUSS_*` may remain
  as fallback or be dropped (Global constraint 3).
- Patch `storage.connection_string` from `STORAGE_DB_*` when
  `STORAGE_DB_HOST` is set.
- Write the resolved YAML to `/tmp/ogmem.resolved.yaml`, export
  `OGMEM_CONFIG` to it, then start the IndexService background process +
  gunicorn (`--worker-class gthread --workers ${OGMEM_WORKERS:-2}
  --threads ${OGMEM_THREADS:-32} --timeout ${OGMEM_GUNICORN_TIMEOUT:-900}`).

**Acceptance (Workstream 1):** `docker compose --profile with-db
--profile with-m3runner up -d` brings up postgres + ogmem + cv_service +
runner; `curl http://127.0.0.1:${CV_PORT}/health` returns
`{"backend":"insightface",...}`; `docker compose exec runner python3 -c
"import m3, faster_whisper, insightface; print('ok')"` succeeds.

---

## Workstream 2 — Parallelize the M3 pipeline drivers

**Goal:** every per-clip/per-question loop in `m3/` runs on a bounded
thread pool instead of sequentially, preserving deterministic output
order and the existing "collect failures, keep going" semantics.

### 2.1 `m3/_parallel.py` (NEW) — the shared primitive

This is the single dependency every parallelized driver imports. It must
exist first.

Public API:
- `env_max_workers(*, default=8, env="OGMEM_M3_MAX_WORKERS") -> int` —
  reads the env var, clamps to `[1, 128]`, warns on garbage. `default=8`.
- `face_max_workers() -> int` — `env_max_workers(default=4,
  env="OGMEM_M3_FACE_WORKERS")`. Face detection is an HTTP fan-out against
  a single cv_service whose ORT session serializes `detect()`; a small cap
  avoids request pile-up. Default 4.
- `@dataclass(frozen=True) class UnitError` — carries `exception` + `tb`
  so callers can re-emit faithful `logger.error(..., exc_info=...)` in the
  main thread. `__str__` returns the exception message so existing
  `"...: %s"` logging keeps working.
- `parallel_clips(clips, unit, max_workers=None, *, label="clip(s)") ->
  list[tuple[Any, Any]]`:
  - Empty input → `[]`.
  - `max_workers=None` → `env_max_workers()`; capped at `len(clips)`;
    `workers==1` → synchronous fallback (identical semantics, no pool).
  - Submits `unit(clip)` via `ThreadPoolExecutor(thread_name_prefix="m3clip")`,
    index-keyed futures so final order is deterministic.
  - A unit that raises → result is `UnitError(exc, traceback)` (does NOT
    abort the batch).
  - Returns `[(clip0, result0), ...]` in input order. Callers iterate,
    check `isinstance(result, UnitError)`, log + append to `failed`,
    keep their existing exit-code semantics.
- `class ThreadLocalModels` — build one model-backed object per worker
  thread, lazily, via `threading.local()`. `__init__(factory, *, name)`;
  `get()` returns the per-thread instance (constructs on first call in a
  thread); `warmup(count)` pre-builds in `count` fresh threads (avoids
  serial warm-up when the first batch hits empty slots). **Use this for
  any backend that is not safe to share across threads** (faster-whisper
  CTranslate2, torch models, and — importantly —
  `OGMemoryControlAdapter`, whose `run()` rejects concurrent questions).

**Thread-safety contract (document in the module docstring — callers rely
on it):**
- SQL graph stores + vector indexes: **safe to share** (bounded pools,
  see Workstream 4). Raise pool sizes ≥ worker count.
- OpenAI-compatible LLM clients: **safe to share** (openai SDK client +
  thread-safe `TokenTracker`).
- Local model backends (faster-whisper, torch, pyannote): **NOT safe** —
  one per worker thread via `ThreadLocalModels`.
- `ctx` (`RequestContext`) is a frozen dataclass — pass a per-clip copy
  with a unique `trace_id` via `dataclasses.replace(ctx,
  trace_id=f"{ctx.trace_id}-{video_id}-{clip_id}")` so logs correlate.

### 2.2 Parallelize each driver (MODIFY — same pattern in each)

The transformation is mechanical and identical across drivers. For each
`m3/<driver>.py` `run()`:

1. **Flatten** the nested `for video in manifest: for clip in video:` loop
   into a single `clips: list[dict]` of per-clip descriptors
   (`{video_id, clip_id, video_node, ...}`).
2. **Define `_unit(clip)`** that builds a per-clip `ctx` copy
   (`dataclasses.replace`) and calls the existing per-clip function
   (`extract_audio_for_clip`, `extract_frames_for_clip`,
   `detect_faces_for_frame`, the episodic `_generate_with_retry`, etc.).
3. **Iterate `for clip, result in parallel_clips(clips, _unit):`**; on
   `UnitError` → `logger.error(...); failed.append({...}); continue`;
   else write the output line + update counters exactly as before.

Drivers to transform (in dependency order):
- `m3/extract_audio.py` — `parallel_clips(clips, _unit)` (default workers).
- `m3/extract_frames.py` — same. Each thread opens its own
  `cv2.VideoCapture`, so decode concurrency is limited only by cores.
- `m3/detect_faces.py` — **use `face_max_workers()`** (not the default).
  The unit is per-**frame**, not per-clip (face detection fans out per
  extracted frame). Flatten to a `frames: list[dict]` of
  `{video_id, clip_id, frame_node}`.
- `m3/voice_asr.py` — **`ThreadLocalModels` for the extractor**. The
  faster-whisper backend owns one CTranslate2 model; concurrent
  `transcribe` on one model is unsafe. Build
  `extractor_factory = ThreadLocalModels(lambda: VoiceASRExtractor(...))`
  and call `extractor_factory.get()` inside `_unit`. The unit returns a
  dict `{rows, resumed_segments, failed, cached, executed}` (not raw JSONL
  writes) so the main thread serializes file writes in order. See 2.3
  for the ASR failure-tolerance addition.
- `m3/episodic_memory.py` — `parallel_clips(clips, _unit)`. The unit
  resolves per-clip frame/audio children, calls `_generate_with_retry`,
  returns `(record, retry_count)`. The multimodal LLM call dominates;
  OpenAI client + SQL graph store are safe to share; the read-only
  `frames_index`/`audio_index`/`asr_index` are safe to share.
- `m3/qa.py` (`run_react_qa`) — **`ThreadLocalModels` for the adapter**
  AND `parallel_clips` over questions. `OGMemoryControlAdapter.run()` is
  not re-entrant, so each worker gets its own adapter over the shared
  thread-safe graph/embedder/LLM. Build `pending` (questions not in
  `done_keys`), define `_run_one_question(q)` returning the full output
  row dict (never raises — catches and sets `error`), then
  `parallel_clips(pending, _run_one_question, label="question(s)")`.
  Counters (`answered`/`failed`) updated by the main thread after writing
  each row: a row counts as failed only when it has an error AND no
  predicted answer (the scaffolding-salvage branch answers with an error
  string but a real prediction).
- `m3/eval.py` — two parallel sites, same `ThreadLocalModels`-for-adapter
  pattern: (a) `run()` (answer+judge) — `_answer_and_judge(q)` runs the
  adapter then `judge_answer`; (b) `_run_judge_only()` (judge-only) —
  `_judge_one(q)` returns `(qr, was_missing)`. Both use
  `parallel_clips(pending, ..., label="question(s)")`. The judge LLM is
  shared (OpenAI client is thread-safe).

### 2.3 ASR per-clip failure tolerance (MODIFY `m3/voice_asr.py`)

A few per-clip ASR failures (silent/corrupt segments tripping
faster-whisper numpy bugs) must not kill the whole video. After the
parallel loop, instead of `return 2 if failed else 0`:

```
ratio = float(os.environ.get("OGMEM_ASR_FAIL_RATIO", "0.15"))
max_allowed = 0 if not audio_nodes else max(1, int(len(audio_nodes) * ratio))
if failed and failed > max_allowed:
    logger.error("... >%.0f%% tolerated — failing run", ratio*100); return 2
if failed:
    logger.warning("... within tolerance — continuing")
return 0
```

Mirror this at the **stage layer** in `m3/pipeline/stages.py`
`_audio_driver_result`: the stage is `failed` only if `rc not in (0, 2)`
OR `failed_audio > max(1, attempted_audio * ratio)`. The module returns
`rc=2` when over ratio; the stage treats `rc==2` as failure too (the
ratio check there is belt-and-suspenders for the summary's own
`failed_audio` count). Apply to all audio-driven stages (voice_asr,
speaker_diarization, speaker_embedding).

### 2.4 Account-scoped diarization stages (MODIFY `m3/pipeline/stages.py`)

`speaker_diarization_stage` and `speaker_embedding_stage` previously
omitted `--account-id`, so they silently diarized the default account's
clips. Both must pass `argv.extend(["--account-id", ctx.account_id])`.
(`voice_asr_stage` got the same fix.) This is a correctness bug, not a
perf feature — do not skip it.

### 2.5 Video-parallel bench driver (MODIFY `scripts/run_m3_bench_subset.py`)

Add `--video-workers N` (default 1): one thread per video, each running
the full per-video flow (pipeline subprocess → live per-video judge →
incremental merge/report refresh). Use `ThreadPoolExecutor` +
`as_completed`; only the shared aggregate state
(`summaries`/`done_ids`/`run_report.json`) is touched under a `Lock`.
With `--judge`, score each video's predictions **immediately** after that
video finishes via `judge_video_jsonl` (in-process `m3.eval` judge-only,
questions judged in parallel via `OGMEM_M3_MAX_WORKERS`), writing
per-video `m3_eval_summary.json`. The final `_aggregate` summary is a
merge of the per-video judge rows (no second judge pass). `Wv=1` behaves
exactly like the old serial loop. Resumable: skip a video if its
per-video `m3_qa_results.jsonl` already covers every annotation question
(`--force` to override).

**Acceptance (Workstream 2):** `OGMEM_M3_MAX_WORKERS=4` runs of each
driver finish faster than serial; output JSONL order is identical to a
serial run (deterministic); a single failing clip is logged and skipped,
not fatal. New tests: `tests/unit/m3/test_bench_video_parallel.py`
(video-parallel threads + live judge).

---

## Workstream 3 — Parallel-reliability core changes

**Goal:** the parallel drivers (Workstream 2) hammer the embedder, the
SQL pools, and the schema bootstrap concurrently. Without these core
changes, parallel runs deadlock, saturate, or lose whole chunks.

### 3.1 Bounded standalone connection pool (MODIFY `fs/sql_adapter/pool.py`)

`PoolAdapterMixin` is the base for `SQLGraphStore`, `OpenGaussVectorIndex`,
and the SQL outbox/control-plane/archive stores. The legacy pool limited
only the **idle** queue — N concurrent `get_connection` calls each opened
their own `psycopg2.connect`, blowing past `pool_size`.

Change `_init_pool` to also create:
- `self._lock = threading.Lock()` — guards the `len(...)` check + `append`/
  `pop` (otherwise the pool exceeds `pool_size` under contention).
- `self._semaphore = threading.Semaphore(max(1, pool_size))` — bounds
  **total live** connections (idle + active). Blocking acquire (no
  timeout) so high parallelism queues for a slot rather than
  over-connecting.

Replace `_get_connection`/`_return_connection` with:
- `_standalone_get_connection(self, timeout=None)` — acquire the semaphore
  (blocks up to `timeout`, else `RuntimeError`); under the lock, pop a
  live idle conn or `_create_connection()`; on exception, release the
  semaphore.
- `_standalone_return_connection(self, conn)` — under the lock, return to
  idle queue if `len < pool_size` and live, else close; **always** release
  the semaphore in `finally`.
- `_create_connection(self)` — `import psycopg2; psycopg2.connect(...)`.
  Subclasses that patch `psycopg2` at module level (e.g.
  `opengauss_index`) **must override `_create_connection`** to use their
  module reference, or unit tests that patch the module symbol break.
- `_get_connection(self, timeout=None)` →
  `self._standalone_get_connection(timeout=timeout)`;
  `_return_connection(self, conn)` →
  `self._standalone_return_connection(conn)`.

Objects built via `__new__` that skipped `_init_pool` (unit tests) have no
semaphore/lock → fall back to the legacy pop-or-create path (use
`getattr(self, "_semaphore", None)` guards).

**Update the three subclasses** to drop their own `_get_connection`/
`_return_connection` overrides and delegate to the mixin:
- `providers/graph_store/sql_graph_store.py` — delete the local
  `_get_connection`/`_return_connection` (mixin provides them). Add
  `ON CONFLICT (node_id) DO NOTHING` to the `graph_nodes` INSERT (re-runs
  over an existing DB hit a UniqueViolation otherwise).
- `providers/vector_index/opengauss_index.py` — override only
  `_create_connection` (module psycopg2 ref for test patching) and
  `_return_connection` (openGauss can leave a conn inside an open
  transaction after errors → `rollback()` before returning to pool, else
  close). Delegate `_get_connection` to the mixin.
- `commit/sql_outbox_store.py`, `server/sql_control_plane_store.py`,
  `session/sql_archive_store.py` — delegate `_get_connection`/
  `_return_connection` to the mixin (same pattern as the graph store).

### 3.2 Serialized schema DDL + short-circuit (MODIFY `fs/sql_adapter/schema.py`)

`ensure_schema(conn)` runs `db_schema.sql` idempotently. Two cold-start
problems under parallel workers:
1. Concurrent processes race `CREATE TABLE IF NOT EXISTS` and deadlock on
   `AccessExclusiveLock` (observed: 4× "deadlock detected" with 2 gunicorn
   workers + outbox workers).
2. Fresh video-worker processes re-run the **whole** DDL even when tables
   exist; `CREATE TABLE IF NOT EXISTS` briefly grabs `AccessExclusive` and
   can deadlock against a mid-transaction writer (pagerank UPDATE).

Fix:
- Module-level `_SCHEMA_LOCK_KEY = 741010016003217` (arbitrary stable int).
- `_backend_is_postgres(conn)` — `True` only for real psycopg connections
  (`isinstance(getattr(conn, "server_version", None), int)`); mocks skip
  the lock silently.
- In `ensure_schema`, after the existing `_schema_ensured` process flag:
  set `conn.autocommit = True`; if postgres, `SELECT pg_advisory_lock(KEY)`
  (session-scoped, autocommit-safe — each process waits its turn).
  **Then** `SELECT to_regclass('public.graph_nodes') IS NOT NULL`; if the
  row is True, the schema is fully applied (the script is one idempotent
  block) → set `_schema_ensured=True`, `pg_advisory_unlock`, **return
  without running any DDL**. Only if the table is absent run
  `cur.execute(schema_sql)` under a `try/finally` that always
  `pg_advisory_unlock`. Restore `autocommit` in the outer `finally`.

The `graph_nodes` presence check is the short-circuit: it assumes
`db_schema.sql` is a single idempotent script (if the target splits DDL
into per-table migrations, pick the last/keystone table to check).

### 3.3 Embedder concurrency + retry (MODIFY `providers/embedder/openai_embedder.py`)

Under a storm of concurrent requests the embedder endpoint saturates and
every query embed times out, emptying the working set for hours. Three
defenses:
1. **Max-concurrency semaphore** — module-level
   `_EMBED_SEMAPHORE = threading.BoundedSemaphore(OGMEM_EMBED_MAX_CONCURRENCY)`
   (env, default 8). Wrap every `embeddings.create` / multimodal embed in
   `with _EMBED_SEMAPHORE:`.
2. **Jittered exponential backoff** — disable the SDK's own retries
   (`client_kwargs["max_retries"] = 0`) and implement `_run_with_retry(fn,
   what)`: attempts = 1 + `OGMEM_EMBED_RETRIES` (default 2); on a
   **transient** error (`_is_transient`: `APITimeoutError`,
   `APIConnectionError`, `RateLimitError`, or `APIStatusError` with
   `status_code >= 500`) sleep `base_delay * 2**(attempt-1) *
   uniform(0.7, 1.3)` (`OGMEM_EMBED_BASE_DELAY`, default 1.0s) and retry;
   non-transient (auth/4xx) fail fast.
3. **Caller degrades** (Workstream 3.4) — the embedder still raises after
   exhausting retries; the retriever catches and degrades.

### 3.4 Retrieval degrades to BM25 on embed failure (MODIFY
`retrieval/seed_retriever.py` + `retrieval/pipeline.py`)

`SeedRetriever.search`: instead of `query_vector =
self.embedder.embed_texts([text])[0]` at the top (which raises and empties
the working set), wrap the embed in `try/except`; on failure, log a
warning and proceed with `query_vector = []`. If `query_vector` is empty,
skip vector search and do **keyword/BM25-only** retrieval
(`self._fuse_results([], bm25_results)` with the same fusion formula,
vector weight off) — so downstream score-threshold filtering behaves
consistently. If BM25 also fails, log and return empty (do not raise).

`RetrievalPipeline` (the layer above): in the `seed_retriever.search`
`try/except`, **do not re-raise** — on any exception, fall back to
`SeedResult` rooted at the typed query's root URIs (score 0.0) so the
search run continues instead of failing every turn during outages.

### 3.5 LLM client timeout env-tunable (MODIFY `providers/llm/openai_llm.py`)

Hardcoded `OpenAI(timeout=120.0, ...)` is too tight when many video
pipelines hit the endpoint concurrently (queue wait + decode exceeds 120s
→ non-retryable "Request timed out" → per-clip flash failures). Make it
`timeout_s = float(os.environ.get("OGMEM_LLM_TIMEOUT", "120.0"))`.

Also fix usage tracking: in the tool-call enforcement retry loop, call
`self._track_usage(response)` **only for the accepted response** (after
the retry loop accepts the tool calls), not on every attempt — otherwise
retried attempts and the final `RuntimeError` inflate the token counters.
Move the `_track_usage` call to after the retry loop's success branch.

### 3.6 Pool-size env knobs (MODIFY `m3/config.py`, `m3/load_clips.py`, `m3/resume.py`)

- `OGMEM_GRAPH_POOL_SIZE` (default 32) — `SQLGraphStore` pool size in both
  `m3/load_clips.build_graph_service` and `m3/resume.build_graph_service`.
  Must be ≥ the parallel-clip worker count or threads serialize on
  checkout.
- `OGMEM_VECTOR_POOL_SIZE` (default: YAML `m3.vector_index.pool_size` or
  5) — read in `m3/config.load_m3_runtime_config` via
  `int(os.environ.get("OGMEM_VECTOR_POOL_SIZE", yaml_value or 5))`. Env
  wins over YAML.

### 3.7 Vectorized face clustering (MODIFY `m3/face_clustering.py`)

The pure-Python O(N²) double loop took 30+ min per video at 14k account
faces. Replace with numpy-blocked cosine similarity:
- Stack embeddings into `embs` (float64), L2-normalize to `unit`.
- Assign integer `group_ids` per distinct `(video_id, clip_id)` (faces
  missing either coordinate → -1, never masked — matches the legacy
  per-row check).
- Block the GEMM: for `i0` in `range(0, n, 2048)`, `sim = unit[i0:i1] @
  unit.T`, `dist = 1 - sim`. Mask out self-pairs (`j < i` within block)
  and within-clip pairs (`group_ids[i0:i1,None] == group_ids[None,:]` and
  both != -1). `keep = dist < threshold`. `np.argwhere(keep)` →
  `uf.union(i0+r, c)` for each pair.
- Memory bounded at N×2048 rows; time is LAPACK GEMM. 14k faces in ~37s
  (≈2000×). Exact-match equivalent to the legacy loop.

### 3.8 Env-var reference (document all of these)

| Env var | Default | Where | Purpose |
|---------|---------|-------|---------|
| `OGMEM_M3_MAX_WORKERS` | 8 | `m3/_parallel.env_max_workers` | Per-clip + per-question thread-pool size (drivers, QA, eval judge) |
| `OGMEM_M3_FACE_WORKERS` | 4 | `m3/_parallel.face_max_workers` | Per-frame face-detection fan-out cap (cv_service serializes detect) |
| `OGMEM_GRAPH_POOL_SIZE` | 32 | `m3/load_clips`, `m3/resume` | SQL graph-store connection pool; keep ≥ worker count |
| `OGMEM_VECTOR_POOL_SIZE` | 5 (or YAML) | `m3/config.load_m3_runtime_config` | Vector-index connection pool |
| `OGMEM_EMBED_MAX_CONCURRENCY` | 8 | `providers/embedder/openai_embedder` | Bound in-flight embed calls |
| `OGMEM_EMBED_RETRIES` | 2 | embedder | Jittered backoff attempts on transient embed failures |
| `OGMEM_EMBED_BASE_DELAY` | 1.0 | embedder | Base backoff delay (seconds) |
| `OGMEM_EXTRACT_MAX_WORKERS` | 4 | `extraction/span_extractor` | Phase-2 span dual-run parallelism |
| `OGMEM_EXTRACT_SPAN_TIMEOUT` | 180 | span_extractor | Per-span dual-run budget (seconds); stragglers dropped |
| `OGMEM_EXTRACT_SPAN_RETRIES` | 1 | span_extractor | Extra attempts per failed span before dropping |
| `OGMEM_LLM_TIMEOUT` | 120 | `providers/llm/openai_llm` | OpenAI client read/connect timeout (seconds) |
| `OGMEM_ASR_FAIL_RATIO` | 0.15 | `m3/voice_asr`, `m3/pipeline/stages` | Tolerated per-clip ASR failure ratio before failing the video |
| `OGMEM_TENANT_SCOPE` | m3bench | compose (ogmem service) | Multi-tenant guard for background workers |
| `ORTSESSION_INTRA_OP_THREADS` | 8 | `scripts/cv_service` | Cap onnxruntime intra-op threads (avoids 192-thread thrash) |
| `OMP_NUM_THREADS` | 8 | compose (runner) | torch intra-op threads for pyannote (23× faster than default) |

**Acceptance (Workstream 3):** parallel ingest stress (the LoCoMo small
dataset or a 100-video M3 run) shows ~0 embed/extract errors in the logs;
no "deadlock detected" at cold start; no `AccessExclusiveLock` deadlock
mid-run; a saturated embedder degrades to BM25 (answers still come back,
not "cannot determine" for everything); `OGMEM_LLM_TIMEOUT=600` lets a
96-concurrent-call burst complete without "Request timed out".

---

## Workstream 4 — Span-structuring resilience (MODIFY `extraction/span_extractor.py`)

This is the conversation-memory extractor (not the M3 video pipeline), but
it's part of the same parallel-reliability bundle and must be replicated.

**Problem.** Under LLM saturation, per-span extraction calls time out. The
old code: Phase-1 failure raised (aborting the whole chunk); Phase-2 used
`ThreadPoolExecutor(max_workers=4)` with `as_completed`, and one failing
span re-raised in `_structure_spans`, discarding the entire chunk's
candidates (observed: 270 "Extractor failed" = entire chunk extractions
lost).

**Fix.**
- Module-level defaults from env: `_DEFAULT_SPAN_WORKERS` (`max(1,
  OGMEM_EXTRACT_MAX_WORKERS or 4)`, `_DEFAULT_SPAN_TIMEOUT` (`max(10.0,
  OGMEM_EXTRACT_SPAN_TIMEOUT or 180.0)`), `_DEFAULT_SPAN_RETRIES` (`max(0,
  OGMEM_EXTRACT_SPAN_RETRIES or 1)`).
- `Extractor.__init__` accepts `max_workers`, `span_timeout_seconds`,
  `span_retries` (None → env defaults); store as `self._max_workers`,
  `self._span_timeout_seconds`, `self._span_retries`.
- **Phase 1 (`_identify_spans`)**: on `Exception`, do NOT raise — log a
  warning and `return []` so `extract()` falls through to its
  full-conversation fallback span. (Degrade instead of abort.)
- **Phase 2 (`_structure_spans`)**: replace the `as_completed` loop with:
  - Single span → call `_run_span_dual` directly (no pool; retry logic
    inside still applies).
  - Multiple spans → `ThreadPoolExecutor(max_workers=self._max_workers)`;
    submit **one** `_run_span_dual` future per prepared span (not two
    futures per span — the dual run is inside `_run_span_dual`). Use
    `wait(pending, timeout=self._span_timeout_seconds,
    return_when=FIRST_COMPLETED)` in a loop: collect `done` results,
    drop stragglers when nothing completes within the budget
    (`pool.shutdown(wait=False, cancel_futures=True)`). Failed/timed-out
    spans go into a `failed` list; successful spans keep their candidates
    (partial extraction). Log a summary warning with succeeded/failed
    counts.
- **`_run_span_dual(prepared, ctx, archive_id, enable_line_lineage) ->
  ExtractionResult | None`** (NEW): the dual run (default pass + temp=0
  pass) with bounded retry. `attempts = 1 + self._span_retries`. On pass-1
  exception: if attempts remain, `sleep(min(2**attempt, 8.0))` and retry;
  else log error + `return None` (caller drops the span, keeps others).
  On pass-2 exception: log a warning and **keep pass-1 output** (a partial
  dual-run is better than losing the span). Return
  `ExtractionResult(r1.candidates + r2.candidates, r1.relations +
  r2.relations)`.

**Acceptance:** under LLM load, chunks produce *partial* extractions
(some spans' candidates retained) instead of total loss; no "Extractor
failed" for the whole chunk. New test:
`tests/unit/fs/test_schema_shortcircuit.py` (schema) + extend existing
extractor tests for the span-timeout/drop path.

---

## Workstream 5 — Stage-bounded account wipe (NEW `m3/wipe_stages.py`)

**Goal:** rebuild only the LLM side of the M3 graph without redoing the
deterministic signal-extraction stages (frames, faces, audio, ASR,
diarization). The constructive counterpart of "use a fresh account":
wipe stages at/after a boundary in the same namespace, keep everything
before it.

**Design.**
- `ORDERED_STAGES` — the dependency order (NOT YAML order): `extract_audio
  → extract_frames → detect_faces → face_clustering → voice_segmentation
  → voice_asr → speaker_diarization → speaker_embedding → episodic_memory
  → memory.from_episodic → index_episodic →
  semantic_memory_consolidator → qa → eval`. Wiping stage X wipes X and
  everything after it.
- `STAGE_SPECS: dict[str, StageWipe]` — what each stage owns:
  `extractor_types` (e.g. `("audio_extractor",)`), `raw_types` (e.g.
  `("audio",)`), `memory_nodes` (bool), `drop_same_identity_edges`,
  `clear_episodic_embeddings`, `vector_index`, `qa_files`. Map every M3
  stage to the `graph_nodes.extractor_type` / `raw_type` / `node_type` it
  produces (verify against the live store — the spec doc lists the
  verified mapping).
- `seed_where(stages)` — builds the SQL predicate selecting the *direct*
  seeds of the wipe (`extractor_type = ANY(%(e)s) OR raw_type = ANY(%(r)s)
  OR node_type = 'memory'`).
- `plan_sql` / `delete_sql` — a recursive CTE over `graph_nodes`:
  `WITH RECURSIVE doomed(node_id) AS (SELECT ... WHERE account_id=%(a)s
  AND (<seed>) UNION SELECT c.node_id FROM graph_nodes c JOIN doomed d ON
  c.source_raw_id = d.node_id WHERE c.account_id=%(a)s)`. The recursive
  closure follows `source_raw_id` (extracted → raw parent); `graph_edges`
  `ON DELETE CASCADE` removes every edge touching a deleted node
  (extracted_from/references/supports/evolves/same_identity).
- `post_ops(stages)` — after the node delete: drop `same_identity` edges
  only when faces SURVIVE (boundary == face_clustering); clear episodic
  node embeddings only when episodic nodes SURVIVE (boundary ==
  index_episodic); drop vector rows (needs the vector DSN).
- `find_qa_files(root)` — rglob `m3_qa_results.jsonl`,
  `all_m3_qa_results.jsonl`, `run_report.json`, `m3_eval_summary.json`.
- `run(conn, *, account, boundary, vector_conn=None, clear_qa_dir=None,
  execute=False)` — plan (dry-run) or delete; `account_id` is mandatory;
  dry-run is the default; every write limited to one account.
- CLI: `--account` (required), `--wipe-from STAGE` | `--keep-until STAGE`
  (mutually exclusive, one required), `--list-stages`, `--yes` (execute),
  `--connection-string` (env `OGMEM_GRAPH_CONNECTION_STRING`),
  `--wipe-vector` + `--vector-connection-string` (env
  `OGMEM_VECTOR_CONNECTION_STRING`), `--clear-qa-dir DIR`.

**Common boundaries:** `--wipe-from m3.episodic_memory` (re-prompt the
LLM-side graph, reuse all signal stages), `m3.speaker_diarization` (new
voice model), `m3.face_clustering` (only same_identity edges),
`m3.extract_frames` (full media-to-graph rebuild).

**Acceptance:** dry-run prints the doomed-node counts by kind; `--yes`
deletes exactly those rows for the one account; smart-resume probes reuse
the kept stages on the next run. New test: `tests/unit/m3/test_wipe_stages.py`
(stage order, boundary math, SQL builders, post-ops — DB-free).

---

## Workstream 6 — Docs, config sync, tests

### 6.1 `AGENTS.md` (MODIFY — the mind-map index)

Document, concisely (index-like, not explanatory): the Docker-first quick
start, the compose profiles, the directory map, the M3-Bench run commands
(direct `docker run` and the compose `with-m3runner` path), the
speaker-diarization offline-enablement notes (adapted to upstream — drop
the Huawei/refs-main specifics, keep the `OMP_NUM_THREADS=8` win and the
account-scoping bug fix), the parallel-reliability knobs table (3.8), the
LoCoMo verification overview, the clean-DB snippets (Postgres + openGauss
vector wipe), and the stage-bounded wipe usage.

### 6.2 Config sync

- `deploy/ogmemory.example.yaml` — sync the embedding block to whatever
  embedder the target uses (the source switched to a local
  Qwen3-Embedding-0.6B on :8002, 1024-dim). The dimension must match
  `vector_db.dimension` and stay ≤ 2000 (openGauss HNSW cap).
- `config/openclaw/ogmem.template.json` — minor wiring sync (see diff).

### 6.3 Tests to add/extend

- `tests/unit/fs/test_schema_shortcircuit.py` (NEW) — `ensure_schema`
  skips DDL when `graph_nodes` exists; runs DDL when absent; non-postgres
  backends skip the advisory lock. Uses a `_FakeCursor` capturing executed
  SQL.
- `tests/unit/m3/test_wipe_stages.py` (NEW) — stage order is dependency
  order; `stages_from`/`resolve_boundary` math; `seed_where` predicate
  params; `plan_sql`/`delete_sql` text; `post_ops` edge/embedding/vector
  logic; `find_qa_files` rglob. DB-free.
- `tests/unit/m3/test_bench_video_parallel.py` (NEW) — `--video-workers`
  threads, live per-video judge, incremental merge. Uses a fake
  `run_one_video` writing a per-video JSONL.
- `tests/unit/m3/test_eval_harness.py` (MODIFY) — parallel question
  judging via `parallel_clips`; failure classification unchanged.
- `tests/unit/m3/test_qa_runtime_wiring.py` (MODIFY) — `qa_params_argv`
  maps `qa.params.transcript_top_k=20` → `["--transcript-top-k","20"]`;
  QA-only resume honors `--set qa.params.*`.
- `tests/unit/m3/test_resume.py` (MODIFY) — graph pool sizing via
  `OGMEM_GRAPH_POOL_SIZE`.

### 6.4 QA-only resume retrieval knobs (MODIFY `m3/resume.py` + `m3/run_video_pipeline.py`)

`run_qa_only` gains a `qa_params: list[str] | None` param appended **last**
to the `m3.qa` argv (argparse last-occurrence wins, so explicit
`--set qa.params.*` override driver defaults). `qa_params_argv(set_values)`
maps `qa.params.<name>=<value>` → `["--"+name.replace("_","-"), value]`,
ignoring non-`qa.params.*` entries. `run_video_pipeline.run_pipeline`
calls `qa_params_argv(getattr(args,"set",None) or [])` and passes it
through. This fixes graph-aware QA-only resume ignoring retrieval knobs.

---

## Replication order (dependency-aware)

1. **Workstream 3.1–3.2 first** (pool + schema) — everything parallel
   depends on a bounded pool and non-deadlocking schema. Land these
   before any parallel driver.
2. **Workstream 2.1** (`m3/_parallel.py`) — the primitive every driver
   imports.
3. **Workstream 2.2–2.5** (parallelize drivers + video-parallel bench)
   + **3.6** (pool-size knobs) together — the knobs must exist when the
   drivers fan out.
4. **Workstream 3.3–3.5** (embedder retry, retrieval degrade, LLM
   timeout) — can land in parallel with 2; they're independent defenses.
5. **Workstream 4** (span extractor) — independent of M3; land anytime.
6. **Workstream 3.7** (vectorized face clustering) — independent perf
   fix; land anytime before the full-corpus run.
7. **Workstream 1** (Docker) — land last; it packages everything. The
   Dockerfile must include the deps added in 2–5 (rank-bm25, nltk,
   faster-whisper, insightface, pyannote/speechbrain if diarization is
   enabled).
8. **Workstream 5** (wipe_stages) — land anytime after the stage
   vocabulary is stable.
9. **Workstream 6** (docs, config sync, tests) — continuously, alongside
   each workstream.

---

## What to explicitly NOT replicate

- Huawei apt/pypi mirrors (Global constraint 1).
- The `docker/py_wheels` offline wheel install + `huggingface_hub==0.26.3`
  pin + `refs/main` no-newline workaround (Global constraint 2). Use the
  public index and current `huggingface_hub`.
- `sitecustomize.py`'s `torch.load` patch — only if the target also pins
  torch≥2.6 with vendored checkpoints; otherwise omit.
- `deploy/docker-compose.huawei.yml` (untracked, Huawei-specific).
- The `ant-trail-multi-modal-memory.txt` design doc (internal, not code).
- Any `swr.cn-north-4.myhuaweicloud.com/...` image default.

---

## Verification checklist (run after each workstream)

- [ ] `docker compose --profile with-db --profile with-m3runner up -d` —
      all services healthy.
- [ ] `curl http://127.0.0.1:${CV_PORT}/health` → insightface backend.
- [ ] `docker compose exec runner python3 -c "import m3, faster_whisper,
      insightface, rank_bm25, nltk; print('ok')"`.
- [ ] Smoke: one video through `m3.run_video_pipeline` (episodic + QA)
      completes; JSONL non-empty.
- [ ] `OGMEM_M3_MAX_WORKERS=4` multi-clip run: output order identical to
      serial; a single injected clip failure is skipped, not fatal.
- [ ] Parallel ingest stress: ~0 embed/extract errors in logs; no
      "deadlock detected"; no `AccessExclusiveLock` mid-run.
- [ ] Saturated embedder → retrieval degrades to BM25 (answers still
      returned, not all "cannot determine").
- [ ] `python -m m3.wipe_stages --account X --wipe-from
      m3.episodic_memory` dry-run prints counts; `--yes` deletes only that
      account's post-boundary nodes; next run reuses kept stages.
- [ ] All new unit tests pass: `test_schema_shortcircuit`,
      `test_wipe_stages`, `test_bench_video_parallel`, updated
      `test_eval_harness`/`test_qa_runtime_wiring`/`test_resume`.
- [ ] `cd deploy && bash check.sh` (if the target has an equivalent) —
      containers, endpoints, DB connectivity, port listening, no error
      log lines.

