# Parallel LoCoMo Runbook (rychkov stack)

Everything required to run LoCoMo (small or locomo10) **truly in parallel**
(32-way ingest + QA) on the shared-host rychkov compose stack — every knob,
where it lives, which phase needs it, and why. This is the committed,
canonical record; `AGENTS.md` (untracked workspace notes) points here.

**Enforcement:** run the preflight script before every phase — it checks the
config files AND the live containers, so stale-env restarts and silently
dropped gateway edits get caught instead of discovered mid-run:

```bash
cd /home/rychkov/projects/oGMemory
python3 scripts/check_locomo_parallel.py --phase ingest   # before ingest
python3 scripts/check_locomo_parallel.py --phase qa \
    --config /home/rychkov/projects/locomo-test/configs/rychkov-full-parallel.toml
```

Exit code 1 = fix the failures first. Requires docker group access for the
live checks (without it they degrade to warnings).

Reference result: full locomo10, 32 workers — **65.91% (1015/1540)**,
ingest 3h57m, QA ~2h15m equivalent, vs ~2–3 days serial ingest.

---

## 1. The knob table

| Knob | Value | Lives in | Phase | Why |
|---|---|---|---|---|
| `OGMEM_AUTO_CAPTURE` | `true` / `false` | `deploy/.env` → openclaw container env | **ingest / qa** (differs!) | ingest needs after_turn extraction or it stalls silently; QA must have it off — eager re-capture of 60–120k-token recall injections blocks every turn in 503-retry loops |
| `OGMEM_AUTO_RECALL` | `true` | `deploy/.env` → openclaw container env | both | otherwise no memory injection during compose |
| `OGMEM_GUNICORN_TIMEOUT` | `3600` | `deploy/.env` → ogmem container env | both | default 900s kills long composes under 32-way load mid-request |
| `service.threads` | `32` | `deploy/ogmemory.yaml` | both | gunicorn thread pool must cover the parallel lanes (default starves them) |
| `llm.max_tokens` | `32768` | `deploy/ogmemory.yaml` | both | deepseek-v4.1-flash is a reasoning model: ~7k tokens of reasoning per extraction span; 8192 → `finish_reason=length` → extraction fails hard (LLMCallError) |
| `agents.defaults.maxConcurrent` | `32` | **live `openclaw.json` only** — no env knob | both | openclaw default 4 starves the lanes |
| `agents.defaults.model.primary` | `openai/<LLM_MODEL>` | live `openclaw.json` (env-generated) | both | must match `.env` `LLM_MODEL` |
| `OGMEM_ACCOUNT_ID` / `OGMEM_USER_ID` / `OGMEM_AGENT_ID` | set, matching locomo-test config | `deploy/.env` | both | the gateway does NOT forward the `/v1/responses` `user` field to the context-engine plugin — identity must be pinned in env |
| `embedding.base_url` | `http://127.0.0.1:18000/v1` (note `/v1`) | `deploy/ogmemory.yaml` | both | openai SDK appends `/embeddings`; a bare host gives Flask 404s |
| embedder instance | **dedicated** `rychkov-embedding` on `:18000` | docker (outside compose) | both | the shared `:8000` service serializes all requests on one CPU process (8 concurrent 2 KB embeds → 10–16 s each) → `APITimeoutError` retry storms → 30–130 s compose hangs; dedicated does 8 in ~2.5 s |
| `vector_db.dimension` | `1024` | `deploy/ogmemory.yaml` | both | must equal embedder output dims (Qwen3-Embedding-0.6B → 1024) |
| `general.parallel` | `32` | locomo-test run config `.toml` | both | worker count in the runner |
| no-tool QA instructions | (code) | locomo-test `82ed75d` | qa | MaaS deepseek tool-loops (sqlite3/python/tesseract "investigations") bloat sessions to 150+ messages → context-overflow compaction loops → gateway 500s and garbage answers |

### Gotchas baked into the table

- **Compose env changes need RECREATE, not restart**: "a restart does not
  update container environment" (see the compose comment on
  `OGMEM_AUTO_CAPTURE`). After editing `deploy/.env`, run
  `docker compose up -d` so the containers are recreated.
- **`maxConcurrent` has no env knob.** The openclaw entrypoint regenerates
  `openclaw.json` from env on every start; manual `agents.defaults.*` edits
  are supposed to be preserved by the migrate-in-place logic, but have been
  observed lost after restarts/resets (found 2026-10-20: the live config had
  no `maxConcurrent` at all). Re-apply after any gateway reset and re-run the
  preflight — it reads the **live** file via `docker exec`.
- The host-side `openclaw.json` (`deploy/_rychkov/openclaw-home/`) is owned by
  uid 1000 with mode 600 — edit it via `docker exec`, not from the host.

## 2. Phase procedure

```bash
cd /home/rychkov/projects/oGMemory/deploy

# 0. (fresh run only) full reset — wipes DB + gateway sessions
sg docker -c "docker compose -p ogmem-rychkov --profile with-pg --profile with-openclaw down -v"
sg docker -c "docker run --rm -v $PWD/_rychkov/openclaw-home:/oc alpine sh -c 'rm -rf /oc/*'"

# 1. INGEST phase: auto-capture ON
#    edit deploy/.env -> OGMEM_AUTO_CAPTURE=true
sg docker -c "docker compose -p ogmem-rychkov --profile with-pg --profile with-openclaw up -d"
# re-apply gateway maxConcurrent (see gotcha above), then:
python3 ../scripts/check_locomo_parallel.py --phase ingest

# 2. QA phase: auto-capture OFF — flip .env, RECREATE containers
#    edit deploy/.env -> OGMEM_AUTO_CAPTURE=false
sg docker -c "docker compose -p ogmem-rychkov --profile with-pg --profile with-openclaw up -d"
python3 ../scripts/check_locomo_parallel.py --phase qa --config <locomo-test toml>
```

The runner itself must run as uid 1000 with docker-socket access (the
openclaw `agents/` dir is 700/node-owned) — use the `locomo-runner:rychkov`
container recipe in `AGENTS.md` (Stage 1).

## 3. Failure taxonomy (symptom → fix)

All of these were hit and fixed during the 32-way ingest; kept for pattern
recognition. Fixes 1–3 are commits on `rychkov/m3-parallel` — they are
**required**, not optional:

| Symptom | Root cause | Fix |
|---|---|---|
| Perpetual CAS version-race losses on hot entity nodes (a person updated by many sessions) | seconds-long LLM semantic merge sits between plan and CAS write | `09688fce` per-URI write serialization in `commit/context_writer.py` |
| Index outbox events DLQ'd on hot nodes → account-wide fail-closed index barrier → QA gate refuses | 5 s per-URI advisory-lock timeout | `75377947` timeout 5 s → 300 s in `commit/sql_outbox_store.py` |
| Optimistic-lock retry exhaustion under write stampede | retry budget 3 too shallow | `7f4a3fdc` budget 3 → 8, exponential backoff |
| Embedding request timeouts under load | shared `:8000` embedder serializes everything | dedicated `rychkov-embedding` on `:18000` |
| Gateway read timeouts on long composes | gunicorn 900 s default | `OGMEM_GUNICORN_TIMEOUT=3600` |
| QA lanes starved (4 concurrent) | openclaw `maxConcurrent` default | `agents.defaults.maxConcurrent: 32` in live `openclaw.json` |
| QA sessions bloat → `SessionWriteLockTimeoutError` + compaction loops | MaaS deepseek tool-loops on QA questions | no-tool instructions on QA sends (locomo-test `82ed75d`) |
| Ingest stalls, nothing extracted | `OGMEM_AUTO_CAPTURE=false` during ingest, or session buffer below `memory.after_turn_threshold` (200 tokens) | flip to true, recreate containers |
| Extraction fails `finish_reason=length` | `llm.max_tokens` too small for reasoning model | 32768 |
| Empty BM25 results → QA degrades to "cannot determine" | NLTK `punkt_tab`/`stopwords` lazily downloaded, fails offline | baked into `ogmemory-m3-runner:rychkov` (m3 path; locomo path unaffected) |

## 4. Mid-run ops

- **Resume**: the runner skips executed records; delete bad rows (timeout
  answers, tool-call transcripts land in `qa_results.csv` and grade wrong)
  and re-run with `--resume`. Failed ingest chunks reconcile via checkpoints
  in `output/<run>/.ogmem_chunks/` (status=extracted with `archive_ids`
  means writes committed — re-verify, never re-send).
- **LLM quota death mid-run** (ark resets weekly Mondays; MaaS is the
  current endpoint): switch `LLM_BASE_URL`/`LLM_API_KEY` in `deploy/.env` +
  gateway `openclaw.json` + locomo-test `configs/env.toml [judge]`, restart
  ogmem+gateway **preserving the postgres volume** (no `down -v`), `--resume`.
- **Gateway session hygiene before/during QA**: purge `:qa-` mappings from
  `sessions.json` + their `.jsonl` files (gateway must be stopped; sessions
  with unextracted messages can't be evicted via API). Interrupted runs leave
  bloated jsonl files that hit write-lock timeouts on resume.
- **Contamination cleanup** (QA-era junk before a final QA pass — all
  account-scoped): `DELETE FROM session_archives WHERE created_at > <qa-start>`;
  delete `context_nodes` in `session_archive`/`state` categories created after
  qa-start; delete orphaned `vector_index` rows; delete stuck `outbox_events`
  (`UPSERT_CONTEXT` events with 100–300 KB payloads of deleted nodes retried
  forever — they hammer the embedder).

## 5. Where things live

| Artifact | Path | Tracked? |
|---|---|---|
| env knobs | `deploy/.env` | gitignored — see `deploy/.env.parallel-locomo.example` for the committed template |
| ogmem config | `deploy/ogmemory.yaml` | gitignored — `deploy/ogmemory.example.yaml` is the committed base |
| gateway live config | `deploy/_rychkov/openclaw-home/openclaw.json` (uid-1000, mode 600) | runtime state, untracked |
| runner configs | `locomo-test/configs/rychkov-*.toml` | committed (locomo-test repo) |
| runner env | `locomo-test/configs/env.toml` | committed (locomo-test repo) |
| preflight | `scripts/check_locomo_parallel.py` (this repo) | committed |
| full spec | `specs/parallel_dockerize.md` (this repo) | committed |

## 6. History

- 2026-10-14: LoCoMo small, serial — 91.4% (32/35).
- 2026-10-20: full locomo10, 32-way parallel — 65.91% (1015/1540);
  cat1 55.67%, cat2 66.67%, cat3 57.29%, cat4 70.04%. Ingest 3h57m
  (~15.3M LLM tokens), QA ~2h15m equivalent (40.7M tokens), judge 14 min.
- 2026-10-20: this runbook + preflight script created after discovering the
  gateway `maxConcurrent: 32` edit had been silently lost to a restart.
