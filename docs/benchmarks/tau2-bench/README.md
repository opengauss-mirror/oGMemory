# tau2-bench: oGMemory + OpenClaw CE

Agent experience memory raises weighted task success from **70.00% to 77.92% (+7.92 percentage points)** in this paired evaluation.

## Results

| Domain | Without memory | With oGMemory + OpenClaw CE | Gain |
| --- | ---: | ---: | ---: |
| Retail | 232/320 (72.50%) | 261/320 (81.56%) | +9.06pp |
| Airline | 104/160 (65.00%) | 113/160 (70.63%) | +5.63pp |
| Weighted overall | 336/480 (70.00%) | 374/480 (77.92%) | +7.92pp |

All 960 scored rollouts completed. The score audit checked the full task/seed grid, uniqueness, and each summary reward against its raw task result. Rates are pooled success fractions over eight repeats; the overall score weights Retail and Airline by their rollout counts. Displayed rates and gains are rounded independently from the exact fractions.

Machine-readable [results and audit summary](results.json) and [runtime manifest](manifest.json) accompany this report. The original local evidence bundle is `tau2-ce-lite-20261028`, validated on 2026-10-30; it contains the raw trajectories, summaries, score audit and provider logs.

## Protocol

| Setting | Value |
| --- | --- |
| Agent and user-simulator model | `doubao-seed-2.1-lite` |
| Thinking / temperature | Disabled / 0 |
| Test tasks | Retail 40; Airline 20 |
| Repeats | Eight per arm; seeds 300–307 |
| Maximum steps | 200 |
| Baseline | OpenClaw without the CE memory plugin |
| Memory treatment | OpenClaw with native oGMemory CE prefetch and compose |
| Extraction | Single-pass eager; output limit 16,384 tokens |
| Enabled memory types | Agent-owned case, pattern, skill and tool |
| Compose budget | 32,768 tokens, with previously injected memory deduplicated from history |
| Training | One rollout per training task; successful trajectories only |
| Training memory indexed | Retail 53/53 trajectories; Airline 17/17; idle and indexing drain verified |
| Test memory | Read-only; `OGMEM_AFTER_TURN_ENABLED=false` |
| First user message | Shared cached fixture per scenario; subsequent simulator turns live |
| Execution concurrency | Baseline 3; CE 1; ingest 1 |

Training and test splits are disjoint. Training transcripts include the domain policy and role/tool messages. External tools and rewards are handled by the tau2 orchestrator. The two arms share the model, task/seed grid and first-user fixtures. Experience is extracted from the training split before testing.

The provider usage audit covers logged gateway agent and user-simulator calls across the four scored cells, including interrupted attempts. All 18,354 calls with an explicit reasoning counter reported zero reasoning tokens. Native internal extraction was configured with thinking disabled; the counter audit above covers gateway calls.

## Versions and scope

- Documentation branch base: remote `opengauss/oGMemory` dev, `a21092314a7567dda9e7eac7717e57530fea5a55`.
- Evaluated native memory runtime: `3b8c5ee7acfb37bc8737311225b90047bdeb6e11`; selected runtime file hashes and the CE plugin hash are recorded in the manifest.
- tau2 revision: `79dbf0c18ac7637aedf869cb3122babcd57aaf17`, using the PR #297 confirmation-aware user simulator.
- OpenViking protocol reference: `b7aa01d227c249407971c6693f73ef71c436f439`, fixed-first-user, eight-repeat setup.

These are historical measurements of the recorded runtime and benchmark adapter. The documentation branch is based on current dev; the experiment was performed on the runtime above. The initial default CE treatment encountered context growth and compaction failure; final scores cover the bounded-budget, deduplicated treatment. Pilot and interrupted diagnostic runs are retained in the local evidence bundle and excluded from the final 960 scored rollouts.

The protocol follows the same evaluation family as [OpenViking's benchmark](https://github.com/volcengine/OpenViking#proof-it-works). Models and memory implementations differ, so published cross-project scores are reference points rather than a controlled head-to-head comparison.
