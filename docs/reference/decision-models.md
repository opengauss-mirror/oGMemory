# Jev-compatible decision model

`decision` is independent of the generative `llm`. Omit it or set it to `null` to retain native behavior. Setting `OGMEM_DECISION_API_KEY` alone does not enable it. A configured block requires a non-empty model and HTTP(S) base URL; empty or unknown configuration is rejected.

```yaml
decision:
  provider: jev
  base_url: https://ai-gateway.vercel.sh/typesafe
  model: typesafe-ai/jev
  # Omit api_key to read OGMEM_DECISION_API_KEY from the environment.
  timeout_seconds: 30
```

`provider: jev` selects the protocol, not a particular hosted model. TypeSafe direct access can use `https://api.typesafe.ai` with its model name. A self-hosted Laya service can use its HTTP base URL and `model: multilingual`. Omit `api_key` to read `OGMEM_DECISION_API_KEY` from the environment. An explicitly configured key takes precedence; an explicit empty string disables authentication. YAML `${...}` placeholders are not expanded by this loader. A non-empty key is sent as a Bearer token and is never inherited from `llm`. Keep keys in environment variables rather than committed YAML. `/v1/systemone`, a `/v1` base, and a host/protocol base are accepted; the endpoint suffix is appended once.

Only `model`, `state`, and `questions` are sent. There are no mandatory Laya `max_len`/`head_max_len` fields or Vercel-only response checks. Both Noul relevance probabilities and Choice action distributions are validated. Missing, invalid, non-finite and explicitly truncated answers fail. Extra response metadata is tolerated, including Laya's `laya-rl-agent` wrapper model and `routing.model`.

The client makes one request per invocation; the caller owns retries/recovery. Configure the final service URL: automatic HTTP redirects (including same-origin redirects) fail as `DecisionError` without a follow-up request or forwarding authentication. `timeout_seconds` is the configured per-request I/O timeout. Inside an existing operation deadline, the network timeout is capped by the remaining time; expiry is reported as a deadline error rather than a provider error. This does not create a new overall pipeline budget. `DecisionBatch` returns elapsed time and the provider response (including any usage metadata) for diagnostics. Transport errors expose safe error classes and HTTP status; the client records one Decision usage event per actual request. Service binding persists that event under the current account/owner, separately from LLM and embedding usage.

## Optional retrieval reranking

Configuring `decision` alone does not enable retrieval reranking. Enable it explicitly:

```yaml
retrieval:
  decision_rerank_enabled: true
```

The switch defaults to `false` and requires a valid `decision` block when enabled. YAML takes precedence over `OGMEM_RETRIEVAL_DECISION_RERANK_ENABLED`. It applies independently of eager/lazy extraction mode. Only ACL-hydrated ordinary-memory candidates are reordered; native scores, code positions, candidate admission and injection budgets remain unchanged. Provider failures preserve native order; operation deadline errors and usage persistence failures propagate. Large inputs can still exceed the provider context limit; failures expose safe HTTP status in `decision_rerank` statistics.

Decision usage reads non-negative integer `usage.input_tokens` and `usage.output_tokens` from the provider response. Their sum contributes to the known-token total, independently of generative LLM usage. Missing or invalid fields are marked as missing usage; available valid fields are retained without estimation. Provider/protocol failures record a failed call, and requests skipped by the switch or candidate count record no call. Responses without usage are not evidence of zero consumption. Persistent and in-memory model metrics expose a separate `decision` category, with calls, failures and missing-usage counts. No additional cache tokens are inferred or added.

## Decision-driven lazy extraction

With `memory.extraction.mode: lazy`, a valid `decision` block selects the JEV controller. An absent or null `decision` block selects the native LLM controller.

```yaml
memory:
  extraction:
    mode: lazy
    read_probability_threshold: 0.70
decision:
  provider: jev
  base_url: https://decision.example.internal
  model: multilingual
  timeout_seconds: 30
```

JEV chooses whether to read old memory, form memory operations, or finish. It scores authorized unread candidates independently in one request. Lazy extraction reads all candidates at or above `read_probability_threshold` in one tool round. The threshold accepts values in (0, 1] and defaults to 0.70. Calibrate it on representative extraction samples.

For memory formation, the LLM uses the preselected categories, source, read snapshots, and recorded progress to generate a batch of native operations. Native execution validates targets and versions and records feedback for JEV. When the loop requires final JSON, native safety checks run first. JEV then checks whether the recorded and tentative operations cover the complete source. An incomplete batch enters native full-source recovery.

Model transport failures and usage persistence errors propagate to the caller. Output and coverage failures use native recovery. Each physical model request uses the existing usage ledger. Cost comparisons include Decision calls and recovery calls.
