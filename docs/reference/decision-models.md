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

The client makes one request per invocation; the caller owns retries/recovery. Configure the final service URL: automatic HTTP redirects (including same-origin redirects) fail as `DecisionError` without a follow-up request or forwarding authentication. `timeout_seconds` is the configured per-request I/O timeout. Inside an existing operation deadline, the network timeout is capped by the remaining time; expiry is reported as a deadline error rather than a provider error. This does not create a new overall pipeline budget. `DecisionBatch` returns elapsed time and the provider response (including any usage metadata) for caller-owned accounting. Transport errors expose safe error classes and HTTP status; the client does not duplicate caller logging or usage accounting.
