"""Decision accounting using the shared ledger and request ownership scope."""

from providers.token_tracker import TokenTracker, UsageTracker


class DecisionTokenTracker(TokenTracker):
    def __init__(self, model):
        super().__init__(model=model)
        self._status = UsageTracker()

    def record_response(self, raw, *, failed=False):
        usage = raw.get("usage") if isinstance(raw, dict) else None
        usage = usage if isinstance(usage, dict) else {}
        values = [usage.get(key) for key in ("input_tokens", "output_tokens")]
        valid = [type(value) is int and value >= 0 for value in values]
        inp, out = [value if known else 0 for value, known in zip(values, valid)]
        missing = int(not all(valid))
        self._persist(
            "decision",
            {
                "input_tokens": inp,
                "output_tokens": out,
                "llm_calls": 1,
                "failed_calls": int(failed),
                "missing_usage_calls": missing,
            },
        )
        # Reuse scoped counters, without adding Decision tokens to LLM perf buckets.
        UsageTracker.record_llm(self, inp, out)
        self._status.record_llm(int(failed), missing)

    def decision_status(self, account_id="", owner_spaces=()):
        snapshot = (
            self._status.scoped_snapshot(account_id, owner_spaces)
            if account_id
            else self._status.snapshot()
        )
        return {
            "failed_calls": snapshot.input_tokens,
            "missing_usage_calls": snapshot.output_tokens,
        }
