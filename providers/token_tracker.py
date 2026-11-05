"""Thread-safe token usage tracker for LLM and Embedding providers.

Providers SHOULD feed the raw ``response.usage`` block returned by their
SDK directly into ``record_llm`` / ``record_embed`` from provider implementations.

Per-context buckets
-------------------
Beyond the process-level cumulative counters, this module maintains a
``ContextVar``-based **stack** of :class:`TokenBucket` objects. Every
``record_*`` call simultaneously updates the global tracker AND every
bucket on the active context's stack. Each thread / asyncio task / perf
span can push its own bucket and read isolated, accurate per-context
totals — eliminating the cross-thread pollution that snapshot-diff has
under concurrency.

Use :func:`push_bucket` to start a new attribution scope and
:func:`pop_bucket` (with the returned token) to end it. Buckets nest
naturally: when nested, every ``record_*`` is added to the inner bucket
*and* all enclosing buckets, so parent spans still see the full total.
"""

from __future__ import annotations

import contextvars
import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any
from uuid import uuid4


@dataclass
class TokenUsageSnapshot:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0
    llm_calls: int = 0
    embed_tokens: int = 0
    embed_calls: int = 0
    tool_stats: dict = field(default_factory=dict)
    local_cache_hits: int = 0
    local_cache_misses: int = 0
    local_cache_saved_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.embed_tokens

    def to_dict(self) -> dict:
        data = {
            "llm": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "cache_read": self.cache_read,
                "cache_write": self.cache_write,
                "total_tokens": self.input_tokens + self.output_tokens,
                "calls": self.llm_calls,
            },
            "embedding": {
                "total_tokens": self.embed_tokens,
                "calls": self.embed_calls,
            },
            "total_tokens": self.total_tokens,
        }
        if self.tool_stats:
            data["tools"] = self.tool_stats
        return data


# ---------------------------------------------------------------------------
# Tool/session tracking helpers
# ---------------------------------------------------------------------------


def _empty_tool_counters() -> dict:
    return {
        "call_count": 0,
        "success_count": 0,
        "fail_count": 0,
        "total_duration_ms": 0.0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "llm_call_ids": [],
    }


def _empty_session_counters() -> dict:
    return {
        "llm_calls": 0,
        "tool_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "tool_tokens": 0,
    }


def _empty_daily_counters() -> dict[str, int]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read": 0,
        "cache_write": 0,
        "llm_calls": 0,
        "embed_tokens": 0,
        "embed_calls": 0,
    }


_active_usage_scope: contextvars.ContextVar[tuple[str, str] | None] = contextvars.ContextVar(
    "ogmem_usage_scope", default=None
)

_usage_distribution = contextvars.ContextVar('ogmem_usage_distribution', default=None)
_embedding_inputs = contextvars.ContextVar('ogmem_embedding_inputs', default=None)


@contextmanager
def distributed_usage_scope(weights, *, inputs=None):
    """Attribute one provider request across the owners participating in a batch."""
    token = _usage_distribution.set(weights)
    input_token = _embedding_inputs.set(inputs)
    try:
        yield
    finally:
        _usage_distribution.reset(token)
        _embedding_inputs.reset(input_token)


@contextmanager
def embedding_input_scope(indices):
    inputs = _embedding_inputs.get()
    if inputs is None:
        yield
        return
    selected = [inputs[i] for i in indices]
    with distributed_usage_scope(embedding_scope_weights(selected), inputs=selected):
        yield


def embedding_scope_weights(inputs):
    weights = {}
    for scope, weight in inputs:
        weights[scope] = weights.get(scope, 0) + weight
    return weights


def embedding_usage_tokens(usage):
    for key in ('total_tokens', 'prompt_tokens'):
        value = usage.get(key) if isinstance(usage, dict) else getattr(usage, key, None)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


class UsagePersistenceError(RuntimeError):
    """A completed provider call could not be durably accounted for."""

_DAILY_USAGE_RETENTION_DAYS = 90


@contextmanager
def usage_scope(account_id: str, owner_space: str = ""):
    """Attribute provider usage in this execution context to one tenant owner."""
    token = _active_usage_scope.set((str(account_id or ""), str(owner_space or "")))
    try:
        yield
    finally:
        _active_usage_scope.reset(token)


def request_usage_scope(ctx):
    owner_space = ctx.agent_space_name() if ctx.agent_id else ctx.user_space_name()
    return usage_scope(ctx.account_id, owner_space)


def _copy_tool_stats(stats: dict) -> dict:
    copied: dict = {}
    for group_name, group in stats.items():
        if not isinstance(group, dict):
            continue
        copied[group_name] = {}
        for key, value in group.items():
            if isinstance(value, dict):
                copied[group_name][key] = {
                    inner_key: list(inner_value) if isinstance(inner_value, list) else inner_value
                    for inner_key, inner_value in value.items()
                }
            else:
                copied[group_name][key] = value
    return copied


# ---------------------------------------------------------------------------
# Per-context bucket — used by perf module to attribute tokens to a span
# ---------------------------------------------------------------------------


@dataclass
class TokenBucket:
    """Per-context accumulator written to by every active provider call.

    A bucket is pushed onto :data:`_active_buckets` (a ``ContextVar``
    stack) at the start of a perf span and popped at the end. Because
    the stack is a ``ContextVar``, each thread / asyncio task gets its
    own independent stack — concurrent extractions in background
    threads will each see only their own LLM/embed calls.

    Buckets are mutable but guarded by a per-bucket lock; updates from
    the network thread that issued an LLM request and from a
    surrounding perf span run on the same thread, but we keep the lock
    cheap so the design also supports providers that fan out work to
    background threads.
    """

    llm_input: int = 0
    llm_output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    llm_calls: int = 0
    embed_tokens: int = 0
    embed_calls: int = 0
    llm_model: str | None = None
    embed_model: str | None = None
    local_cache_hits: int = 0
    local_cache_misses: int = 0
    local_cache_saved_tokens: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def add_llm(
        self,
        input_tokens: int,
        output_tokens: int,
        cache_read: int,
        cache_write: int,
        model: str | None,
    ) -> None:
        with self._lock:
            self.llm_input += int(input_tokens or 0)
            self.llm_output += int(output_tokens or 0)
            self.cache_read += int(cache_read or 0)
            self.cache_write += int(cache_write or 0)
            self.llm_calls += 1
            if model and not self.llm_model:
                self.llm_model = model

    def add_embed(self, total_tokens: int, model: str | None) -> None:
        with self._lock:
            self.embed_tokens += int(total_tokens or 0)
            self.embed_calls += 1
            if model and not self.embed_model:
                self.embed_model = model

    def add_local_cache_hit(self, tokens_saved: int) -> None:
        with self._lock:
            self.local_cache_hits += 1
            self.local_cache_saved_tokens += int(tokens_saved or 0)

    def add_local_cache_miss(self) -> None:
        with self._lock:
            self.local_cache_misses += 1

    def to_attribution_dict(self) -> dict[str, Any]:
        """Render in the shape used by perf SpanEvent.tokens."""
        with self._lock:
            return {
                "llm": {
                    "input_tokens": self.llm_input,
                    "output_tokens": self.llm_output,
                    "cache_read": self.cache_read,
                    "cache_write": self.cache_write,
                    "llm_calls": self.llm_calls,
                    "embed_tokens": 0,
                    "embed_calls": 0,
                    "local_cache_hits": self.local_cache_hits,
                    "local_cache_misses": self.local_cache_misses,
                    "local_cache_saved_tokens": self.local_cache_saved_tokens,
                },
                "embed": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_read": 0,
                    "cache_write": 0,
                    "llm_calls": 0,
                    "embed_tokens": self.embed_tokens,
                    "embed_calls": self.embed_calls,
                },
                "llm_model": self.llm_model,
                "embed_model": self.embed_model,
            }


# Module-level stack of active buckets.
_active_buckets: contextvars.ContextVar[tuple[TokenBucket, ...]] = contextvars.ContextVar(
    "ogmem_token_buckets", default=()
)


def push_bucket() -> tuple[TokenBucket, contextvars.Token]:
    """Push a fresh bucket onto the current context's stack."""
    bucket = TokenBucket()
    current = _active_buckets.get()
    token = _active_buckets.set(current + (bucket,))
    return bucket, token


def pop_bucket(token: contextvars.Token) -> None:
    """Restore the bucket stack to its state before the matching ``push_bucket``."""
    _active_buckets.reset(token)


def current_buckets() -> tuple[TokenBucket, ...]:
    """Return the active bucket stack for the current context (may be empty)."""
    return _active_buckets.get()


# ---------------------------------------------------------------------------
# UsageTracker — tracks tool calls and session-level usage
# ---------------------------------------------------------------------------


class UsageTracker:
    """Thread-safe provider and tool usage tracker.

    This is the unified tracker used by both provider-level token accounting
    and session-level tool usage accounting. ``TokenTracker`` remains as a
    backward-compatible name for provider code.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._input = 0
        self._output = 0
        self._cache_read = 0
        self._cache_write = 0
        self._llm_calls = 0
        self._embed_tokens = 0
        self._embed_calls = 0
        self._local_cache_hits = 0
        self._local_cache_misses = 0
        self._local_cache_saved_tokens = 0
        self._tool_by_name: dict[str, dict] = {}
        self._tool_by_category: dict[str, dict] = {}
        self._usage_by_session: dict[str, dict] = {}
        self._daily_usage: dict[str, dict[str, int]] = {}
        self._usage_by_scope: dict[tuple[str, str], dict[str, int]] = {}
        self._daily_usage_by_scope: dict[tuple[str, str], dict[str, dict[str, int]]] = {}
        self._daily_usage_pruned_on: date | None = None

    @property
    def tool_stats(self) -> dict[str, dict]:
        """Per-tool aggregate used by extraction prompts."""
        with self._lock:
            return _copy_tool_stats({"by_tool": self._tool_by_name}).get("by_tool", {})

    def record_llm(
        self,
        input_tokens: int,
        output_tokens: int,
        cache_read: int = 0,
        cache_write: int = 0,
        *,
        session_id: str | None = None,
        call_id: str | None = None,
    ) -> str:
        call_id = call_id or f"llm_{uuid4().hex[:12]}"
        with self._lock:
            today = datetime.now(UTC).date()
            self._prune_daily_usage_unlocked(today)
            self._input += input_tokens
            self._output += output_tokens
            self._cache_read += cache_read
            self._cache_write += cache_write
            self._llm_calls += 1
            daily = self._daily_usage.setdefault(today.isoformat(), _empty_daily_counters())
            daily["input_tokens"] += input_tokens
            daily["output_tokens"] += output_tokens
            daily["cache_read"] += cache_read
            daily["cache_write"] += cache_write
            daily["llm_calls"] += 1
            self._record_scoped_llm(input_tokens, output_tokens, cache_read, cache_write)
            if session_id:
                session_stats = self._usage_by_session.setdefault(
                    session_id, _empty_session_counters()
                )
                session_stats["llm_calls"] += 1
                session_stats["input_tokens"] += input_tokens
                session_stats["output_tokens"] += output_tokens
        return call_id

    def record_embed(self, total_tokens: int) -> None:
        with self._lock:
            today = datetime.now(UTC).date()
            self._prune_daily_usage_unlocked(today)
            self._embed_tokens += total_tokens
            self._embed_calls += 1
            daily = self._daily_usage.setdefault(today.isoformat(), _empty_daily_counters())
            daily["embed_tokens"] += total_tokens
            daily["embed_calls"] += 1
            self._record_scoped_embed(total_tokens)

    def record_local_cache_hit(self, tokens_saved: int) -> None:
        with self._lock:
            self._local_cache_hits += 1
            self._local_cache_saved_tokens += int(tokens_saved or 0)

    def record_local_cache_miss(self) -> None:
        with self._lock:
            self._local_cache_misses += 1

    def record_tool_call(
        self,
        *,
        tool_name: str,
        category: str = "",
        session_id: str = "",
        status: str = "",
        duration_ms: float | None = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        llm_call_id: str | None = None,
    ) -> None:
        if not tool_name:
            return

        success = status in {"completed", "success"}
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
        with self._lock:
            self._add_tool_call(
                self._tool_by_name.setdefault(tool_name, _empty_tool_counters()),
                success=success,
                duration_ms=duration_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                llm_call_id=llm_call_id,
            )
            if category:
                self._add_tool_call(
                    self._tool_by_category.setdefault(category, _empty_tool_counters()),
                    success=success,
                    duration_ms=duration_ms,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    llm_call_id=llm_call_id,
                )
            if session_id:
                session_stats = self._usage_by_session.setdefault(
                    session_id, _empty_session_counters()
                )
                session_stats["tool_calls"] += 1
                session_stats["tool_tokens"] += total_tokens

    def snapshot(self) -> TokenUsageSnapshot:
        with self._lock:
            return self._snapshot_unlocked()

    def snapshot_and_reset(self) -> TokenUsageSnapshot:
        with self._lock:
            snap = self._snapshot_unlocked()
            self._reset_unlocked()
            return snap

    def daily_snapshot(
        self,
        days: int = 14,
        *,
        account_id: str = "",
        owner_spaces: tuple[str, ...] = (),
    ) -> list[dict[str, int | str]]:
        if days < 1 or days > 90:
            raise ValueError("days must be between 1 and 90")
        end = datetime.now(UTC).date()
        dates = [(end - timedelta(days=offset)).isoformat() for offset in range(days - 1, -1, -1)]
        with self._lock:
            if account_id:
                selected = self._matching_scope_keys(account_id, owner_spaces)
                return [
                    {
                        "date": date_key,
                        **self._sum_counters(
                            self._daily_usage_by_scope.get(key, {}).get(date_key)
                            for key in selected
                        ),
                    }
                    for date_key in dates
                ]
            return [
                {"date": date_key, **self._daily_usage.get(date_key, _empty_daily_counters())}
                for date_key in dates
            ]

    def scoped_snapshot(
        self, account_id: str, owner_spaces: tuple[str, ...] = ()
    ) -> TokenUsageSnapshot:
        with self._lock:
            counters = self._sum_counters(
                self._usage_by_scope.get(key)
                for key in self._matching_scope_keys(account_id, owner_spaces)
            )
        return TokenUsageSnapshot(
            input_tokens=counters["input_tokens"],
            output_tokens=counters["output_tokens"],
            cache_read=counters["cache_read"],
            cache_write=counters["cache_write"],
            llm_calls=counters["llm_calls"],
            embed_tokens=counters["embed_tokens"],
            embed_calls=counters["embed_calls"],
        )

    def record_scoped_embed_distribution(
        self,
        total_tokens: int,
        embed_calls: int,
        scope_weights: dict[tuple[str, str], int],
    ) -> None:
        """Attribute tokens and batch participation to every involved scope.

        Scoped ``embed_calls`` count participations, so aggregating owner scopes can
        exceed the number of provider requests. Global counters remain unchanged.
        """
        positive = {scope: max(0, int(weight)) for scope, weight in scope_weights.items()}
        positive = {scope: weight for scope, weight in positive.items() if weight}
        if not positive:
            return
        weight_total = sum(positive.values())
        allocations = {
            scope: max(0, int(total_tokens)) * weight // weight_total
            for scope, weight in positive.items()
        }
        largest_scope = max(positive, key=positive.get)
        allocations[largest_scope] += max(0, int(total_tokens)) - sum(allocations.values())
        today = datetime.now(UTC).date()
        date_key = today.isoformat()
        embed_calls = max(0, int(embed_calls))
        with self._lock:
            self._prune_daily_usage_unlocked(today)
            for scope, tokens in allocations.items():
                counters = self._usage_by_scope.setdefault(scope, _empty_daily_counters())
                daily = self._daily_usage_by_scope.setdefault(scope, {}).setdefault(
                    date_key, _empty_daily_counters()
                )
                for target in (counters, daily):
                    target["embed_tokens"] += tokens
                    target["embed_calls"] += embed_calls

    def reset(self) -> None:
        with self._lock:
            self._reset_unlocked()

    def replace_tool_stats(self, stats: dict[str, dict]) -> None:
        """Replace per-tool stats from legacy ``SessionBuffer.tool_usage_stats``."""
        with self._lock:
            self._tool_by_name = {}
            for name, raw in (stats or {}).items():
                counters = _empty_tool_counters()
                counters.update(
                    {
                        "call_count": int(raw.get("call_count", 0) or 0),
                        "success_count": int(raw.get("success_count", 0) or 0),
                        "fail_count": int(raw.get("fail_count", 0) or 0),
                        "total_duration_ms": float(raw.get("total_duration_ms", 0.0) or 0.0),
                        "total_prompt_tokens": int(raw.get("total_prompt_tokens", 0) or 0),
                        "total_completion_tokens": int(raw.get("total_completion_tokens", 0) or 0),
                        "total_tokens": int(raw.get("total_tokens", 0) or 0),
                        "llm_call_ids": list(raw.get("llm_call_ids") or []),
                    }
                )
                self._tool_by_name[name] = counters

    def merge_tool_stats(self, stats: dict[str, dict]) -> None:
        """Merge per-tool stats from legacy ``SessionBuffer.tool_usage_stats``."""
        with self._lock:
            for name, raw in (stats or {}).items():
                counters = self._tool_by_name.setdefault(name, _empty_tool_counters())
                counters["call_count"] += int(raw.get("call_count", 0) or 0)
                counters["success_count"] += int(raw.get("success_count", 0) or 0)
                counters["fail_count"] += int(raw.get("fail_count", 0) or 0)
                counters["total_duration_ms"] += float(raw.get("total_duration_ms", 0.0) or 0.0)
                counters["total_prompt_tokens"] += int(raw.get("total_prompt_tokens", 0) or 0)
                counters["total_completion_tokens"] += int(
                    raw.get("total_completion_tokens", 0) or 0
                )
                counters["total_tokens"] = (
                    counters["total_prompt_tokens"] + counters["total_completion_tokens"]
                )
                existing_ids = counters.get("llm_call_ids", [])
                for call_id in raw.get("llm_call_ids", []) or []:
                    if call_id not in existing_ids:
                        existing_ids.append(call_id)
                counters["llm_call_ids"] = existing_ids

    def merge_snapshot(self, snapshot: TokenUsageSnapshot) -> None:
        with self._lock:
            self._input += snapshot.input_tokens
            self._output += snapshot.output_tokens
            self._cache_read += snapshot.cache_read
            self._cache_write += snapshot.cache_write
            self._llm_calls += snapshot.llm_calls
            self._embed_tokens += snapshot.embed_tokens
            self._embed_calls += snapshot.embed_calls
            self._local_cache_hits += snapshot.local_cache_hits
            self._local_cache_misses += snapshot.local_cache_misses
            self._local_cache_saved_tokens += snapshot.local_cache_saved_tokens
            for name, stats in snapshot.tool_stats.get("by_tool", {}).items():
                self._merge_tool_counters(
                    self._tool_by_name.setdefault(name, _empty_tool_counters()),
                    stats,
                )
            for category, stats in snapshot.tool_stats.get("by_category", {}).items():
                self._merge_tool_counters(
                    self._tool_by_category.setdefault(category, _empty_tool_counters()),
                    stats,
                )
            for session_id, stats in snapshot.tool_stats.get("by_session", {}).items():
                target = self._usage_by_session.setdefault(session_id, _empty_session_counters())
                for key in target:
                    target[key] += stats.get(key, 0) or 0

    def _snapshot_unlocked(self) -> TokenUsageSnapshot:
        return TokenUsageSnapshot(
            input_tokens=self._input,
            output_tokens=self._output,
            cache_read=self._cache_read,
            cache_write=self._cache_write,
            llm_calls=self._llm_calls,
            embed_tokens=self._embed_tokens,
            embed_calls=self._embed_calls,
            tool_stats={
                "by_tool": _copy_tool_stats({"by_tool": self._tool_by_name}).get("by_tool", {}),
                "by_category": _copy_tool_stats({"by_category": self._tool_by_category}).get(
                    "by_category", {}
                ),
                "by_session": _copy_tool_stats({"by_session": self._usage_by_session}).get(
                    "by_session", {}
                ),
            },
            local_cache_hits=self._local_cache_hits,
            local_cache_misses=self._local_cache_misses,
            local_cache_saved_tokens=self._local_cache_saved_tokens,
        )

    def _reset_unlocked(self) -> None:
        self._input = 0
        self._output = 0
        self._cache_read = 0
        self._cache_write = 0
        self._llm_calls = 0
        self._embed_tokens = 0
        self._embed_calls = 0
        self._local_cache_hits = 0
        self._local_cache_misses = 0
        self._local_cache_saved_tokens = 0
        self._tool_by_name = {}
        self._tool_by_category = {}
        self._usage_by_session = {}

    def _record_scoped_llm(
        self, input_tokens: int, output_tokens: int, cache_read: int, cache_write: int
    ) -> None:
        scope = _active_usage_scope.get()
        if not scope or not scope[0]:
            return
        counters = self._usage_by_scope.setdefault(scope, _empty_daily_counters())
        daily = self._daily_usage_by_scope.setdefault(scope, {}).setdefault(
            datetime.now(UTC).date().isoformat(), _empty_daily_counters()
        )
        for target in (counters, daily):
            target["input_tokens"] += input_tokens
            target["output_tokens"] += output_tokens
            target["cache_read"] += cache_read
            target["cache_write"] += cache_write
            target["llm_calls"] += 1

    def _record_scoped_embed(self, total_tokens: int) -> None:
        scope = _active_usage_scope.get()
        if not scope or not scope[0]:
            return
        counters = self._usage_by_scope.setdefault(scope, _empty_daily_counters())
        daily = self._daily_usage_by_scope.setdefault(scope, {}).setdefault(
            datetime.now(UTC).date().isoformat(), _empty_daily_counters()
        )
        for target in (counters, daily):
            target["embed_tokens"] += total_tokens
            target["embed_calls"] += 1

    def _prune_daily_usage_unlocked(self, today: date) -> None:
        if self._daily_usage_pruned_on == today:
            return
        cutoff = (today - timedelta(days=_DAILY_USAGE_RETENTION_DAYS - 1)).isoformat()
        self._daily_usage = {
            date_key: counters
            for date_key, counters in self._daily_usage.items()
            if date_key >= cutoff
        }
        for scope, daily in list(self._daily_usage_by_scope.items()):
            retained = {
                date_key: counters for date_key, counters in daily.items() if date_key >= cutoff
            }
            if retained:
                self._daily_usage_by_scope[scope] = retained
            else:
                del self._daily_usage_by_scope[scope]
        self._daily_usage_pruned_on = today

    def _matching_scope_keys(
        self, account_id: str, owner_spaces: tuple[str, ...]
    ) -> list[tuple[str, str]]:
        allowed = set(owner_spaces)
        return [
            key
            for key in self._usage_by_scope
            if key[0] == account_id and (not allowed or key[1] in allowed)
        ]

    @staticmethod
    def _sum_counters(rows) -> dict[str, int]:
        total = _empty_daily_counters()
        for row in rows:
            if not row:
                continue
            for key in total:
                total[key] += int(row.get(key, 0) or 0)
        return total

    @staticmethod
    def _add_tool_call(
        counters: dict,
        *,
        success: bool,
        duration_ms: float | None,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        llm_call_id: str | None,
    ) -> None:
        counters["call_count"] += 1
        if success:
            counters["success_count"] += 1
        else:
            counters["fail_count"] += 1
        counters["total_duration_ms"] += duration_ms or 0
        counters["total_prompt_tokens"] += prompt_tokens or 0
        counters["total_completion_tokens"] += completion_tokens or 0
        counters["total_tokens"] += total_tokens
        if llm_call_id and llm_call_id not in counters["llm_call_ids"]:
            counters["llm_call_ids"].append(llm_call_id)

    @staticmethod
    def _merge_tool_counters(target: dict, source: dict) -> None:
        for key in (
            "call_count",
            "success_count",
            "fail_count",
            "total_duration_ms",
            "total_prompt_tokens",
            "total_completion_tokens",
            "total_tokens",
        ):
            target[key] += source.get(key, 0) or 0
        for call_id in source.get("llm_call_ids", []) or []:
            if call_id not in target["llm_call_ids"]:
                target["llm_call_ids"].append(call_id)


# ---------------------------------------------------------------------------
# TokenTracker — extends UsageTracker with bucket integration for perf
# ---------------------------------------------------------------------------


class TokenTracker(UsageTracker):
    """Accumulates token usage from API responses.

    Extends UsageTracker with per-context bucket support for perf span
    token attribution. Call ``record_llm`` / ``record_embed`` from provider
    implementations, and ``snapshot_and_reset`` from the service layer.
    """

    def __init__(self, model: str | None = None) -> None:
        super().__init__()
        self._model = model
        self._usage_store = None

    def set_usage_store(self, store):
        self._usage_store = store

    def _persist(self, kind, counters, *, event_id=None, model=None, deadline=None, events=None):
        if self._usage_store is None:
            return
        weights = _usage_distribution.get() if kind.startswith('embedding') else None
        if not weights:
            scope = _active_usage_scope.get() or ('', '')
            if not scope[0]:
                logging.getLogger(__name__).error('Model usage has no account attribution')
            weights = {scope: 1}
        weights = {s: int(w) for s, w in weights.items() if int(w) > 0}
        total_weight = sum(weights.values())
        allocations = {s: dict(counters) for s in weights}
        largest = max(weights, key=weights.get)
        for key in ('input_tokens', 'output_tokens', 'cache_read', 'cache_write', 'embed_tokens'):
            amount = int(counters.get(key, 0))
            for scope, weight in weights.items():
                allocations[scope][key] = amount * weight // total_weight
            allocations[largest][key] += amount - sum(a[key] for a in allocations.values())
        event = {'id': event_id or uuid4().hex, 'model': model or self._model or 'unknown',
                 'kind': kind, 'at': datetime.now(UTC), 'allocations': allocations}
        if events is not None:
            events.append(event)
        else:
            self._write_events([event], deadline)

    def _write_events(self, events, deadline):
        if not events:
            return
        record = self._usage_store.record if len(events) == 1 else self._usage_store.record_many
        payload = events[0] if len(events) == 1 else events
        try:
            if deadline is None:
                record(payload)
            else:
                record(payload, deadline=deadline)
        except TimeoutError:
            raise
        except Exception as exc:
            logging.getLogger(__name__).exception('Model usage persistence failed: event_id=%s', events[0]['id'])
            raise UsagePersistenceError('Model usage persistence failed') from exc

    def record_failure(self, kind, *, deadline=None):
        if kind == 'llm':
            from providers.llm_observation import record_observed_usage
            record_observed_usage(model=self._model, failed=True, usage_known=False)
        self._persist(kind, {'failed_calls': 1}, deadline=deadline)

    def record_cache_hit(self, kind, *, deadline=None):
        self._persist(kind + '_cache', {'local_cache_hits': 1}, deadline=deadline)

    def record_embedding_cache_hits(self, indices, *, deadline=None):
        events = []
        for index in indices:
            with embedding_input_scope([index]):
                self._persist('embedding_cache', {'local_cache_hits': 1}, events=events)
        self._write_events(events, deadline)

    @property
    def model(self) -> str | None:
        """Model identifier used for per-bucket attribution (may be None)."""
        return self._model

    def set_model(self, model: str | None) -> None:
        """Late-bind a provider model name (used by providers that resolve model after init)."""
        self._model = model

    def record_llm(
        self,
        input_tokens: int,
        output_tokens: int,
        cache_read: int = 0,
        cache_write: int = 0,
        *,
        event_id: str | None = None,
        usage_known: bool = True,
        model: str | None = None,
    ) -> str:
        from providers.llm_observation import record_observed_usage
        record_observed_usage(input_tokens=input_tokens, output_tokens=output_tokens,
                              model=model or self._model, usage_known=usage_known, event_id=event_id)
        self._persist('llm', dict(input_tokens=input_tokens, output_tokens=output_tokens,
                                 cache_read=cache_read, cache_write=cache_write, llm_calls=1,
                                 missing_usage_calls=int(not usage_known)),
                      event_id=event_id, model=model)
        call_id = super().record_llm(input_tokens, output_tokens, cache_read, cache_write)
        for bucket in _active_buckets.get():
            bucket.add_llm(input_tokens, output_tokens, cache_read, cache_write, self._model)
        return call_id

    def record_embed(self, total_tokens: int, *, event_id=None, usage_known=True, deadline=None) -> None:
        self._persist('embedding', dict(embed_tokens=total_tokens, embed_calls=1,
                                       missing_usage_calls=int(not usage_known)), event_id=event_id,
                      deadline=deadline)
        super().record_embed(total_tokens)
        for bucket in _active_buckets.get():
            bucket.add_embed(total_tokens, self._model)

    def record_local_cache_hit(self, tokens_saved: int) -> None:
        super().record_local_cache_hit(tokens_saved)
        for bucket in _active_buckets.get():
            bucket.add_local_cache_hit(tokens_saved)

    def record_local_cache_miss(self) -> None:
        super().record_local_cache_miss()
        for bucket in _active_buckets.get():
            bucket.add_local_cache_miss()
