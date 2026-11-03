"""Deterministic bounded evidence projection for arbitrarily long traces."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from skills.artifact import validate_skill_name
from skills.sanitizer import StreamingTraceSanitizer, without_correlation_ids

_CORRELATION_KEY_ORDER = ("tool_use_id", "tool_call_id", "call_id")
_POLICY_VERSION = 4
_FIELD_CHARS = 4000
_LIST_ITEMS = 20
_DICT_ITEMS = 32
_MAX_DEPTH = 16
_CHECKPOINT_BYTES_PER_TRACE_CHAR = 8
_TOOL_PAIRS = 24
_FAILURE_PAIRS = 4
_SKILL_ACTIVATION_PAIRS = 2
_PENDING_STARTS = 24
_PENDING_STARTS_PER_CALL = 4
_PENDING_CHARS_PER_TRACE_CHAR = 2


def _json_chars(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _json_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _truncated_text(value: str, retained_chars: int) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    side = retained_chars // 2
    return (
        value[:side]
        + f"\n[TRUNCATED chars={len(value)} sha256={digest}]\n"
        + (value[-(retained_chars - side):] if retained_chars > side else "")
    )


def _bounded(value: Any, metrics: dict[str, int], *, depth: int = 0) -> Any:
    if depth >= _MAX_DEPTH and isinstance(value, (dict, list)):
        metrics["truncated_fields"] += 1
        return (
            f"[TRUNCATED type={type(value).__name__} depth={depth} "
            f"sha256={_json_digest(value)}]"
        )
    if isinstance(value, str):
        if len(value) <= _FIELD_CHARS:
            return value
        metrics["truncated_fields"] += 1
        return _truncated_text(value, _FIELD_CHARS - 96)
    if isinstance(value, list):
        items = value
        if len(items) > _LIST_ITEMS:
            metrics["truncated_fields"] += 1
            half = _LIST_ITEMS // 2
            items = items[:half] + [{"truncated_items": len(value) - _LIST_ITEMS}] + items[-half:]
        return [_bounded(item, metrics, depth=depth + 1) for item in items]
    if isinstance(value, dict):
        items = list(value.items())
        truncated = max(0, len(items) - _DICT_ITEMS)
        if truncated:
            metrics["truncated_fields"] += 1
            half = _DICT_ITEMS // 2
            items = items[:half] + items[-half:]
        result = {
            str(key): _bounded(item, metrics, depth=depth + 1)
            for key, item in items
        }
        if truncated:
            marker = "truncated_keys"
            while marker in result:
                marker = "_" + marker
            result[marker] = truncated
        return result
    return value


def _call_key(event: dict[str, Any]) -> str:
    tool = str(event.get("tool_name") or "")
    call_id = next(
        (
            event.get(field).strip()
            for field in _CORRELATION_KEY_ORDER
            if isinstance(event.get(field), str) and event.get(field).strip()
        ),
        "",
    )
    identity = hashlib.sha256(call_id.encode("utf-8")).hexdigest()[:24] if call_id else "none"
    return f"{tool}|{identity}"


def _activated_skill_name(pair: list[dict[str, Any]]) -> str | None:
    """Return the valid skill named by one successful read-skill pair."""
    if len(pair) < 2:
        return None
    start = pair[0].get("event") if isinstance(pair[0], dict) else None
    outcome = pair[-1].get("event") if isinstance(pair[-1], dict) else None
    if not isinstance(start, dict) or not isinstance(outcome, dict):
        return None
    tool = str(start.get("tool_name") or "").casefold()
    if not (
        tool == "ogm_read_skill"
        or any(tool.endswith(separator + "ogm_read_skill") for separator in (".", "__", "_", "-"))
    ):
        return None
    if outcome.get("event_type") != "tool_success":
        return None
    for key in ("tool_input", "input", "arguments", "args"):
        value = start.get(key)
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                continue
        if not isinstance(value, dict) or not isinstance(value.get("name"), str):
            continue
        try:
            return validate_skill_name(value["name"])
        except ValueError:
            return None
    return None


def _required_event_projection(event: dict[str, Any], text_limit: int) -> dict[str, Any]:
    """Keep the identifying type and a bounded primary message for required evidence."""
    event_type = str(event.get("event_type") or "")
    text_field = "prompt" if event_type == "user_message" else "assistant_message"
    text = event.get(text_field)
    projected = {"event_type": event_type}
    if not isinstance(text, str):
        return projected
    if len(text) <= text_limit:
        projected[text_field] = text
        return projected
    projected[text_field] = _truncated_text(text, text_limit)
    return projected


def _fit_required_events(events: list[dict[str, Any]], max_chars: int) -> list[dict[str, Any]]:
    """Project required messages into the aggregate JSON budget."""
    payload = [item["event"] for item in events]
    if _json_chars(payload) <= max_chars:
        return payload

    longest_text = max(
        (
            len(value)
            for item in payload
            for key in ("prompt", "assistant_message")
            if isinstance((value := item.get(key)), str)
        ),
        default=0,
    )
    low, high = 0, longest_text
    best = [_required_event_projection(item, 0) for item in payload]
    while low <= high:
        text_limit = (low + high) // 2
        proposal = [_required_event_projection(item, text_limit) for item in payload]
        if _json_chars(proposal) <= max_chars:
            best = proposal
            low = text_limit + 1
        else:
            high = text_limit - 1
    return best


def _fit_failure_event(
    selected: list[dict[str, Any]], failure: dict[str, Any], max_chars: int,
) -> dict[str, Any] | None:
    """Retain the failure type and a bounded primary error when its pair cannot fit."""
    full_payload = [
        item["event"]
        for item in sorted(selected + [failure], key=lambda item: item["ordinal"])
    ]
    if _json_chars(full_payload) <= max_chars:
        return failure
    event = failure["event"]
    projected = {
        key: event[key] for key in ("event_type", "tool_name") if key in event
    }
    text_field = next(
        (
            key for key in ("error", "message", "output", "result")
            if isinstance(event.get(key), str)
        ),
        None,
    )
    text = event.get(text_field) if text_field else None
    low, high = 0, len(text) if isinstance(text, str) else 0
    best = None
    while low <= high:
        text_limit = (low + high) // 2
        candidate = dict(projected)
        if isinstance(text, str):
            candidate[text_field] = (
                text if len(text) <= text_limit else _truncated_text(text, text_limit)
            )
        wrapped = {"ordinal": failure["ordinal"], "event": candidate}
        payload = [
            item["event"]
            for item in sorted(selected + [wrapped], key=lambda item: item["ordinal"])
        ]
        if _json_chars(payload) <= max_chars:
            best = wrapped
            low = text_limit + 1
        else:
            high = text_limit - 1
    return best


class TraceEvidenceCompactor:
    """Accumulate high-value sanitized events within an assessor JSON budget."""

    policy_version = _POLICY_VERSION

    def __init__(self, *, max_chars: int, checkpoint: dict[str, Any] | None = None) -> None:
        if max_chars < 1024:
            raise ValueError("max_chars must be at least 1024")
        self.max_chars = max_chars
        self.first_user: dict[str, Any] | None = None
        self.latest_user: dict[str, Any] | None = None
        self.final_assistant: dict[str, Any] | None = None
        self.pending: dict[str, list[dict[str, Any]]] = {}
        self.pending_dropped: dict[str, int] = {}
        self.tool_pairs: list[list[dict[str, Any]]] = []
        self.failure_pairs: list[list[dict[str, Any]]] = []
        self.skill_activation_pairs: dict[str, list[dict[str, Any]]] = {}
        self.exact_events: list[dict[str, Any]] | None = []
        self.metrics = {
            "events_seen": 0,
            "raw_chars_seen": 0,
            "truncated_fields": 0,
            "deduplicated_events": 0,
        }
        self.next_ordinal = 0
        if checkpoint:
            self._restore(checkpoint)

    def _restore(self, checkpoint: dict[str, Any]) -> None:
        if checkpoint.get("policy_version") != self.policy_version:
            return
        state = checkpoint.get("accumulator")
        if not isinstance(state, dict):
            return
        self.first_user = state.get("first_user")
        self.latest_user = state.get("latest_user")
        self.final_assistant = state.get("final_assistant")
        self.pending = {
            str(key): list(value) for key, value in state.get("pending", {}).items()
            if isinstance(value, list)
        }
        self.pending_dropped = {
            str(key): int(value)
            for key, value in state.get("pending_dropped", {}).items()
            if isinstance(value, int) and value > 0
        }
        self.tool_pairs = [list(value) for value in state.get("tool_pairs", []) if isinstance(value, list)]
        self.failure_pairs = [
            list(value) for value in state.get("failure_pairs", []) if isinstance(value, list)
        ]
        restored_activations = state.get("skill_activation_pairs")
        if isinstance(restored_activations, dict):
            for name, value in restored_activations.items():
                if len(self.skill_activation_pairs) >= _SKILL_ACTIVATION_PAIRS:
                    break
                if isinstance(name, str) and isinstance(value, list):
                    try:
                        self.skill_activation_pairs[validate_skill_name(name)] = list(value)
                    except ValueError:
                        continue
        exact = state.get("exact_events")
        self.exact_events = list(exact) if isinstance(exact, list) else None
        stored_metrics = checkpoint.get("metrics")
        if isinstance(stored_metrics, dict):
            for key in self.metrics:
                value = stored_metrics.get(key)
                if isinstance(value, int) and value >= 0:
                    self.metrics[key] = value
        self.next_ordinal = max(0, int(checkpoint.get("next_ordinal", 0)))
        self._bound_pending()

    def _drop_oldest_pending(self, key: str) -> None:
        starts = self.pending[key]
        starts.pop(0)
        if not starts:
            self.pending.pop(key)
        self.pending_dropped[key] = self.pending_dropped.get(key, 0) + 1
        self.metrics["deduplicated_events"] += 1

    def _bound_pending(self) -> None:
        while self.pending:
            oversized_key = next(
                (
                    key for key, starts in self.pending.items()
                    if len(starts) > _PENDING_STARTS_PER_CALL
                ),
                None,
            )
            total = sum(len(starts) for starts in self.pending.values())
            over_budget = _json_chars(self.pending) > (
                self.max_chars * _PENDING_CHARS_PER_TRACE_CHAR
            )
            if oversized_key is None and total <= _PENDING_STARTS and not over_budget:
                break
            key = oversized_key or min(
                self.pending,
                key=lambda item: self.pending[item][0]["ordinal"],
            )
            self._drop_oldest_pending(key)
        self.pending_dropped = {
            key: count for key, count in self.pending_dropped.items()
            if key in self.pending
        }

    def feed(
        self, *, ordinal: int, raw_event: dict[str, Any], sanitized_event: dict[str, Any],
    ) -> None:
        exact_event = without_correlation_ids(sanitized_event)
        event = _bounded(exact_event, self.metrics)
        wrapped = {"ordinal": ordinal, "event": event}
        self.metrics["events_seen"] += 1
        self.metrics["raw_chars_seen"] += _json_chars(raw_event)
        self.next_ordinal = ordinal

        if self.exact_events is not None:
            if event != exact_event:
                self.exact_events = None
            else:
                proposed = self.exact_events + [event]
                if _json_chars(proposed) <= self.max_chars:
                    self.exact_events = proposed
                else:
                    self.exact_events = None

        event_type = event.get("event_type")
        if event_type == "user_message":
            self.first_user = self.first_user or wrapped
            self.latest_user = wrapped
        elif event_type in {"assistant_message", "turn_end"}:
            self.final_assistant = wrapped
        elif event_type == "tool_start":
            self.pending.setdefault(_call_key(raw_event), []).append(wrapped)
            self._bound_pending()
        elif event_type in {"tool_success", "tool_failure"}:
            key = _call_key(raw_event)
            dropped = self.pending_dropped.get(key, 0)
            if dropped:
                start = None
                if dropped == 1:
                    self.pending_dropped.pop(key)
                else:
                    self.pending_dropped[key] = dropped - 1
            else:
                starts = self.pending.get(key) or []
                start = starts.pop(0) if starts else None
                if not starts:
                    self.pending.pop(key, None)
            pair = ([start] if start else []) + [wrapped]
            target = self.tool_pairs if event_type == "tool_success" else self.failure_pairs
            target.append(pair)
            if event_type == "tool_success":
                skill_name = _activated_skill_name(pair)
                if (
                    skill_name
                    and skill_name not in self.skill_activation_pairs
                    and len(self.skill_activation_pairs) < _SKILL_ACTIVATION_PAIRS
                ):
                    self.skill_activation_pairs[skill_name] = [
                        {
                            "ordinal": pair[0]["ordinal"],
                            "event": {
                                "event_type": "tool_start",
                                "tool_name": pair[0]["event"].get("tool_name", "ogm_read_skill"),
                                "arguments": {"name": skill_name},
                            },
                        },
                        {
                            "ordinal": pair[-1]["ordinal"],
                            "event": {
                                "event_type": "tool_success",
                                "tool_name": pair[-1]["event"].get("tool_name", "ogm_read_skill"),
                            },
                        },
                    ]
            limit = _TOOL_PAIRS if event_type == "tool_success" else _FAILURE_PAIRS
            if len(target) > limit:
                self.metrics["deduplicated_events"] += len(target.pop(0))

    @staticmethod
    def _events(items: list[dict[str, Any] | None]) -> list[dict[str, Any]]:
        return [item for item in items if isinstance(item, dict)]

    def events(self) -> list[dict[str, Any]]:
        if self.exact_events is not None:
            return list(self.exact_events)
        required = self._events([self.first_user, self.latest_user, self.final_assistant])
        required_by_ordinal = {item["ordinal"]: item for item in required}
        required = sorted(required_by_ordinal.values(), key=lambda item: item["ordinal"])
        if _json_chars([item["event"] for item in required]) > self.max_chars:
            return _fit_required_events(required, self.max_chars)
        selected_by_ordinal = dict(required_by_ordinal)
        for pair in self.skill_activation_pairs.values():
            proposal = list(selected_by_ordinal.values()) + pair
            payload = [item["event"] for item in sorted(proposal, key=lambda item: item["ordinal"])]
            if _json_chars(payload) <= self.max_chars:
                selected_by_ordinal.update({item["ordinal"]: item for item in pair})
            else:
                self.metrics["deduplicated_events"] += len(pair)
        selected = list(selected_by_ordinal.values())
        pairs = list(reversed(self.failure_pairs)) + list(reversed(self.tool_pairs))
        for pair in pairs:
            unseen = [item for item in pair if item["ordinal"] not in selected_by_ordinal]
            proposal = selected + unseen
            payload = [item["event"] for item in sorted(proposal, key=lambda item: item["ordinal"])]
            if _json_chars(payload) <= self.max_chars:
                selected = proposal
                selected_by_ordinal.update({item["ordinal"]: item for item in unseen})
            else:
                if pair[-1]["event"].get("event_type") == "tool_failure":
                    fitted = _fit_failure_event(selected, pair[-1], self.max_chars)
                    if fitted is not None:
                        selected.append(fitted)
                        selected_by_ordinal[fitted["ordinal"]] = fitted
                        self.metrics["deduplicated_events"] += max(0, len(unseen) - 1)
                        continue
                self.metrics["deduplicated_events"] += len(unseen)
        return [item["event"] for item in sorted(selected, key=lambda item: item["ordinal"])]

    def checkpoint(self, *, sanitizer: dict[str, Any], done: bool) -> dict[str, Any]:
        checkpoint = {
            "policy_version": self.policy_version,
            "next_ordinal": self.next_ordinal,
            "done": done,
            "sanitizer": sanitizer,
            "accumulator": {
                "first_user": self.first_user,
                "latest_user": self.latest_user,
                "final_assistant": self.final_assistant,
                "pending": self.pending,
                "pending_dropped": self.pending_dropped,
                "tool_pairs": self.tool_pairs,
                "failure_pairs": self.failure_pairs,
                "skill_activation_pairs": self.skill_activation_pairs,
                "exact_events": self.exact_events,
            },
            "metrics": dict(self.metrics),
        }
        sanitizer_bytes = len(
            json.dumps(sanitizer, ensure_ascii=False, sort_keys=True).encode("utf-8")
        )
        if sanitizer_bytes > StreamingTraceSanitizer.MAX_CHECKPOINT_BYTES:
            raise ValueError(
                "trace sanitizer checkpoint exceeds persistent byte budget "
                f"({sanitizer_bytes} > {StreamingTraceSanitizer.MAX_CHECKPOINT_BYTES})"
            )
        compactor_state = {key: value for key, value in checkpoint.items() if key != "sanitizer"}
        compactor_bytes = len(
            json.dumps(compactor_state, ensure_ascii=False, sort_keys=True).encode("utf-8")
        )
        max_compactor_bytes = self.max_chars * _CHECKPOINT_BYTES_PER_TRACE_CHAR
        if compactor_bytes > max_compactor_bytes:
            raise ValueError(
                "trace compaction checkpoint exceeds persistent byte budget "
                f"({compactor_bytes} > {max_compactor_bytes})"
            )
        return checkpoint

    def public_metrics(self, assessment_chars: int) -> dict[str, Any]:
        return {
            "compaction_policy_version": self.policy_version,
            "trace_events_seen": self.metrics["events_seen"],
            "raw_trace_chars_seen": self.metrics["raw_chars_seen"],
            "assessment_trace_chars": assessment_chars,
            "trace_compacted": self.exact_events is None,
            "deduplicated_event_count": self.metrics["deduplicated_events"],
            "truncated_field_count": self.metrics["truncated_fields"],
        }
