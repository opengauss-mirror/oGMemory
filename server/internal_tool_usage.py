# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from __future__ import annotations

import threading
from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from server.control_plane_store import ControlPlaneStore


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(value: str | None):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _empty_summary() -> dict:
    return {
        "llm_tool_rounds": 0,
        "tool_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read": 0,
        "cache_write": 0,
        "total_tokens": 0,
    }


def _empty_tool_counter() -> dict:
    return {
        "call_count": 0,
        "success_count": 0,
        "fail_count": 0,
        "total_duration_ms": 0.0,
        "round_ids": set(),
        "total_tokens": 0,
        "attributed_tokens": 0,
        "allocated_tokens": 0,
    }


def _tool_response(tool_name: str, raw: dict) -> dict:
    call_count = int(raw.get("call_count", 0) or 0)
    success_count = int(raw.get("success_count", 0) or 0)
    total_duration = float(raw.get("total_duration_ms", 0.0) or 0.0)
    round_ids = raw.get("round_ids", set()) or set()
    if not isinstance(round_ids, set):
        round_ids = set(round_ids)
    return {
        "tool_name": tool_name,
        "call_count": call_count,
        "success_count": success_count,
        "fail_count": int(raw.get("fail_count", 0) or 0),
        "success_rate": success_count / call_count if call_count else 0.0,
        "avg_duration_ms": total_duration / call_count if call_count else 0.0,
        "round_count": len(round_ids),
        "attributed_tokens": int(raw.get("attributed_tokens", raw.get("total_tokens", 0)) or 0),
        "allocated_tokens": int(raw.get("allocated_tokens", 0) or 0),
        # Backward-compatible alias: total_tokens historically meant attributed
        # round tokens for this tool and may double count across tools.
        "total_tokens": int(raw.get("total_tokens", raw.get("attributed_tokens", 0)) or 0),
    }


class InternalToolUsageTracker:
    """In-memory usage tracker for ogmem internal tool calls."""

    def __init__(self, max_rounds_per_session: int = 1000) -> None:
        self._lock = threading.Lock()
        self._max_rounds = max(1, int(max_rounds_per_session or 1000))
        self._sessions: dict[str, dict] = {}

    def record_round(
        self,
        *,
        account_id: str,
        session_id: str,
        pipeline_name: str,
        round_id: str,
        tool_names: list[str],
        user_id: str = "",
        model: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read: int = 0,
        cache_write: int = 0,
        started_at: str = "",
        ended_at: str = "",
    ) -> None:
        if not session_id or not round_id or not tool_names:
            return
        with self._lock:
            session = self._get_session(account_id, session_id, user_id=user_id)
            is_new_round = round_id not in session["round_ids"]
            if is_new_round:
                session["round_ids"].add(round_id)
                session["summary"]["llm_tool_rounds"] += 1
                session["summary"]["input_tokens"] += input_tokens or 0
                session["summary"]["output_tokens"] += output_tokens or 0
                session["summary"]["cache_read"] += cache_read or 0
                session["summary"]["cache_write"] += cache_write or 0
                session["summary"]["total_tokens"] += (input_tokens or 0) + (output_tokens or 0)
            round_record = {
                "round_id": round_id,
                "session_id": session_id,
                "user_id": user_id or session.get("user_id", ""),
                "pipeline": pipeline_name,
                "model": model,
                "input_tokens": input_tokens or 0,
                "output_tokens": output_tokens or 0,
                "cache_read": cache_read or 0,
                "cache_write": cache_write or 0,
                "total_tokens": (input_tokens or 0) + (output_tokens or 0),
                "tool_names": list(tool_names),
                "tool_call_count": len(tool_names),
                "started_at": started_at,
                "ended_at": ended_at or _now_iso(),
            }
            session["rounds"].append(round_record)
            session["rounds_truncated"] = session["rounds_truncated"] or (
                len(session["rounds"]) == self._max_rounds
                and session["summary"]["llm_tool_rounds"] > self._max_rounds
            )

    def record_tool_call(
        self,
        *,
        account_id: str,
        session_id: str,
        pipeline_name: str,
        round_id: str,
        tool_name: str,
        status: str,
        duration_ms: float | int | None = 0,
        error_type: str = "",
    ) -> None:
        if not session_id or not tool_name:
            return
        success = status in {"success", "completed", "recorded"}
        with self._lock:
            session = self._get_session(account_id, session_id)
            session["summary"]["tool_calls"] += 1
            pipeline_tools = session["tools_by_pipeline"].setdefault(pipeline_name or "", {})
            counter = pipeline_tools.setdefault(tool_name, _empty_tool_counter())
            counter["call_count"] += 1
            if success:
                counter["success_count"] += 1
            else:
                counter["fail_count"] += 1
            counter["total_duration_ms"] += duration_ms or 0
            if round_id:
                counter["round_ids"].add(round_id)
            session["tool_events"].append({
                "tool_name": tool_name,
                "pipeline": pipeline_name,
                "round_id": round_id,
                "status": status,
                "duration_ms": duration_ms or 0,
                "error_type": error_type,
                "created_at": _now_iso(),
            })

    def get_stats(
        self,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        pipeline_name: str | None = None,
        include_rounds: bool = False,
    ) -> dict:
        with self._lock:
            sessions = {
                sid: deepcopy(data)
                for sid, data in self._sessions.items()
                if not session_id or sid == session_id
            }
        return self._format_stats(
            sessions,
            session_id=session_id,
            user_id=user_id,
            start_time=start_time,
            end_time=end_time,
            pipeline_name=pipeline_name,
            include_rounds=include_rounds,
        )

    def session_snapshot(self, session_id: str) -> dict | None:
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return None
            sessions = {session_id: deepcopy(session)}
        stats = self._format_stats(sessions, session_id=session_id, include_rounds=True)
        if not stats["summary"]["llm_tool_rounds"] and not stats["summary"]["tool_calls"]:
            return None
        stats["account_id"] = session.get("account_id", "")
        return stats

    def reset(self) -> None:
        with self._lock:
            self._sessions = {}

    def clear_session(self, session_id: str) -> None:
        if not session_id:
            return
        with self._lock:
            self._sessions.pop(session_id, None)

    def count_snapshot(self) -> dict:
        stats = self.get_stats()
        summary = stats["summary"]
        return {
            "rounds": summary["llm_tool_rounds"],
            "tool_calls": summary["tool_calls"],
            "input_tokens": summary.get("input_tokens", 0),
            "output_tokens": summary.get("output_tokens", 0),
            "cache_read": summary.get("cache_read", 0),
            "cache_write": summary.get("cache_write", 0),
            "total_tokens": summary.get("total_tokens", 0),
        }

    def _get_session(self, account_id: str, session_id: str, *, user_id: str = "") -> dict:
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "account_id": account_id,
                "user_id": user_id,
                "session_id": session_id,
                "created_at": _now_iso(),
                "updated_at": _now_iso(),
                "summary": _empty_summary(),
                "tools_by_pipeline": {},
                "rounds": deque(maxlen=self._max_rounds),
                "round_ids": set(),
                "rounds_truncated": False,
                "tool_events": deque(maxlen=self._max_rounds),
            }
        session = self._sessions[session_id]
        session["updated_at"] = _now_iso()
        if account_id and not session.get("account_id"):
            session["account_id"] = account_id
        if user_id and not session.get("user_id"):
            session["user_id"] = user_id
        return session

    def _format_stats(
        self,
        sessions: dict[str, dict],
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        pipeline_name: str | None = None,
        include_rounds: bool = False,
    ) -> dict:
        summary = _empty_summary()
        tools: dict[str, dict] = {}
        session_rows = []
        rounds = []
        rounds_truncated = False
        rounds_limit = self._max_rounds
        start_dt = _parse_iso(start_time)
        end_dt = _parse_iso(end_time)

        for sid, session in sessions.items():
            if user_id and session.get("user_id") != user_id:
                continue
            session_summary = _empty_summary()
            session_rounds = [
                dict(item) for item in session.get("rounds", [])
                if (
                    (not pipeline_name or item.get("pipeline") == pipeline_name)
                    and self._round_in_time_range(item, start_dt, end_dt)
                )
            ]
            allowed_round_ids = {item["round_id"] for item in session_rounds}
            apply_round_filter = bool(pipeline_name or start_dt or end_dt)
            session_tools = self._aggregate_session_tools(
                session,
                pipeline_name,
                allowed_round_ids=allowed_round_ids if apply_round_filter else None,
            )
            if pipeline_name:
                session_summary["llm_tool_rounds"] = len({r["round_id"] for r in session_rounds})
                session_summary["tool_calls"] = sum(
                    int(counter.get("call_count", 0) or 0)
                    for counter in session_tools.values()
                )
                self._add_round_tokens(session_summary, session_rounds)
            else:
                if start_dt or end_dt:
                    session_summary["llm_tool_rounds"] = len(allowed_round_ids)
                    session_summary["tool_calls"] = sum(
                        int(counter.get("call_count", 0) or 0)
                        for counter in session_tools.values()
                    )
                    self._add_round_tokens(session_summary, session_rounds)
                else:
                    session_summary.update(session.get("summary", _empty_summary()))

            summary["llm_tool_rounds"] += session_summary["llm_tool_rounds"]
            summary["tool_calls"] += session_summary["tool_calls"]
            summary["input_tokens"] += session_summary.get("input_tokens", 0) or 0
            summary["output_tokens"] += session_summary.get("output_tokens", 0) or 0
            summary["cache_read"] += session_summary.get("cache_read", 0) or 0
            summary["cache_write"] += session_summary.get("cache_write", 0) or 0
            summary["total_tokens"] += session_summary.get("total_tokens", 0) or 0
            for name, counter in session_tools.items():
                target = tools.setdefault(name, _empty_tool_counter())
                target["call_count"] += counter.get("call_count", 0) or 0
                target["success_count"] += counter.get("success_count", 0) or 0
                target["fail_count"] += counter.get("fail_count", 0) or 0
                target["total_duration_ms"] += counter.get("total_duration_ms", 0.0) or 0.0
                target["round_ids"].update(counter.get("round_ids", set()) or set())
                target["total_tokens"] += counter.get("total_tokens", 0) or 0
                target["attributed_tokens"] += counter.get("attributed_tokens", counter.get("total_tokens", 0)) or 0
                target["allocated_tokens"] += counter.get("allocated_tokens", 0) or 0

            session_rows.append({
                "session_id": sid,
                "user_id": session.get("user_id", ""),
                "llm_tool_rounds": session_summary["llm_tool_rounds"],
                "tool_calls": session_summary["tool_calls"],
            })
            if include_rounds:
                rounds.extend(session_rounds)
                rounds_truncated = rounds_truncated or bool(session.get("rounds_truncated", False))

        result = {
            "ok": True,
            "source": "internal",
            "filters": {
                "session_id": session_id,
                "user_id": user_id,
                "start_time": start_time,
                "end_time": end_time,
                "pipeline": pipeline_name,
            },
            "summary": summary,
            "tools": [_tool_response(name, tools[name]) for name in sorted(tools)],
            "sessions": sorted(session_rows, key=lambda item: item["session_id"]),
        }
        if include_rounds:
            result["rounds"] = rounds[-rounds_limit:]
            result["rounds_limit"] = rounds_limit
            result["rounds_truncated"] = rounds_truncated or len(rounds) > rounds_limit
        return result

    @staticmethod
    def _round_in_time_range(round_record: dict, start_dt, end_dt) -> bool:
        ended_at = _parse_iso(round_record.get("ended_at"))
        if ended_at is None:
            return not (start_dt or end_dt)
        if start_dt and ended_at < start_dt:
            return False
        if end_dt and ended_at > end_dt:
            return False
        return True

    @staticmethod
    def _add_round_tokens(summary: dict, rounds: list[dict]) -> None:
        for round_record in rounds:
            summary["input_tokens"] += int(round_record.get("input_tokens", 0) or 0)
            summary["output_tokens"] += int(round_record.get("output_tokens", 0) or 0)
            summary["cache_read"] += int(round_record.get("cache_read", 0) or 0)
            summary["cache_write"] += int(round_record.get("cache_write", 0) or 0)
            summary["total_tokens"] += int(round_record.get("total_tokens", 0) or 0)

    @staticmethod
    def _round_token_map(session: dict) -> dict[str, int]:
        return {
            str(round_record.get("round_id", "")): int(round_record.get("total_tokens", 0) or 0)
            for round_record in session.get("rounds", [])
            if round_record.get("round_id")
        }

    @classmethod
    def _aggregate_session_tools(
        cls,
        session: dict,
        pipeline_name: str | None,
        *,
        allowed_round_ids: set[str] | None = None,
    ) -> dict[str, dict]:
        round_tokens = cls._round_token_map(session)
        if allowed_round_ids is not None:
            combined: dict[str, dict] = {}
            for event in session.get("tool_events", []):
                if pipeline_name and event.get("pipeline") != pipeline_name:
                    continue
                round_id = event.get("round_id", "")
                if round_id not in allowed_round_ids:
                    continue
                target = combined.setdefault(event.get("tool_name", ""), _empty_tool_counter())
                target["call_count"] += 1
                if event.get("status") in {"success", "completed", "recorded"}:
                    target["success_count"] += 1
                else:
                    target["fail_count"] += 1
                target["total_duration_ms"] += event.get("duration_ms", 0.0) or 0.0
                if round_id:
                    target["round_ids"].add(round_id)
            round_tool_counts = cls._round_tool_call_counts(
                session, pipeline_name, allowed_round_ids=allowed_round_ids,
            )
            cls._attach_tool_tokens(combined, round_tokens, round_tool_counts)
            return combined

        if pipeline_name:
            combined = deepcopy(session.get("tools_by_pipeline", {}).get(pipeline_name, {}))
            round_tool_counts = cls._round_tool_call_counts(session, pipeline_name)
            cls._attach_tool_tokens(combined, round_tokens, round_tool_counts)
            return combined
        combined: dict[str, dict] = {}
        for pipeline_tools in session.get("tools_by_pipeline", {}).values():
            for name, counter in pipeline_tools.items():
                target = combined.setdefault(name, _empty_tool_counter())
                target["call_count"] += counter.get("call_count", 0) or 0
                target["success_count"] += counter.get("success_count", 0) or 0
                target["fail_count"] += counter.get("fail_count", 0) or 0
                target["total_duration_ms"] += counter.get("total_duration_ms", 0.0) or 0.0
                target["round_ids"].update(counter.get("round_ids", set()) or set())
        round_tool_counts = cls._round_tool_call_counts(session, pipeline_name=None)
        cls._attach_tool_tokens(combined, round_tokens, round_tool_counts)
        return combined

    @staticmethod
    def _round_tool_call_counts(
        session: dict,
        pipeline_name: str | None,
        *,
        allowed_round_ids: set[str] | None = None,
    ) -> dict[str, dict[str, int]]:
        counts: dict[str, dict[str, int]] = {}
        for event in session.get("tool_events", []):
            if pipeline_name and event.get("pipeline") != pipeline_name:
                continue
            round_id = str(event.get("round_id", ""))
            if not round_id:
                continue
            if allowed_round_ids is not None and round_id not in allowed_round_ids:
                continue
            tool_name = str(event.get("tool_name", ""))
            if not tool_name:
                continue
            per_round = counts.setdefault(round_id, {})
            per_round[tool_name] = per_round.get(tool_name, 0) + 1
        return counts

    @staticmethod
    def _attach_tool_tokens(
        tools: dict[str, dict],
        round_tokens: dict[str, int],
        round_tool_counts: dict[str, dict[str, int]],
    ) -> None:
        for tool_name, counter in tools.items():
            attributed = sum(
                round_tokens.get(round_id, 0)
                for round_id in counter.get("round_ids", set())
            )
            allocated = 0.0
            for round_id in counter.get("round_ids", set()):
                per_round = round_tool_counts.get(round_id, {})
                total_calls = sum(per_round.values())
                if not total_calls:
                    continue
                allocated += (
                    round_tokens.get(round_id, 0)
                    * per_round.get(tool_name, 0)
                    / total_calls
                )
            counter["attributed_tokens"] = int(attributed)
            counter["allocated_tokens"] = int(round(allocated))
            counter["total_tokens"] = int(attributed)


class InternalToolUsageStore:
    """JSON store for per-session internal tool usage snapshots."""

    def __init__(self, store: ControlPlaneStore):
        self._store = store

    def write_session(self, snapshot: dict) -> None:
        account_id = snapshot["account_id"] if "account_id" in snapshot else self._account_from_snapshot(snapshot)
        session_id = snapshot["filters"]["session_id"]
        self._store.write_json(
            self._tool_calls_path(account_id, session_id),
            {
                "version": 1,
                "account_id": account_id,
                "session_id": session_id,
                "user_id": snapshot.get("sessions", [{}])[0].get("user_id", ""),
                "created_at": _now_iso(),
                "updated_at": _now_iso(),
                "summary": snapshot.get("summary", _empty_summary()),
                "tools": snapshot.get("tools", []),
            },
        )
        self._store.write_json(
            self._rounds_path(account_id, session_id),
            {
                "version": 1,
                "account_id": account_id,
                "session_id": session_id,
                "user_id": snapshot.get("sessions", [{}])[0].get("user_id", ""),
                "rounds_limit": snapshot.get("rounds_limit", 1000),
                "rounds_truncated": snapshot.get("rounds_truncated", False),
                "rounds": snapshot.get("rounds", []),
            },
        )

    def read_session(self, account_id: str, session_id: str, *, include_rounds: bool = False) -> dict:
        tool_calls = self._store.read_json(
            self._tool_calls_path(account_id, session_id),
            {
                "summary": _empty_summary(),
                "tools": [],
            },
        )
        result = {
            "ok": True,
            "source": "internal",
            "filters": {"session_id": session_id, "user_id": None, "start_time": None, "end_time": None, "pipeline": None},
            "summary": tool_calls.get("summary", _empty_summary()),
            "tools": tool_calls.get("tools", []),
            "sessions": [{
                "session_id": session_id,
                "user_id": tool_calls.get("user_id", ""),
                "llm_tool_rounds": tool_calls.get("summary", {}).get("llm_tool_rounds", 0),
                "tool_calls": tool_calls.get("summary", {}).get("tool_calls", 0),
            }],
        }
        if include_rounds:
            rounds = self._store.read_json(
                self._rounds_path(account_id, session_id),
                {"rounds": [], "rounds_limit": 1000, "rounds_truncated": False},
            )
            result["rounds"] = rounds.get("rounds", [])
            result["rounds_limit"] = rounds.get("rounds_limit", 1000)
            result["rounds_truncated"] = rounds.get("rounds_truncated", False)
        return result

    def _tool_calls_path(self, account_id: str, session_id: str) -> str:
        return self._store._join("accounts", account_id, "sessions", session_id, "tool_usage", "tool_calls.json")

    def _rounds_path(self, account_id: str, session_id: str) -> str:
        return self._store._join("accounts", account_id, "sessions", session_id, "tool_usage", "rounds.json")

    @staticmethod
    def _account_from_snapshot(snapshot: dict) -> str:
        rounds = snapshot.get("rounds") or []
        if rounds:
            return str(rounds[0].get("account_id") or "")
        return ""
