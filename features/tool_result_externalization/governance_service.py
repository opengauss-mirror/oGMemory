"""Lifecycle coordination for governed tool-result Artifacts."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re as _re
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Callable
from uuid import uuid4

from core.models import RequestContext
from features.tool_result_externalization.store_base import (
    ToolResultExternalizationDisabledError,
)

logger = logging.getLogger("ogmem.service")

_TOOL_RESULT_GOVERNANCE_FALLBACK_STATE = "tool_result_governance_raw_fallbacks"
_MAX_TOOL_RESULT_GOVERNANCE_FALLBACK_SOURCE_IDS = 8_192


@dataclass(frozen=True)
class ComposeGovernanceState:
    params: dict
    messages: list
    ctx: RequestContext | None
    ledger_source_groups: tuple[frozenset[str], ...] = ()
    ledger_hot_seen_artifacts: dict[str, str] | None = None
    fallback_source_ids: frozenset[str] = frozenset()
    cache_delay_decision: Any = None


@dataclass(frozen=True)
class AfterTurnGovernanceState:
    params: dict
    new_messages: list
    semantic_messages: list | None
    ledger_registration: Any = None
    fallback_original: bool = False


@dataclass(frozen=True)
class AfterTurnGovernanceResult:
    failed: bool = False
    fallback_token: str = ""
    source_message_ids: tuple[str, ...] = ()


class ToolResultGovernanceService:
    """Coordinate tool-result governance across Session lifecycle entry points."""

    def __init__(
        self,
        *,
        config,
        externalizer,
        store_provider: Callable[[], Any],
        turn_ledger,
        io_runner,
        source_attestation_key: bytes,
        session_manager_provider: Callable[[], Any],
        context_builder: Callable[[dict], RequestContext],
    ) -> None:
        self._cfg = config
        self._tool_result_externalizer = externalizer
        self._tool_result_store_provider = store_provider
        self._tool_result_turn_ledger = turn_ledger
        self._tool_result_io_runner = io_runner
        self._tool_result_source_attestation_key = source_attestation_key
        self._session_manager_provider = session_manager_provider
        self._context_builder = context_builder
        self._tool_result_cache_delay = None

    def initialize_cache_delay(self) -> None:
        """Build the independently gated lazy TTL controller."""
        if not getattr(self._cfg, "tool_result_cache_delay_enabled", False):
            return

        from features.tool_result_externalization.externalizer import (
            ToolResultExternalizer,
        )
        from features.tool_result_externalization.normalizer import (
            ToolResultNormalizer,
        )
        from features.tool_result_externalization.policy import ToolResultPolicy
        from features.tool_result_externalization.synopsis import (
            ToolResultSynopsisBuilder,
        )
        from features.tool_result_optimization.cache_delay import (
            ToolResultCacheDelay,
        )

        store = self._tool_result_store

        def externalizer(threshold: int) -> ToolResultExternalizer:
            return ToolResultExternalizer(
                ToolResultNormalizer(),
                ToolResultPolicy(threshold),
                ToolResultSynopsisBuilder(
                    self._cfg.tool_result_synopsis_max_chars
                ),
                store,
                retry_attempts=self._cfg.tool_result_retry_attempts,
                write_timeout_ms=self._cfg.tool_result_write_timeout_ms,
                summary_per_artifact_chars=(
                    self._cfg.tool_result_synopsis_max_chars
                ),
                summary_per_turn_chars=(
                    self._cfg.tool_result_stub_summary_turn_chars
                ),
                source_attestation_key=self._tool_result_source_attestation_key,
                io_runner=self._tool_result_io_runner,
            )

        self._tool_result_cache_delay = ToolResultCacheDelay(
            persistence_externalizer=externalizer(
                self._cfg.tool_result_cache_delay_chars
            ),
            replay_externalizer=externalizer(self._cfg.tool_result_single_chars),
            store=store,
            delayed_result_chars=self._cfg.tool_result_cache_delay_chars,
            immediate_result_chars=self._cfg.tool_result_single_chars,
            idle_ttl_seconds=self._cfg.tool_result_cache_delay_idle_ttl_seconds,
            cache_hit_price_ratio=self._cfg.tool_result_cache_hit_price_ratio,
        )

    def get_session_manager(self):
        return self._session_manager_provider()

    @property
    def _tool_result_store(self):
        return self._tool_result_store_provider()

    def build_context(self, params: dict) -> RequestContext:
        return self._context_builder(params)

    def validate_fallback_token(
        self,
        token,
        params: dict,
        ctx: RequestContext,
        session_id: str,
    ) -> bool:
        return self._validate_tool_result_governance_fallback_token(
            token,
            params,
            ctx,
            session_id,
        )

    def finalize_projection(
        self,
        messages: list,
        *,
        state: ComposeGovernanceState | None = None,
        ctx: RequestContext | None = None,
    ) -> list:
        if state is None:
            if self._tool_result_externalizer is None:
                return messages
            return self._strip_tool_result_private_sources(messages)
        return self._finalize_tool_result_cache_projection(
            messages,
            session_id=(
                state.params.get("sessionId")
                or getattr(ctx or state.ctx, "session_id", "")
            ),
            ctx=ctx or state.ctx,
            decision=state.cache_delay_decision,
            ledger_hot_seen_artifacts=state.ledger_hot_seen_artifacts,
        )

    def prepare_compose(
        self,
        params: dict,
        messages: list,
    ) -> ComposeGovernanceState:
        """Project and govern one model-facing transcript before composition."""
        ctx = params.get("_ctx")
        original_message_count = len(messages)
        messages, _original_indices, retained_prefix_counts = (
            self._project_tool_result_messages(
                messages,
                ctx=ctx,
                session_id=(
                    params.get("sessionId")
                    or getattr(ctx, "session_id", None)
                ),
                drop_ogmem=True,
            )
        )
        if (
            len(messages) != original_message_count
            and "prePromptMessageCount" in params
        ):
            params = {
                **params,
                "prePromptMessageCount": self._map_projected_message_boundary(
                    retained_prefix_counts,
                    params["prePromptMessageCount"],
                ),
            }

        ledger_source_groups: tuple[frozenset[str], ...] = ()
        ledger_hot_seen_artifacts: dict[str, str] = {}
        fallback_source_ids: frozenset[str] = frozenset()
        cache_delay_decision = None
        try:
            ctx = ctx or self.build_context(params)
            session_id = params.get("sessionId") or ctx.session_id
            messages = self._strip_unverified_semantic_sources(messages)
            restored_turns = self._restore_user_message_tool_turns(
                messages,
                ctx=ctx,
                session_id=session_id,
            )
            if restored_turns is not None:
                messages = restored_turns.messages
                ledger_source_groups = restored_turns.source_groups
                ledger_hot_seen_artifacts = restored_turns.hot_seen_artifacts
            fallback_source_ids = self._tool_result_governance_fallback_source_ids(
                session_id,
                ctx,
            )
            cache_delay_decision = self._prepare_tool_result_cache_delay(
                session_id,
                ctx,
            )
            messages = self._govern_model_facing_messages(
                messages,
                params=params,
                ctx=ctx,
                session_id=session_id,
                validated_source_groups=ledger_source_groups,
                fallback_source_ids=fallback_source_ids,
            )
            messages = self._apply_tool_result_cold_view(
                messages,
                session_id=session_id,
                ctx=ctx,
                decision=cache_delay_decision,
                ledger_hot_seen_artifacts=ledger_hot_seen_artifacts,
                ranking_prompt=params.get("prompt", ""),
            )
        except Exception as exc:
            logger.warning("compose tool-result governance unavailable: %s", exc)
            for message in messages:
                if isinstance(message, dict):
                    message.pop("_ogmem_source", None)
        return ComposeGovernanceState(
            params=params,
            messages=messages,
            ctx=ctx,
            ledger_source_groups=ledger_source_groups,
            ledger_hot_seen_artifacts=ledger_hot_seen_artifacts,
            fallback_source_ids=fallback_source_ids,
            cache_delay_decision=cache_delay_decision,
        )

    def finalize_compose(
        self,
        messages: list,
        *,
        state: ComposeGovernanceState,
        ctx: RequestContext,
    ) -> list:
        """Run the final safety scan and remove private transport fields."""
        try:
            messages = self._govern_model_facing_messages(
                messages,
                params=state.params,
                ctx=ctx,
                session_id=state.params.get("sessionId") or ctx.session_id,
                validated_source_groups=state.ledger_source_groups,
                fallback_source_ids=state.fallback_source_ids,
            )
        except Exception as exc:
            logger.warning("compose final tool-result scan failed open: %s", exc)
        return self._finalize_tool_result_cache_projection(
            messages,
            session_id=state.params.get("sessionId") or ctx.session_id,
            ctx=ctx,
            decision=state.cache_delay_decision,
            ledger_hot_seen_artifacts=state.ledger_hot_seen_artifacts,
        )

    def prepare_after_turn(
        self,
        params: dict,
        messages: list,
        *,
        pre_prompt_count: int,
        ctx: RequestContext,
        session_id: str,
    ) -> AfterTurnGovernanceState:
        """Validate and normalize one complete user-message Turn."""
        fallback_token = params.get("toolResultGovernanceFallbackToken")
        fallback_original = bool(fallback_token)
        if fallback_original and not self._validate_tool_result_governance_fallback_token(
            fallback_token,
            params,
            ctx,
            session_id,
        ):
            raise ValueError("invalid tool-result governance fallback token")

        (
            stripped_messages,
            _retained_indices,
            retained_prefix_counts,
            projection_changed,
        ) = self._strip_internal_artifact_list_roundtrips(
            messages,
            ctx=ctx,
            session_id=session_id,
        )
        if projection_changed:
            pre_prompt_count = self._map_projected_message_boundary(
                retained_prefix_counts,
                pre_prompt_count,
            )
            params = {**params, "prePromptMessageCount": pre_prompt_count}

        new_messages = stripped_messages[pre_prompt_count:]
        self._tool_result_semantic_mode(params)
        ledger_registration = self._register_user_message_tool_turn(
            new_messages,
            start_boundary=pre_prompt_count,
            ctx=ctx,
            session_id=session_id,
        )
        semantic_messages = None
        if ledger_registration is not None:
            semantic_messages = ledger_registration.messages
            new_messages = ledger_registration.messages
            if (
                not ledger_registration.changed
                and ledger_registration.entry.get("governance_complete") is True
            ):
                new_messages = []
        return AfterTurnGovernanceState(
            params=params,
            new_messages=new_messages,
            semantic_messages=semantic_messages,
            ledger_registration=ledger_registration,
            fallback_original=fallback_original,
        )

    def persist_after_turn(
        self,
        state: AfterTurnGovernanceState,
        *,
        buf,
        ctx: RequestContext,
        session_id: str,
    ) -> AfterTurnGovernanceResult:
        """Persist governed Turn messages and update Ledger state atomically."""
        from extraction.tool_collector import (
            parse_message_parts,
            update_buffer_stats,
        )

        buf.begin_operation()
        try:
            fallback_reuse = []
            if state.semantic_messages is None:
                with buf.extraction_lock:
                    fallback_reuse = list(
                        buf.messages[buf.extraction_watermark:]
                    )
            governance_messages = (
                self._mark_validated_semantic_sources(state.semantic_messages)
                if state.semantic_messages is not None
                else self._attach_server_fallback_sources(
                    self._strip_unverified_semantic_sources(state.new_messages),
                    reuse_from=fallback_reuse,
                )
            )
            governed = self._govern_request_messages(
                governance_messages,
                ctx=ctx,
                session_id=session_id,
                persistence=True,
                fallback_original=state.fallback_original,
            )
            if governed is None:
                return AfterTurnGovernanceResult()

            failed_count = int(getattr(governed, "failed_count", 0) or 0)
            fallback_applied = bool(
                getattr(governed, "fallback_original", False)
            )
            self._upsert_governed_messages(
                session_id,
                governed,
                ctx,
                tail_fallback=True,
            )
            if fallback_applied:
                self._record_tool_result_governance_fallback(
                    buf,
                    governed.source_message_ids,
                )
            failed = failed_count > 0
            registration = state.ledger_registration
            if registration is not None and not failed_count:
                if self._mark_user_message_turn_governed(
                    registration,
                    ctx=ctx,
                    session_id=session_id,
                ):
                    if not fallback_applied:
                        self._record_user_message_turn_artifacts(
                            registration,
                            ctx=ctx,
                            session_id=session_id,
                        )
                else:
                    failed = not fallback_applied
            if not failed:
                for message in state.new_messages:
                    parsed = parse_message_parts(
                        message.get("content", "")
                    )
                    if parsed["tool_calls"]:
                        update_buffer_stats(
                            buf.usage_stats,
                            parsed["tool_calls"],
                            session_id=session_id,
                        )
            return AfterTurnGovernanceResult(
                failed=failed,
                source_message_ids=tuple(governed.source_message_ids),
                fallback_token=(
                    self._issue_tool_result_governance_fallback_token(
                        state.params,
                        ctx,
                        session_id,
                    )
                    if failed
                    else ""
                ),
            )
        finally:
            buf.end_operation()


    def defer_user_message_ingest(
        self,
        params: dict,
        messages: list,
        *,
        ctx: RequestContext,
        session_id: str,
    ) -> dict:
        """Persist only non-tool messages until terminal afterTurn supplies identity."""
        from extraction.tool_collector import parse_message_parts

        self._tool_result_semantic_mode(params)
        normalized = self._tool_result_externalizer.normalizer.normalize(messages)
        deferred_indexes = {
            item.message_index for item in normalized if item.tool_results
        }
        mgr = self.get_session_manager()
        count = 0
        for message_index, message in enumerate(messages):
            if message_index in deferred_indexes:
                continue
            role = (
                message.get("role", "user")
                if isinstance(message, dict)
                else getattr(message, "role", "user")
            )
            content_raw = (
                message.get("content", "")
                if isinstance(message, dict)
                else getattr(message, "content", "")
            )
            content = parse_message_parts(content_raw).get("text", "")
            created_at = (
                message.get("created_at")
                if isinstance(message, dict)
                else getattr(message, "created_at", None)
            )
            if content:
                mgr.add_message(
                    session_id,
                    role,
                    content,
                    ctx,
                    created_at=created_at,
                )
                count += 1
        return {
            "ingested": True,
            "count": count,
            "externalized": 0,
            "governance_deferred": True,
            "deferred_tool_result_messages": len(deferred_indexes),
        }

    def _govern_request_messages(
        self,
        messages: list,
        *,
        ctx: RequestContext,
        session_id: str,
        persistence: bool,
        ranking_prompt: str = "",
        fallback_original: bool = False,
    ):
        """Apply the single provider-neutral governance path to a request copy."""
        if self._tool_result_externalizer is None:
            return None
        # Binding the session through SessionManager provides the same complete
        # account/user/agent/session visibility check as the ordinary buffer.
        self.get_session_manager().get_or_create(session_id, ctx=ctx)
        if fallback_original:
            return self._tool_result_externalizer.govern_messages(
                messages,
                ctx=ctx,
                session_id=session_id,
                persistence=persistence,
                ranking_prompt=ranking_prompt,
                fallback_original=True,
            )
        if persistence and hasattr(self._tool_result_store, "cleanup_stale_writes"):
            try:
                self._tool_result_store.cleanup_stale_writes(
                    session_id,
                    ctx,
                    self._cfg.tool_result_artifact_temp_grace_seconds,
                )
            except Exception as exc:
                logger.debug("tool-result stale marker cleanup skipped: %s", exc)
        if persistence and self._tool_result_cache_delay is not None:
            return self._tool_result_cache_delay.govern_persistent_messages(
                messages,
                ctx=ctx,
                session_id=session_id,
                ranking_prompt=ranking_prompt,
            )
        if not persistence and self._tool_result_cache_delay is not None:
            return self._tool_result_cache_delay.govern_replay_messages(
                messages,
                ctx=ctx,
                session_id=session_id,
                ranking_prompt=ranking_prompt,
            )
        return self._tool_result_externalizer.govern_messages(
            messages,
            ctx=ctx,
            session_id=session_id,
            persistence=persistence,
            ranking_prompt=ranking_prompt,
        )

    def _tool_result_semantic_mode(self, params: dict) -> str:
        """Validate the explicit, currently supported semantic Turn lifecycle."""
        configured = getattr(
            self._cfg,
            "tool_result_semantic_turn_mode",
            "user_message",
        )
        if configured != "user_message":
            raise ValueError(
                "tool_result_semantic_turn_mode currently supports only "
                "'user_message'"
            )
        requested = params.get("semanticTurnMode")
        if requested not in (None, "", "user_message"):
            raise ValueError(
                "semanticTurnMode currently supports only 'user_message'"
            )
        return "user_message"

    def _tool_result_governance_fallback_payload(
        self,
        params: dict,
        ctx: RequestContext,
        session_id: str,
    ) -> bytes:
        """Bind one raw-persistence fallback to the exact failed user Turn."""
        payload = {
            "schema_version": 1,
            "owner": {
                "account_id": ctx.account_id,
                "user_id": ctx.user_id,
                "agent_id": ctx.agent_id,
                "session_id": session_id,
            },
            "messages": params.get("messages", []),
            "pre_prompt_message_count": params.get("prePromptMessageCount", 0),
            "semantic_turn_mode": self._tool_result_semantic_mode(params),
        }
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    def _issue_tool_result_governance_fallback_token(
        self,
        params: dict,
        ctx: RequestContext,
        session_id: str,
    ) -> str:
        return hmac.new(
            self._tool_result_source_attestation_key,
            b"tool-result-governance-fallback-v1\0"
            + self._tool_result_governance_fallback_payload(params, ctx, session_id),
            hashlib.sha256,
        ).hexdigest()

    def _validate_tool_result_governance_fallback_token(
        self,
        token,
        params: dict,
        ctx: RequestContext,
        session_id: str,
    ) -> bool:
        if not isinstance(token, str) or _re.fullmatch(r"[a-f0-9]{64}", token) is None:
            return False
        expected = self._issue_tool_result_governance_fallback_token(
            params,
            ctx,
            session_id,
        )
        return hmac.compare_digest(token, expected)

    def _register_user_message_tool_turn(
        self,
        messages: list,
        *,
        start_boundary: int,
        ctx: RequestContext,
        session_id: str,
    ):
        """Persist one complete user-message Turn before issuing strong ids."""
        ledger = self._tool_result_turn_ledger
        if ledger is None:
            return None
        mgr = self.get_session_manager()
        buf = mgr.get_or_create(session_id, ctx=ctx)
        with buf.semantic_turn_lock:
            previous = deepcopy(buf.runtime_extensions.get(ledger.STATE_KEY))
            registration = ledger.register_user_message_turn(
                previous,
                messages,
                start_boundary=start_boundary,
                ctx=ctx,
                session_id=session_id,
            )
            if registration is None:
                return None
            if not registration.changed:
                return registration
            buf.runtime_extensions[ledger.STATE_KEY] = registration.state
            if mgr.save_session_state(session_id, ctx):
                return registration
            if previous is None:
                buf.runtime_extensions.pop(ledger.STATE_KEY, None)
            else:
                buf.runtime_extensions[ledger.STATE_KEY] = previous
            logger.warning(
                "tool-result Turn Ledger unavailable; using safe fallback identity "
                "session=%s",
                session_id[:8],
            )
            return None

    def _restore_user_message_tool_turns(
        self,
        messages: list,
        *,
        ctx: RequestContext,
        session_id: str,
    ):
        """Restore only uniquely matching, owner-bound Ledger identities."""
        ledger = self._tool_result_turn_ledger
        if ledger is None:
            return None
        mgr = self.get_session_manager()
        buf = mgr.get_or_create(session_id, ctx=ctx)
        with buf.semantic_turn_lock:
            raw = deepcopy(buf.runtime_extensions.get(ledger.STATE_KEY))
            return ledger.restore_completed_turns(
                raw,
                messages,
                ctx=ctx,
                session_id=session_id,
            )

    def _record_user_message_turn_artifacts(
        self,
        registration,
        *,
        ctx: RequestContext,
        session_id: str,
    ) -> None:
        """Record verified Artifact ids and initial hot visibility best-effort."""
        ledger = self._tool_result_turn_ledger
        if ledger is None or registration is None or self._tool_result_store is None:
            return
        from features.tool_result_externalization.externalizer import (
            ToolResultExternalizer,
        )
        from features.tool_result_externalization.normalizer import (
            canonical_tool_call_id,
        )

        entry = registration.entry
        artifacts: dict[tuple[int, str, int], tuple[str, str]] = {}
        controller = self._tool_result_cache_delay
        observed_at = ""
        if controller is not None:
            try:
                observed_at = controller.clock().astimezone(UTC).isoformat()
            except Exception:
                observed_at = datetime.now(UTC).isoformat()
        deadline = time.monotonic() + self._tool_result_externalizer.write_timeout_ms / 1000
        for result in entry.get("results", []):
            artifact_id = ToolResultExternalizer.strong_artifact_id(
                ctx.account_id,
                ctx.user_id,
                ctx.agent_id,
                session_id,
                entry["semantic_turn_id"],
                result["source_message_id"],
                result.get("tool_call_key", ""),
                result["part_path"],
                str(result["occurrence_index"]),
            )
            try:
                metadata, output = self._tool_result_io_runner.run(
                    lambda artifact_id=artifact_id: self._tool_result_store.read_all(
                        artifact_id,
                        session_id,
                        ctx,
                    ),
                    deadline=deadline,
                    operation_key=session_id,
                )
            except Exception:
                continue
            if not (
                metadata.identity_strength == "strong"
                and metadata.turn_id == entry["semantic_turn_id"]
                and metadata.source_message_id == result["source_message_id"]
                and canonical_tool_call_id(metadata.tool_call_id)
                == result.get("tool_call_key", "")
                and metadata.part_path == result["part_path"]
                and metadata.occurrence_index == result["occurrence_index"]
                and metadata.original_chars == result["original_chars"]
                and metadata.sha256 == result["content_sha256"]
                and hashlib.sha256(output.encode("utf-8")).hexdigest()
                == result["content_sha256"]
            ):
                continue
            hot_seen_at = ""
            if (
                controller is not None
                and controller.delayed_result_chars <= result["original_chars"]
                < controller.immediate_result_chars
            ):
                hot_seen_at = observed_at
            artifacts[(
                result["message_ordinal"],
                result["part_path"],
                result["occurrence_index"],
            )] = (artifact_id, hot_seen_at)
        if not artifacts:
            return

        mgr = self.get_session_manager()
        buf = mgr.get_or_create(session_id, ctx=ctx)
        # Fixed lock order for operations touching both extension states:
        # semantic Turn Ledger first, cache-delay state second.
        with buf.semantic_turn_lock:
            with buf.tool_result_cache_delay_lock:
                previous_ledger = deepcopy(
                    buf.runtime_extensions.get(ledger.STATE_KEY)
                )
                previous_cache = None
                updated_ledger = ledger.update_entry_results(
                    previous_ledger,
                    turn_key=entry["turn_key"],
                    artifacts=artifacts,
                    ctx=ctx,
                    session_id=session_id,
                )
                buf.runtime_extensions[ledger.STATE_KEY] = updated_ledger
                if controller is not None:
                    from features.tool_result_optimization.cache_delay import (
                        CacheDelayState,
                    )

                    previous_cache = deepcopy(
                        buf.runtime_extensions.get(controller.STATE_KEY)
                    )
                    cache_state = CacheDelayState.from_mapping(previous_cache)
                    ledger_seen = ledger.hot_seen_artifacts(
                        updated_ledger,
                        ctx=ctx,
                        session_id=session_id,
                    )
                    seen = {
                        artifact_id: seen_at
                        for artifact_id, seen_at in cache_state.hot_seen_artifacts.items()
                        if artifact_id not in ledger_seen
                    }
                    anchored_cache_state = controller.anchor_turn_completion(
                        cache_state.to_dict(),
                        observed_at,
                    )
                    buf.runtime_extensions[controller.STATE_KEY] = CacheDelayState(
                        schema_version=cache_state.schema_version,
                        last_model_request_at=(
                            anchored_cache_state.last_model_request_at
                        ),
                        cold_cutover_before=(
                            anchored_cache_state.cold_cutover_before
                        ),
                        hot_seen_artifacts=seen,
                    ).to_dict()
                if mgr.save_session_state(session_id, ctx):
                    return
                if previous_ledger is None:
                    buf.runtime_extensions.pop(ledger.STATE_KEY, None)
                else:
                    buf.runtime_extensions[ledger.STATE_KEY] = previous_ledger
                if controller is not None:
                    if previous_cache is None:
                        buf.runtime_extensions.pop(controller.STATE_KEY, None)
                    else:
                        buf.runtime_extensions[controller.STATE_KEY] = previous_cache
            logger.warning(
                "tool-result Turn Ledger Artifact update was not durable; "
                "compose will re-verify from the store session=%s",
                session_id[:8],
            )

    def _mark_user_message_turn_governed(
        self,
        registration,
        *,
        ctx: RequestContext,
        session_id: str,
    ) -> bool:
        """Durably close the exact-retry ingest boundary for one Ledger Turn."""
        ledger = self._tool_result_turn_ledger
        if ledger is None or registration is None:
            return False
        mgr = self.get_session_manager()
        buf = mgr.get_or_create(session_id, ctx=ctx)
        with buf.semantic_turn_lock:
            previous = deepcopy(buf.runtime_extensions.get(ledger.STATE_KEY))
            updated, changed = ledger.mark_governance_complete(
                previous,
                turn_key=registration.entry["turn_key"],
                ctx=ctx,
                session_id=session_id,
            )
            entry = next(
                (
                    row for row in updated.get("completed_turns", [])
                    if row.get("turn_key") == registration.entry["turn_key"]
                ),
                None,
            )
            if entry is None or entry.get("governance_complete") is not True:
                return False
            if not changed:
                return True
            buf.runtime_extensions[ledger.STATE_KEY] = updated
            if mgr.save_session_state(session_id, ctx):
                return True
            if previous is None:
                buf.runtime_extensions.pop(ledger.STATE_KEY, None)
            else:
                buf.runtime_extensions[ledger.STATE_KEY] = previous
            logger.warning(
                "tool-result Turn Ledger governance marker was not durable; "
                "afterTurn remains retryable session=%s",
                session_id[:8],
            )
            return False

    @staticmethod
    def _strip_unverified_semantic_sources(messages: list) -> list:
        """Drop every caller-asserted private identity before fallback governance.

        A normalizer treats a source/turn pair as stable, so retaining even an
        envelope without ``semantic_turn_attested`` would let an arbitrary HTTP
        caller manufacture a strong Artifact identity. Turn-Ledger matches
        bypass this helper and are reattached only after complete server-side
        validation of the user-message Turn.
        """
        cleaned = deepcopy(messages)
        for message in cleaned:
            if not isinstance(message, dict):
                continue
            message.pop("_ogmem_source", None)
            message.pop("_ogmem_semantic_source_validated", None)
        return cleaned

    @staticmethod
    def _attach_server_fallback_sources(
        messages: list,
        *,
        reuse_from: list | None = None,
    ) -> list:
        """Bind fallback persistence to fresh server-issued Buffer ids.

        These ids do not make an Artifact strong and are never accepted from a
        caller. They only prove to later internal safety scans that a mixed
        ``stub + inline`` projection was generated for this exact Buffer
        message, avoiding an Artifact-of-Artifact while keeping TTL disabled.
        """
        from features.tool_result_externalization.externalizer import (
            render_safe_message_content,
        )

        bound = deepcopy(messages)
        candidates = list(reuse_from or [])
        consumed: set[str] = set()
        for message in bound:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "user"))
            original_content = render_safe_message_content(message)
            matched_id = ""
            for existing in candidates:
                existing_id = str(getattr(existing, "id", "") or "")
                if (
                    existing_id
                    and existing_id not in consumed
                    and getattr(existing, "role", None) == role
                    and getattr(existing, "content", None) == original_content
                ):
                    matched_id = existing_id
                    consumed.add(existing_id)
                    break
            message["_ogmem_source"] = {
                "source_message_id": matched_id or f"msg_{uuid4().hex[:12]}",
                "identity_strength": "fallback",
                "server_fallback_allocated": True,
            }
            message.pop("_ogmem_semantic_source_validated", None)
        return bound

    @staticmethod
    def _normalized_tool_result_governance_fallback_ids(raw_state) -> list[str]:
        if not isinstance(raw_state, dict) or raw_state.get("schema_version") != 1:
            return []
        source_ids = raw_state.get("source_message_ids")
        if not isinstance(source_ids, list):
            return []
        normalized = []
        seen = set()
        for value in source_ids:
            if (
                not isinstance(value, str)
                or not value
                or len(value) > 512
                or value in seen
            ):
                continue
            normalized.append(value)
            seen.add(value)
        return normalized[-_MAX_TOOL_RESULT_GOVERNANCE_FALLBACK_SOURCE_IDS:]

    def _record_tool_result_governance_fallback(
        self,
        buf,
        source_message_ids,
    ) -> None:
        """Exclude only one failed user Turn from future governance attempts."""
        new_ids = [
            value for value in source_message_ids
            if isinstance(value, str) and value and len(value) <= 512
        ]
        if not new_ids:
            return
        with buf.extraction_lock:
            existing = self._normalized_tool_result_governance_fallback_ids(
                buf.runtime_extensions.get(
                    _TOOL_RESULT_GOVERNANCE_FALLBACK_STATE
                )
            )
            combined = list(dict.fromkeys([*existing, *new_ids]))
            buf.runtime_extensions[_TOOL_RESULT_GOVERNANCE_FALLBACK_STATE] = {
                "schema_version": 1,
                "source_message_ids": combined[
                    -_MAX_TOOL_RESULT_GOVERNANCE_FALLBACK_SOURCE_IDS:
                ],
            }

    def _tool_result_governance_fallback_source_ids(
        self,
        session_id: str,
        ctx: RequestContext,
    ) -> frozenset[str]:
        buf = self.get_session_manager().get_or_create(session_id, ctx=ctx)
        with buf.extraction_lock:
            return frozenset(
                self._normalized_tool_result_governance_fallback_ids(
                    buf.runtime_extensions.get(
                        _TOOL_RESULT_GOVERNANCE_FALLBACK_STATE
                    )
                )
            )

    @staticmethod
    def _mark_validated_semantic_sources(messages: list) -> list:
        marked = deepcopy(messages)
        for message in marked:
            if isinstance(message, dict):
                message["_ogmem_semantic_source_validated"] = True
        return marked

    def _govern_model_facing_messages(
        self,
        messages: list,
        *,
        params: dict,
        ctx: RequestContext,
        session_id: str,
        validated_source_ids: frozenset[str] | None = None,
        validated_source_groups: tuple[frozenset[str], ...] | None = None,
        fallback_source_ids: frozenset[str] | None = None,
    ) -> list:
        """Govern history individually and the current semantic turn as one budget."""
        groups = list(validated_source_groups or ())
        if validated_source_ids is not None:
            groups.append(validated_source_ids)
        groups = [group for group in groups if group]
        fallback_ids = frozenset(fallback_source_ids or ())
        if fallback_ids:
            output = []
            cursor = 0
            while cursor < len(messages):
                source = (
                    messages[cursor].get("_ogmem_source")
                    if isinstance(messages[cursor], dict)
                    else None
                )
                source_id = (
                    source.get("source_message_id")
                    if isinstance(source, dict)
                    else None
                )
                bypass = source_id in fallback_ids
                end = cursor + 1
                while end < len(messages):
                    candidate_source = (
                        messages[end].get("_ogmem_source")
                        if isinstance(messages[end], dict)
                        else None
                    )
                    candidate_id = (
                        candidate_source.get("source_message_id")
                        if isinstance(candidate_source, dict)
                        else None
                    )
                    if (candidate_id in fallback_ids) != bypass:
                        break
                    end += 1
                segment = messages[cursor:end]
                if bypass:
                    output.extend(deepcopy(segment))
                else:
                    segment_ids = frozenset(
                        str(message.get("_ogmem_source", {}).get(
                            "source_message_id", ""
                        ))
                        for message in segment
                        if isinstance(message, dict)
                        and isinstance(message.get("_ogmem_source"), dict)
                    )
                    segment_groups = tuple(
                        frozenset(group & segment_ids)
                        for group in groups
                        if group & segment_ids
                    )
                    output.extend(self._govern_model_facing_messages(
                        segment,
                        params=params,
                        ctx=ctx,
                        session_id=session_id,
                        validated_source_groups=segment_groups,
                    ))
                cursor = end
            return output
        if not groups:
            governed = self._govern_request_messages(
                messages,
                ctx=ctx,
                session_id=session_id,
                persistence=False,
                ranking_prompt=params.get("prompt", ""),
            )
            return governed.messages if governed is not None else messages

        ranges: list[tuple[int, int]] = []
        for group in groups:
            semantic_indices = [
                index for index, message in enumerate(messages)
                if isinstance(message, dict)
                and isinstance(message.get("_ogmem_source"), dict)
                and message["_ogmem_source"].get("source_message_id") in group
            ]
            if not semantic_indices:
                continue
            first, last = semantic_indices[0], semantic_indices[-1]
            if semantic_indices != list(range(first, last + 1)):
                continue
            ranges.append((first, last + 1))
        ranges.sort()
        if not ranges or any(
            ranges[index][0] < ranges[index - 1][1]
            for index in range(1, len(ranges))
        ):
            governed = self._govern_request_messages(
                messages,
                ctx=ctx,
                session_id=session_id,
                persistence=False,
                ranking_prompt=params.get("prompt", ""),
            )
            return governed.messages if governed is not None else messages

        output = []
        cursor = 0
        segments: list[list] = []
        for first, end in ranges:
            if cursor < first:
                segments.append(messages[cursor:first])
            segments.append(messages[first:end])
            cursor = end
        if cursor < len(messages):
            segments.append(messages[cursor:])
        for segment in segments:
            if not segment:
                continue
            governed = self._govern_request_messages(
                segment,
                ctx=ctx,
                session_id=session_id,
                persistence=False,
                ranking_prompt=params.get("prompt", ""),
            )
            output.extend(governed.messages if governed is not None else segment)
        return output

    def _project_tool_result_messages(
        self,
        messages: list,
        *,
        ctx: RequestContext | None = None,
        session_id: str | None = None,
        drop_ogmem: bool = False,
    ) -> tuple[list, tuple[int, ...], tuple[int, ...]]:
        """Project trusted synthetic removals while retaining original positions."""
        projected = []
        original_indices: list[int] = []
        retained_prefix_counts = [0]
        for original_index, message in enumerate(messages):
            if drop_ogmem and isinstance(message, dict) and message.get("_ogmem"):
                retained = []
            else:
                retained = [message]
            projected.extend(retained)
            original_indices.extend([original_index] * len(retained))
            retained_prefix_counts.append(len(projected))
        return (
            projected,
            tuple(original_indices),
            tuple(retained_prefix_counts),
        )

    @staticmethod
    def _map_projected_message_boundary(
        retained_prefix_counts: tuple[int, ...],
        boundary: int,
    ) -> int:
        if not isinstance(boundary, int):
            raise TypeError("message boundary must be an integer")
        original_count = max(0, len(retained_prefix_counts) - 1)
        clamped = min(max(0, boundary), original_count)
        return retained_prefix_counts[clamped]

    @staticmethod
    def _canonical_internal_tool_name(value) -> str:
        return str(value or "").rsplit("__", 1)[-1].casefold()

    def _strip_internal_artifact_list_roundtrips(
        self,
        messages: list,
        *,
        ctx: RequestContext,
        session_id: str,
    ) -> tuple[list, tuple[int, ...], tuple[int, ...], bool]:
        """Drop only uniquely paired, attested list navigation after terminal output."""
        identity = tuple(range(len(messages)))
        unchanged_prefix = tuple(range(len(messages) + 1))
        if self._tool_result_externalizer is None or not messages:
            return messages, identity, unchanged_prefix, False

        from features.tool_result_externalization.normalizer import (
            canonical_tool_call_id,
        )
        from features.tool_result_externalization.turn_ledger import (
            _terminal_assistant,
        )

        if (
            not isinstance(messages[-1], dict)
            or not _terminal_assistant(messages[-1])
        ):
            return messages, identity, unchanged_prefix, False

        copied = deepcopy(messages)
        all_calls: dict[str, list[tuple[int, str, int]]] = {}
        list_calls: dict[str, list[tuple[int, str, int]]] = {}
        call_types = {"tool_call", "toolcall", "tool_use", "tooluse"}

        def record_call(
            message_index: int,
            field: str,
            item_index: int,
            call_id,
            tool_name,
        ) -> None:
            key = canonical_tool_call_id(call_id)
            if not key:
                return
            location = (message_index, field, item_index)
            all_calls.setdefault(key, []).append(location)
            if self._canonical_internal_tool_name(tool_name) == (
                "list_tool_result_artifacts"
            ):
                list_calls.setdefault(key, []).append(location)

        for message_index, message in enumerate(copied):
            if not isinstance(message, dict) or str(
                message.get("role", "")
            ).casefold() != "assistant":
                continue
            content = message.get("content")
            if isinstance(content, list):
                for item_index, block in enumerate(content):
                    if not isinstance(block, dict):
                        continue
                    if str(block.get("type", "")).casefold() not in call_types:
                        continue
                    call_id = next((
                        block.get(key)
                        for key in (
                            "id", "tool_call_id", "toolCallId",
                            "tool_use_id", "toolUseId",
                        )
                        if block.get(key) not in (None, "")
                    ), None)
                    tool_name = next((
                        block.get(key)
                        for key in ("name", "tool_name", "toolName")
                        if block.get(key) not in (None, "")
                    ), None)
                    record_call(
                        message_index, "content", item_index, call_id, tool_name,
                    )
            for field in ("tool_calls", "toolCalls"):
                calls = message.get(field)
                if not isinstance(calls, list):
                    continue
                for item_index, call in enumerate(calls):
                    if not isinstance(call, dict):
                        continue
                    function = (
                        call.get("function")
                        if isinstance(call.get("function"), dict)
                        else {}
                    )
                    call_id = next((
                        call.get(key)
                        for key in (
                            "id", "tool_call_id", "toolCallId",
                            "tool_use_id", "toolUseId",
                        )
                        if call.get(key) not in (None, "")
                    ), None)
                    tool_name = next((
                        call.get(key)
                        for key in ("name", "tool_name", "toolName")
                        if call.get(key) not in (None, "")
                    ), None) or function.get("name")
                    record_call(
                        message_index, field, item_index, call_id, tool_name,
                    )

        if not list_calls:
            return messages, identity, unchanged_prefix, False

        normalized = self._tool_result_externalizer.normalizer.normalize(copied)
        all_results: dict[str, list[tuple[int, object]]] = {}
        trusted_list_results: dict[str, list[tuple[int, object]]] = {}
        for normalized_message in normalized:
            for part in normalized_message.tool_results:
                key = canonical_tool_call_id(part.tool_call_id)
                if not key:
                    continue
                occurrence = (normalized_message.message_index, part)
                all_results.setdefault(key, []).append(occurrence)
                if self._tool_result_externalizer.validates_artifact_list_result(
                    part,
                    ctx,
                    session_id,
                ):
                    trusted_list_results.setdefault(key, []).append(occurrence)

        call_removals: dict[int, dict[str, set[int]]] = {}
        result_block_removals: dict[int, set[int]] = {}
        whole_message_removals: set[int] = set()
        terminal_index = len(copied) - 1
        for key, candidate_calls in list_calls.items():
            candidates = trusted_list_results.get(key, [])
            if (
                len(candidate_calls) != 1
                or len(all_calls.get(key, [])) != 1
                or len(candidates) != 1
                or len(all_results.get(key, [])) != 1
            ):
                continue
            call_location = candidate_calls[0]
            result_message_index, result_part = candidates[0]
            if not (
                call_location[0] < result_message_index < terminal_index
            ):
                continue
            result_message = copied[result_message_index]
            role = str(result_message.get("role", "")).casefold()
            remove_whole_result = role in {
                "tool", "toolresult", "tool_result",
            }
            result_block_index = None
            payload_path = tuple(getattr(result_part, "payload_path", ()))
            if (
                not remove_whole_result
                and len(payload_path) >= 2
                and payload_path[0] == "content"
                and isinstance(payload_path[1], int)
            ):
                result_block_index = payload_path[1]
            if not remove_whole_result and result_block_index is None:
                continue
            call_removals.setdefault(call_location[0], {}).setdefault(
                call_location[1], set(),
            ).add(call_location[2])
            if remove_whole_result:
                whole_message_removals.add(result_message_index)
                continue
            result_block_removals.setdefault(
                result_message_index, set(),
            ).add(result_block_index)

        if not call_removals:
            return messages, identity, unchanged_prefix, False

        retained_messages = []
        retained_indices: list[int] = []
        retained_prefix_counts = [0]
        touched_indices = set(call_removals) | set(result_block_removals)
        for message_index, message in enumerate(copied):
            if message_index in whole_message_removals:
                retained_prefix_counts.append(len(retained_messages))
                continue
            for field, item_indices in call_removals.get(
                message_index, {},
            ).items():
                items = message.get(field)
                if isinstance(items, list):
                    for item_index in sorted(item_indices, reverse=True):
                        if 0 <= item_index < len(items):
                            items.pop(item_index)
                    if not items and field != "content":
                        message.pop(field, None)
            content = message.get("content")
            if isinstance(content, list):
                for item_index in sorted(
                    result_block_removals.get(message_index, set()),
                    reverse=True,
                ):
                    if 0 <= item_index < len(content):
                        content.pop(item_index)
            if message_index in touched_indices:
                content = message.get("content")
                has_content = (
                    isinstance(content, str) and bool(content.strip())
                ) or (isinstance(content, list) and bool(content))
                has_calls = any(
                    isinstance(message.get(field), list) and message[field]
                    for field in ("tool_calls", "toolCalls")
                )
                if not has_content and not has_calls:
                    retained_prefix_counts.append(len(retained_messages))
                    continue
            retained_messages.append(message)
            retained_indices.append(message_index)
            retained_prefix_counts.append(len(retained_messages))

        return (
            retained_messages,
            tuple(retained_indices),
            tuple(retained_prefix_counts),
            True,
        )

    def _prepare_tool_result_cache_delay(
        self,
        session_id: str,
        ctx: RequestContext,
    ):
        """Persist one lazy TTL transition; return None on any state failure."""
        controller = self._tool_result_cache_delay
        if controller is None:
            return None
        mgr = self.get_session_manager()
        buf = mgr.get_or_create(session_id, ctx=ctx)
        with buf.tool_result_cache_delay_lock:
            previous = deepcopy(
                buf.runtime_extensions.get(controller.STATE_KEY, {})
            )
            try:
                mgr.load_session_state(session_id, ctx)
                previous = deepcopy(
                    buf.runtime_extensions.get(controller.STATE_KEY, {})
                )
                decision = controller.advance(previous)
                buf.runtime_extensions[controller.STATE_KEY] = (
                    decision.state.to_dict()
                )
                if not mgr.save_session_state(session_id, ctx):
                    if previous:
                        buf.runtime_extensions[controller.STATE_KEY] = previous
                    else:
                        buf.runtime_extensions.pop(controller.STATE_KEY, None)
                    logger.warning(
                        "tool-result cache-delay state unavailable; preserving "
                        "hot view session=%s",
                        session_id[:8],
                    )
                    return None
                logger.debug(
                    "tool-result cache-delay session=%s cutoff=%s "
                    "transitioned=%s idle=%.3f hit_ratio=%.3f",
                    session_id[:8],
                    decision.active_cutoff,
                    decision.transitioned,
                    decision.observed_idle_seconds,
                    decision.cache_hit_price_ratio,
                )
                return decision
            except Exception as exc:
                if previous:
                    buf.runtime_extensions[controller.STATE_KEY] = previous
                else:
                    buf.runtime_extensions.pop(controller.STATE_KEY, None)
                logger.warning(
                    "tool-result cache-delay state failed open session=%s: %s",
                    session_id[:8],
                    exc,
                )
                return None

    def _apply_tool_result_cold_view(
        self,
        messages: list,
        *,
        session_id: str,
        ctx: RequestContext,
        decision,
        ledger_hot_seen_artifacts: dict[str, str] | None = None,
        ranking_prompt: str = "",
    ) -> list:
        """Apply a verified cold replay view or preserve the caller projection."""
        if (
            self._tool_result_cache_delay is None
            or decision is None
            or not decision.active_cutoff
        ):
            return messages
        try:
            hot_seen_artifacts = dict(decision.state.hot_seen_artifacts)
            hot_seen_artifacts.update(ledger_hot_seen_artifacts or {})
            rewritten = self._tool_result_cache_delay.rewrite_cold_view(
                messages,
                ctx=ctx,
                session_id=session_id,
                cutoff=decision.active_cutoff,
                hot_seen_artifacts=hot_seen_artifacts,
                ranking_prompt=ranking_prompt,
            )
            return rewritten.messages
        except Exception as exc:
            logger.warning(
                "tool-result cache-delay rewrite failed open session=%s: %s",
                session_id[:8],
                exc,
            )
            return messages

    def _record_tool_result_hot_view(
        self,
        messages: list,
        *,
        session_id: str,
        ctx: RequestContext,
        decision,
        ledger_hot_seen_artifacts: dict[str, str] | None = None,
    ) -> None:
        """Persist stable Artifacts that survived into the final raw projection."""
        controller = self._tool_result_cache_delay
        if controller is None or decision is None:
            return
        mgr = self.get_session_manager()
        buf = mgr.get_or_create(session_id, ctx=ctx)
        with buf.tool_result_cache_delay_lock:
            previous = deepcopy(
                buf.runtime_extensions.get(controller.STATE_KEY, {})
            )
            try:
                observation = controller.observe_hot_view(
                    messages,
                    state=previous,
                    ctx=ctx,
                    session_id=session_id,
                    observed_at=decision.state.last_model_request_at,
                    ledger_hot_seen_artifacts=ledger_hot_seen_artifacts,
                )
                if observation.overflow_count:
                    logger.warning(
                        "tool-result cache-delay hot-seen capacity reached; "
                        "new Artifacts remain hot session=%s overflow=%d",
                        session_id[:8],
                        observation.overflow_count,
                    )
                if not (observation.observed_count or observation.pruned_count):
                    return
                buf.runtime_extensions[controller.STATE_KEY] = (
                    observation.state.to_dict()
                )
                if mgr.save_session_state(session_id, ctx):
                    return
            except Exception as exc:
                logger.warning(
                    "tool-result cache-delay hot-seen state failed open "
                    "session=%s: %s",
                    session_id[:8],
                    exc,
                )
            if previous:
                buf.runtime_extensions[controller.STATE_KEY] = previous
            else:
                buf.runtime_extensions.pop(controller.STATE_KEY, None)
            logger.warning(
                "tool-result cache-delay hot-seen state unavailable; "
                "new Artifacts remain hot session=%s",
                session_id[:8],
            )

    @staticmethod
    def _strip_tool_result_private_sources(messages: list) -> list:
        for message in messages:
            if isinstance(message, dict):
                message.pop("_ogmem_source", None)
                message.pop("_ogmem_semantic_source_validated", None)
        return messages

    def _finalize_tool_result_cache_projection(
        self,
        messages: list,
        *,
        session_id: str,
        ctx: RequestContext | None,
        decision,
        ledger_hot_seen_artifacts: dict[str, str] | None = None,
    ) -> list:
        if self._tool_result_externalizer is None:
            return messages
        if ctx is not None:
            self._record_tool_result_hot_view(
                messages,
                session_id=session_id,
                ctx=ctx,
                decision=decision,
                ledger_hot_seen_artifacts=ledger_hot_seen_artifacts,
            )
        return self._strip_tool_result_private_sources(messages)

    def _upsert_governed_messages(
        self,
        session_id: str,
        governed,
        ctx: RequestContext,
        *,
        tail_fallback: bool,
    ) -> int:
        """Write governed flat projections using stable ids or one-to-one tail matching."""
        from features.tool_result_externalization.externalizer import (
            render_safe_message_content,
        )

        mgr = self.get_session_manager()
        buf = mgr.get_or_create(session_id, ctx=ctx)
        fallback_original = bool(
            getattr(governed, "fallback_original", False)
        )
        validation_deadline = (
            time.monotonic()
            + self._tool_result_externalizer.write_timeout_ms / 1000
        )
        preserved_ids = set(getattr(governed, "preserved_message_ids", ()) or ())
        for message_id in preserved_ids:
            existing = next(
                (message for message in buf.messages if message.id == message_id),
                None,
            )
            if existing is None:
                continue
            existing_status = (
                self._tool_result_externalizer.embedded_stub_projection_status(
                    existing.content,
                    ctx,
                    session_id,
                    deadline=validation_deadline,
                    expected_source_message_id=existing.id,
                )
            )
            if existing_status in {"valid", "unavailable"}:
                continue
            # Only a verified last-known-good stub is preservable. An older raw
            # value that newly became eligible must still be removed fail closed.
            mgr.remove_message(session_id, message_id, ctx)
        for message_id in governed.removed_message_ids:
            if message_id in preserved_ids:
                continue
            mgr.remove_message(session_id, message_id, ctx)

        fallback_candidates = []
        if tail_fallback:
            fallback_count = len(governed.messages) + len(
                governed.removed_fallback_messages
            )
            start = max(
                buf.extraction_watermark,
                len(buf.messages) - fallback_count,
            )
            fallback_candidates = list(buf.messages[start:])
        consumed_tail: set[str] = set()

        for role, original_content in governed.removed_fallback_messages:
            for existing in fallback_candidates:
                if existing.id in consumed_tail:
                    continue
                if existing.role == role and existing.content == original_content:
                    consumed_tail.add(existing.id)
                    mgr.remove_message(session_id, existing.id, ctx)
                    break

        count = 0
        for message, source_message_id, original_content in zip(
            governed.messages,
            governed.source_message_ids,
            governed.fallback_original_contents,
            strict=True,
        ):
            role = str(message.get("role", "user"))
            content = render_safe_message_content(message)
            if not content:
                continue
            created_at = message.get("created_at")
            source = message.get("_ogmem_source")
            semantic_attested = bool(
                isinstance(source, dict)
                and source.get("semantic_turn_attested") is True
            )
            semantic_validated = bool(
                message.get("_ogmem_semantic_source_validated") is True
            )
            if semantic_attested and not semantic_validated:
                source_message_id = None
            if source_message_id and semantic_attested:
                existing = next(
                    (
                        item for item in buf.messages
                        if item.id == source_message_id
                    ),
                    None,
                )
                existing_status = (
                    self._tool_result_externalizer.stub_sequence_status(
                        existing.content,
                        ctx,
                        session_id,
                        deadline=validation_deadline,
                    )
                    if existing is not None
                    else "invalid"
                )
                incoming_status = self._tool_result_externalizer.stub_sequence_status(
                    content,
                    ctx,
                    session_id,
                    deadline=validation_deadline,
                )
                if (
                    existing is not None
                    and existing_status in {"valid", "unavailable"}
                    and incoming_status != "valid"
                    and not fallback_original
                ):
                    # A late, shorter checkpoint may no longer exceed the turn
                    # budget. Never let it downgrade an already verified stub
                    # for the same adapter-attested append-only source message.
                    continue
            if not source_message_id and tail_fallback:
                matched = None
                for existing in fallback_candidates:
                    if existing.id in consumed_tail:
                        continue
                    if existing.role == role and existing.content == original_content:
                        matched = existing
                        consumed_tail.add(existing.id)
                        break
                if matched is not None:
                    if matched.content != content:
                        mgr.add_message(
                            session_id,
                            role,
                            content,
                            ctx,
                            created_at=created_at,
                            message_id=matched.id,
                        )
                        count += 1
                    continue
            mgr.add_message(
                session_id,
                role,
                content,
                ctx,
                created_at=created_at,
                message_id=source_message_id,
            )
            count += 1
        return count

    @staticmethod
    def _first_view_identity_value(
        value,
        *,
        field: str,
        max_chars: int = 512,
    ) -> str:
        """Validate one trusted-adapter identity component without echoing it."""
        if not isinstance(value, str):
            raise ValueError(f"{field} must be a string")
        normalized = value.strip()
        if not normalized or len(normalized) > max_chars or "\x00" in normalized:
            raise ValueError(f"invalid {field}")
        return normalized

    def govern_first_view_tool_result(self, params: dict) -> dict:
        """Govern a trusted post-tool/pre-model result without touching Buffer.

        Artifact persistence is synchronous, while the model-facing projection
        is fail-open. This path deliberately uses the primary single-result
        threshold for first-view replacement.
        """
        content = deepcopy(params.get("content", []))
        if self._tool_result_externalizer is None:
            return {
                "ok": True,
                "externalized": 0,
                "failed": 0,
                "content": content,
                "reason": "externalization_disabled",
            }
        if not getattr(self._cfg, "tool_result_first_view_enabled", True):
            return {
                "ok": True,
                "externalized": 0,
                "failed": 0,
                "content": content,
                "reason": "first_view_disabled",
            }

        session_id = self._first_view_identity_value(
            params.get("sessionId"), field="sessionId",
        )
        adapter_id = self._first_view_identity_value(
            params.get("adapterId"), field="adapterId", max_chars=64,
        )
        turn_id = self._first_view_identity_value(
            params.get("turnId"), field="turnId",
        )
        tool_call_id = self._first_view_identity_value(
            params.get("toolCallId"), field="toolCallId",
        )
        tool_name = self._first_view_identity_value(
            params.get("toolName"), field="toolName", max_chars=256,
        )
        if _re.fullmatch(r"[a-zA-Z0-9_.-]+", adapter_id) is None:
            raise ValueError("invalid adapterId")

        ctx = params.get("_ctx") or self.build_context(params)
        turn_digest = hashlib.sha256("\0".join((
            adapter_id,
            ctx.account_id,
            ctx.user_id,
            ctx.agent_id,
            session_id,
            turn_id,
        )).encode("utf-8")).hexdigest()
        message_digest = hashlib.sha256("\0".join((
            turn_digest,
            tool_call_id,
        )).encode("utf-8")).hexdigest()
        trusted_turn_id = f"ogmfv_turn_{turn_digest[:32]}"
        trusted_source_id = f"ogmfv_msg_{message_digest[:32]}"
        message = {
            "role": "toolResult",
            "toolCallId": tool_call_id,
            "toolName": tool_name,
            "content": content,
            "_ogmem_source": {
                "source_message_id": trusted_source_id,
                "turn_id": trusted_turn_id,
                "identity_strength": "strong",
                "first_view_adapter_attested": True,
            },
            "_ogmem_semantic_source_validated": True,
        }
        ranking_prompt = str(params.get("rankingPrompt") or "")[:32_768]

        mgr = self.get_session_manager()
        buf = mgr.begin_managed_operation(session_id, ctx)
        try:
            governed = self._tool_result_externalizer.govern_messages(
                [message],
                ctx=ctx,
                session_id=session_id,
                # Artifact writes still occur. False selects model-facing
                # fail-open behavior and avoids a duplicate Buffer write before
                # the complete Turn reaches afterTurn.
                persistence=False,
                ranking_prompt=ranking_prompt,
            )
        finally:
            buf.end_operation()

        rewritten_content = content
        if governed.messages:
            rewritten_content = deepcopy(governed.messages[0].get("content", content))
        externalized = int(governed.externalized_count or 0)
        failed = int(governed.failed_count or 0)
        return {
            "ok": True,
            "externalized": externalized,
            "failed": failed,
            "content": rewritten_content,
            "identityStrength": "strong" if externalized else None,
            "sourceMessageId": trusted_source_id if externalized else None,
            "turnId": trusted_turn_id if externalized else None,
            "reason": (
                "externalized"
                if externalized and not failed
                else "partial_fail_open"
                if externalized and failed
                else "fail_open"
                if failed
                else "below_threshold"
            ),
        }

    def govern_pending_buffer(
        self,
        session_id: str,
        ctx: RequestContext,
        *,
        _operation_held: bool = False,
    ) -> dict:
        """Fail-closed safety net for unextracted messages that bypassed ingress."""
        if self._tool_result_externalizer is None:
            return {"governed": False, "changed": 0}
        mgr = self.get_session_manager()
        operation_started = False
        if not _operation_held:
            buf = mgr.begin_managed_operation(session_id, ctx)
            operation_started = True
        else:
            buf = mgr.get_or_create(session_id, ctx=ctx)
        try:
            with buf.extraction_lock:
                pending = list(buf.messages[buf.extraction_watermark:])
                fallback_source_ids = frozenset(
                    self._normalized_tool_result_governance_fallback_ids(
                        buf.runtime_extensions.get(
                            _TOOL_RESULT_GOVERNANCE_FALLBACK_STATE
                        )
                    )
                )
            if not pending:
                return {"governed": True, "changed": 0}
            governable_pending = []
            for message in pending:
                if message.id in fallback_source_ids:
                    continue
                status = self._tool_result_externalizer.embedded_stub_projection_status(
                    message.content,
                    ctx,
                    session_id,
                    expected_source_message_id=message.id,
                )
                if status != "valid":
                    governable_pending.append(message)
            if not governable_pending:
                return {"governed": True, "changed": 0}
            messages = [
                {
                    "role": message.role,
                    "content": message.content,
                    "created_at": message.created_at,
                    "_ogmem_source": {
                        "source_message_id": message.id,
                        "identity_strength": "fallback",
                    },
                }
                for message in governable_pending
            ]
            governed = self._govern_request_messages(
                messages,
                ctx=ctx,
                session_id=session_id,
                persistence=True,
            )
            if governed is None:
                return {"governed": False, "changed": 0}
            before = [(message.id, message.role, message.content) for message in pending]
            self._upsert_governed_messages(
                session_id,
                governed,
                ctx,
                tail_fallback=False,
            )
            with buf.extraction_lock:
                after = [
                    (message.id, message.role, message.content)
                    for message in buf.messages[buf.extraction_watermark:]
                ]
            failed_count = int(getattr(governed, "failed_count", 0) or 0)
            if failed_count:
                return {
                    "governed": False,
                    "changed": int(before != after),
                    "failed_count": failed_count,
                    "retryable": True,
                }
            return {"governed": True, "changed": int(before != after)}
        finally:
            if operation_started:
                buf.end_operation()

    def add_session_message(self, params: dict) -> dict:
        """Govern the direct session-message route before it reaches the buffer."""
        session_id = params.get("sessionId", "unknown")
        ctx = params.get("_ctx") or self.build_context(params)
        if self._tool_result_externalizer is None:
            return self.get_session_manager().add_message(
                session_id,
                params.get("role", "user"),
                params.get("content", ""),
                ctx,
                created_at=params.get("created_at"),
            )
        message = params.get("message")
        if not isinstance(message, dict):
            message = {
                "role": params.get("role", "user"),
                "content": params.get("content", ""),
                "created_at": params.get("created_at"),
            }
            if params.get("messageId"):
                message["_ogmem_source"] = {
                    "source_message_id": params["messageId"],
                    "turn_id": params.get("turnId"),
                }
        # Direct session-message callers cannot attest private source identity.
        # Stable ids are issued only by a validated semantic Turn or Turn Ledger.
        message = self._attach_server_fallback_sources(
            self._strip_unverified_semantic_sources([message])
        )[0]
        mgr = self.get_session_manager()
        buf = mgr.begin_managed_operation(session_id, ctx)
        try:
            governed = self._govern_request_messages(
                [message], ctx=ctx, session_id=session_id,
                persistence=True,
            )
            if governed is None:
                from extraction.tool_collector import parse_message_parts

                return mgr.add_message(
                    session_id,
                    str(message.get("role", "user")),
                    parse_message_parts(message.get("content", "")).get(
                        "text", ""
                    ),
                    ctx,
                    created_at=message.get("created_at"),
                )
            count = self._upsert_governed_messages(
                session_id, governed, ctx, tail_fallback=False,
            )
            if int(getattr(governed, "failed_count", 0) or 0) > 0:
                return {
                    "ok": False,
                    "written": bool(count),
                    "count": count,
                    "status": "tool_result_governance_pending",
                    "retryable": True,
                }
            return {
                "ok": True,
                "written": bool(count),
                "count": count,
                "externalized": governed.externalized_count,
            }
        finally:
            buf.end_operation()

    def commit_session_buffer(self, params: dict) -> dict:
        """Commit facade kept as the sole transport entry point."""
        session_id = params.get("sessionId", "unknown")
        ctx = params.get("_ctx") or self.build_context(params)
        mgr = self.get_session_manager()
        if self._tool_result_externalizer is None:
            return mgr.commit(
                session_id, ctx, wait=params.get("wait", False),
            )
        buf = mgr.begin_managed_operation(session_id, ctx)
        try:
            governance = self.govern_pending_buffer(
                session_id,
                ctx,
                _operation_held=True,
            )
            if governance.get("retryable"):
                return {
                    "ok": False,
                    "status": "tool_result_governance_pending",
                    "retryable": True,
                    **governance,
                }
            return mgr.commit(
                session_id, ctx, wait=bool(params.get("wait", False)),
            )
        finally:
            buf.end_operation()
    def list_tool_result_artifacts(self, params: dict) -> dict:
        if self._tool_result_externalizer is None or self._tool_result_store is None:
            raise ToolResultExternalizationDisabledError(
                "tool-result externalization is disabled"
            )
        from features.tool_result_externalization.store_base import ArtifactNotVisibleError

        session_id = params.get("sessionId") or params.get("session_id") or ""
        ctx = params.get("_ctx") or self.build_context(params)
        try:
            mgr = self.get_session_manager()
            buf = mgr.begin_managed_operation(session_id, ctx)
        except PermissionError as exc:
            raise ArtifactNotVisibleError("not_found_or_not_visible") from exc
        try:
            deadline = (
                time.monotonic()
                + self._cfg.tool_result_search_deadline_ms / 1000
            )
            result = self._tool_result_io_runner.run(
                lambda: self._tool_result_store.list(
                    session_id,
                    ctx,
                    limit=int(params.get("limit", 50)),
                    cursor=params.get("cursor"),
                ),
                deadline=deadline,
                operation_key=session_id,
            )
            private_fields = {
                "fact_candidates",
                "evidence_candidates",
                "record_candidates",
                "record_layout",
            }
            return {
                **result,
                "artifacts": [
                    {
                        key: value
                        for key, value in row.items()
                        if key not in private_fields
                    }
                    for row in result.get("artifacts", [])
                    if isinstance(row, dict)
                ],
            }
        finally:
            buf.end_operation()

    def read_tool_result_artifact(self, params: dict) -> dict:
        if self._tool_result_externalizer is None or self._tool_result_store is None:
            raise ToolResultExternalizationDisabledError(
                "tool-result externalization is disabled"
            )
        import hashlib
        from features.tool_result_externalization.store_base import ArtifactNotVisibleError

        session_id = params.get("sessionId") or params.get("session_id") or ""
        ctx = params.get("_ctx") or self.build_context(params)
        try:
            mgr = self.get_session_manager()
            buf = mgr.begin_managed_operation(session_id, ctx)
        except PermissionError as exc:
            raise ArtifactNotVisibleError("not_found_or_not_visible") from exc
        artifact_id = params.get("artifactId") or params.get("artifact_id") or ""
        ref = params.get("ref")
        if ref and not artifact_id:
            artifact_id = str(ref).removeprefix("ogmemory://tool-results/")
        try:
            deadline = (
                time.monotonic()
                + self._tool_result_externalizer.write_timeout_ms / 1000
            )
            result = self._tool_result_io_runner.run(
                lambda: self._tool_result_store.read_range(
                    artifact_id,
                    session_id,
                    ctx,
                    int(params.get("offset_bytes", 0)),
                    int(params.get(
                        "length_bytes",
                        self._cfg.tool_result_read_default_bytes,
                    )),
                ),
                deadline=deadline,
                operation_key=session_id,
            )
        finally:
            buf.end_operation()
        metadata = result.pop("metadata")
        content = result.pop("content")
        total_bytes = int(metadata["original_bytes"])
        next_offset = int(result["next_offset_bytes"])
        continuation_required = not bool(result["eof"])
        remaining_bytes = max(0, total_bytes - next_offset)
        next_length = min(
            remaining_bytes,
            max(1, int(result.get("actual_length_bytes", 0) or self._cfg.tool_result_read_default_bytes)),
        )
        envelope = {
            "_ogmemory_artifact_source": True,
            "schema_version": 1,
            "source_tool": "read_tool_result_artifact",
            "source_tool_call_id": params.get("source_tool_call_id"),
            "artifact_ref": result.pop("ref"),
            "offset_bytes": result["actual_offset_bytes"],
            "artifact_sha256": metadata["sha256"],
            "total_bytes": total_bytes,
            "continuation_required": continuation_required,
            "remaining_bytes": remaining_bytes,
            "next_request": (
                {
                    "offset_bytes": next_offset,
                    "length_bytes": next_length,
                }
                if continuation_required else None
            ),
            "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            **result,
            "content": content,
        }
        return self.attest_tool_result_source(envelope, ctx)

    def attest_tool_result_source(self, envelope: dict, ctx: RequestContext) -> dict:
        """Attach a same-process trusted alternative when MCP call IDs are unavailable."""
        if self._tool_result_externalizer is None:
            raise ToolResultExternalizationDisabledError(
                "tool-result externalization is disabled"
            )
        from features.tool_result_externalization.externalizer import attest_source_envelope

        return attest_source_envelope(
            envelope,
            self._tool_result_source_attestation_key,
            ctx,
            ctx.session_id,
        )

    def search_tool_result_artifacts(self, params: dict) -> dict:
        if self._tool_result_externalizer is None or self._tool_result_store is None:
            raise ToolResultExternalizationDisabledError(
                "tool-result externalization is disabled"
            )
        from features.tool_result_externalization.externalizer import search_artifacts
        from features.tool_result_externalization.store_base import ArtifactNotVisibleError

        session_id = params.get("sessionId") or params.get("session_id") or ""
        ctx = params.get("_ctx") or self.build_context(params)
        try:
            mgr = self.get_session_manager()
            buf = mgr.begin_managed_operation(session_id, ctx)
        except PermissionError as exc:
            raise ArtifactNotVisibleError("not_found_or_not_visible") from exc
        try:
            result = search_artifacts(
                self._tool_result_store,
                session_id=session_id,
                query=str(params.get("query", "")),
                ctx=ctx,
                top_k=int(params.get("top_k", self._cfg.tool_result_search_top_k)),
                max_chars=int(params.get("max_chars", 20_000)),
                max_artifacts=self._cfg.tool_result_search_max_artifacts,
                chunk_bytes=self._cfg.tool_result_search_chunk_bytes,
                overlap_bytes=self._cfg.tool_result_search_overlap_bytes,
                deadline_ms=self._cfg.tool_result_search_deadline_ms,
                io_runner=self._tool_result_io_runner,
            )
        finally:
            buf.end_operation()
        source_call_id = params.get("source_tool_call_id")
        envelopes = []
        for row in result.get("results", []):
            envelope = {
                "_ogmemory_artifact_source": True,
                "schema_version": 2,
                "source_tool": "search_tool_result_artifacts",
                "source_tool_call_id": source_call_id,
                "artifact_ref": row.get("ref"),
                "offset_bytes": row.get("offset_bytes", 0),
                "actual_offset_bytes": row.get("offset_bytes", 0),
                "actual_length_bytes": row.get("actual_length_bytes", 0),
                "next_offset_bytes": row.get("next_offset_bytes", 0),
                "decoded_chars": row.get("decoded_chars", 0),
                "artifact_sha256": row.get("artifact_sha256", ""),
                "content_sha256": row.get("content_sha256", row.get("chunk_sha256", "")),
                "content": row.get("content", ""),
                "relevance_score": row.get("relevance_score", 0),
                "fragment_type": row.get("fragment_type", "text_segment"),
                "fragment_complete": bool(row.get("fragment_complete", False)),
                "match_offset_bytes": row.get("match_offset_bytes", 0),
                "match_length_bytes": row.get("match_length_bytes", 0),
                "context_before": row.get("context_before", ""),
                "context_before_offset_bytes": row.get("context_before_offset_bytes", 0),
                "context_before_sha256": row.get("context_before_sha256", ""),
                "context_after": row.get("context_after", ""),
                "context_after_offset_bytes": row.get("context_after_offset_bytes", 0),
                "context_after_sha256": row.get("context_after_sha256", ""),
                "returned_chars": row.get("returned_chars", 0),
            }
            envelopes.append(self.attest_tool_result_source(envelope, ctx))
        result["results"] = envelopes
        return result
