"""Context writer for persisting ContextNodes.

Orchestrates the write process: extraction → planning → building → writing → outbox.
"""

import hashlib
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import cast

from commit.archive_builder import ArchiveBuilder
from commit.outbox_store import OutboxStore
from commit.policy_router import PolicyRouter
from core.errors import ConcurrentModificationError
from core.interfaces import LLM, ContextFS
from core.logging_config import get_logger
from core.models import CandidateMemory, RequestContext, WriteAction, WritePlan

logger = get_logger(__name__)


class ContextWriter:
    """Writes context memories to storage.

    Orchestrates the full write pipeline:
    1. Plan: Determine write action via MergePolicy
    2. Build: Construct ContextNode via ArchiveBuilder
    3. Write: Persist via ContextFS
    4. Outbox: Register event for async index synchronization
    """

    def __init__(
        self,
        fs: ContextFS,
        llm: LLM,
        policy_router: PolicyRouter | None = None,
        archive_builder: ArchiveBuilder | None = None,
        outbox_store: OutboxStore | None = None,
        directory_summary_enabled: bool = True,
    ):
        """Initialize ContextWriter.

        Args:
            fs: ContextFS for persisting nodes
            llm: LLM instance for semantic merge (required, passed to ArchiveBuilder)
            policy_router: PolicyRouter for selecting merge policies
            archive_builder: ArchiveBuilder for building ContextNode
            outbox_store: OutboxStore for registering index events (optional)
            directory_summary_enabled: Whether to register directory summary events
        """
        self._fs = fs
        self._llm = llm
        if policy_router is None:
            raise ValueError(
                "ContextWriter requires a PolicyRouter — "
                "pass schema_registry to MemoryWriteAPI"
            )
        self._policy_router = policy_router
        self._archive_builder = archive_builder or ArchiveBuilder(llm=llm)
        self._outbox_store = outbox_store
        self._directory_summary_enabled = directory_summary_enabled
        self._executed_ops: set[str] = set()

    @staticmethod
    def _get_real_method(
        instance: object | None,
        name: str,
    ) -> Callable[..., object] | None:
        """Return a class-defined method without accepting dynamic mock attributes."""
        if instance is None or not callable(getattr(type(instance), name, None)):
            return None
        method = getattr(instance, name, None)
        if not callable(method):
            return None
        return cast(Callable[..., object], method)

    def _operation_fingerprint(self, plan: WritePlan, ctx: RequestContext, content_hash: str = "") -> str:
        """Generate operation fingerprint for idempotency tracking.

        Args:
            plan: WritePlan for this operation
            ctx: RequestContext for this operation
            content_hash: Optional hash of the content to distinguish
                          same-URI different-content updates

        Returns:
            SHA256 hash fingerprint unique to this operation
        """
        payload = f"{plan.target_uri}:{plan.action}:{ctx.session_id}:{content_hash}"
        return hashlib.sha256(payload.encode()).hexdigest()

    @staticmethod
    def _is_code_resource(node: object) -> bool:
        """Return whether *node* is governed by the durable code contract."""
        return (
            getattr(node, "context_type", None) == "RESOURCE"
            and getattr(node, "metadata", {}).get("resource_type") == "code"
        )

    def write_candidate(
        self,
        candidate: CandidateMemory,
        ctx: RequestContext
    ) -> WritePlan:
        """Write a single candidate memory.

        Args:
            candidate: CandidateMemory to write
            ctx: RequestContext for this operation

        Returns:
            WritePlan showing what action was taken
        """
        # In authenticated mode an agent-scoped candidate requires an explicit,
        # authorized agent context. Without agent_id we skip instead of building
        # malformed agent URIs like .../agents//...
        if candidate.owner_scope == "agent" and not ctx.agent_id:
            logger.info(
                "Skipping agent-scoped candidate without agent context: %s/%s",
                candidate.category,
                candidate.routing_key,
            )
            return WritePlan(
                action="skip",
                target_uri="",
                merged_fields={},
                relation_edges=[],
            )

        # Retry loop for concurrent modification (optimistic locking)
        max_retries = 3
        retry_delays = [0.1, 0.2, 0.4]  # Exponential backoff in seconds

        if self._policy_router is None:
            raise RuntimeError(
                "ContextWriter requires a PolicyRouter — "
                "pass schema_registry to MemoryWriteAPI"
            )

        for attempt in range(max_retries):
            try:
                # Step 1: Plan - determine write action
                plan = self._policy_router.plan(candidate, ctx)

                # Skip if action is skip
                if plan.action == "skip":
                    return plan

                # Step 1.5: Idempotency check - prevent duplicate operations
                fingerprint_content = candidate.content or ""
                from index.activation_projection import ACTIVATION_VERSION
                descriptor = candidate.metadata.get("update_activation")
                if (candidate.metadata.get("update_activation_version") == ACTIVATION_VERSION
                        and isinstance(descriptor,str) and descriptor.strip()
                        and "\n" not in descriptor.strip() and "\r" not in descriptor.strip()):
                    # A changed connection key is a real graph projection
                    # update, even when ordinary content stays identical.
                    import json
                    fingerprint_content = json.dumps([fingerprint_content,descriptor.strip(),ACTIVATION_VERSION])
                content_hash = hashlib.sha256(fingerprint_content.encode()).hexdigest()[:16]
                fp = self._operation_fingerprint(plan, ctx, content_hash)
                if fp in self._executed_ops:
                    logger.info(f"Skipping duplicate operation: {plan.target_uri}")
                    return plan

                # Handle archive operation
                if plan.action == WriteAction.ARCHIVE:
                    self._executed_ops.add(fp)
                    return self._handle_archive(plan, ctx)

                # Handle delete operation
                if plan.action == WriteAction.DELETE:
                    self._executed_ops.add(fp)
                    return self._handle_delete(plan, ctx)

                # Step 2: Build - construct ContextNode
                node = self._archive_builder.build(candidate, plan, ctx)

                # Step 3+4: Write node + outbox event atomically (if supported)
                atomic_write = self._get_real_method(self._fs, "write_node_with_outbox")
                supports_atomic_write = self._get_real_method(
                    self._fs,
                    "supports_atomic_outbox_write",
                )
                build_write_event = self._get_real_method(
                    self._outbox_store,
                    "build_write_event",
                )
                atomic_write_allowed = (
                    supports_atomic_write is None
                    or bool(supports_atomic_write(node))
                )
                if self._is_code_resource(node) and (
                    atomic_write is None
                    or not atomic_write_allowed
                    or build_write_event is None
                ):
                    raise RuntimeError(
                        "code resource writes require an atomic Outbox backend"
                    )
                if (
                    atomic_write is not None
                    and atomic_write_allowed
                    and build_write_event is not None
                    and plan.action in ("create", "merge")
                ):
                    use_atomic = True
                    outbox_event = build_write_event(node)
                    atomic_write(node, ctx, outbox_event)
                else:
                    use_atomic = False
                    # Generic non-code resources retain their legacy fallback.
                    self._fs.write_node(node, ctx)

                # Mark as executed only after successful write
                self._executed_ops.add(fp)

                # Step 4: Outbox - register event for async index synchronization
                # Skip when atomic path already inserted the event in the same txn
                if not use_atomic and self._outbox_store and plan.action in ("create", "merge"):
                    try:
                        self._outbox_store.register_write(node, ctx)
                    except Exception as e:
                        # Log error but don't fail the write
                        logger.warning(
                            "Failed to register outbox event for %s: %s",
                            node.uri, e
                        )

                # Step 5: Directory - trigger directory summary update
                # Skip for profile: parent is memories/, not a category directory
                if (
                    self._directory_summary_enabled
                    and self._outbox_store
                    and plan.action in ("create", "merge")
                ):
                    if candidate.category != "profile":
                        try:
                            self._outbox_store.register_directory(node, ctx)
                        except Exception as e:
                            # Log error but don't fail the write
                            logger.warning(
                                "Failed to register directory event for %s: %s",
                                node.uri, e
                            )

                return plan

            except ConcurrentModificationError as e:
                if attempt < max_retries - 1:
                    # Retry with exponential backoff
                    delay = retry_delays[attempt]
                    logger.info(
                        "Concurrent modification detected for %s (attempt %d), "
                        "retrying after %.1fs: expected v%d, found v%d",
                        candidate.routing_key, attempt + 1, delay, e.expected_version, e.actual_version
                    )
                    time.sleep(delay)
                    # Re-plan on retry (will read fresh version)
                    continue
                else:
                    # Max retries exceeded
                    logger.error(
                        "Max retries exceeded for concurrent modification at %s",
                        candidate.routing_key
                    )
                    raise

        # Should be unreachable — loop always returns or raises
        raise RuntimeError("Unreachable: write_candidate loop exited unexpectedly")  # pragma: no cover

    def write_candidates(
        self,
        candidates: list[CandidateMemory],
        ctx: RequestContext
    ) -> list[WritePlan]:
        """Write multiple candidate memories.

        Args:
            candidates: List of CandidateMemory to write
            ctx: RequestContext for these operations

        Returns:
            List of WritePlan showing what action was taken for each
        """
        plans = []

        for candidate in candidates:
            try:
                plan = self.write_candidate(candidate, ctx)
                plans.append(plan)
            except Exception as e:
                # Log error but continue with other candidates
                logger.error(
                    "Failed to write candidate %s/%s: %s",
                    candidate.category, candidate.routing_key, e,
                    exc_info=True
                )

        return plans

    def write_candidates_parallel(
        self,
        candidates: list[CandidateMemory],
        ctx: RequestContext,
        max_workers: int = 4
    ) -> list[WritePlan]:
        """Write multiple candidates in parallel.

        Args:
            candidates: List of CandidateMemory to write
            ctx: RequestContext for these operations
            max_workers: Maximum parallel write operations

        Returns:
            List of WritePlan showing what action was taken for each
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        plans = []
        executor = ThreadPoolExecutor(max_workers=max_workers)

        try:
            # Submit all tasks
            futures = {
                executor.submit(self.write_candidate, candidate, ctx): candidate
                for candidate in candidates
            }

            # Collect results
            for future in as_completed(futures):
                candidate = futures[future]
                try:
                    plan = future.result()
                    plans.append(plan)
                except Exception as e:
                    logger.error(
                        "Failed to write candidate %s/%s: %s",
                        candidate.category, candidate.routing_key, e,
                        exc_info=True
                    )

        finally:
            # Ensure proper shutdown even if interrupted
            # cancel_futures=True cancels any pending futures
            executor.shutdown(wait=True, cancel_futures=True)

        return plans

    def _handle_archive(self, plan: WritePlan, ctx: RequestContext) -> WritePlan:
        """Handle archive operation: soft delete with status update.

        Args:
            plan: WritePlan with action="archive"
            ctx: RequestContext for this operation

        Returns:
            WritePlan showing what action was taken
        """
        try:
            # Read existing node
            node = self._fs.read_node(plan.target_uri, ctx)

            # Update metadata to mark as archived
            node.metadata["status"] = "ARCHIVED"
            node.metadata["archived_at"] = datetime.now(UTC).isoformat()

            # Write back the updated node
            self._fs.write_node(node, ctx)

            # Register archive event for index synchronization
            if self._outbox_store:
                try:
                    self._outbox_store.register_archive(plan.target_uri, ctx)
                except Exception as e:
                    logger.warning(
                        "Failed to register archive event for %s: %s",
                        plan.target_uri, e
                    )

            logger.info("Archived node: %s", plan.target_uri)
            return plan

        except Exception as e:
            logger.error("Failed to archive node %s: %s", plan.target_uri, e)
            raise

    def _handle_delete(self, plan: WritePlan, ctx: RequestContext) -> WritePlan:
        """Handle delete operation: permanent removal.

        Args:
            plan: WritePlan with action="delete"
            ctx: RequestContext for this operation

        Returns:
            WritePlan showing what action was taken
        """
        try:
            # Delete node from filesystem
            self._fs.delete_node(plan.target_uri, ctx)

            # Register delete event for index synchronization
            if self._outbox_store:
                try:
                    self._outbox_store.register_delete(plan.target_uri, ctx)
                except Exception as e:
                    logger.warning(
                        "Failed to register delete event for %s: %s",
                        plan.target_uri, e
                    )

            logger.info("Deleted node: %s", plan.target_uri)
            return plan

        except Exception as e:
            logger.error("Failed to delete node %s: %s", plan.target_uri, e)
            raise
