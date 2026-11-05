"""OperatorService — high-level facade for operator execution.

Responsibilities:

1. Resolve raw nodes by ID (via the existing ``GraphService``)
2. Run a single operator OR a pipeline of operators
3. Persist each step's result as an ``ExtractedNode`` in the
   heterogeneous graph, automatically creating the ``extracted_from``
   edge (so the existing retrieval / PageRank machinery works).
4. Track async tasks (via ``TaskStore``) and expose status queries.

The service is the only place that ties together the ``OperatorRegistry``,
``OperatorPipeline``, and ``GraphService``. Operators themselves do not
know about the graph; the graph does not know about operators.
"""

from __future__ import annotations

import copy
import logging
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from core.errors import NodeNotFoundError
from core.graph_enums import RawDataType
from core.graph_models import RawNode
from extraction.operators import (
    OperatorPipeline,
    OperatorRegistry,
    PipelineResult,
)
from extraction.operators.base import ExtractionResult
from extraction.operators.materialize import (
    MaterializationReport,
    MaterializedChild,
    lightweight_summary,
    materialize_children,
)
from extraction.operators.task_store import (
    InMemoryTaskStore,
    TaskStore,
)

if TYPE_CHECKING:
    from core.models import RequestContext
    from service.graph_service import GraphService

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class OperatorServiceError(Exception):
    """Base class for OperatorService errors."""


class OperatorServiceBusyError(OperatorServiceError):
    """The process-local asynchronous worker limit has been reached."""


class RawNodeNotFoundError(NodeNotFoundError, OperatorServiceError):
    """Raised when a referenced raw node does not exist."""


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class OperatorService:
    """Coordinates operator execution, graph persistence, and async tracking.

    The service is stateless w.r.t. user data — it is safe to share
    one instance across all request threads.

    Args:
        graph_service: Used to read raw nodes and write extracted nodes.
        registry: Operator registry; defaults to ``default_registry()``.
        task_store: Backend for async task tracking; defaults to an
            in-memory store suitable for single-process deployments.
        max_async_workers: Process-local admission limit (default 4). Excess
            requests fail immediately with OperatorServiceBusyError; no queue.
    """

    def __init__(
        self,
        graph_service: GraphService,
        *,
        registry: OperatorRegistry | None = None,
        task_store: TaskStore | None = None,
        max_async_workers: int = 4,
    ) -> None:
        from extraction.operators import default_registry
        self._graph = graph_service
        self._registry = registry or default_registry()
        self._tasks = task_store or InMemoryTaskStore()
        if max_async_workers < 1:
            raise ValueError("max_async_workers must be positive")
        self._async_slots = threading.BoundedSemaphore(max_async_workers)
        self._async_lock = threading.Lock()
        self._async_threads: dict[str, threading.Thread] = {}

    # ------------------------------------------------------------------
    # Sync API
    # ------------------------------------------------------------------

    def list_operators(
        self, raw_type: RawDataType | None = None,
    ) -> list[dict]:
        """Return public metadata for every registered operator."""
        return [
            {
                "operator_type": info.operator_type,
                "display_name": info.display_name,
                "description": info.description,
                "supported_raw_types": [t.value for t in info.supported_raw_types],
                "schema_name": info.schema_name,
                "version": info.version,
                "config_schema": info.config_schema,
            }
            for info in self._registry.list(raw_type=raw_type)
        ]

    def describe_operator(self, operator_type: str) -> dict:
        info = self._registry.get_info(operator_type)
        return {
            "operator_type": info.operator_type,
            "display_name": info.display_name,
            "description": info.description,
            "supported_raw_types": [t.value for t in info.supported_raw_types],
            "schema_name": info.schema_name,
            "version": info.version,
            "config_schema": info.config_schema,
        }

    def run_operator(
        self,
        raw_node_id: str,
        operator_type: str,
        ctx: RequestContext,
        *,
        params: dict[str, Any] | None = None,
        bytes_writer: Callable[[dict, bytes], str] | None = None,
        write_to_graph: bool = True,
    ) -> dict:
        """Run a single operator against an existing raw node.

        Args:
            raw_node_id: ID of the raw node to extract from.
            operator_type: Registered operator id.
            ctx: RequestContext for access control.
            params: Optional per-call parameters stored under
                ``extracted_content._params`` as metadata. Note: ``params``
                here do NOT reconfigure the operator — use
                :meth:`persist_extraction` directly if you need per-call
                operator configuration (e.g. ``VideoFrameExtractor(fps=...)``).
            bytes_writer: Optional callable ``(item, raw_bytes) -> path``
                invoked when the operator's materialization policy declares
                a ``bytes_field`` (e.g. base64 JPEG frames). The returned
                path is stored in the child ``RawNode.metadata`` under
                the policy's ``bytes_metadata_key``. Required for operators
                that materialize binary payloads; ignored otherwise.
            write_to_graph: If True (default), persist the result as
                an ExtractedNode and create the ``extracted_from`` edge.
                When the operator declares a materialization policy, this
                also writes child ``RawNode``s + edges and replaces the
                heavy per-item collection in ``extracted_content`` with a
                lightweight ``child_node_ids`` summary.

        Returns:
            Dict with extraction details, the new ``extracted_node_id``
            (if written), and ``materialized_children`` (a list of
            :class:`MaterializedChild` records).
        """
        raw = self._get_raw_node(raw_node_id, ctx)
        operator = self._registry.get(operator_type)

        extraction = operator.extract(raw, ctx)
        # Apply per-call params to the extracted_content metadata
        if params and extraction.error is None:
            extraction.extracted_content.setdefault("_params", {}).update(params)

        return self.persist_extraction(
            raw_node=raw,
            operator=operator,
            extraction=extraction,
            ctx=ctx,
            bytes_writer=bytes_writer,
            write_to_graph=write_to_graph,
        )

    def persist_extraction(
        self,
        *,
        raw_node: RawNode,
        operator: Any,  # MultimodalExtractor; Any avoids circular import
        extraction: ExtractionResult,
        ctx: RequestContext,
        bytes_writer: Callable[[dict, bytes], str] | None = None,
        write_to_graph: bool = True,
        desired_node_id: str | None = None,
        refresh_extracted_children: bool = False,
    ) -> dict:
        """Persist a pre-computed ``ExtractionResult`` to the graph.

        This is the graph-side half of an operator call. Use it when the
        caller needs to configure the operator itself with per-call
        parameters (e.g. ``VideoFrameExtractor(fps=5, max_frames=10)``)
        rather than relying on the registry's default instance — the
        :meth:`run_operator` helper looks up the registered instance and
        ignores caller-supplied construction kwargs.

        Args:
            refresh_extracted_children: Opt-in replacement of existing
                extracted child payloads during refresh. Raw children retain
                their existing behavior. This does not remove obsolete nodes;
                the caller must reconcile them after complete persistence.
            desired_node_id: If set, the ``ExtractedNode`` is created with
                this deterministic id. Useful for idempotent re-runs
                where the caller wants to find the same node by id later
                (e.g. ``m3.extract_frames`` checks for an existing
                ``ExtractedNode`` before doing real work).

        Behaviour matches :meth:`run_operator`'s graph side:

        * If ``operator.info.materialization`` is set, materialize child
          ``RawNode``s + ``REFERENCES`` edges and replace the heavy
          per-item collection in ``extraction.extracted_content`` with
          a lightweight ``child_node_ids`` summary.
        * Write the (possibly lightweight) ``ExtractedNode`` to the graph.
        * Return a dict mirroring :meth:`run_operator`'s shape.
        """
        result: dict[str, Any] = {
            "operator": operator.extractor_type,
            "raw_node_id": raw_node.node_id,
            "ok": extraction.error is None,
            "confidence": extraction.confidence,
            "extracted_content": extraction.extracted_content,
            "schema_used": extraction.schema_used,
            "error": extraction.error,
            "materialized_children": [],
        }

        if not write_to_graph or extraction.error is not None:
            return result

        operator_info = operator.info
        policy = operator_info.materialization
        materialized: list[MaterializedChild] = []
        materialization_report: MaterializationReport | None = None
        content_to_write = extraction.extracted_content
        if policy is not None:
            materialized, materialization_report = materialize_children(
                self._graph,
                ctx,
                source_raw_id=raw_node.node_id,
                source_metadata=raw_node.metadata,
                extracted_content=extraction.extracted_content,
                policy=policy,
                bytes_writer=bytes_writer,
                refresh_extracted=refresh_extracted_children,
            )
            content_to_write = lightweight_summary(
                extraction.extracted_content, policy,
                [c.child_node_id for c in materialized],
            )
            content_to_write["materialization"] = materialization_report.to_dict()

        try:
            extracted = self._graph.create_extracted_node(
                source_raw_id=raw_node.node_id,
                extractor_type=operator.extractor_type,
                extraction_schema=extraction.schema_used,
                ctx=ctx,
                extracted_content=content_to_write,
                node_id=desired_node_id,
            )
        except FileExistsError:
            extracted = self._graph.get_extracted_node(desired_node_id, ctx) if desired_node_id else None
            if (extracted is None or extracted.source_raw_id != raw_node.node_id
                    or extracted.extractor_type != operator.extractor_type
                    or extracted.extraction_schema != extraction.schema_used):
                raise
        # Mark the node as COMPLETED so M3 drivers and other callers can
        # distinguish a real persisted result from a leftover pending
        # row written by a previous failed run. ``create_extracted_node``
        # defaults to PENDING, so the explicit update is required.
        self._graph.mark_extraction_completed(
            extracted.node_id,
            extracted_content=content_to_write,
            confidence=extraction.confidence,
            ctx=ctx,
        )
        result["extracted_node_id"] = extracted.node_id
        # Update the response payload with the lightweight summary so
        # callers don't accidentally see the heavy per-item data.
        result["extracted_content"] = content_to_write
        result["materialized_children"] = [
            {
                "child_node_id": c.child_node_id,
                "edge_id": c.edge_id,
                "metadata": c.payload,
                "resumed": c.resumed,
                "child_node_kind": c.child_node_kind,
            }
            for c in materialized
        ]
        if materialization_report is not None:
            result["materialization"] = materialization_report.to_dict()
        return result

    def run_pipeline(
        self,
        raw_node_id: str,
        pipeline_spec: dict,
        ctx: RequestContext,
        *,
        write_to_graph: bool = True,
    ) -> dict:
        """Run a full pipeline of operators against a raw node.

        Args:
            raw_node_id: ID of the raw node to extract from.
            pipeline_spec: A JSON-serialisable pipeline spec (see
                :class:`OperatorPipeline` for the format).
            ctx: RequestContext.
            write_to_graph: Persist each step's result as an ExtractedNode.

        Returns:
            Dict with the full pipeline result (steps, errors, node IDs).
        """
        raw = self._get_raw_node(raw_node_id, ctx)
        pipe = OperatorPipeline.from_spec(pipeline_spec)
        result: PipelineResult = pipe.execute(raw, ctx, registry=self._registry)

        out = result.to_dict()
        if write_to_graph:
            out["created_nodes"] = self._persist_pipeline(result, ctx)
        return out

    # ------------------------------------------------------------------
    # Async API
    # ------------------------------------------------------------------

    def run_pipeline_async(
        self,
        raw_node_id: str,
        pipeline_spec: dict,
        ctx: RequestContext,
        *,
        write_to_graph: bool = True,
    ) -> str:
        """Run a pipeline on a background thread; return task_id.

        The caller can poll ``get_task_status(task_id, ctx)`` for the result.
        """
        # No task records or threads are created for invalid/unauthorized input.
        # Admission also bounds concurrent preflight database reads.
        if not self._async_slots.acquire(blocking=False):
            raise OperatorServiceBusyError("Async pipeline capacity reached; retry later")
        try:
            raw_node = self._get_raw_node(raw_node_id, ctx)
            pipeline_spec = copy.deepcopy(pipeline_spec)
            pipe = OperatorPipeline.from_spec(pipeline_spec)
            pipe.validate(self._registry, raw_node.raw_type)
            rec = self._tasks.submit(
                raw_node_id=raw_node_id,
                operator_service="OperatorService.run_pipeline_async",
                trace_id=ctx.trace_id,
                account_id=ctx.account_id,
            )
        except BaseException:
            self._async_slots.release()
            raise

        def _worker():
            try:
                self._tasks.mark_running(rec.task_id)
                result = self.run_pipeline(
                    raw_node_id, pipeline_spec, ctx,
                    write_to_graph=write_to_graph,
                )
                self._tasks.mark_completed(rec.task_id, result)
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Async pipeline failed for task %s", rec.task_id)
                self._tasks.mark_failed(rec.task_id, str(exc))
            finally:
                # Drop the thread reference — without this, _async_threads
                # grows unboundedly with each run_pipeline_async call.
                with self._async_lock:
                    self._async_threads.pop(rec.task_id, None)
                # Reap terminal records older than the cutoff so the
                # task store doesn't accumulate stale entries forever.
                # Swallow prune errors — a failure here must not crash
                # the worker thread, which is the only owner of the task.
                try:
                    self._tasks.prune()
                except Exception:  # pragma: no cover - defensive
                    logger.exception(
                        "prune() failed in worker cleanup for %s", rec.task_id,
                    )
                finally:
                    self._async_slots.release()

        try:
            t = threading.Thread(
                target=_worker, daemon=True,
                name=f"op-pipeline-{rec.task_id[:8]}",
            )
            with self._async_lock:
                self._async_threads[rec.task_id] = t
            t.start()
        except BaseException:
            with self._async_lock:
                self._async_threads.pop(rec.task_id, None)
            self._async_slots.release()
            self._tasks.mark_failed(rec.task_id, "Unable to start pipeline worker")
            raise
        return rec.task_id

    def get_task_status(self, task_id: str, ctx: RequestContext) -> dict | None:
        """Return only tasks owned by the requesting account, including on failure."""
        rec = self._tasks.get(task_id)
        if rec is None or not rec.account_id or rec.account_id != ctx.account_id:
            return None
        return rec.to_dict()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_raw_node(
        self, raw_node_id: str, ctx: RequestContext
    ) -> RawNode:
        node = self._graph.get_raw_node(raw_node_id, ctx)
        if node is None:
            raise RawNodeNotFoundError(raw_node_id)
        return node

    def _persist_pipeline(
        self, result: PipelineResult, ctx: RequestContext,
    ) -> list[dict]:
        """Write each step's extraction to the graph; return node IDs.

        Skips skipped steps and steps whose extraction raised an error
        (since writing a failed extraction would be misleading).
        """
        created: list[dict] = []
        for step in result.steps:
            if step.skipped:
                continue
            if step.extraction.error is not None:
                continue
            # For a parallel step, write each sub-step
            if step.sub_results:
                for sub in step.sub_results:
                    if sub.skipped or sub.extraction.error is not None:
                        continue
                    self._write_extracted(
                        raw_id=result.raw_node_id,
                        operator_type=sub.operator,
                        schema_used=sub.extraction.schema_used,
                        extracted_content=sub.extraction.extracted_content,
                        confidence=sub.extraction.confidence,
                        ctx=ctx,
                        out=created,
                    )
            else:
                self._write_extracted(
                    raw_id=result.raw_node_id,
                    operator_type=step.operator,
                    schema_used=step.extraction.schema_used,
                    extracted_content=step.extraction.extracted_content,
                    confidence=step.extraction.confidence,
                    ctx=ctx,
                    out=created,
                )
        return created

    def _write_extracted(
        self,
        *,
        raw_id: str,
        operator_type: str,
        schema_used: str,
        extracted_content: dict,
        confidence: float,
        ctx: RequestContext,
        out: list[dict],
    ) -> None:
        # Skip the synthetic "<parallel>" wrapper label
        if operator_type == "<parallel>":
            return
        extracted = self._graph.create_extracted_node(
            source_raw_id=raw_id,
            extractor_type=operator_type,
            extraction_schema=schema_used,
            ctx=ctx,
            extracted_content=extracted_content,
        )
        self._graph.mark_extraction_completed(
            extracted.node_id, extracted_content=extracted_content,
            confidence=confidence, ctx=ctx,
        )
        out.append({
            "operator": operator_type,
            "extracted_node_id": extracted.node_id,
            "schema_used": schema_used,
        })
