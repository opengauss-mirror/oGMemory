"""Directory Event Handler: Process UPSERT_DIRECTORY events with DAG execution.

Implements DAG execution for directory summaries:
1. Accept UPSERT_DIRECTORY(root_dir) event
2. Trigger DAG execution for bottom-up processing
3. Wait for subdirectory summaries before generating parent summaries
4. Use fallback summaries when LLM fails
5. Vectorize and store to database after summary generation

This ensures summaries are generated bottom-up:
    leaf files → subdirs → parent dirs → root
"""

import asyncio
import logging
from dataclasses import dataclass, field, replace

from core.errors import NodeNotFoundError
from core.interfaces import LLM, ContextFS, Embedder, VectorIndex
from core.models import ContextNode, IndexRecord, OutboxEvent, RequestContext
from index.directory_summarizer import (
    ChildSummary,
    DirectorySummarizer,
    DirectorySummary,
    is_directory_uri,
)
from index.index_record_builder import build_directory_index_records
from index.vector_projection import VectorProjection
from service.evidence_directory import publication_guard, source_refs, summary_source_refs

logger = logging.getLogger(__name__)


@dataclass
class DirNode:
    """Directory node state for DAG execution."""

    uri: str
    children_dirs: list[str]
    file_children: list[str]
    child_index: dict[str, int]
    children_summaries: list[ChildSummary | None]
    pending: int
    parent_uri: str | None = None
    summary_scheduled: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass
class DagStats:
    """Statistics for DAG execution."""

    total_dirs: int = 0
    pending_dirs: int = 0
    completed_dirs: int = 0
    failed_dirs: int = 0
    record_ids: list = field(default_factory=list)
    written_dir_uris: list = field(default_factory=list)


@dataclass
class DirectoryEventResult:
    """Result of processing a directory event."""

    success: bool
    root_uri: str
    stats: DagStats
    error_message: str = ""
    record_ids: list = field(default_factory=list)
    written_dir_uris: list = field(default_factory=list)


class DirectoryEventHandler:
    """Handle UPSERT_DIRECTORY events with DAG-style bottom-up processing.

    Flow:
    1. Receive UPSERT_DIRECTORY(root_dir) event
    2. Recursively dispatch all subdirectories
    3. Wait for subdirectory summaries (pending == 0)
    4. Generate parent summary with complete child information
    5. Use fallback when LLM fails
    6. Vectorize and store to database

    Usage:
        handler = DirectoryEventHandler(fs, llm, embedder, vector_index)
        result = handler.process_directory_event(event, ctx)
    """

    def __init__(
        self,
        fs: ContextFS,
        llm: LLM,
        embedder: Embedder | None,
        vector_index: VectorIndex | None,
        max_children: int = 50,
        max_concurrent_llm: int = 10,
        dag_timeout_seconds: float = 300.0,
    ):
        """Initialize handler.

        Args:
            fs: ContextFS for reading/writing nodes
            llm: LLM for generating summaries
            embedder: Embedder for vectorization
            vector_index: VectorIndex for upsert
            max_children: Max children per directory
            max_concurrent_llm: Max concurrent LLM calls
            dag_timeout_seconds: Max seconds to wait for root completion
        """
        self._fs = fs
        self._llm = llm
        self._projection = (
            VectorProjection(vector_index, embedder)
            if vector_index is not None
            else None
        )
        self._max_children = max_children
        self._max_concurrent_llm = max_concurrent_llm
        self._dag_timeout_seconds = dag_timeout_seconds
        self._summarizer = DirectorySummarizer(fs, llm, max_children)

    def process_directory_event(
        self,
        event: OutboxEvent,
        ctx: RequestContext,
    ) -> DirectoryEventResult:
        """Process UPSERT_DIRECTORY event synchronously.

        Args:
            event: OutboxEvent with directory URI
            ctx: RequestContext for access control

        Returns:
            DirectoryEventResult with success status and stats
        """
        root_uri = event.uri

        if not is_directory_uri(root_uri):
            return DirectoryEventResult(
                success=False,
                root_uri=root_uri,
                stats=DagStats(),
                error_message=f"URI {root_uri} is not a directory",
            )

        try:
            result = asyncio.run(self._run_dag(root_uri, ctx, event.payload.get("evidence_rebuild_directories", ())))
            return result
        except Exception as e:
            logger.error("Failed to process directory event %s: %s", root_uri, e, exc_info=True)
            return DirectoryEventResult(
                success=False,
                root_uri=root_uri,
                stats=DagStats(),
                error_message=str(e),
            )

    async def _run_dag(
        self,
        root_uri: str,
        ctx: RequestContext,
        rebuild_directories=(),
    ) -> DirectoryEventResult:
        """Run DAG execution asynchronously."""
        self._nodes: dict[str, DirNode] = {}
        self._rebuild_directories = rebuild_directories
        self._parent_map: dict[str, str | None] = {}
        self._root_done: asyncio.Event | None = None
        self._stats = DagStats()
        self._llm_sem = asyncio.Semaphore(self._max_concurrent_llm)

        self._root_done = asyncio.Event()
        self._nodes.clear()
        self._parent_map.clear()
        self._stats = DagStats()

        await self._dispatch_dir(root_uri, parent_uri=None, ctx=ctx)
        try:
            await asyncio.wait_for(
                self._root_done.wait(),
                timeout=self._dag_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"Directory DAG timed out after "
                f"{self._dag_timeout_seconds:g}s for {root_uri}"
            ) from exc

        return DirectoryEventResult(
            success=True,
            root_uri=root_uri,
            stats=DagStats(
                total_dirs=self._stats.total_dirs,
                pending_dirs=self._stats.pending_dirs,
                completed_dirs=self._stats.completed_dirs,
                failed_dirs=self._stats.failed_dirs,
                record_ids=list(self._stats.record_ids),
                written_dir_uris=list(self._stats.written_dir_uris),
            ),
            record_ids=list(self._stats.record_ids),
            written_dir_uris=list(self._stats.written_dir_uris),
        )

    async def _dispatch_dir(
        self,
        dir_uri: str,
        parent_uri: str | None,
        ctx: RequestContext,
    ) -> None:
        """Dispatch a directory for processing.

        If already dispatched, skip.
        Otherwise, list children and schedule subtasks.
        """
        if dir_uri in self._nodes:
            logger.warning(
                "Skipping already-dispatched directory %s while processing parent %s",
                dir_uri, parent_uri,
            )
            if parent_uri:
                await self._on_child_done(parent_uri, dir_uri, None, ctx)
            return

        self._parent_map[dir_uri] = parent_uri

        try:
            children = [
                child_uri
                for child_uri in self._fs.list_children(dir_uri, ctx)
                if not self._same_uri(child_uri, dir_uri)
            ]
            # Cleanup knows affected ancestors even when a storage adapter lists
            # category nodes without a trailing slash (or omits directory rows).
            rebuilt_children = {uri for uri in self._rebuild_directories
                                if uri.rstrip("/").rsplit("/", 1)[0] == dir_uri.rstrip("/")}
            children = [uri for uri in children if uri.rstrip("/") + "/" not in rebuilt_children]
            children.extend(sorted(rebuilt_children))

            if len(children) > self._max_children:
                logger.warning(
                    "Directory %s has %d children, limiting to %d",
                    dir_uri, len(children), self._max_children
                )
                children = children[:self._max_children]

            children_dirs: list[str] = []
            file_children: list[str] = []

            for child_uri in children:
                if is_directory_uri(child_uri):
                    children_dirs.append(child_uri)
                else:
                    file_children.append(child_uri)

            child_index = {uri: idx for idx, uri in enumerate(children_dirs)}
            pending = len(children_dirs) + len(file_children)

            node = DirNode(
                uri=dir_uri,
                children_dirs=children_dirs,
                file_children=file_children,
                child_index=child_index,
                children_summaries=[None] * len(children_dirs),
                pending=pending,
                parent_uri=parent_uri,
            )
            self._nodes[dir_uri] = node
            self._stats.total_dirs += 1
            self._stats.pending_dirs += 1

            if pending == 0:
                self._schedule_summary(dir_uri, ctx)
                return

            for file_uri in file_children:
                asyncio.create_task(self._file_task(dir_uri, file_uri, ctx))

            for child_uri in children_dirs:
                asyncio.create_task(self._dispatch_dir(child_uri, dir_uri, ctx))

        except Exception as e:
            logger.error("Failed to dispatch directory %s: %s", dir_uri, e, exc_info=True)
            self._stats.failed_dirs += 1
            if parent_uri:
                await self._on_child_done(parent_uri, dir_uri, None, ctx)
            elif self._root_done:
                self._root_done.set()

    @staticmethod
    def _same_uri(left: str, right: str) -> bool:
        return left.rstrip("/") == right.rstrip("/")

    async def _file_task(
        self,
        parent_uri: str,
        file_uri: str,
        ctx: RequestContext,
    ) -> None:
        """Process a file child.

        Files are considered "done" immediately since they already have abstracts.
        """
        await self._on_file_done(parent_uri, file_uri, ctx)

    async def _on_file_done(
        self,
        parent_uri: str,
        file_uri: str,
        ctx: RequestContext,
    ) -> None:
        """Handle file completion, decrement parent pending."""
        node = self._nodes.get(parent_uri)
        if not node:
            return

        async with node.lock:
            node.pending -= 1
            if node.pending == 0 and not node.summary_scheduled:
                node.summary_scheduled = True
                asyncio.create_task(self._summary_task(parent_uri, ctx))

    async def _on_child_done(
        self,
        parent_uri: str,
        child_uri: str,
        child_summary: ChildSummary | None,
        ctx: RequestContext,
    ) -> None:
        """Handle child directory completion.

        Args:
            parent_uri: Parent directory URI
            child_uri: Child directory URI that just completed
            child_summary: Summary of the child (or None if failed)
            ctx: RequestContext
        """
        node = self._nodes.get(parent_uri)
        if not node:
            return

        async with node.lock:
            idx = node.child_index.get(child_uri)
            if idx is not None:
                node.children_summaries[idx] = child_summary
            node.pending -= 1
            if node.pending == 0 and not node.summary_scheduled:
                node.summary_scheduled = True
                asyncio.create_task(self._summary_task(parent_uri, ctx))

    def _schedule_summary(self, dir_uri: str, ctx: RequestContext) -> None:
        """Schedule summary generation for a directory."""
        node = self._nodes.get(dir_uri)
        if not node or node.summary_scheduled:
            return

        node.summary_scheduled = True
        asyncio.create_task(self._summary_task(dir_uri, ctx))

    def _collect_file_summaries(
        self,
        node: DirNode,
        ctx: RequestContext,
    ) -> list[ChildSummary]:
        """Collect summaries from file children."""
        summaries: list[ChildSummary] = []

        for file_uri in node.file_children:
            try:
                file_node = self._fs.read_node(file_uri, ctx)
                if file_node and file_node.abstract:
                    summaries.append(ChildSummary(
                        uri=file_uri,
                        name=file_uri.split("/")[-1],
                        abstract=file_node.abstract,
                        category=file_node.category or "unknown",
                        is_directory=False,
                        has_abstract=True,
                        source_refs=summary_source_refs(self._fs, file_node, ctx),
                    ))
            except Exception as e:
                logger.warning("Failed to read file %s: %s", file_uri, e)

        return summaries

    def _collect_children_summaries(self, node: DirNode) -> list[ChildSummary]:
        """Collect summaries from child directories (already completed)."""
        summaries: list[ChildSummary] = []

        for idx, child_uri in enumerate(node.children_dirs):
            item = node.children_summaries[idx]
            if item is not None:
                summaries.append(item)
            else:
                summaries.append(ChildSummary(
                    uri=child_uri,
                    name=child_uri.rstrip("/").split("/")[-1],
                    abstract="",
                    category="directory",
                    is_directory=True,
                    has_abstract=False,
                ))

        return summaries

    async def _summary_task(self, dir_uri: str, ctx: RequestContext) -> None:
        """Generate summary for a directory.

        This is called only when all children (files + subdirs) are done.
        """
        node = self._nodes.get(dir_uri)
        if not node:
            return

        file_summaries = self._collect_file_summaries(node, ctx)
        children_summaries = self._collect_children_summaries(node)

        all_summaries = file_summaries + children_summaries
        valid_summaries = [s for s in all_summaries if s.has_abstract]

        try:
            async with self._llm_sem:
                summary = await asyncio.to_thread(
                    self._generate_summary_with_fallback,
                    dir_uri,
                    valid_summaries,
                )

            owner_space = self._extract_owner_space_from_uri(dir_uri, ctx)

            dir_node = ContextNode(
                uri=dir_uri,
                context_type="MEMORY",
                category=dir_uri.rstrip("/").split("/")[-1] or "directory",
                level=0,
                owner_space=owner_space,
                abstract=summary.abstract,
                overview=summary.overview,
                content="",
                metadata={
                    "node_type": "directory",
                    "child_count": summary.child_count,
                    "categories": summary.categories,
                    "evidence_source_refs": [ref for child in valid_summaries for ref in child.source_refs],
                },
            )

            dir_node = await asyncio.to_thread(self._publish_directory, dir_node, ctx)
            self._stats.written_dir_uris.append(dir_uri)
            logger.info("Generated summary for directory %s", dir_uri)

            if self._projection is not None:
                await self._index_directory(dir_node, ctx)

            self._stats.completed_dirs += 1
            self._stats.pending_dirs = max(0, self._stats.pending_dirs - 1)

            child_summary = ChildSummary(
                uri=dir_uri,
                name=dir_uri.rstrip("/").split("/")[-1],
                abstract=summary.abstract,
                category="directory",
                is_directory=True,
                has_abstract=True,
                source_refs=source_refs(dir_node),
            )

        except Exception as e:
            logger.error("Failed to generate summary for %s: %s", dir_uri, e, exc_info=True)
            self._stats.failed_dirs += 1
            self._stats.pending_dirs = max(0, self._stats.pending_dirs - 1)
            child_summary = None

        parent_uri = self._parent_map.get(dir_uri)
        if parent_uri is None:
            if self._root_done:
                self._root_done.set()
            return

        await self._on_child_done(parent_uri, dir_uri, child_summary, ctx)

    def _publish_directory(self, node, ctx):
        with publication_guard(self._fs, node, ctx):
            if callable(getattr(type(self._fs), "memory_identity_lock", None)):
                try:
                    current = self._fs.read_node(node.uri, ctx)
                except NodeNotFoundError:
                    current = None
                if current is not None:
                    node = replace(node, metadata={**node.metadata, "_expected_version": current.metadata.get("version", 0)})
            self._fs.write_node(node, ctx)
            return self._fs.read_node(node.uri, ctx) if source_refs(node) else node

    def _publish_vectors(self, node, records, ctx):
        with publication_guard(self._fs, node, ctx, revision=node.metadata.get("version") if source_refs(node) else None):
            self._projection.upsert(records)

    def _extract_owner_space_from_uri(self, uri: str, ctx: RequestContext) -> str:
        """Extract owner_space from URI.

        URI patterns:
        - ctx://{account}/users/{user}/memories/... → "user:{user}"
        - ctx://{account}/agents/{agent}/memories/... → "agent:{agent}"

        Args:
            uri: ContextNode URI
            ctx: RequestContext (fallback to user_space if URI parsing fails)

        Returns:
            owner_space string
        """
        parts = uri.split("/")
        if len(parts) >= 4:
            if parts[3] == "users" and len(parts) >= 5:
                return f"user:{parts[4]}"
            elif parts[3] == "agents" and len(parts) >= 5:
                return f"agent:{parts[4]}"

        return ctx.user_space_name()

    def _generate_summary_with_fallback(
        self,
        dir_uri: str,
        child_summaries: list[ChildSummary],
    ) -> DirectorySummary:
        """Generate summary with fallback when LLM fails.

        Reuses DirectorySummarizer's fallback methods for consistency.

        Args:
            dir_uri: Directory URI
            child_summaries: List of child summaries

        Returns:
            DirectorySummary (from LLM or fallback)
        """
        if not child_summaries:
            return DirectorySummary(
                abstract="空目录",
                overview="",
                child_count=0,
                categories=[],
            )

        summaries_dict = [
            {
                "uri": s.uri,
                "abstract": s.abstract,
                "category": s.category,
            }
            for s in child_summaries
        ]

        try:
            summary = self._summarizer._generate_summary(dir_uri, summaries_dict)
            return summary
        except Exception as e:
            logger.warning("LLM failed for %s, using fallback: %s", dir_uri, e)
            abstract = self._summarizer._fallback_abstract(summaries_dict)
            overview = self._summarizer._fallback_overview(summaries_dict)
            categories = list(set(s.category for s in child_summaries))
            return DirectorySummary(
                abstract=abstract[:200],
                overview=overview,
                child_count=len(child_summaries),
                categories=categories,
            )

    async def _index_directory(
        self,
        dir_node: ContextNode,
        ctx: RequestContext,
    ) -> None:
        """Project the directory summary into the configured indexes."""
        projection = self._projection
        assert projection is not None
        records = build_directory_index_records(dir_node, ctx.account_id)
        if projection.has_embeddings:
            embeddings = await asyncio.to_thread(projection.embed, records)
            records = [
                IndexRecord(record.id, record.uri, record.level, record.text, record.filters,
                            {**record.metadata, "_embedding": embedding})
                for record, embedding in zip(records, embeddings)
            ]
        await asyncio.to_thread(self._publish_vectors, dir_node, records, ctx)
        self._stats.record_ids.extend(record.id for record in records)
        logger.info("Indexed directory %s with %d records", dir_node.uri, len(records))
