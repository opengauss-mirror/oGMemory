"""Candidate extraction pipeline.

Runs all CandidateExtractor implementations in parallel and aggregates results.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context

from core.candidate_dedup import merge_duplicate_event_sources
from core.interfaces import CandidateExtractor
from core.logging_config import get_logger
from core.memory_names import candidate_identity_key
from core.models import CandidateMemory, RequestContext
from core.validation import validate_attribution, validate_candidate

logger = get_logger(__name__)


class CandidatePipeline:
    """Pipeline for parallel candidate extraction.

    Runs all extractors concurrently and aggregates results.
    """

    def __init__(self, extractors: list[CandidateExtractor] | None = None):
        """Initialize CandidatePipeline.

        Args:
            extractors: List of CandidateExtractor instances.
                       If None, must be set before calling extract().
        """
        self._extractors = extractors or []

    def set_extractors(self, extractors: list[CandidateExtractor]) -> None:
        """Set the extractors to use.

        Args:
            extractors: List of CandidateExtractor instances
        """
        self._extractors = extractors

    def extract(
        self,
        messages: list[dict],
        ctx: RequestContext,
        parallel: bool = True,
        session_time=None,
        session_summary: str = "",
        tool_stats_text: str = "",
        archive_id: str | None = None,
    ) -> list[CandidateMemory]:
        """Extract candidates from messages using all extractors.

        Args:
            messages: List of message dicts with "role" and "content"
            ctx: RequestContext for this extraction
            parallel: If True, run extractors in parallel (default: True)
            session_time: Optional datetime for temporal resolution
            session_summary: Optional summary of previously extracted content
            tool_stats_text: Optional tool usage statistics text
            archive_id: Optional archive_id for provenance tracking

        Returns:
            List of CandidateMemory from all extractors

        Raises:
            Exception: If any extractor fails. Do not commit partial extraction
                or advance the session watermark after a failed model call.
        """
        if not self._extractors:
            return []
        if parallel:
            return self._extract_parallel(
                messages, ctx, session_time, session_summary, tool_stats_text,
                archive_id,
            )
        else:
            return self._extract_serial(
                messages, ctx, session_time, session_summary, tool_stats_text,
                archive_id,
            )

    def _extract_parallel(
        self,
        messages: list[dict],
        ctx: RequestContext,
        session_time=None,
        session_summary: str = "",
        tool_stats_text: str = "",
        archive_id: str | None = None,
    ) -> list[CandidateMemory]:
        """Run extractors in parallel using ThreadPoolExecutor.

        Args:
            messages: List of message dicts
            ctx: RequestContext
            session_time: Optional datetime for temporal resolution
            session_summary: Optional summary of previously extracted content
            tool_stats_text: Optional tool usage statistics text
            archive_id: Optional archive_id for provenance tracking

        Returns:
            List of CandidateMemory from all extractors
        """
        all_candidates = []

        with ThreadPoolExecutor(max_workers=len(self._extractors)) as executor:
            # Submit all extraction jobs
            futures = {
                executor.submit(
                    copy_context().run, self._call_extractor,
                    extractor, messages, ctx, session_time, session_summary,
                    tool_stats_text, archive_id,
                ): extractor
                for extractor in self._extractors
            }

            # Collect results as they complete
            for future in as_completed(futures):
                extractor = futures[future]
                try:
                    candidates = future.result()
                    all_candidates.extend(candidates)
                except Exception as e:
                    logger.error(
                        "Extractor %s failed: %s",
                        extractor.__class__.__name__, e,
                        exc_info=True
                    )
                    raise

        # Validate all candidates
        return self._validate_candidates(all_candidates, ctx)

    def _extract_serial(
        self,
        messages: list[dict],
        ctx: RequestContext,
        session_time=None,
        session_summary: str = "",
        tool_stats_text: str = "",
        archive_id: str | None = None,
    ) -> list[CandidateMemory]:
        """Run extractors serially.

        Args:
            messages: List of message dicts
            ctx: RequestContext
            session_time: Optional datetime for temporal resolution
            session_summary: Optional summary of previously extracted content
            tool_stats_text: Optional tool usage statistics text
            archive_id: Optional archive_id for provenance tracking

        Returns:
            List of CandidateMemory from all extractors
        """
        all_candidates = []

        for extractor in self._extractors:
            try:
                candidates = self._call_extractor(
                    extractor, messages, ctx, session_time,
                    session_summary, tool_stats_text, archive_id,
                )
                all_candidates.extend(candidates)
            except Exception as e:
                logger.error(
                    "Extractor %s failed: %s",
                    extractor.__class__.__name__, e,
                    exc_info=True
                )
                raise

        # Validate all candidates
        return self._validate_candidates(all_candidates, ctx)

    @staticmethod
    def _validate_candidates(candidates, ctx):
        return [validate_candidate(validate_attribution(candidate, user_id=ctx.user_id if ctx else None))
                for candidate in candidates]

    @staticmethod
    def _call_extractor(
        extractor,
        messages,
        ctx,
        session_time,
        session_summary,
        tool_stats_text,
        archive_id,
    ):
        """Call one extractor through the stable candidate-extraction contract."""
        return extractor.extract(
            messages,
            ctx,
            session_time,
            session_summary,
            tool_stats_text,
            archive_id,
        )

    def filter_by_confidence(
        self,
        candidates: list[CandidateMemory],
        threshold: float = 0.5
    ) -> list[CandidateMemory]:
        """Filter candidates by confidence threshold.

        Args:
            candidates: List of CandidateMemory
            threshold: Minimum confidence score (default: 0.5)

        Returns:
            Filtered list of CandidateMemory
        """
        return [c for c in candidates if c.confidence >= threshold]

    def deduplicate(
        self,
        candidates: list[CandidateMemory]
    ) -> list[CandidateMemory]:
        """Deduplicate matching candidate identities and abstract prefixes.

        Keeps the candidate with the highest confidence per group.
        This catches duplicates that slip through dual-run merge,
        e.g. same fact extracted from different spans.
        Unbound event drafts require exact equality and retain all source IDs.
        Unbound cases lack reliable run origin here and remain separate, including
        exact drafts from independent spans or extractors.
        """
        seen: dict[tuple[str, ...] | int, CandidateMemory] = {}
        for index, c in enumerate(candidates):
            if c.category in {"event", "case"} and not c.target_uri and (
                c.category == "case" or c.write_operation_id
            ):
                seen[index] = c
                continue
            key = (*candidate_identity_key(c), c.abstract[:80])
            if key in seen and c.category == "event" and not c.target_uri:
                seen[key] = merge_duplicate_event_sources(seen[key], c)
            elif key not in seen or c.confidence > seen[key].confidence:
                seen[key] = c
        return list(seen.values())
