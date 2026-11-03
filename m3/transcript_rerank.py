"""BM25 rerank over ``speech_transcript`` for the M3 [Search] tool.

Pure vector top-k retrieval (``GraphService.search_by_embedding``)
smooths away the exact words a speech-anchored M3-Bench question needs
("album", "fetch the pictures in my bag"). The verbatim Whisper transcript
carries those words. This module fuses a BM25 score over each hit's
``speech_transcript`` into the existing vector score so a clip whose
transcript matches the query's tokens gets rescued up the ranking —
without a second store query (the transcript is already on the in-hand
top-k hits).

Reuses the project's existing lexical infra rather than rebuilding it:

* :func:`retrieval.bm25_index._tokenize` — the NLTK (lowercase → Porter
  stem → English stopword removal) tokenizer. ``rank-bm25`` and ``nltk``
  are already declared deps.
* :meth:`retrieval.seed_retriever.SeedRetriever._fuse_results` — the
  Vector-Anchored Fusion formula this mirrors:
  ``sat = raw_bm25 / (raw_bm25 + k)``,
  ``final = alpha * vec + (1 - alpha) * sat``.

Unlike :class:`retrieval.bm25_index.Bm25Index` (which is ChromaDB-coupled
and keeps a global long-lived index with upsert/delete lifecycle), the
M3 candidate set is the small per-question vector top-k already in hand.
A per-query ``BM25L`` (BM25-Lucene, the small-corpus-safe variant —
see :func:`rerank_with_transcript_bm25` for why not ``BM25Okapi``) over
those few transcripts is the right shape and avoids a second store call
and a second index to keep in sync.
"""

from __future__ import annotations

import dataclasses
import logging

from service.graph_service import EvidenceHit

logger = logging.getLogger(__name__)


def _hit_transcript_text(hit: EvidenceHit) -> str:
    """Flatten one hit's ``speech_transcript`` segments into a string.

    ``speech_transcript`` is a list of ``{start_sec, end_sec, text, ...}``
    dicts (see :func:`m3.index_episodic._format_transcript_block`). We
    only need the ``text`` for BM25; timing is irrelevant to lexical
    match. Non-dict segments are skipped defensively.
    """
    segs = (hit.node.extracted_content or {}).get("speech_transcript") or []
    return " ".join(
        str(seg.get("text", ""))
        for seg in segs
        if isinstance(seg, dict)
    )


def rerank_with_transcript_bm25(
    query: str,
    hits: list[EvidenceHit],
    *,
    alpha: float = 0.5,
    saturation_k: float = 1.0,
    lex_query: str | None = None,
) -> list[EvidenceHit]:
    """Vector-anchored BM25 rerank over each hit's speech transcript.

    Tokenizes the query and every hit's transcript with the shared
    :func:`retrieval.bm25_index._tokenize` (NLTK stem + stopword removal),
    builds a per-query ``BM25L`` over the in-hand hits' transcripts, and
    fuses each raw BM25 score with the hit's existing vector score:

        sat_bm25 = raw_bm25 / (raw_bm25 + saturation_k)
        fused    = alpha * vec_score + (1 - alpha) * sat_bm25

    ``alpha`` defaults to ``0.5`` (vs the global retriever's ``0.7``)
    because here BM25 scores the *verbatim speech* that M3 questions
    anchor on and the vector score is the weaker signal being rescued.
    A 2-hit per-query corpus has a weak IDF, so BM25 is given equal
    weight to keep the lexical signal decisive in the small-candidate
    regime the [Search] tool actually operates in.

    ``saturation_k`` defaults to ``1.0`` (vs the global retriever's
    ``5.0``) because ``BM25L`` on a 2-8-doc corpus emits small raw scores
    (≈1.6-2.2 for a full two-token match); the global ``k=5`` was
    calibrated for ``BM25Okapi`` over the full ChromaDB index where raw
    scores run 5-15. A large ``k`` over-saturates here — it compresses
    the rescuing lexical signal below the vector gap so the rescue never
    fires. ``k=1.0`` keeps ``sat = raw/(raw+1)`` decisive in this raw
    range while still bounding the floor (a speech-free hit stays at
    ``sat=0``, so the rerank only lifts, never sinks).

    Edge cases (all preserved as invariants, pinned by tests):

    * Empty ``query`` or empty ``hits`` → input returned unchanged.
    * No hit has any transcript → hits returned with their vector scores
      (no crash, no spurious reordering).
    * A speech-free hit keeps ``sat = 0`` so its fused score is
      ``alpha * vec`` — it is never penalized below its vector rank by
      the rerank, only lifted when its transcript matches.

    Args:
        query: The natural-language query the LLM passed to [Search].
            Used for the vector embedding (caller's responsibility) and
            as the default BM25 query when ``lex_query`` is ``None``.
        hits:  The vector top-k hits (already cosine-scored).
        alpha: Vector weight in ``[0, 1]`` (default ``0.5``).
        saturation_k: BM25 saturation constant (default ``1.0``, tuned
            for ``BM25L``'s small-corpus raw-score range).
        lex_query: Optional alternative query for BM25 tokenization. Use
            this when the natural-language query contains tokens the
            transcripts don't carry (typical case: the question names a
            person with one spelling while Whisper heard the name
            differently — e.g. "Lewis" vs transcript "Louis/Louise/
            Luis"). Those tokens are 0-hit on the lexical side and only
            dilute the BM25 signal; passing a stripped query lets the
            load-bearing content words ("decoration", "earring") do the
            matching. Defaults to ``query`` for backward compatibility.

    Returns:
        A NEW list of ``EvidenceHit`` (``EvidenceHit`` is frozen, so the
        fused score is set via :func:`dataclasses.replace`). Sorted by
        fused score descending. The input list is not mutated.
    """
    if not hits or not query or not query.strip():
        return list(hits)

    transcript_texts = [_hit_transcript_text(h) for h in hits]
    if not any(text.strip() for text in transcript_texts):
        return list(hits)

    # ``lex_query`` decouples the BM25 token side from the embedder
    # query. Falls back to ``query`` so existing callers (and the
    # project-wide default of "same string for both sides") keep working.
    bm25_query = lex_query if lex_query is not None else query
    # If the caller passed an empty/whitespace-only ``lex_query``
    # (typically the stripping layer ran out of content), fall back to
    # ``query`` so BM25 still fires with the un-stripped tokens. The
    # alternative — short-circuiting on empty lex_query — would silently
    # degrade the rerank to vector-only and lose the rescue signal that
    # is the entire point of this function.
    if not bm25_query or not bm25_query.strip():
        bm25_query = query

    # Reuse the project tokenizer (NLTK stem + stopword removal). Imported
    # lazily so the NLTK data download in ``_tokenize`` only fires when a
    # Search actually runs — matching how ``bm25_index`` defers NLTK.
    from retrieval.bm25_index import _tokenize

    query_tokens = _tokenize(bm25_query)
    if not query_tokens:
        return list(hits)

    transcript_tokens = [_tokenize(t) for t in transcript_texts]

    # If no hit carries any transcript tokens, BM25 can't add signal —
    # return vector-ranked hits unchanged (don't reorder by a zero BM25
    # floor that could tie-break arbitrarily).
    if not any(transcript_tokens):
        return list(hits)

    # BM25L (Lucenel2000), not BM25Okapi: the [Search] rerank operates on
    # the in-hand vector top-k — a 2-8 document corpus. BM25Okapi's IDF,
    # ``log((N - df + 0.5) / (df + 0.5))``, collapses to 0 at 50% doc
    # frequency (df=1, N=2 → log(1) = 0), so a query token that the
    # transcript actually carries scores *zero* and the rescue the whole
    # point of this rerank — never fires. (This is invisible in the
    # production ``Bm25Index`` only because ChromaDB corpora are large
    # enough that df << N.) BM25L lifts the lower-bound IDF floor so a
    # 50%-frequency term still contributes positive score, which is the
    # correct behaviour in the small-candidate regime here.
    from rank_bm25 import BM25L

    bm25 = BM25L(transcript_tokens)
    raw_scores = bm25.get_scores(query_tokens)

    fused: list[EvidenceHit] = []
    for hit, raw_bm25 in zip(hits, raw_scores):
        vec_score = hit.score
        sat_bm25 = (
            raw_bm25 / (raw_bm25 + saturation_k)
            if raw_bm25 > 0
            else 0.0
        )
        fused_score = alpha * vec_score + (1.0 - alpha) * sat_bm25
        fused.append(dataclasses.replace(hit, score=float(fused_score)))

    fused.sort(key=lambda h: h.score, reverse=True)
    return fused
