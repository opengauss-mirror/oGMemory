"""Bounded message retrieval and verified memory hints shared by all transports."""

from __future__ import annotations

import json
import re
import time

from core.errors import NodeNotFoundError
from core.evidence import evidence_uri
from core.evidence_query import EvidenceQuery, MessageEvidenceResult, facts_supported_by
from core.operation_deadline import remaining_seconds
from core.evidence_timing import operation, request_id, stage, timed
from providers.token_tracker import request_usage_scope
from retrieval.evidence.ranking import (
    fact_matches,
    lexical_scores,
    query_branches,
    query_topics,
    rank_lexical,
    select_groups,
    supplement_scores,
    terms,
    topic_terms,
)
from service.evidence_store import EvidenceStore


def _add_archive_hints(node, sources, memory_texts):
    from core.provenance_resolver import ProvenanceResolver

    for provenance in node.metadata.get("provenance_ids", []):
        try:
            parsed = ProvenanceResolver.parse_id(provenance)
        except ValueError:
            continue
        if parsed["source_type"] != "archive":
            continue
        for source in sources.values():
            if parsed["source_id"] in source["archive_ids"] and source["message_id"] in parsed["detail"]:
                for unit in source["units"]:
                    memory_texts.setdefault(unit["evidence_unit_id"], node.content)


class EvidenceService:
    def __init__(self, fs):
        self.store = EvidenceStore(fs)

    def search(self, query, ctx, **options):
        from extraction.deterministic.nlp import reuse_documents

        request = EvidenceQuery(query=query, **options)
        with request_usage_scope(ctx), operation("READ_EVIDENCE", ctx.trace_id), reuse_documents():
            return self._search(request, ctx)

    def _search(self, request, ctx):
        query = request.query
        started = time.monotonic()
        deadline = started + min(20, remaining_seconds() or 20)
        diagnostics = {"partial": False, "truncated": False, "scanned_sources": 0, "scanned_cards": 0,
                       "verified_facts": 0, "rejected_facts": 0, "unverifiable_cards": 0, "channels": {
                           name: {"enabled": getattr(request, "use_" + name), "available": True}
                           for name in ("memories", "events", "time")}}
        with stage("source_scan"):
            units, sources, source_bytes = {}, {}, 0
            source_uris = self.store.iter_uris(evidence_uri(ctx, "sources"), ctx, 10001)
            for uri, read_source in self.store.iter_sources(source_uris, ctx, read_limit=10000, deadline=deadline):
                if diagnostics["scanned_sources"] >= 10000 or time.monotonic() >= deadline:
                    diagnostics.update(partial=True, truncated=True)
                    break
                diagnostics["scanned_sources"] += 1
                source = read_source()
                if source is None:
                    continue
                source_bytes += len(source["text"].encode("utf-8"))
                if source_bytes > 16 * 1024 * 1024:
                    diagnostics.update(partial=True, truncated=True)
                    break
                sources[source["source_key"]] = source
                for unit in source["units"]:
                    units[unit["evidence_unit_id"]] = {
                        **unit, **{name: source[name] for name in ("source_key", "session_id", "message_id",
                            "content_hash", "recorded_at", "sequence", "source_node_version")},
                        "source_uri": uri,
                        "sequence_verified": source.get("sequence_verified", True),
                    }
        with stage("lexical_fusion"):
            lexical = lexical_scores(query, units)
        with stage("fusion"):
            scores = rank_lexical(query, units, lexical, request.candidate_limit)
            # Bounded subqueries add missing query branches by strongest support only.
            branches = query_branches(query)
            supplementary = [part for part, _ in branches[:request.max_supplementary_queries]] if len(branches) > 1 else []
            for branch in supplementary:
                branch_scores = rank_lexical(branch, units, lexical_scores(branch, units), request.candidate_limit)
                for key, score in branch_scores.items():
                    scores[key] = max(scores.get(key, 0), score)
            diagnostics["supplementary_queries"] = supplementary
        records, events, card_bytes = [], [], 0
        topics = query_topics(query, units)
        speaker_terms = {word for unit in units.values() for word in terms(unit.get("speaker", ""))}
        context_terms = topics - speaker_terms
        memory_texts = {}
        context_hints, hint_attempts = set(), 0
        hint_topics_by_unit = {}
        with stage("card_scan"):
            if request.use_memories or request.use_events:
                from extraction.deterministic.verification import record_verifier

                verify = record_verifier(self.store, ctx)
                directories = (["facts", "entities", "profile", "preferences", "cases", "patterns"]
                               if request.use_memories else []) + (["events"] if request.use_events else [])
                for directory in directories:
                    prefix = f"ctx://{ctx.account_id}/users/{ctx.user_id}/memories/{directory}"
                    card_uris = [prefix] if directory == "profile" else self.store.iter_uris(prefix, ctx, 2001)
                    stop_cards = False
                    for page in self.store.node_pages(card_uris, ctx, read_limit=2000 - diagnostics["scanned_cards"], deadline=deadline):
                        for uri, value in page:
                            if diagnostics["scanned_cards"] >= 2000 or time.monotonic() >= deadline:
                                diagnostics.update(partial=True, truncated=True)
                                stop_cards = True
                                break
                            diagnostics["scanned_cards"] += 1
                            if isinstance(value, NodeNotFoundError):
                                continue
                            if isinstance(value, Exception):
                                raise value
                            node = value
                            size = len(node.content.encode("utf-8"))
                            card_bytes += size
                            if size > 2 * 1024 * 1024 or card_bytes + source_bytes > 16 * 1024 * 1024:
                                diagnostics.update(partial=True, truncated=True)
                                stop_cards = True
                                break
                            data = node.metadata.get("deterministic", {})
                            if not data:
                                if request.use_memories and topics & set(terms(node.content)):
                                    if not node.metadata.get("provenance_ids"):
                                        diagnostics["unverifiable_cards"] += 1
                                    _add_archive_hints(node, sources, memory_texts)
                                continue
                            candidates = data.get("fact_records", [])
                            for record in candidates:
                                if time.monotonic() >= deadline:
                                    diagnostics.update(partial=True, truncated=True)
                                    break
                                refs = record.get("source_refs", [])
                                fact_eligible = bool(topics & set(terms(record.get("quote", "")))) and fact_matches(query, record)
                                hints = set()
                                if (not fact_eligible and refs and request.use_memories and node.category != "event"
                                        and hint_attempts < request.candidate_limit
                                        and len(context_hints) < request.candidate_limit):
                                    for ref in refs:
                                        key = ref["evidence_unit_id"]
                                        if key not in units or key in scores or key in context_hints:
                                            continue
                                        unit = units[key]
                                        # Names and greetings alone cannot rescue an unrelated source.
                                        if key not in hint_topics_by_unit:
                                            names = {word.casefold() for word in re.findall(r"\b[A-Z][a-z]+\b", unit["text"])}
                                            hint_topics_by_unit[key] = context_terms & (set(terms(unit["text"])) - names)
                                        if hint_topics_by_unit[key] - set(terms(record.get("subject", ""))):
                                            hints.add(key)
                                if not refs or not (fact_eligible or hints):
                                    diagnostics["rejected_facts"] += 1
                                    continue
                                if hints and not fact_eligible:
                                    hint_attempts += 1
                                if not verify(record):
                                    diagnostics["rejected_facts"] += 1
                                    continue
                                # Weaker than channel heads; no duplicate vote or structured fact.
                                for key in sorted(hints)[:request.candidate_limit - len(context_hints)]:
                                    context_hints.add(key)
                                    memory_texts.setdefault(key, units[key]["text"])
                                if not fact_eligible:
                                    continue
                                if node.category == "event" and request.use_events:
                                    events.append((node.uri, record))
                                elif request.use_memories and node.category != "event":
                                    records.append(record)
                        if stop_cards:
                            break
                    if diagnostics["truncated"]:
                        break
        with stage("fusion"):
            # At most two distinct, source-grounded events. Name-only matches are insufficient.
            admitted_events = set()
            event_support = {}
            def event_priority(item):
                uri, record = item
                topic = topic_terms(record["quote"]) - set(terms(record["subject"]))
                exact = bool(record.get("event_time") and record["event_time"] in query)
                return (-int(exact), -len(topic & topics), uri, record["fact_id"])
            for uri, record in sorted(events, key=event_priority):
                non_subject = topic_terms(record["quote"]) - set(terms(record["subject"]))
                if not non_subject & topics or uri not in admitted_events and len(admitted_events) == 2:
                    continue
                admitted_events.add(uri)
                records.append(record)
                for ref in record["source_refs"]:
                    event_support.setdefault(ref["evidence_unit_id"], set()).add(uri)
            unique = {record["fact_id"]: record for record in records}
            for record in unique.values():
                for ref in record["source_refs"]:
                    key = ref["evidence_unit_id"]
                    if key in units and topics & set(terms(record["quote"])):
                        memory_texts[key] = memory_texts.get(key, "") + "\n" + record["quote"]
            for key, score in supplement_scores(query, units, memory_texts, request.candidate_limit).items():
                scores.setdefault(key, score)
            event_bonus = 0.0
            for key, event_uris in event_support.items():
                if key in scores:
                    bonus = min(.004, .002 * len(event_uris))
                    scores[key] += bonus
                    event_bonus += bonus
            diagnostics.update(context_hints=len(context_hints), context_hint_attempts=hint_attempts,
                               events_admitted=len(admitted_events), events_matched=len(events),
                               event_bonus=round(event_bonus, 6))
        with stage("auxiliary"):
            from retrieval.evidence.auxiliary import fuse_auxiliary
            scores, auxiliary = fuse_auxiliary(request, scores, units, sources, deadline=deadline)
            diagnostics["auxiliary"] = auxiliary
        with stage("format"):
            # Count the complete serialized units and their facts, not only raw text.
            @timed("serialize")
            def encoded_size(value):
                return len(json.dumps(value, ensure_ascii=False).encode("utf-8"))
            costs = {key: encoded_size(unit) + 2 for key, unit in units.items()}
            for record in unique.values():
                for ref in record["source_refs"]:
                    if ref["evidence_unit_id"] in costs:
                        costs[ref["evidence_unit_id"]] += encoded_size(record) + 2
            selected, groups = select_groups(
                query, scores, units, request.top_k, request.candidate_limit,
                branch_terms=[words for _, words in branches], costs=costs,
                byte_limit=56 * 1024 - encoded_size(query),
            )
            if len(selected) < min(len(scores), request.top_k):
                diagnostics["truncated"] = True
        with stage("final_source_check"):
            evidence = []
            for key in selected:
                item = units[key]
                current = self.store.read_source(item["source_key"], ctx)
                if current is not None and current["source_node_version"] == item["source_node_version"]:
                    evidence.append(item)
                else:
                    diagnostics["partial"] = True
        with stage("format"):
            available = {item["evidence_unit_id"] for item in evidence}
            groups = [group for group in groups if set(group) <= available]
            retained = {key for group in groups for key in group}
            evidence = [item for item in evidence if item["evidence_unit_id"] in retained]
            selected_ids = {item["evidence_unit_id"] for item in evidence}
            facts = facts_supported_by(unique.values(), selected_ids)
            diagnostics.update(verified_facts=len(facts),
                               selected_evidence_unit_ids=sorted(selected_ids),
                               source_bytes=source_bytes, card_bytes=card_bytes,
                               index_ready=bool(sources), elapsed_ms=round((time.monotonic() - started) * 1000))
            diagnostics["evidence_groups"] = groups
            result = MessageEvidenceResult(ctx.trace_id or request_id(), query, evidence,
                                        sorted(facts, key=lambda record: record["fact_id"]), diagnostics)
            def refresh_coverage():
                final_terms = set().union(*(set(terms(item["text"])) for item in result.evidence))
                diagnostics["uncovered_branches"] = [index for index, branch in enumerate(branches)
                                                      if not branch[1] & final_terms]
            refresh_coverage()
            while groups and encoded_size(result.to_dict()) > 64 * 1024 - 64:
                removed = set(groups.pop()) - {key for group in groups for key in group}
                result.evidence = [item for item in result.evidence if item["evidence_unit_id"] not in removed]
                selected_ids -= removed
                result.facts = facts_supported_by(result.facts, selected_ids)
                diagnostics.update(truncated=True, verified_facts=len(result.facts),
                                   selected_evidence_unit_ids=sorted(selected_ids))
                refresh_coverage()
            diagnostics["response_bytes"] = encoded_size(result.to_dict())
        return result
