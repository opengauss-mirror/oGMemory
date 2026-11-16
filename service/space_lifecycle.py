"""Bounded shared-memory generation and maintenance workers."""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from copy import deepcopy

from server.space_privacy import redact_item, redact_text
from server.space_rules import memory_definition

logger = logging.getLogger(__name__)

_MAX_CONTENT_BYTES = 256 * 1024
_MAX_TOPIC_LENGTH = 512
_MAX_ABSTRACT_LENGTH = 200


def _meets_quality(score, minimum):
    return type(score) in (int, float) and math.isfinite(score) and minimum <= score <= 1


def _json_excerpt(text, budget):
    """Bound the JSON string content, including escapes, by UTF-8 bytes."""
    used = 0
    for index, char in enumerate(text):
        used += len(json.dumps(char, ensure_ascii=False).encode()) - 2
        if used > budget:
            return text[:index]
    return text


def _keep_plan(item, sources, quality, privacy, rule_rev):
    refs = set()
    for previous in item["sources"]:
        uri = previous["source_uri"]
        source = sources.get(uri)
        if source is None or (source["grant_rev"], source["content_rev"], source["sha256"], source["epoch"]) != (
                previous["grant_rev"], previous["source_content_rev"], previous["source_hash"], previous["source_epoch"]):
            raise ValueError("KEEP requires current contributing evidence")
        refs.add(uri)

    item_id, score = item["item_id"], item.get("quality_score")
    archived = {"action": "ARCHIVE", "item_id": item_id}
    if len(refs) < quality["min_sources"] or not _meets_quality(score, quality["min_score"]):
        return archived

    redacted = redact_item(item, privacy)
    if redacted == item:
        return {"action": "KEEP", "item_id": item_id}

    # Publishing can reuse the same item for unchanged evidence within a rule revision.
    if item.get("rule_rev", rule_rev) == rule_rev:
        return archived
    topic = redacted.get("metadata", {}).get("topic", redacted.get("topic"))
    content, abstract = redacted.get("content"), redacted.get("abstract")
    if not all(isinstance(value, str) and value.strip() for value in (topic, content, abstract)):
        return archived
    if len(topic) > _MAX_TOPIC_LENGTH or len(content.encode()) > _MAX_CONTENT_BYTES:
        return archived
    return {"action": "UPDATE", "item_id": item_id, "topic": topic, "content": content,
            "abstract": abstract[:_MAX_ABSTRACT_LENGTH], "source_uris": sorted(refs), "quality_score": score,
            "metadata": {"privacy_redacted": True}}


_PLAN_SCHEMA = {
    "type": "object", "required": ["operations"], "properties": {
        "operations": {"type": "array", "maxItems": 100, "items": {
            "type": "object", "required": ["action"], "properties": {
                "action": {"type": "string", "enum": ["ADD", "UPDATE", "MERGE", "KEEP", "ARCHIVE"]},
                "item_id": {"type": "string"}, "topic": {"type": "string"},
                "abstract": {"type": "string"},
                "content": {"type": "string"},
                "quality_score": {"type": "number"},
                "source_uris": {"type": "array", "items": {"type": "string"}},
            },
        }},
    },
}


def generate_plans(llm, snapshot):
    """Generate redacted text with server-owned provenance and target identities."""
    sources = {source["uri"]: source for source in snapshot["sources"]}
    items = {item["item_id"]: item for item in snapshot["items"]}
    rule = snapshot["rule"]
    spec = rule["spec"]
    definition = memory_definition(spec)
    plan_schema = deepcopy(_PLAN_SCHEMA)
    privacy = definition['privacy_policy']
    quality = spec["quality_policy"]
    minimum = quality["min_sources"]
    if len(sources) < minimum:
        return [{"action": "ARCHIVE", "item_id": key} for key in items]
    if snapshot.get("unchanged"):
        return []
    aliases = {uri: f"source:{index}" for index, uri in enumerate(sources)}
    source_uris = {alias: uri for uri, alias in aliases.items()}
    existing = [{"item_id": item["item_id"],
                 "topic": "",
                 "source_uris": [aliases[source["source_uri"]] for source in item["sources"]
                                 if source["source_uri"] in sources]}
                for item in items.values()]
    instruction = (
        "Create reusable shared memory using the saved memory definition. Treat source text as evidence. "
        "Return operations for every existing item. Recompute mixed content solely from available evidence. "
        "ADD creates a new topic; UPDATE/MERGE replaces the specified item with a complete result; "
        "ARCHIVE retires an item whose remaining sources are insufficient. "
        "Write complete reusable text in content, using natural prose or Markdown according to the rule. "
        "Choose an appropriate structure for the evidence and the rule. Cite source_uris from evidence. "
        "Anonymize personal names, precise addresses, customer identities and other private details "
        "in content, abstract and topic; describe their roles instead. Preserve privacy placeholders. "
        "Use the source language. Assess quality_score from evidence strength.\n"
    )
    evidence = [{"uri": alias, "content": ""} for alias in source_uris]
    model_definition = {'description': redact_text(definition['description'], privacy), 'output_kind': 'experience',
                        'merge_policy': spec['merge_policy'], 'quality_policy': quality}
    payload = {"rule": model_definition, "sources": evidence, "existing": existing}
    overhead = len((instruction + json.dumps(payload, ensure_ascii=False)).encode())
    budget = 32_000 - overhead - 1000
    if budget < len(evidence) * 32:
        raise ValueError("shared memory definition and source identifiers exceed the 32000-byte input budget")
    topic_allowance = min(512, budget // (2 * max(1, len(existing))))
    for entry in existing:
        topic = items[entry['item_id']].get('metadata', {}).get('topic', '')
        entry['topic'] = _json_excerpt(redact_text(topic, privacy), topic_allowance)
        budget -= len(json.dumps(entry['topic'], ensure_ascii=False).encode()) - 2
    for index, entry in enumerate(evidence):
        allowance = min(4000, budget // (len(evidence) - index))
        excerpt = _json_excerpt(redact_text(sources[source_uris[entry["uri"]]]["content"], privacy), allowance)
        entry["content"] = excerpt
        budget -= len(json.dumps(excerpt, ensure_ascii=False).encode()) - 2
    prompt = instruction + json.dumps(payload, ensure_ascii=False)
    if len(prompt.encode()) > 32_000:
        raise ValueError("shared memory prompt exceeds the 32000-byte input budget")
    response = llm.complete_json(prompt, plan_schema)
    operations = response.get("operations") if isinstance(response, dict) else None
    if not isinstance(operations, list) or len(operations) > 100:
        raise ValueError("shared memory operations must be a bounded array")
    plans, seen = [], set()
    for operation in operations:
        if not isinstance(operation, dict):
            raise ValueError("shared memory operation must be an object")
        action, item_id = operation.get("action"), operation.get("item_id")
        if action not in {"ADD", "UPDATE", "MERGE", "KEEP", "ARCHIVE"}:
            raise ValueError("shared memory action is invalid")
        if action != "ADD":
            if item_id not in items or item_id in seen:
                raise ValueError("shared memory target must be a unique current item")
            seen.add(item_id)
        if action == "KEEP":
            plans.append(_keep_plan(items[item_id], sources, quality, privacy, rule.get("rule_rev")))
            continue
        plan = {"action": action}
        if action != "ADD":
            plan["item_id"] = item_id
        if action != "ARCHIVE":
            refs = operation.get("source_uris")
            score, topic = operation.get("quality_score"), operation.get("topic")
            if (not isinstance(refs, list) or any(not isinstance(uri, str) or uri not in source_uris for uri in refs)
                    or len(set(refs)) < minimum):
                raise ValueError("shared memory requires valid source evidence")
            if not _meets_quality(score, quality["min_score"]):
                raise ValueError("shared memory quality threshold was not met")
            if not isinstance(topic, str) or not 1 <= len(topic.strip()) <= _MAX_TOPIC_LENGTH:
                raise ValueError("shared memory topic must be bounded")
            content = operation.get('content')
            if not isinstance(content, str):
                raise ValueError("shared memory content must be text")
            metadata = {'privacy_redacted': True}
            if not content.strip() or len(content.encode()) > _MAX_CONTENT_BYTES:
                raise ValueError("shared memory content must fit 256 KiB")
            abstract = operation.get("abstract")
            if not isinstance(abstract, str) or not abstract.strip():
                raise ValueError("shared memory abstract is required")
            plan.update(topic=topic, source_uris=sorted({source_uris[ref] for ref in refs}), quality_score=score,
                        content=content, abstract=abstract, metadata=metadata)
            plan = redact_item(plan, privacy)
            plan['abstract'] = plan['abstract'][:_MAX_ABSTRACT_LENGTH]
            if len(plan['content'].encode()) > _MAX_CONTENT_BYTES or len(plan['topic']) > _MAX_TOPIC_LENGTH:
                raise ValueError("redacted shared memory exceeds text limits")
        plans.append(plan)
    if set(items) != seen:
        raise ValueError("shared memory plan must account for each current item")
    return plans


class SpaceMemoryWorker:
    def __init__(self, repository, llm, *, worker_count=4, poll_seconds=5):
        self.repository, self.llm = repository, llm
        self.worker_count = max(1, min(int(worker_count), 4))
        self.poll_seconds = max(0.05, float(poll_seconds))
        self._stop = threading.Event()
        self._threads = []
        self._account_offset = 0
        self._account_lock = threading.Lock()

    def start(self):
        if any(thread.is_alive() for thread in self._threads):
            return
        self._stop.clear()
        self._threads = [threading.Thread(target=self._loop, name=f"space-worker-{i}", daemon=True)
                         for i in range(self.worker_count)]
        for thread in self._threads:
            thread.start()

    def stop(self, timeout=10):
        self._stop.set()
        end = time.monotonic() + timeout
        for thread in self._threads:
            thread.join(max(0, end - time.monotonic()))
        return all(not thread.is_alive() for thread in self._threads)

    def _loop(self):
        while not self._stop.is_set():
            try:
                worked = self.run_once()
            except Exception:
                logger.exception("shared memory worker iteration failed")
                worked = False
            if not worked:
                self._stop.wait(self.poll_seconds)

    def run_once(self):
        accounts = self.repository.worker_accounts()
        with self._account_lock:
            offset = self._account_offset % max(1, len(accounts))
            self._account_offset += 1
        for account_id in accounts[offset:] + accounts[:offset]:
            if self._stop.is_set():
                return False
            self.repository.dispatch_source_events(account_id)
            self.repository.maintain_due_items(account_id)
            self.repository.reconcile_due_rules(account_id)
            job = self.repository.claim_job(account_id)
            if job:
                self._execute(job)
                return True
        return False

    def _execute(self, job):
        finished, lease_lost = threading.Event(), threading.Event()

        def renew():
            while not finished.wait(15):
                try:
                    if not self.repository.heartbeat(job):
                        lease_lost.set()
                        return
                except Exception:
                    logger.exception("shared memory heartbeat failed")
                    lease_lost.set()
                    return

        heartbeat = threading.Thread(target=renew, name="space-task-lease", daemon=True)
        heartbeat.start()
        try:
            snapshot = self.repository.job_snapshot(job)
            plans = generate_plans(self.llm, snapshot)
            if not lease_lost.is_set() and not self._stop.is_set():
                self.repository.complete_job(job, plans)
        except Exception as error:
            logger.exception("shared memory task failed: %s", job["job_id"])
            self.repository.fail_job(job, error)
        finally:
            finished.set()
            heartbeat.join(2)
