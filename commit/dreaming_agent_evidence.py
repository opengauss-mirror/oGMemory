"""Assess the source bindings of active structured Agent memory units."""

from hashlib import sha256

from commit.agent_memory import reviewed_agent_units
from extraction.agent_memory import (
    agent_source_body,
    agent_source_id,
    memory_units,
    node_memory_fields,
    project_trace,
)


def _source_events(events, category):
    result = {}
    last_answer = None
    for event in project_trace(events).get(category, ()):
        ordinal = event["ordinal"]
        kind = event["event_type"]
        answer = event.get("assistant_message")
        occurrence = f"event:{ordinal}"
        if tool_id := event.get("tool_use_id"):
            occurrence = f"tool:{tool_id}"
        if kind == "assistant_message" and answer:
            last_answer = event
        if (kind == "turn_end" and last_answer and answer == last_answer["assistant_message"]
                and all(not event.get(field) or event[field] == last_answer.get(field)
                        for field in ("reasoning", "assistant_messages"))):
            occurrence = f"event:{last_answer['ordinal']}"
        result[str(ordinal)] = event, occurrence
    return result


def _proofs(node, unit, read_source, traces):
    if read_source is None:
        return
    provenance = node.metadata.get("provenance_ids") or ()
    prefix = "/".join(node.uri.split("/", 3)[:3]) + "/traces/"
    for source in unit.get("grounding") or []:
        if not isinstance(source, dict) or source.get("role") != "source" or source.get("lane") != node.category:
            continue
        uri = source.get("uri")
        if not isinstance(uri, str) or not uri.startswith(prefix):
            continue
        trace_uri, _, event_id = uri.partition("#event-")
        if trace_uri not in provenance or source.get("source_id") != agent_source_id(uri):
            continue
        if trace_uri not in traces:
            events = read_source(trace_uri)
            traces[trace_uri] = _source_events(events, node.category) if isinstance(events, list) else {}
        source_event = traces[trace_uri].get(event_id)
        if source_event is None:
            continue
        event, occurrence = source_event
        if source.get("origin") != event["event_type"]:
            continue
        body = agent_source_body(event)
        revision = sha256(body.encode()).hexdigest()
        quote = source.get("quote", "")
        if (not body or source.get("content_hash") != revision or source.get("resource_version") != revision
                or not isinstance(quote, str) or (quote and quote not in body)):
            continue
        yield {"source_ref": uri, "occurrence_id": f"{trace_uri}#{occurrence}", "weight": 4,
               "supports": True, "quote": quote, "source_revision": revision}


def agent_units(node, read_source):
    if node.category not in ("tool", "experience") or node.metadata.get("agent_memory_version") != 1:
        return None
    traces = {}
    try:
        items = {item["key"]: item for item in memory_units(node.category, node_memory_fields(node))}
    except (KeyError, ValueError):
        return {}
    reviewed = reviewed_agent_units(items, node.metadata.get("routing_key"),
                                    node.metadata.get("agent_memory_value_review"),
                                    node.metadata.get("agent_memory_unit_reviews"))
    return {key: (unit, list(_proofs(node, unit, read_source, traces)), key in reviewed)
            for key, unit in items.items()}
