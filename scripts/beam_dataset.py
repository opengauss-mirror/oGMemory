"""BEAM parquet readers, keeping source ingestion separate from held-out probes."""
from __future__ import annotations

import ast
import json
from pathlib import Path

from scripts.beam_official import CATEGORIES


def _decoded(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return ast.literal_eval(value)
    return value


def _row(parquet: Path, index: int, columns: list[str]) -> dict:
    if type(index) is not int or index < 0:
        raise ValueError("conversation-index must be a non-negative integer")
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise RuntimeError("Install BEAM dependencies: pip install -r scripts/requirements-beam.txt") from None
    rows = pq.read_table(parquet, columns=columns).slice(index, 1).to_pylist()
    if len(rows) != 1:
        raise ValueError("BEAM conversation row is missing")
    return rows[0]


def load_source_pairs(parquet: Path, index: int = 0) -> tuple[object, list[list[dict]]]:
    """Read ONLY conversation_id/chat; never load questions, answers or rubrics."""
    row = _row(parquet, index, ["conversation_id", "chat"])
    segments = _decoded(row["chat"])
    if not isinstance(segments, list) or not segments:
        raise ValueError("BEAM chat must contain source segments")
    messages = []
    for segment in segments:
        segment = _decoded(segment)
        if not isinstance(segment, list):
            raise ValueError("BEAM source segment must contain messages")
        messages.extend(segment)
    if not messages or len(messages) % 2:
        raise ValueError("Expected complete user/assistant source pairs")
    pairs = [messages[i:i + 2] for i in range(0, len(messages), 2)]
    for pair in pairs:
        if not all(isinstance(message, dict) for message in pair):
            raise ValueError("BEAM source message must be an object")
        if [message.get("role") for message in pair] != ["user", "assistant"]:
            raise ValueError("Unexpected BEAM source pair roles")
        for message in pair:
            if not isinstance(message.get("content"), str) or not message["content"].strip():
                raise ValueError("Source message content must not be empty")
            identity = message.get("id")
            if isinstance(identity, bool) or not isinstance(identity, (int, str)) or identity == "":
                raise ValueError("Source message ID is missing")
    return row["conversation_id"], pairs


def load_questions(parquet: Path, index: int = 0) -> tuple[object, dict]:
    """Load all ten official categories in their pinned, deterministic order."""
    row = _row(parquet, index, ["conversation_id", "probing_questions"])
    probes = _decoded(row["probing_questions"])
    if not isinstance(probes, dict) or set(probes) != set(CATEGORIES):
        raise ValueError("Expected all ten official BEAM categories")
    ordered = {}
    for category in CATEGORIES:
        questions = probes[category]
        if not isinstance(questions, list) or len(questions) != 2:
            raise ValueError("Expected two probes per official BEAM category")
        for question in questions:
            if not isinstance(question, dict) or not isinstance(question.get("question"), str) or not question["question"].strip():
                raise ValueError("Probe question must not be empty")
            rubrics = question.get("rubric")
            if not isinstance(rubrics, list) or not rubrics or not all(
                isinstance(item, str) and item.strip() for item in rubrics
            ):
                raise ValueError("Probe rubrics must contain non-empty strings")
        ordered[category] = questions
    return row["conversation_id"], ordered
