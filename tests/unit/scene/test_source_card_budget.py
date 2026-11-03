"""Source evidence retains the longest JSON-budgeted text prefix."""

import json
from dataclasses import replace

import pytest

from fs.agfs_adapter.agfs_context_fs import AGFSContextFS
from scene.adapters import CanonicalSceneAdapter, _scope_context
from scene.config import SceneConfig
from session.archive_store import SessionArchiveStore
from tests.unit.fs.test_agfs_context_fs import _MemoryAGFSClient
from tests.unit.scene.test_adapters import Embedder, IndexedVectors, Repository, node

CTX = _scope_context("acme", "user:alice")
BUDGET = 1024


def source_with_text(kind, content):
    fs = AGFSContextFS(_MemoryAGFSClient())
    if kind == "memory":
        item = replace(node("source"), content=content)
        fs.write_node(item, CTX)
        reference, text = item.uri, content
    else:
        result = SessionArchiveStore(fs).write_archive(
            "planning", "Plan", "Plan", [{"id": "m1", "role": "user", "content": content}],
            CTX, archive_id="source",
        )
        assert result.success, result.error
        reference, text = result.uri, f"[m1]  user: {content}\n"
    source = CanonicalSceneAdapter(
        Repository(), fs, IndexedVectors([]), Embedder(),
        config=replace(SceneConfig(), candidate_card_budget_tokens=BUDGET // 4),
    )
    return source, reference, text


def assert_maximal_prefix(card, text, budget):
    remaining = budget - len(json.dumps({**card, "text": ""}, ensure_ascii=False))
    end = 0
    for character in text:
        cost = len(json.dumps(character, ensure_ascii=False)) - 2
        if cost > remaining:
            break
        remaining -= cost
        end += 1
    assert card["text"] == text[:end]
    assert len(json.dumps(card, ensure_ascii=False)) <= budget


@pytest.mark.parametrize("kind", ["memory", "archive"])
@pytest.mark.parametrize("repeats", [2, 2000])
@pytest.mark.parametrize("fragment", ["plain", '"', "\\", "\n\t\r", "\x01", "中文😀", 'A"\\\n\x01中😀'])
def test_source_cards_fill_encoded_budget(kind, repeats, fragment):
    source, reference, text = source_with_text(kind, fragment * repeats)
    cards = source.read_source_context("acme", "user:alice", [reference], 1)
    assert len(cards) == 1
    assert cards[0]["source_ref"] == reference
    assert_maximal_prefix(cards[0], text, BUDGET)


@pytest.mark.parametrize("kind", ["memory", "archive"])
def test_source_cards_fill_remaining_shared_budget(kind):
    source, reference, text = source_with_text(kind, 'A"\\\n\x01中😀' * 2000)
    first = replace(node("first"), content='First "evidence"\n')
    source.fs.write_node(first, CTX)
    cards = source.read_source_context("acme", "user:alice", [first.uri, reference], 2)
    assert len(cards) == 2
    assert cards[0]["text"] == first.content
    remaining = BUDGET - len(json.dumps(cards[0], ensure_ascii=False))
    assert_maximal_prefix(cards[1], text, remaining)


def test_archive_budget_preserves_complete_messages_and_partial_next_message():
    source, reference, _ = source_with_text("archive", "initial")
    messages = [{"id": "m1", "role": "user", "content": 'First "evidence"\n'},
                {"id": "m2", "role": "assistant", "content": "Next\t\x01" * 2000},
                {"id": "m3", "role": "user", "content": "Final evidence"}]
    saved = SessionArchiveStore(source.fs).write_archive(
        "planning", "Plan", "Plan", messages, CTX, archive_id="source",
    )
    assert saved.success, saved.error
    cards = source.read_source_context("acme", "user:alice", [reference], 1)
    assert len(cards) == 1
    assert cards[0]["message_ids"] == ["m1", "m2"]
    text = "".join(f"[{message['id']}]  {message['role']}: {message['content']}\n" for message in messages[:2])
    assert_maximal_prefix(cards[0], text, BUDGET)


@pytest.mark.parametrize("field", ["content", "overview", "abstract"])
def test_memory_source_prefix_fills_each_encoding_boundary(field):
    text = "A" + "".join(chr(value) for value in range(32)) + '"\\中文😀'
    item = replace(node("source"), content="", overview="", abstract="")
    setattr(item, field, text)
    complete = CanonicalSceneAdapter._source_node_card(item, item.uri, BUDGET)
    overhead = len(json.dumps({**complete, "text": ""}, ensure_ascii=False))
    for available in range(1, len(json.dumps(text, ensure_ascii=False))):
        card = CanonicalSceneAdapter._source_node_card(item, item.uri, overhead + available)
        assert card is not None
        assert_maximal_prefix(card, text, overhead + available)
