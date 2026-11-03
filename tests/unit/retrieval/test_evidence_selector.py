from types import SimpleNamespace
from unittest.mock import Mock
import pytest

from core.models import ContextNode, RequestContext, RetrievedBlock
from retrieval.context_reader import ContextReader
from retrieval.evidence_selector import EvidenceLimits, hydrate_evidence, select_evidence


@pytest.mark.parametrize("body_field", ["content", "overview"])
@pytest.mark.parametrize("identity", ["Distinct receiver identity. " * 4, "接收器的识别信息。" * 5])
def test_small_budget_delivers_ranked_prefix_with_source(body_field, identity):
    items = [{"uri": f"ctx://acct/resources/rank-{i:02}", "abstract": identity,
              body_field: f"Receiver {i} operates at {37+i} MHz.", "_evidence_diagnostic": {}}
             for i in range(15)]
    text, stats = select_evidence(items, "37 MHz", 1000)
    delivered = [i for i, item in enumerate(items) if item["uri"] in text]
    assert delivered == list(range(len(delivered)))
    assert "Receiver 0 operates at 37 MHz." in text
    assert len(text) <= 1000
    assert stats["output_chars"] == len(text)


def test_full_fit_keeps_unequal_header_sizes():
    from retrieval.evidence_selector import _heading
    items = [{"uri": "ctx://acct/resources/first", "abstract": "identity " * 40, "content": "Fact A."},
             {"uri": "ctx://acct/resources/second", "content": "Fact B."}]
    budget = sum(len(_heading(i)) + 1 + len(i["content"]) for i in items) + 2
    text, stats = select_evidence(items, "Fact", budget)
    assert all(i["content"] in text for i in items)
    assert len(text) == budget
    assert stats["ranking"] == "full_content"


@pytest.mark.parametrize("body_field", ["content", "overview"])
def test_body_delivery_is_confirmed_when_budget_becomes_available(body_field):
    from retrieval.evidence_selector import _heading
    item = {"uri": "ctx://acct/resources/receiver", "abstract": "Receiver identity",
            body_field: "Receiver operates at 37 MHz. " * 20}
    _, first = select_evidence([item], "frequency", len(_heading(item)) + 1)
    assert first["delivered_uris"] == []
    text, second = select_evidence([item], "frequency", 2000)
    assert item[body_field] in text
    assert second["delivered_uris"] == [item["uri"]]


def test_budget_admits_next_rank_when_first_header_exceeds_capacity():
    items = [{"uri": "ctx://acct/resources/large", "abstract": "A" * 512, "content": "First fact."},
             {"uri": "ctx://acct/resources/small", "content": "Second fact."}]
    text, stats = select_evidence(items, "fact", 200)
    assert "Second fact." in text
    assert len(text) <= 200


def test_ranked_prefix_reserves_passage_heading_and_range_marker():
    from retrieval.evidence_selector import _heading
    items = [{"uri": f"ctx://acct/resources/{i}", "content": "x" * 192,
              "_evidence_diagnostic": {}} for i in range(2)]
    budget = 2 * (len(_heading(items[0])) + 67)
    text, stats = select_evidence(items, "x", budget)
    assert items[0]["_evidence_diagnostic"]["delivered_chars"] > 0
    assert len(text) <= budget


def test_near_fit_cards_keep_evidence_with_entry_separators():
    from retrieval.evidence_selector import _heading
    prefix = "# Receiver\n\n## Data\n- It operates at 37 MHz.\n- "
    items = [{"uri": f"ctx://acct/users/user/memories/entities/{key}", "category": "entity",
              "abstract": "A receiver.", "content": prefix + "x" * (size - len(prefix)),
              "_evidence_diagnostic": {}}
             for key, size in (("aaaa", 459), ("bbbb", 458))]
    budget = 2 * (len(_heading(items[0])) + 460)

    text, stats = select_evidence(items, "37 MHz", budget)

    assert text.count("37 MHz") == 2
    assert len(text) <= budget
    assert all(item["_evidence_diagnostic"]["delivered_chars"] > 0 for item in items)
    assert stats["passage_count"] <= EvidenceLimits().max_blocks


def _ctx():
    return RequestContext(
        account_id="acct", user_id="user", agent_id="agent", session_id="session",
        trace_id="trace", visible_owner_spaces=("user:user", "agent:agent"),
    )


def _node(uri, *, context_type="MEMORY"):
    return ContextNode(
        uri=uri, context_type=context_type, category="entity", level=2,
        owner_space="user:user", abstract="Canonical identity",
        overview="Short overview", content="# Card\n\nExact L2 fact.\n",
        metadata={"when": "2026-10-02", "source_uri": "ctx://acct/sessions/source"},
    )


def test_context_reader_returns_canonical_levels_and_metadata():
    uri = "ctx://acct/users/user/memories/entities/reference"
    node = _node(uri)
    fs = SimpleNamespace(read_node=Mock(return_value=node))
    ctx = _ctx()

    block = ContextReader(fs=fs, storage_backend="sql").read(uri + "/content.md", ctx)

    fs.read_node.assert_called_once_with(uri, ctx)
    assert block.abstract == node.abstract
    assert block.overview == node.overview
    assert block.content_excerpt == node.content
    assert block.metadata == node.metadata
    assert block.has_content is True
    block.metadata["when"] = "2026-10-03"
    assert node.metadata["when"] == "2026-10-02"


def test_context_reader_reads_account_resource_with_visible_owner_spaces():
    uri = "ctx://acct/resources/reference"
    node = _node(uri, context_type="RESOURCE")
    fs = SimpleNamespace(read_node=Mock(return_value=node))
    ctx = _ctx()

    block = ContextReader(fs=fs, storage_backend="sql").read(uri, ctx)

    assert block.content_excerpt == node.content
    fs.read_node.assert_called_once_with(uri, ctx)


def test_hydration_deduplicates_and_preserves_search_and_node_metadata():
    uri = "ctx://acct/users/user/memories/entities/reference"
    read = Mock(return_value=RetrievedBlock(
        uri=uri, abstract="Canonical identity", content_excerpt="Full source text",
        metadata={"when": "2026-10-02"},
    ))
    ctx = _ctx()

    items, stats = hydrate_evidence([
        {"uri": uri + "/content.md", "abstract": "Index prefix", "metadata": {"source": "prefetch"}},
        {"uri": uri},
    ], read, ctx)

    read.assert_called_once_with(uri, ctx)
    assert len(items) == 1
    assert items[0]["content"] == "Full source text"
    assert items[0]["abstract"] == "Canonical identity"
    assert items[0]["metadata"] == {"source": "prefetch", "when": "2026-10-02"}
    assert stats["sources"][0]["content_chars"] == len("Full source text")


def test_structured_cards_are_delivered_whole_with_provenance():
    content = "# Reference\n\n## Facts\n- Color: amber\n- Value: 17\n\n## Source\nExact wording.\n"
    items = [
        {"uri": f"ctx://acct/resources/{index}", "abstract": f"Card {index}",
         "content": content, "overview": "overview", "metadata": {"when": "2026-10-02"}}
        for index in range(2)
    ]

    text, stats = select_evidence(items, "color", 1800)

    assert text.count(content) == 2
    assert "Card 0" in text and "Card 1" in text
    assert "when: 2026-10-02" in text
    assert stats["ranking"] == "full_content"
    assert stats["output_bytes"] <= 1800


def test_long_content_selects_late_semantic_fact_with_heading_and_positions():
    content = "# Laboratory\n" + "Routine equipment inventory and maintenance records.\n" * 350
    content += "\n## Experiment result\nThe sample changed to turquoise under ultraviolet light.\n"
    diagnostic = {}
    embedder = SimpleNamespace(embed_texts=Mock(side_effect=lambda texts: [
        [1.0, 0.0] if index == 0 or "turquoise" in text else [0.0, 1.0]
        for index, text in enumerate(texts)
    ]))

    text, stats = select_evidence([
        {"uri": "ctx://acct/resources/lab", "abstract": "Lab record", "content": content,
         "_evidence_diagnostic": diagnostic},
        {"uri": "ctx://acct/resources/other", "abstract": "Related card", "content": "Observer: technician."},
    ], "样本在紫外线下变成什么颜色？", 2000, embedder_factory=lambda: embedder)

    assert "The sample changed to turquoise under ultraviolet light." in text
    assert "## Experiment result" in text
    assert "Observer: technician." in text
    assert any("turquoise" in content[start:end] for start, end in diagnostic["selected_ranges"])
    assert stats["ranking"] == "semantic"
    assert stats["embedding_text_count"] <= 97
    assert stats["embedding_input_bytes"] <= 64 * 1024
    assert stats["output_bytes"] <= 2000


def test_different_sized_nodes_share_available_full_content_budget():
    contents = ["First factual sentence.\n" * 240, "Small factual sentence.\n" * 10]
    items = [{"uri": f"ctx://acct/resources/{index}", "content": content}
             for index, content in enumerate(contents)]

    text, stats = select_evidence(items, "factual sentence", 7000)

    assert all(content in text for content in contents)
    assert stats["ranking"] == "full_content"
    assert stats["output_chars"] <= 7000


def test_evidence_limits_report_admitted_bytes_and_candidates():
    limits = EvidenceLimits(max_candidates=3, max_node_bytes=10, max_total_bytes=12)
    read = Mock(side_effect=lambda uri, ctx: RetrievedBlock(uri=uri, content_excerpt="abcdef"))
    items, stats = hydrate_evidence([
        {"uri": f"ctx://acct/resources/{index}", "abstract": "Identity"}
        for index in range(5)
    ], read, _ctx(), limits)

    assert len(items) == 3
    assert stats["read_count"] == 3
    assert stats["admitted_content_bytes"] == 12
    assert stats["candidate_limit_skipped"] == 2
    assert stats["sources"][2]["reason"] == "total_byte_limit"
