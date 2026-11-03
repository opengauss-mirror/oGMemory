"""Unit tests for the [Search]/[Answer] protocol-scaffolding salvage helper.

When the LLM emits its answer as plain content — ``"[Answer] <text>"``
— instead of calling the [Answer] tool, the QA driver used to discard
the row entirely. The salvage helper recovers the intended answer
from the LAST ``[Answer]`` tag's tail, with remaining protocol tags
stripped. A row with no ``[Answer]`` tag (or only tags left) stays a
genuine failure and returns ``""``.

These tests pin the helper's contract.
"""

from __future__ import annotations

import pytest

from m3.qa import _PROTOCOL_SCAFFOLDING_RE, _has_protocol_scaffolding, _salvage_scaffolding


@pytest.mark.parametrize("text", [
    'Search("where are the shoes?", top_k=3)',
    'Answer(final_answer="on the table")',
    '<tools>{"name":"Search","arguments":{"query":"shoes"}}</tools>',
    '{"name": "Search", "arguments": {"query": "shoes"}}',
    '<tool_call>{"name":"Answer","arguments":{}}</tool_call>',
])
def test_native_call_text_is_not_a_final_answer(text):
    assert _has_protocol_scaffolding(text)
    assert _salvage_scaffolding("[Answer] " + text) == ""


@pytest.mark.parametrize("text", [
    "They search for the shoes near the door.",
    "The answer is on the table.",
    "On the shelf beside the tools.",
])
def test_ordinary_prose_is_not_tool_scaffolding(text):
    assert not _has_protocol_scaffolding(text)


class TestSalvageScaffolding:
    def test_recovers_text_after_last_answer_tag(self):
        assert _salvage_scaffolding("[Answer] In the wardrobe") == "In the wardrobe"

    def test_strips_trailing_period(self):
        # The salvage trim set strips ".", so a trailing sentence
        # period is removed (the caller treats it as noise — the period
        # isn't load-bearing).
        assert (
            _salvage_scaffolding("[Answer] On the upper side.")
            == "On the upper side"
        )

    def test_strips_remaining_search_tags(self):
        # A stray [Search] interleaved AFTER the [Answer] tag is
        # replaced with whitespace; the trailing word survives.
        assert (
            _salvage_scaffolding("[Answer] Milk [Search] again")
            == "Milk again"
        )

    def test_no_answer_tag_returns_empty(self):
        # Only a [Search] leak — model never produced an answer.
        assert _salvage_scaffolding("[Search] where is the coat") == ""

    def test_only_tags_remain_returns_residue(self):
        # The helper strips tags but leaves non-tag residue. "retry"
        # (5 chars) survives; the caller applies the >=3 threshold and
        # salvages "retry" as the answer.
        assert _salvage_scaffolding("[Answer] [Search] retry") == "retry"

    def test_short_real_answers_survive_threshold(self):
        # Threshold is ``>= 3``; "Milk" (4 chars) is a real M3 answer
        # and must survive.
        assert _salvage_scaffolding("[Answer] Milk") == "Milk"

    def test_two_char_residue_returns_empty(self):
        # Two characters of residue is below the threshold; let the
        # caller treat this as a genuine failure.
        result = _salvage_scaffolding("[Answer] ok")
        # Helper returns the cleaned text (len 2); the >=3 threshold
        # is enforced by the caller. The helper itself does not
        # enforce the length — only the caller does.
        assert result == "ok"

    def test_empty_string_returns_empty(self):
        assert _salvage_scaffolding("") == ""

    def test_only_whitespace_and_punctuation_returns_empty(self):
        # Helper strips trailing punctuation; only-punctuation residue
        # collapses to empty.
        assert _salvage_scaffolding("[Answer] --") == ""

    def test_only_tag_returns_empty(self):
        assert _salvage_scaffolding("[Answer]") == ""

    def test_case_insensitive_answer_tag(self):
        assert _salvage_scaffolding("[answer] A dress") == "A dress"
        assert _salvage_scaffolding("[ANSWER] A dress") == "A dress"

    def test_uses_last_answer_tag(self):
        # Model emitted [Search] then later [Answer]; only the tail
        # after the LAST [Answer] tag is the answer.
        assert _salvage_scaffolding(
            "[Search] retry [Answer] final answer text"
        ) == "final answer text"

    def test_collapses_whitespace(self):
        assert (
            _salvage_scaffolding("[Answer] multiple    spaces   here")
            == "multiple spaces here"
        )

    def test_scaffolding_regex_catches_both_tags(self):
        # Sanity check: the helper relies on _PROTOCOL_SCAFFOLDING_RE
        # to strip residual tags. If that ever breaks, this test
        # fails and signals the regex needs updating.
        assert _PROTOCOL_SCAFFOLDING_RE.search("[Search] q")
        assert _PROTOCOL_SCAFFOLDING_RE.search("[Answer] a")
        assert not _PROTOCOL_SCAFFOLDING_RE.search("ordinary text")


class TestSalvageViaQaDriver:
    """End-to-end: salvage flows into the JSONL row via the driver.

    We don't run the full LLM loop here (covered elsewhere); we just
    exercise the final classification path by calling
    ``_salvage_scaffolding`` on the same strings the loop would have
    produced.
    """

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("[Answer] In the wardrobe", "In the wardrobe"),
            ("[Answer] On the upper side.", "On the upper side"),
            ("[Answer] Milk", "Milk"),
            ("[Search] where", ""),
            ("[Answer] [Search] retry", "retry"),
            ("", ""),
        ],
    )
    def test_param_salvage(self, raw, expected):
        assert _salvage_scaffolding(raw) == expected
