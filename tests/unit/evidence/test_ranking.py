"""Generic channel competition and bounded conversational context."""

from retrieval.evidence.ranking import fuse_channels, lexical_scores


def test_name_only_consensus_does_not_displace_strong_lexical_hit():
    units = {"topic": {"text": "I repaired a telescope.", "speaker": "Alex"},
             "greeting": {"text": "Hello Alex!", "speaker": "Morgan"},
             "synonym": {"text": "I fixed the optical instrument.", "speaker": "Alex"},
             "other": {"text": "Tomorrow is sunny.", "speaker": "Morgan"}}
    scores = fuse_channels("Alex repaired a telescope", units,
                           lexical_scores("Alex repaired a telescope", units),
                           {"greeting": .99, "synonym": .9}, 10)
    assert scores["topic"] >= scores["greeting"]
    assert "synonym" in scores  # Semantic-only paraphrases survive.
    assert "other" not in scores


def test_shared_tail_cannot_displace_independent_channel_heads():
    units = {str(i): {"text": "A telescope observation.", "speaker": ""} for i in range(60)}
    units.update({"lex": {"text": "Telescope repair", "speaker": ""},
                  "sem": {"text": "Optical equipment", "speaker": ""}})
    lexical = {"lex": 100, **{str(i): 60 - i for i in range(60)}}
    semantic = {"sem": 100, **{str(i): 60 - i for i in range(60)}}
    scores = fuse_channels("telescope", units, lexical, semantic, 50)
    assert scores["lex"] > scores["48"]
    assert scores["sem"] > scores["48"]
    assert "49" not in scores  # Each channel is capped before fusion.
    assert "sem" in scores and "lex" in scores


def test_fusion_is_deduplicated_and_does_not_reject_name_only_queries():
    units = {"one": {"text": "Alex arrived.", "speaker": "Alex"},
             "two": {"text": "Alex left.", "speaker": "Morgan"}}
    lexical = lexical_scores("Alex", units)
    scores = fuse_channels("Alex", units, lexical, {"one": .9, "outside": 1}, 5)
    assert set(scores) == {"one", "two"}
    assert max(scores.values()) <= 1 / 61  # No topical double vote for the name.


def test_common_terms_do_not_create_a_false_lexical_candidate():
    units = {"one": {"text": "Hello friend, greetings."},
             "two": {"text": "Hello friend, telescope repair."},
             "three": {"text": "Hello friend, lunch time."}}
    query = "Hello friend telescope"
    scores = fuse_channels(query, units, lexical_scores(query, units), {}, 5)
    assert set(scores) == {"two"}


def unit(text, sequence, *, session="session", speaker=""):
    return {"text": text, "sequence": sequence, "session_id": session,
            "speaker": speaker, "parent_span": [0, len(text)]}


def select(units, anchor, *, top_k=4, **kwargs):
    from retrieval.evidence.ranking import select_groups

    return select_groups("How long has Morgan kept a studio?", {anchor: 1}, units, top_k, 50, **kwargs)


def test_duration_answer_without_shared_topic_is_grouped_in_both_directions():
    units = {"q": unit("I love ceramics. How long have you kept a studio? Just curious.", 0),
             "a": unit("Seven years already!", 1),
             "other": unit("It is raining outside.", 2)}
    for anchor in ("q", "a"):
        selected, _, groups = select(units, anchor)
        assert selected == ["q", "a"]
        assert groups == [["q", "a"]]


def test_caption_referring_question_and_short_answer_stay_together():
    units = {"caption": unit("Photo: our ceramics studio during its opening.", 0),
             "q": unit("How long have you worked in it?", 1),
             "a": unit("Five years now.", 2),
             "other": unit("She repaired a piano.", 3)}
    for anchor in ("q", "a"):
        assert select(units, anchor)[0] == ["caption", "q", "a"]


def test_local_reference_requires_independent_topic_or_media_support():
    units = {"greeting": unit("Hello Morgan!", 0, speaker="Alex"),
             "q": unit("How long have you worked in it?", 1, speaker="Alex"),
             "a": unit("Five years now.", 2, speaker="Morgan")}
    assert select(units, "a")[0] == ["q", "a"]
    units["greeting"] = unit("I rented a ceramics studio.", 0, speaker="Morgan")
    assert select(units, "a")[0] == ["greeting", "q", "a"]


def test_typed_answer_rejects_unrelated_short_sentence_and_long_response():
    from retrieval.evidence.ranking import _short_answer

    assert _short_answer("How many tickets did you buy", "Four.")
    assert _short_answer("你在那里工作多久了", "三年了。")
    assert _short_answer("Did you photograph it", "We did it yesterday.")
    assert _short_answer("What is your cat's name", "Juniper.")
    assert not _short_answer("How long have you kept a studio", "She repaired a piano.")
    assert not _short_answer("How long have you kept a studio", "She spent five years repairing pianos.")
    assert not _short_answer("How many tickets did you buy", "She repaired three pianos.")
    assert not _short_answer("How long have you kept a studio", "Five years " * 100)
    assert not _short_answer("Did you photograph it", "When did you photograph it?")


def test_groups_do_not_cross_sessions_or_sequence_gaps():
    units = {"q": unit("How long have you kept a studio?", 0),
             "a": unit("Seven years now.", 3)}
    assert select(units, "q")[0] == ["q"]
    units["a"] = unit("Seven years now.", 1, session="another-session")
    assert select(units, "q")[0] == ["q"]


def test_whole_groups_respect_count_and_serialized_byte_limits():
    units = {"q": unit("How long have you kept a studio?", 0),
             "a": unit("Seven years now.", 1)}
    assert select(units, "q", top_k=1)[0] == []
    assert select(units, "q", costs={"q": 60, "a": 60}, byte_limit=100)[0] == []
    assert select(units, "q", costs={"q": 50, "a": 50}, byte_limit=100)[0] == ["q", "a"]
    units.update({str(i): unit("It was a studio.", i) for i in range(2, 8)})
    assert len(select(units, "q", top_k=10)[0]) == 4


def test_functional_overlap_and_query_operators_cannot_establish_a_topic():
    from retrieval.evidence.ranking import supplement_scores, terms

    units = {"repair": {"text": "I've repaired different bicycles with my neighbour."},
             "other": {"text": "I’ve been cooking different soups with my sister."},
             "semantic": {"text": "Restoring two-wheel transport."}}
    query = "How many different bicycles have I been repairing with my neighbour?"
    scores = fuse_channels(query, units, lexical_scores(query, units), {"semantic": .9}, 5)
    assert set(scores) == {"repair", "semantic"}
    supplements = supplement_scores(query, units, {key: u["text"] for key, u in units.items()}, 5)
    assert set(supplements) == {"repair"}
    assert supplements["repair"] < 1 / 65
    assert terms("I've repaired") == terms("I have repaired") == terms("I’ve repaired")


def test_supplements_have_distinct_bounded_ranks_and_keep_chinese_topics():
    from retrieval.evidence.ranking import supplement_scores

    units = {"a": {"text": "我参加了摄影课程。"}, "b": {"text": "我参加了陶艺课程。"},
             "c": {"text": "我修理了自行车。"}}
    scores = supplement_scores("摄影课程", units, {key: u["text"] for key, u in units.items()}, 1)
    assert list(scores) == ["a"]
    scores = supplement_scores("课程", units, {key: u["text"] for key, u in units.items()}, 5)
    assert set(scores) == {"a", "b"}
    assert len(set(scores.values())) == 2


def test_independent_facts_fit_before_optional_long_reply():
    from retrieval.evidence.ranking import select_groups

    units = {"first": unit("I completed four pottery courses. Any other course ideas?", 0),
             "reply": unit("That course progress is excellent. " * 100, 1),
             "second": unit("I completed two photography courses.", 0, session="other")}
    selected, _, groups = select_groups("How many courses did I complete?",
        {"first": 1, "second": .9}, units, 2, 50,
        costs={"first": 80, "reply": 3000, "second": 80}, byte_limit=160)
    assert selected == ["first", "second"]
    assert groups == [["first"], ["second"]]


def test_complete_answer_does_not_require_a_repeated_question():
    units = {"q": unit("How long have you kept a studio?", 0),
             "a": unit("I have kept a studio for seven years.", 1)}
    assert select(units, "a", top_k=1)[0] == ["a"]


def test_referring_seed_keeps_antecedent_and_does_not_cross_a_gap():
    units = {"prior": unit("I rented a studio.", 0),
             "ref": unit("It is my favourite studio.", 1)}
    assert select(units, "ref", top_k=1)[0] == []
    assert select(units, "ref", top_k=2)[0] == ["prior", "ref"]
    units["ref"]["sequence"] = 4
    assert select(units, "ref", top_k=1)[0] == ["ref"]


def test_optional_context_is_available_after_seeds_and_retains_dependency_ids():
    from retrieval.evidence.ranking import select_groups

    units = {"first": unit("I opened a studio.", 0),
             "reply": unit("That studio sounds wonderful.", 1),
             "second": unit("I closed another studio.", 0, session="other")}
    selected, _, groups = select_groups("studio", {"first": 1, "second": .9}, units, 3, 50)
    assert selected == ["first", "second", "reply"]
    assert groups == [["first"], ["second"], ["first", "reply"]]


def test_topic_label_before_a_question_does_not_make_its_answer_optional():
    units = {"q": unit("Studio update: How long have you kept it?", 0),
             "a": unit("Seven years now.", 1)}
    assert select(units, "q", top_k=1)[0] == []
    assert select(units, "q", top_k=2)[0] == ["q", "a"]


def test_ambiguous_function_words_remain_searchable_as_literal_names():
    from retrieval.evidence.ranking import supplement_scores

    units = {"one": {"text": "May repaired a telescope.", "speaker": "May"},
             "two": {"text": "Will repaired a piano.", "speaker": "Will"}}
    for query, expected in [("May", "one"), ("Will", "two"), ("may", "one"), ("will", "two")]:
        scores = fuse_channels(query, units, lexical_scores(query, units), {}, 5)
        assert set(scores) == {expected}
        assert set(supplement_scores(query, units, {key: u["text"] for key, u in units.items()}, 5)) == {expected}
    assert set(fuse_channels("What piano may I repair?", units,
                            lexical_scores("What piano may I repair?", units), {}, 5)) == {"two"}


def test_negative_contractions_and_demonstratives_are_not_independent_topics():
    from retrieval.evidence.ranking import terms, topic_terms

    assert terms("I haven’t repaired it") == terms("I have not repaired it")
    assert topic_terms("That was different from what I couldn't do") == set()
    units = {"one": {"text": "I can't repair the bicycle."},
             "two": {"text": "I can’t attend the exhibition."}}
    query = "Which bicycle can't I repair?"
    assert set(fuse_channels(query, units, lexical_scores(query, units), {}, 5)) == {"one"}
