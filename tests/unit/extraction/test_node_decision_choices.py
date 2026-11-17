"""Bounded summaries and explicit related choices complete normal writes."""

from dataclasses import replace
from copy import deepcopy

import pytest
from jsonschema import Draft202012Validator

from tests.unit.extraction.test_node_identity_contract import operation, person
from tests.unit.extraction.test_node_processing import node_payload, setup


@pytest.mark.parametrize("answered", [(0, 1), (1, 3), ()])
def test_answered_drafts_commit_once_and_omissions_are_logged(answered, caplog):
    ctx, fs, _, api = setup(lambda _: [])
    drafts = [person(name, f"{name} enjoys swimming.") for name in ("Alice", "Bob", "Carol", "Dan")]
    source = "\n".join(draft.abstract for draft in drafts)

    def decide(payload):
        assert len(payload["drafts"]) == 4
        return {"operations": [operation([index], drafts[index]) for index in answered],
                "related_reconciliations": {}}

    with caplog.at_level("INFO"):
        result, _ = commit_choices(api, ctx, drafts, source, decide)

    assert result["writes_completed"] == len(answered)
    assert {node.content for node in fs.nodes.values()} == {drafts[index].content for index in answered}
    assert f"skipped_drafts={[i for i in range(4) if i not in answered]}" in caplog.text


@pytest.mark.parametrize("facts,replacement", [
    ("They watched a beautiful sunset together in silence; the colors in the sky were special.", ""),
    (["They watched a beautiful sunset together in silence; the colors in the sky were special."], ""),
    (["They watched a beautiful sunset together in silence; the colors in the sky were special."],
     "They watched a beautiful sunset together in silence; the colors in the sky were special."),
])
def test_section_format_variants_commit_all_facts_in_one_model_call(facts, replacement):
    ctx, fs, _, api = setup(lambda _: [])
    source = "They watched a beautiful sunset together in silence; the colors in the sky were special."
    draft = person("Sunset outing", source)
    response = {"operations": [operation([0], draft)], "related_reconciliations": {}}
    response["operations"][0]["sections"] = [{
        "heading": "Outing", "facts": ["They went to the beach.", facts], "replacement": replacement}]
    original = deepcopy(response)

    result, _ = commit_choices(api, ctx, [draft], source, lambda _: response)

    assert result["writes_completed"] == 1
    node = next(iter(fs.nodes.values()))
    assert node.content == "# Sunset outing\n\n## Outing\n- They went to the beach.\n- " + source
    assert response == original


def test_related_section_formats_preserve_target_and_all_facts():
    ctx, fs, _, api = setup(lambda _: [])
    owner = person("Nate", "Nate owns two turtles.")
    target = api.write_memory(owner, ctx)["target_uri"]
    source = "Nate: I adopted Moss today. Moss enjoys swimming."
    draft = replace(person("Moss", "Moss is Nate's third turtle."), source_contexts=[source])

    def decide(payload):
        key = next(iter(payload["related_questions"]))
        answer = update_answer(source, owner.abstract, "Nate owns three turtles.")
        answer["sections"] = [{"heading": "Pets", "facts": [["Moss enjoys swimming."]],
                               "replacement": "Moss enjoys swimming."}]
        op = operation([0], draft)
        op["sections"] = [{"heading": "Facts", "facts": draft.abstract}]
        return {"operations": [op], "related_reconciliations": {key: answer}}

    result, _ = commit_choices(api, ctx, [draft], source, decide)

    assert result["writes_completed"] == 2
    node = fs.read_node(target, ctx)
    assert node.metadata["version"] == 2
    assert node.content == "# Nate\n\n## Facts\n- Nate owns three turtles.\n## Pets\n- Moss enjoys swimming."


def commit_choices(api, ctx, drafts, source, decide):
    calls = []
    api._pipeline.extract = lambda *args, **kwargs: drafts

    def complete(**call):
        calls.append(call)
        payload = node_payload(call)
        schema = call["tools"][0]["input_schema"]
        questions = payload["related_questions"]
        answers = schema["properties"]["related_reconciliations"]
        assert set(answers["required"]) == set(answers["properties"]) == set(questions)
        result = decide(payload)
        return [{"tool": "process_memory_nodes", "input": result}]

    api._llm.complete_with_tools = complete
    result = api.commit_session([{"role": "user", "content": source}], ctx)
    assert len(calls) == 1
    return result, calls[0]


def update_answer(source, before, after):
    return {"decision": "update", "source_draft_ids": [0], "source_quote": source,
            "abstract": after, "overview": [after],
            "edits": [{"search": before, "replacement": after}], "sections": []}


def test_full_facts_persist_with_bounded_primary_and_related_summaries():
    ctx, fs, _, api = setup(lambda _: [])
    owner = person("Nate", "Nate owns two turtles.")
    target = api.write_memory(owner, ctx)["target_uri"]
    source = ("Nate adopted Moss as his third turtle. " + "Moss enjoys swimming in the large pond. " * 8).strip()
    draft = replace(person("Moss", source), source_contexts=[source])

    def decide(payload):
        key, question = next(iter(payload["related_questions"].items()))
        assert question == {"target": target, "source_draft_ids": [0],
                            "decisions": ["update", "unchanged"]}
        return {"operations": [operation([0], draft)],
                "related_reconciliations": {key: update_answer(source, owner.abstract, source)}}

    result, _ = commit_choices(api, ctx, [draft], source, decide)
    assert result["writes_completed"] == 2
    for node in fs.nodes.values():
        assert node.abstract == source[:200]
        assert node.overview == "- " + source[:200]
        assert source in node.content


def test_related_question_keeps_owner_and_commits_new_pet():
    ctx, fs, _, api = setup(lambda _: [])
    owner = person("Nate", "Nate owns a turtle named Moss.")
    target = api.write_memory(owner, ctx)["target_uri"]
    before = fs.read_node(target, ctx)
    source = "Nate: My turtle Moss enjoys swimming."
    draft = replace(person("Moss", "Moss is Nate's turtle and enjoys swimming."), source_contexts=[source])

    def decide(payload):
        return {"operations": [operation([0], draft)], "related_reconciliations": {
            key: {"decision": "unchanged"} for key in payload["related_questions"]}}

    result, call = commit_choices(api, ctx, [draft], source, decide)
    assert result["writes_completed"] == 1
    assert fs.read_node(target, ctx) == before
    assert len(fs.nodes) == 2
    answer = decide(node_payload(call))
    Draft202012Validator(call["tools"][0]["input_schema"]).validate(answer)


def test_unmatched_quote_preserves_owner_while_other_choices_commit(caplog):
    ctx, fs, _, api = setup(lambda _: [])
    owner = person("Nate", "Nate owns two turtles.")
    author = person("Joanna", "Joanna completed two screenplays.")
    targets = {card.object_name: api.write_memory(card, ctx)["target_uri"] for card in (owner, author)}
    before = fs.read_node(targets["Nate"], ctx)
    source = "Nate: I adopted Moss today.\nJoanna: I finished my third screenplay."
    draft = replace(person("Moss", "Moss is Nate's pet turtle."), source_contexts=[source])

    def decide(payload):
        answers = {}
        for key, question in payload["related_questions"].items():
            if question["target"] == targets["Nate"]:
                answers[key] = update_answer("Nate adopted Moss today.", owner.abstract,
                                             "Nate owns three turtles.")
            else:
                answers[key] = update_answer("Joanna: I finished my third screenplay.", author.abstract,
                                             "Joanna completed three screenplays.")
        return {"operations": [operation([0], draft)], "related_reconciliations": answers}

    with caplog.at_level("INFO"):
        result, _ = commit_choices(api, ctx, [draft], source, decide)
    assert result["writes_completed"] == 2
    assert fs.read_node(targets["Nate"], ctx) == before
    assert fs.read_node(targets["Joanna"], ctx).content.endswith("- Joanna completed three screenplays.")
    assert len(fs.nodes) == 3
    assert "reason=source_quote_unmatched" in caplog.text


def test_related_keep_choice_applies_the_main_update_once():
    ctx, fs, _, api = setup(lambda _: [])
    owner = person("Nate", "Nate owns two turtles.")
    target = api.write_memory(owner, ctx)["target_uri"]
    source = "Nate adopted Moss as his third turtle."
    drafts = [replace(person("Nate", "Nate owns three turtles."), source_contexts=[source]),
              replace(person("Moss", "Moss is Nate's third turtle."), source_contexts=[source])]

    def decide(payload):
        answers = {}
        for key, question in payload["related_questions"].items():
            assert question["target"] == target
            assert question["decisions"] == ["update", "unchanged"]
            answers[key] = {"decision": "unchanged"}
        return {"operations": [operation([0], drafts[0], target=target, action="update", edits=[{
            "search": owner.abstract, "replacement": drafts[0].abstract}]), operation([1], drafts[1])],
                "related_reconciliations": answers}

    result, _ = commit_choices(api, ctx, drafts, source, decide)
    assert result["writes_completed"] == 2
    assert fs.read_node(target, ctx).metadata["version"] == 2
    assert fs.read_node(target, ctx).content == "# Nate\n\n## Facts\n- Nate owns three turtles."


def test_scope_quote_skip_preserves_snapshot_and_commits_independent_draft():
    ctx, fs, _, api = setup(lambda _: [])
    owner = replace(person("Nate", "Nate lives in Seattle."), scope_ref="Nate profile")
    target = api.write_memory(owner, ctx)["target_uri"]
    before = fs.read_node(target, ctx)
    source = "Nate: I moved to Boston. I own a turtle called Moss."
    drafts = [replace(person("Nate", "Nate lives in Boston."), source_contexts=[source]),
              replace(person("Moss", "Moss is a pet turtle."), source_contexts=[source])]

    def decide(payload):
        op = operation([0], drafts[0], target=target, action="update", edits=[{
            "search": owner.abstract, "replacement": drafts[0].abstract}])
        op["scope_resolution"] = {"stored_scope_role": "self", "incoming_scope_role": "self",
                                  "source_quotes": ["Nate moved to Boston."]}
        return {"operations": [op, operation([1], drafts[1])],
                "related_reconciliations": {key: {"decision": "unchanged"}
                                            for key in payload["related_questions"]}}

    result, _ = commit_choices(api, ctx, drafts, source, decide)
    assert result["writes_completed"] == 1
    assert fs.read_node(target, ctx) == before
    assert len(fs.nodes) == 2
