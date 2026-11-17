"""Normal subject updates preserve identity and reconcile associated facts."""

from dataclasses import replace

from core.models import CandidateMemory
from tests.unit.extraction.test_node_processing import related_targets, setup
from tests.unit.extraction.test_node_reconciliation import ReconciliationLLM


def person(name, fact, *, source_id="message"):
    return CandidateMemory(
        "entity", "user", name.lower(), fact, "- " + fact,
        f"# {name}\n\n## Facts\n- {fact}", .95, object_name=name,
        provenance_ids=[f"prov:1:archive:identity:{source_id}"],
        source_contexts=[fact],
    )


def operation(ids, draft, *, target="", action="create", edits=()):
    return {
        "draft_ids": ids, "action": action, "target": target,
        "routing_key": draft.routing_key, "object_name": draft.object_name,
        "abstract": draft.abstract, "overview": [draft.abstract],
        "edits": list(edits),
        "sections": [] if edits or action == "noop" else [
            {"heading": "Facts", "facts": [draft.abstract]}],
    }


def commit(api, ctx, drafts, source, decide):
    api._pipeline.extract = lambda *args, **kwargs: drafts
    provider = ReconciliationLLM(decide)
    api._llm = provider
    result = api.commit_session([{"role": "user", "content": source}], ctx)
    assert len(provider.calls) == 1
    return result


def test_people_receive_their_own_updates_in_one_decision():
    ctx, fs, _, api = setup(lambda _: [])
    originals = [person("Nate", "Nate owns two turtles."),
                 person("Joanna", "Joanna is a screenwriter.")]
    targets = {draft.object_name: api.write_memory(draft, ctx)["target_uri"]
               for draft in originals}
    drafts = [person("Nate", "Nate owns three turtles.", source_id="nate"),
              person("Joanna", "Joanna submitted her screenplay.", source_id="joanna")]

    def decide(payload):
        joanna = operation([1], drafts[1], target=targets["Joanna"], action="update")
        joanna.update(abstract="Joanna is a screenwriter who submitted her screenplay.",
                      overview=[originals[1].abstract, drafts[1].abstract])
        return {"operations": [
            operation([0], drafts[0], target=targets["Nate"], action="update",
                      edits=[{"search": "Nate owns two turtles.",
                              "replacement": drafts[0].abstract}]),
            joanna,
        ], "related_reconciliations": []}

    result = commit(api, ctx, drafts, "Nate owns three turtles. Joanna submitted her screenplay.", decide)
    assert result["writes_completed"] == 2
    nate, joanna = (fs.read_node(targets[name], ctx) for name in ("Nate", "Joanna"))
    assert nate.content == "# Nate\n\n## Facts\n- Nate owns three turtles."
    assert joanna.content == ("# Joanna\n\n## Facts\n- Joanna is a screenwriter.\n"
                              "## Facts\n- Joanna submitted her screenplay.")
    assert nate.metadata["object_name_aliases"] == ["Nate"]
    assert joanna.metadata["object_name_aliases"] == ["Joanna"]


def test_same_subject_drafts_commit_one_complete_card():
    ctx, fs, _, api = setup(lambda _: [])
    drafts = [person("Nate", "Nate is a gamer.", source_id="job"),
              person("Nate", "Nate owns three turtles.", source_id="pets")]

    def decide(payload):
        op = operation([0, 1], drafts[0])
        op["sections"][0]["facts"] = [draft.abstract for draft in drafts]
        op.update(abstract="Nate is a gamer who owns three turtles.",
                  overview=[draft.abstract for draft in drafts])
        return {"operations": [op], "related_reconciliations": []}

    result = commit(api, ctx, drafts, "Nate is a gamer. Nate owns three turtles.", decide)
    node, = fs.nodes.values()
    assert result["writes_completed"] == 1
    assert node.content == "# Nate\n\n## Facts\n- Nate is a gamer.\n- Nate owns three turtles."
    assert set(node.metadata["provenance_ids"]) == {source for draft in drafts for source in draft.provenance_ids}
    assert node.metadata["object_name_aliases"] == ["Nate"]


def test_confirmed_alias_updates_the_canonical_card():
    ctx, fs, _, api = setup(lambda _: [])
    original = replace(person("Nate", "Nathan uses the nickname Nate and lives in Seattle."),
                       verified_object_aliases=["Nathan"], verified_routing_aliases=["nathan"])
    target = api.write_memory(original, ctx)["target_uri"]
    draft = person("Nathan", "Nathan moved from Seattle to Portland.", source_id="move")

    def decide(payload):
        op = operation([0], original, target=target, action="update", edits=[{
            "search": original.abstract,
            "replacement": "Nathan uses the nickname Nate and moved from Seattle to Portland."}])
        op.update(abstract="Nathan, nicknamed Nate, moved from Seattle to Portland.",
                  overview=["Nathan uses the nickname Nate.", draft.abstract])
        return {"operations": [op], "related_reconciliations": []}

    result = commit(api, ctx, [draft], draft.abstract, decide)
    node = fs.read_node(target, ctx)
    assert result["writes_completed"] == len(fs.nodes) == 1
    assert node.metadata["object_name"] == "Nate"
    assert set(node.metadata["object_name_aliases"]) == {"Nate", "Nathan"}
    assert node.content.endswith("- Nathan uses the nickname Nate and moved from Seattle to Portland.")


def test_bound_pronoun_keeps_the_read_identity_and_aliases():
    ctx, fs, _, api = setup(lambda _: [])
    original = replace(person("Nate", "Nate lives in Seattle."),
                       verified_object_aliases=["Nathan"], verified_routing_aliases=["nathan"])
    target = api.write_memory(original, ctx)["target_uri"]
    source = "Nathan, nicknamed Nate, uses NateTheGamer online. He owns three turtles."
    draft = replace(person("He", "He owns three turtles.", source_id="bound-pets"),
                    target_uri=target, expected_version=1, source_contexts=[source],
                    verified_object_aliases=["NateTheGamer"], verified_routing_aliases=["nate_the_gamer"])

    def decide(payload):
        op = operation([0], original, target=target, action="update")
        op["sections"][0]["facts"] = ["Nate owns three turtles."]
        op.update(abstract="Nate lives in Seattle and owns three turtles.",
                  overview=[original.abstract, "Nate owns three turtles."])
        return {"operations": [op], "related_reconciliations": []}

    result = commit(api, ctx, [draft], source, decide)
    node = fs.read_node(target, ctx)
    assert result["writes_completed"] == len(fs.nodes) == 1
    assert node.metadata["object_name"] == "Nate"
    assert set(node.metadata["object_name_aliases"]) == {"Nate", "Nathan", "NateTheGamer"}
    assert set(node.metadata["routing_key_aliases"]) == {"nate", "nathan", "nate_the_gamer"}
    assert node.content.endswith("- Nate owns three turtles.")


def test_new_pet_updates_its_owners_current_pet_count():
    ctx, fs, _, api = setup(lambda _: [])
    owner = person("Nate", "Nate owns two turtles, Pip and Dot.")
    target = api.write_memory(owner, ctx)["target_uri"]
    source = "Nate adopted Moss as his third pet turtle; Pip and Dot still live with him."
    draft = replace(person("Moss", "Moss is Nate's third pet turtle.", source_id="adoption"),
                    source_contexts=[source])

    def decide(payload):
        assert target in related_targets(payload)
        update = operation([], owner, target=target, action="update", edits=[{
            "search": owner.abstract, "replacement": "Nate owns three turtles, Pip, Dot, and Moss."}])
        update.pop("action")
        update.pop("draft_ids")
        update.update(source_draft_ids=[0], decision="update", reason="correct_current_fact",
                      source_quote=source, abstract="Nate owns three turtles, Pip, Dot, and Moss.",
                      overview=["Nate owns three turtles, Pip, Dot, and Moss."])
        return {"operations": [operation([0], draft)], "related_reconciliations": [update]}

    result = commit(api, ctx, [draft], source, decide)
    node = fs.read_node(target, ctx)
    pet = next(node for node in fs.nodes.values() if node.metadata["object_name"] == "Moss")
    assert result["writes_completed"] == len(fs.nodes) == 2
    assert node.content == "# Nate\n\n## Facts\n- Nate owns three turtles, Pip, Dot, and Moss."
    assert node.metadata["object_name_aliases"] == ["Nate"]
    assert pet.content == "# Moss\n\n## Facts\n- Moss is Nate's third pet turtle."
    assert pet.metadata["object_name_aliases"] == ["Moss"]


def test_new_dog_reconciles_its_owners_existing_pet_status():
    ctx, fs, _, api = setup(lambda _: [])
    owner = person("Andrew", "Andrew currently has no pets.")
    target = api.write_memory(owner, ctx)["target_uri"]
    source = "Andrew: I adopted a dog named Toby today."
    draft = replace(person("Toby", "Toby is Andrew's dog.", source_id="dog-adoption"),
                    source_contexts=[source])
    fact = "Andrew owns a dog named Toby."

    def decide(payload):
        assert related_targets(payload) == [target]
        update = operation([], owner, target=target, action="update", edits=[{
            "search": owner.abstract, "replacement": fact}])
        update.pop("action")
        update.pop("draft_ids")
        update.update(source_draft_ids=[0], decision="update", reason="correct_current_fact",
                      source_quote=source, abstract=fact, overview=[fact])
        return {"operations": [operation([0], draft)], "related_reconciliations": [update]}

    result = commit(api, ctx, [draft], source, decide)
    stored = fs.read_node(target, ctx)
    dog = next(node for node in fs.nodes.values() if node.metadata["object_name"] == "Toby")
    assert result["writes_completed"] == len(fs.nodes) == 2
    assert stored.metadata["version"] == 2
    assert stored.content == "# Andrew\n\n## Facts\n- " + fact
    assert stored.metadata["object_name_aliases"] == ["Andrew"]
    assert dog.content == "# Toby\n\n## Facts\n- Toby is Andrew's dog."
    assert dog.metadata["object_name_aliases"] == ["Toby"]


def test_third_turtle_delta_updates_the_existing_turtle_group():
    ctx, fs, _, api = setup(lambda _: [])
    group = replace(person("Nate's pet turtles", "Nate's pet turtles are Pip and Dot."),
                    routing_key="nates_pet_turtles", verified_object_aliases=["Nate's turtles"])
    target = api.write_memory(group, ctx)["target_uri"]
    source = "Nate: I adopted a third turtle today, named Moss. Pip and Dot still live with me."
    group_fact = "Nate has three pet turtles: Pip, Dot, and Moss."
    drafts = [replace(group, abstract=group_fact, overview="- " + group_fact,
                      content="# Nate's pet turtles\n\n## Facts\n- " + group_fact,
                      provenance_ids=["prov:1:archive:identity:turtle-group"], source_contexts=[source]),
              replace(person("Moss", "Moss is Nate's third pet turtle.", source_id="new-turtle"),
                      source_contexts=[source])]

    def decide(payload):
        return {"operations": [
            operation([0], drafts[0], target=target, action="update", edits=[{
                "search": group.abstract, "replacement": group_fact}]),
            operation([1], drafts[1]),
        ], "related_reconciliations": []}

    result = commit(api, ctx, drafts, source, decide)
    stored = fs.read_node(target, ctx)
    pet = next(node for node in fs.nodes.values() if node.metadata["object_name"] == "Moss")
    assert result["writes_completed"] == len(fs.nodes) == 2
    assert stored.metadata["version"] == 2
    assert stored.content == "# Nate's pet turtles\n\n## Facts\n- " + group_fact
    assert set(stored.metadata["object_name_aliases"]) == {"Nate's pet turtles", "Nate's turtles"}
    assert stored.metadata["provenance_ids"] == [
        "prov:1:archive:identity:message", "prov:1:archive:identity:turtle-group"]
    assert pet.content == "# Moss\n\n## Facts\n- Moss is Nate's third pet turtle."


def test_complete_speaker_quote_updates_turtles_and_owner_in_one_decision():
    ctx, fs, _, api = setup(lambda _: [])
    group = replace(person("Nate's pet turtles", "Nate's pet turtles are Pip and Dot."),
                    routing_key="nates_pet_turtles", verified_object_aliases=["Nate's turtles"])
    owner = person("Nate", "Nate owns two turtles and lives in Seattle.")
    listener = person("Joanna", "Joanna enjoys landscape photography.")
    targets = {draft.object_name: api.write_memory(draft, ctx)["target_uri"]
               for draft in (group, owner, listener)}
    quote = ("Nate: The lake looked beautiful this morning. "
             "I stopped by the pet shop on the way home. "
             "I adopted Moss as my third turtle; Pip and Dot are still with me.")
    source = quote + "\nJoanna: I love that lake too. Your turtles must be fun to watch."
    group_fact = "Nate has three pet turtles: Pip, Dot, and Moss."
    owner_fact = "Nate owns three turtles and lives in Seattle."
    draft = replace(group, abstract=group_fact, overview="- " + group_fact,
                    content="# Nate's pet turtles\n\n## Facts\n- " + group_fact,
                    provenance_ids=["prov:1:archive:identity:third-turtle"], source_contexts=[source])

    def decide(payload):
        assert set(related_targets(payload)) == {targets["Nate"], targets["Joanna"]}
        update = operation([], owner, target=targets["Nate"], action="update", edits=[{
            "search": owner.abstract, "replacement": owner_fact}])
        update.pop("action")
        update.pop("draft_ids")
        update.update(source_draft_ids=[0], decision="update", reason="correct_current_fact",
                      source_quote=quote, abstract=owner_fact, overview=[owner_fact])
        unchanged = operation([], listener, target=targets["Joanna"], action="noop")
        unchanged.pop("action")
        unchanged.pop("draft_ids")
        unchanged.update(source_draft_ids=[0], decision="unchanged", reason="different_fact",
                         source_quote="", abstract="", overview=[])
        return {"operations": [operation([0], draft, target=targets[group.object_name],
                                         action="update", edits=[{
                                             "search": group.abstract, "replacement": group_fact}])],
                "related_reconciliations": [update, unchanged]}

    result = commit(api, ctx, [draft], source, decide)
    stored_group, stored_owner, stored_listener = (
        fs.read_node(targets[original.object_name], ctx) for original in (group, owner, listener))
    assert result["writes_completed"] == 2
    assert len(fs.nodes) == 3
    assert stored_group.content == "# Nate's pet turtles\n\n## Facts\n- " + group_fact
    assert stored_owner.content == "# Nate\n\n## Facts\n- " + owner_fact
    assert stored_listener.content == listener.content
    assert [node.metadata["version"] for node in (stored_group, stored_owner, stored_listener)] == [2, 2, 1]
    assert set(stored_group.metadata["object_name_aliases"]) == {"Nate's pet turtles", "Nate's turtles"}
    assert stored_owner.metadata["object_name_aliases"] == ["Nate"]
    assert stored_listener.metadata["object_name_aliases"] == ["Joanna"]


def test_update_and_covered_draft_share_one_target_write():
    ctx, fs, _, api = setup(lambda _: [])
    original = person("Nate", "Nate lives in Seattle.")
    target = api.write_memory(original, ctx)["target_uri"]
    drafts = [person("Nate", "Nate owns three turtles.", source_id="pets"),
              replace(original, provenance_ids=["prov:1:archive:identity:location"])]

    def decide(payload):
        update = operation([0], drafts[0], target=target, action="update")
        update.update(abstract="Nate lives in Seattle and owns three turtles.",
                      overview=[original.abstract, drafts[0].abstract])
        return {"operations": [
            update,
            operation([1], drafts[1], target=target, action="noop"),
        ], "related_reconciliations": []}

    result = commit(api, ctx, drafts, "Nate owns three turtles and still lives in Seattle.", decide)
    node = fs.read_node(target, ctx)
    assert result["writes_completed"] == len(fs.nodes) == 1
    assert node.metadata["version"] == 2
    assert node.content == ("# Nate\n\n## Facts\n- Nate lives in Seattle.\n"
                            "## Facts\n- Nate owns three turtles.")
