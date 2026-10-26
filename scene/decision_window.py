"""Select complete canonical decision units within the shared prompt budget."""

from collections.abc import Callable, Iterable
from typing import Any

from scene.decision_sources import resolve_sources, with_source_groups


def decision_window(
    packet: dict[str, Any], requested: Iterable[str], fits: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    """Prioritize requested evidence, then admit whole candidate scenes and their sources."""
    if fits(packet):
        return packet
    cards = {card["uri"]: card for card in packet["memories"]}
    ordered = list(dict.fromkeys(uri for uri in requested if uri in cards))
    priority = set(ordered)

    context = [*cards.values(), *(card for result in packet.get("tool_evidence", [])
                                 for card in result.get("results", []) if isinstance(card, dict))]

    def closure(uris):
        selected = set(uris) & cards.keys()
        _, reached = resolve_sources([cards[uri] for uri in selected], context)
        return selected | {identifier for kind, identifier in reached if kind == "memory" and identifier in cards}

    def project(selected):
        existing = [card for card in packet["existing_scenes"] if set(card["member_uris"]) <= selected]
        return with_source_groups({
            **packet,
            "memories": [card for card in packet["memories"] if card["uri"] in selected],
            "offered_member_uris": [uri for uri in packet["offered_member_uris"] if uri in selected],
            "existing_scenes": existing,
            "candidate_groups": [group for group in packet.get("candidate_groups", [])
                                 if set(group["member_uris"]) <= selected],
            "decision_window": {
                "available_memories": len(cards), "selected_memories": len(selected),
                "deferred_requested_memories": len(priority - selected),
                "deferred_scene_ids": [card["scene_id"] for card in packet["existing_scenes"]
                                       if not set(card["member_uris"]) <= selected],
                "policy": "Requested evidence and its sources first; candidate scenes require every member."
            },
        })

    selected = closure(priority)
    if not fits(project(selected)):
        selected = set()
        for uri in ordered:
            proposed = selected | closure([uri])
            if fits(project(proposed)):
                selected = proposed
    for scene in packet["existing_scenes"]:
        proposed = selected | closure(scene["member_uris"])
        if fits(project(proposed)):
            selected = proposed
    scene_members = {uri for scene in packet["existing_scenes"] for uri in scene["member_uris"]}
    for uri in sorted(cards.keys() - priority - scene_members):
        proposed = selected | closure([uri])
        if fits(project(proposed)):
            selected = proposed
    return project(selected)
