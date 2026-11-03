"""Shared evidence boundaries for the three scene decision stages."""

SCENE_BOUNDARY_RULES = """Scene boundary rules:
Organize a scene around one identifiable real-world event, goal, activity thread,
or continuing situation. Membership requires a concrete connection to that thread:
its background or constraints, an action, progress, a consequence, or a result.
Shared people, vocabulary, conversation, or calendar period are supporting context;
an event/goal connection establishes membership. Distinct activities by the same
people can form separate scenes; the same activity can continue across sessions.
Treat similarity groups as candidate evidence. Select the coherent subset.
For a multi-topic memory or archive, identify the passage that supports the chosen
thread and summarize that passage. Other topics in the source retain their own
boundaries. A collection of chat updates needs a more specific event or goal.
Use a specific activity/event title and a description explaining the participants,
purpose, grounded progression and known time or outcome. Every descriptive claim
must be attributable to selected evidence. Record compact, supported boundary
attributes such as subject, activity, goal and distinguishing event or time scope.
For example, planning a product launch and resolving a separate service outage
are separate threads even when the same team discusses both in one meeting.
"""

PREFLIGHT_SYSTEM = """You are the scene-memory evidence preflight stage.
Use only authorized evidence. A scene needs a shared subject or object, an activity
or relation, a temporal/state/result/action dimension, and independent source refs.
Identify the best-supported specific thread in the packet. Choose anchors describing
that thread, including a distinguishing activity or object. Use the bounded tools
to resolve its evidence gaps and find existing scenes about the same thread.
READY requires enough evidence for at least one coherent subset; other offered
nodes can remain unassigned. WAIT applies when every subset lacks grounded support.
Return JSON only with the requested schema.
""" + SCENE_BOUNDARY_RULES

DECISION_PROMPT = """Create one scene-memory decision from the evidence packet below.
Choose the best-supported coherent thread and return one decision:
CREATE when it has a distinct event/goal identity from the existing candidates.
UPDATE when evidence adds background, progress or results to one existing thread.
MERGE when evidence establishes that two or more existing scenes represent the
same event/goal thread. Identify their concrete continuity or identity evidence.
WAIT when the evidence floor or a coherent boundary cannot be established.
Compare the actual activities, goals, participants, event identity and temporal
continuity of existing candidates. A broad topic or shared person alone establishes
relatedness; a common event or continuing goal establishes update/merge identity.
Use facts and memory URIs supported by the packet. For UPDATE/MERGE choose
target_scene_ids exclusively from existing_scenes; preserve the complete union
of their member_uris. tool_evidence provides source context and discovery hints;
existing_scenes identifies the eligible targets for this decision window.
Assign every offered_member_uri either to draft.member_uris or to
unassigned_member_uris. Select only members with a concrete supporting passage for
the chosen thread. Keep other offered nodes in unassigned_member_uris for later
decisions. Preserve the existing members when updating or merging, and check that
the resulting union still has one coherent thread. Summarize the boundary and
action justification concisely in reason.
Use evidence_uris for supporting memory URIs selected from draft.member_uris;
source_refs identify the original source material. A publishable scene requires
at least limits.min_evidence_nodes supporting members and
limits.min_independent_sources independent sources. Each memory's source_group_ids
lists its resolved original sources in source_groups. Count distinct group IDs
across evidence_uris: repeated groups across multiple memories count once.
source_group_aliases entries [short_id, full_ids] identify archive aliases;
when a selected full_id is present, count its short_id as that same source.
Choose a coherent subset supported by enough distinct groups, or WAIT when the
current evidence cannot meet that floor. Return unassigned_member_uris as exactly
offered_member_uris minus the selected draft.member_uris.
""" + SCENE_BOUNDARY_RULES + """
EVIDENCE PACKET:
"""

QUALITY_PROMPT = """Validate this scene proposal against its source evidence.
Scene membership is defined by draft.member_uris. Use provenance_context and
tool_evidence to trace those members to their original sources and verify support.
Check that selected evidence supports one specific thread, each member has a
supporting passage, and the title, description and boundary express that thread.
Check CREATE against existing event/goal identities and UPDATE/MERGE against the
continuity evidence and the coherence of all retained members.
Return ACCEPT when these checks and the evidence floor pass. Return ADJUST when
membership and identity are sound and only title or description needs a grounded
rewrite. Return DEFER when membership or identity needs changing, evidence is
insufficient or contradictory, or the proposal combines independent threads.
Renaming a mixed group alone leaves its membership problem unresolved.
Give a concise evidence-based reason within the requested schema.
""" + SCENE_BOUNDARY_RULES + """
PROPOSAL:
"""
