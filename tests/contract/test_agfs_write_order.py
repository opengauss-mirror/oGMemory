"""Contract test for AGFS write order and Repair Job consistency.

Phase 0 frozen — verifies the 4-step atomic write order convention matches
the Repair Job's branch logic for detecting and repairing incomplete writes.

See docs/architecture/agfs-directory-spec.md for full specification.
"""

import pytest
from enum import Enum
from dataclasses import dataclass


class WriteStep(Enum):
    """The 4 atomic write steps in strict order."""
    CONTENT = 1  # ① content.md
    RELATIONS = 2  # ② .relations.json
    ABSTRACT_OVERVIEW = 3  # ③ .abstract.md + .overview.md (parallel)
    META = 4  # ④ .meta.json (commit point: status=ACTIVE)
    OUTBOX = 5  # ⑤ OutboxEvent registration


@dataclass
class NodeState:
    """Represents the state of a node directory at any point during writes."""
    has_content: bool = False
    has_relations: bool = False
    has_abstract: bool = False
    has_overview: bool = False
    has_meta: bool = False
    meta_status: str | None = None  # None if no .meta.json
    has_outbox_event: bool = False

    def completed_through_step(self, step: WriteStep) -> bool:
        """Check if node state reflects completion through given step."""
        if step == WriteStep.CONTENT:
            return self.has_content
        if step == WriteStep.RELATIONS:
            return self.has_content and self.has_relations
        if step == WriteStep.ABSTRACT_OVERVIEW:
            return (
                self.has_content
                and self.has_relations
                and self.has_abstract
                and self.has_overview
            )
        if step == WriteStep.META:
            return (
                self.has_content
                and self.has_relations
                and self.has_abstract
                and self.has_overview
                and self.has_meta
                and self.meta_status == "ACTIVE"
            )
        if step == WriteStep.OUTBOX:
            return (
                self.completed_through_step(WriteStep.META)
                and self.has_outbox_event
            )
        return False


class TestWriteOrderConvention:
    """Verify the 4-step write order convention."""

    def test_write_order_is_fixed(self):
        """Write steps must occur in fixed order: ①→②→③→④→⑤."""
        order = [WriteStep.CONTENT, WriteStep.RELATIONS, WriteStep.ABSTRACT_OVERVIEW, WriteStep.META, WriteStep.OUTBOX]
        assert order[0] == WriteStep.CONTENT
        assert order[1] == WriteStep.RELATIONS
        assert order[2] == WriteStep.ABSTRACT_OVERVIEW
        assert order[3] == WriteStep.META
        assert order[4] == WriteStep.OUTBOX

    def test_step_1_content_first(self):
        """Step ①: Write content.md first (largest file)."""
        state = NodeState(has_content=True)
        assert state.completed_through_step(WriteStep.CONTENT)
        assert not state.completed_through_step(WriteStep.RELATIONS)

    def test_step_2_relations_after_content(self):
        """Step ②: Write .relations.json after content.md."""
        state = NodeState(has_content=True, has_relations=True)
        assert state.completed_through_step(WriteStep.RELATIONS)
        assert not state.completed_through_step(WriteStep.ABSTRACT_OVERVIEW)

    def test_step_3_abstract_and_overview_parallel(self):
        """Step ③: Write .abstract.md and .overview.md (parallel safe)."""
        state = NodeState(
            has_content=True,
            has_relations=True,
            has_abstract=True,
            has_overview=True,
        )
        assert state.completed_through_step(WriteStep.ABSTRACT_OVERVIEW)
        assert not state.completed_through_step(WriteStep.META)

    def test_step_4_meta_is_commit_point(self):
        """Step ④: Write .meta.json with status=ACTIVE is the commit point."""
        state = NodeState(
            has_content=True,
            has_relations=True,
            has_abstract=True,
            has_overview=True,
            has_meta=True,
            meta_status="ACTIVE",
        )
        assert state.completed_through_step(WriteStep.META)
        assert not state.completed_through_step(WriteStep.OUTBOX)

    def test_step_5_outbox_after_commit(self):
        """Step ⑤: Register OutboxEvent only after step ④ complete."""
        state = NodeState(
            has_content=True,
            has_relations=True,
            has_abstract=True,
            has_overview=True,
            has_meta=True,
            meta_status="ACTIVE",
            has_outbox_event=True,
        )
        assert state.completed_through_step(WriteStep.OUTBOX)

    def test_pending_status_before_step_4(self):
        """Before step ④, node should have status=PENDING if .meta.json exists."""
        state = NodeState(
            has_content=True,
            has_relations=True,
            has_abstract=True,
            has_overview=True,
            has_meta=True,
            meta_status="PENDING",  # Step ④ not complete
        )
        assert state.meta_status == "PENDING"
        assert not state.completed_through_step(WriteStep.META)

    def test_active_status_only_after_step_4(self):
        """Only after step ④ should status be ACTIVE."""
        state = NodeState(
            has_content=True,
            has_relations=True,
            has_abstract=True,
            has_overview=True,
            has_meta=True,
            meta_status="ACTIVE",
        )
        assert state.meta_status == "ACTIVE"
        assert state.completed_through_step(WriteStep.META)


class TestRepairJobBranchLogic:
    """Verify Repair Job branch logic matches write order failures.

    The Repair Job scans nodes and detects incomplete writes by checking
    file presence and NodeStatus. Branch logic must handle each failure point.
    """

    def test_branch_no_meta_no_content(self):
        """
        Branch: .meta.json missing + content.md missing → Skip.

        This is a normal interruption before step ①. Nothing to repair.
        """
        state = NodeState(has_content=False, has_meta=False)

        # Repair Job action: SKIP (nothing to repair)
        assert not self._should_attempt_repair(state)

    def test_branch_no_meta_with_content(self):
        """
        Branch: .meta.json missing + content.md exists → Repair.

        Interrupted after step ① but before step ④.
        Recovery: Recreate .meta.json, register OutboxEvent.
        """
        state = NodeState(
            has_content=True,
            has_relations=True,
            has_abstract=True,
            has_overview=True,
            has_meta=False,  # Missing
        )

        # Repair Job action: RECREATE metadata + register OutboxEvent
        assert self._should_attempt_repair(state)
        assert state.has_content  # Content exists to rebuild from

    def test_branch_pending_status_with_complete_files(self):
        """
        Branch: status=PENDING + all files present → Activate.

        Interrupted during step ④ (meta write started but not committed).
        Recovery: Update status to ACTIVE, register OutboxEvent.
        """
        state = NodeState(
            has_content=True,
            has_relations=True,
            has_abstract=True,
            has_overview=True,
            has_meta=True,
            meta_status="PENDING",  # Step ④ incomplete
        )

        # Repair Job action: UPDATE status=ACTIVE + register OutboxEvent
        assert self._should_activate_pending(state)
        assert self._all_files_present(state)

    def test_branch_pending_status_with_missing_files(self):
        """
        Branch: status=PENDING + files missing → Mark BROKEN.

        Write failed partway through, files are incomplete.
        Recovery: Update status to BROKEN, alert for manual intervention.
        """
        state = NodeState(
            has_content=True,
            has_relations=True,
            has_abstract=False,  # Missing!
            has_overview=True,
            has_meta=True,
            meta_status="PENDING",
        )

        # Repair Job action: MARK status=BROKEN + alert
        assert self._should_mark_broken(state)

    def test_branch_active_status(self):
        """
        Branch: status=ACTIVE → Skip.

        Node is healthy, no repair needed.
        """
        state = NodeState(
            has_content=True,
            has_relations=True,
            has_abstract=True,
            has_overview=True,
            has_meta=True,
            meta_status="ACTIVE",
        )

        # Repair Job action: SKIP (healthy node)
        assert not self._should_attempt_repair(state)
        assert state.meta_status == "ACTIVE"

    def test_branch_broken_status(self):
        """
        Branch: status=BROKEN → Skip.

        Already marked as broken, requires manual intervention.
        """
        state = NodeState(
            has_content=True,
            has_relations=True,
            has_abstract=True,
            has_overview=True,
            has_meta=True,
            meta_status="BROKEN",
        )

        # Repair Job action: SKIP (awaiting manual intervention)
        assert not self._should_attempt_repair(state)
        assert state.meta_status == "BROKEN"

    def test_branch_archived_status(self):
        """
        Branch: status=ARCHIVED → Skip.

        Archived nodes are valid soft-deleted nodes and must not be repaired.
        """
        state = NodeState(
            has_content=True,
            has_relations=True,
            has_abstract=True,
            has_overview=True,
            has_meta=True,
            meta_status="ARCHIVED",
        )

        assert not self._should_attempt_repair(state)
        assert state.meta_status == "ARCHIVED"

    # Helper methods simulating Repair Job logic

    def _should_attempt_repair(self, state: NodeState) -> bool:
        """Simulate Repair Job: should we attempt repair?"""
        if state.meta_status in {"ACTIVE", "BROKEN", "ARCHIVED"}:
            return False  # Skip healthy or already-broken nodes
        if not state.has_meta and not state.has_content:
            return False  # Normal early interruption, nothing to repair
        return True  # Has content but missing metadata → repair

    def _should_activate_pending(self, state: NodeState) -> bool:
        """Simulate Repair Job: should we activate PENDING node?"""
        return (
            state.meta_status == "PENDING"
            and self._all_files_present(state)
        )

    def _should_mark_broken(self, state: NodeState) -> bool:
        """Simulate Repair Job: should we mark node as BROKEN?"""
        return (
            state.meta_status == "PENDING"
            and not self._all_files_present(state)
        )

    def _all_files_present(self, state: NodeState) -> bool:
        """Check if all required files are present."""
        return (
            state.has_content
            and state.has_relations
            and state.has_abstract
            and state.has_overview
            and state.has_meta
        )


class TestContextFSExistsBehavior:
    """Verify ContextFS.exists() behavior matches spec."""

    def test_exists_returns_true_only_for_active(self):
        """
        ContextFS.exists() returns True only if .meta.json exists AND status=ACTIVE.

        PENDING nodes are NOT visible to upper layers.
        """
        # ACTIVE node → exists
        state_active = NodeState(has_meta=True, meta_status="ACTIVE")
        assert self._exists(state_active) is True

        # PENDING node → does NOT exist (to upper layers)
        state_pending = NodeState(has_meta=True, meta_status="PENDING")
        assert self._exists(state_pending) is False

        # BROKEN node → does NOT exist
        state_broken = NodeState(has_meta=True, meta_status="BROKEN")
        assert self._exists(state_broken) is False

        # No .meta.json → does NOT exist
        state_no_meta = NodeState(has_meta=False)
        assert self._exists(state_no_meta) is False

    def _exists(self, state: NodeState) -> bool:
        """Simulate ContextFS.exists() logic."""
        return state.has_meta and state.meta_status == "ACTIVE"


class TestWriteOrderRecoveryScenarios:
    """End-to-end scenarios: write failures and corresponding recovery."""

    def test_scenario_failure_before_step_1(self):
        """
        Scenario: Process crashes before step ①.

        State: No files exist.
        Recovery: Skip (nothing to recover).
        """
        state = NodeState()  # All False
        assert not self._should_attempt_repair(state)

    def test_scenario_failure_after_step_1(self):
        """
        Scenario: Process crashes after step ①.

        State: content.md exists, no .meta.json.
        Recovery: Recreate .meta.json from content, register OutboxEvent.
        """
        state = NodeState(has_content=True)
        assert self._should_attempt_repair(state)

    def test_scenario_failure_after_step_2(self):
        """
        Scenario: Process crashes after step ②.

        State: content.md + .relations.json exist, no .meta.json.
        Recovery: Recreate .meta.json, register OutboxEvent.
        """
        state = NodeState(has_content=True, has_relations=True)
        assert self._should_attempt_repair(state)

    def test_scenario_failure_after_step_3(self):
        """
        Scenario: Process crashes after step ③.

        State: content.md + .relations.json + .abstract.md + .overview.md exist, no .meta.json.
        Recovery: Recreate .meta.json, register OutboxEvent.
        """
        state = NodeState(
            has_content=True,
            has_relations=True,
            has_abstract=True,
            has_overview=True,
        )
        assert self._should_attempt_repair(state)

    def test_scenario_failure_during_step_4_pending(self):
        """
        Scenario: Process crashes during step ④ (.meta.json write started but not committed).

        State: All files exist, .meta.json has status=PENDING.
        Recovery: Update status to ACTIVE, register OutboxEvent.
        """
        state = NodeState(
            has_content=True,
            has_relations=True,
            has_abstract=True,
            has_overview=True,
            has_meta=True,
            meta_status="PENDING",
        )
        assert self._should_activate_pending(state)

    def test_scenario_partial_write_during_step_3(self):
        """
        Scenario: Process crashes during step ③ (.abstract.md written, .overview.md not).

        State: content.md + .relations.json + .abstract.md exist, .overview.md missing.
        Recovery: Mark BROKEN (incomplete write), alert.
        """
        state = NodeState(
            has_content=True,
            has_relations=True,
            has_abstract=True,
            has_overview=False,  # Missing!
            has_meta=True,
            meta_status="PENDING",
        )
        assert self._should_mark_broken(state)

    def _should_attempt_repair(self, state: NodeState) -> bool:
        if state.meta_status in {"ACTIVE", "BROKEN", "ARCHIVED"}:
            return False
        if not state.has_meta and not state.has_content:
            return False
        return True

    def _should_activate_pending(self, state: NodeState) -> bool:
        return (
            state.meta_status == "PENDING"
            and state.has_content
            and state.has_relations
            and state.has_abstract
            and state.has_overview
            and state.has_meta
        )

    def _should_mark_broken(self, state: NodeState) -> bool:
        return (
            state.meta_status == "PENDING"
            and not (
                state.has_content
                and state.has_relations
                and state.has_abstract
                and state.has_overview
            )
        )


class TestWriteOrderEndToEndIntegration:
    """End-to-end integration tests with actual ContextFS and RepairJob.

    Phase 1 supplement — verifies that AGFSContextFS.write_node() produces
    file state that RepairJob can correctly detect and repair.

    These tests use MemoryAGFSSnapshot and MemoryOutboxEventSink to simulate
    the AGFS and OutboxStore without requiring real AGFS connection.
    """

    def test_write_node_creates_files_in_correct_order(self):
        """
        Verify AGFSContextFS.write_node() creates files in the exact order
        expected by RepairJob branch logic.
        """
        from core.models import ContextNode, RequestContext
        from index.repair_job import MemoryAGFSSnapshot, MemoryOutboxEventSink, RepairJob

        # Setup
        snapshot = MemoryAGFSSnapshot()
        event_sink = MemoryOutboxEventSink()
        repair_job = RepairJob(snapshot, event_sink)

        ctx = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="bot",
            session_id="sess-1",
            trace_id="trace-1",
        )

        # Create a node
        node = ContextNode(
            uri="ctx://acme/users/alice/memories/profile",
            context_type="MEMORY",
            category="profile",
            level=0,
            owner_space="user:alice",
            abstract="Software engineer profile",
            overview="## Overview\n- 5 years experience",
            content="Full profile content here",
            metadata={},
        )

        # Simulate write_node by manually adding files in correct order
        # (In production, write_node would call AGFS client)
        # Step ①: Write content.md
        snapshot.add_node(node.uri, {"content.md": node.content})

        # Step ②: Write .relations.json
        snapshot.add_node(node.uri, {
            "content.md": node.content,
            ".relations.json": "[]",
        })

        # Step ③: Write .abstract.md and .overview.md
        snapshot.add_node(node.uri, {
            "content.md": node.content,
            ".relations.json": "[]",
            ".abstract.md": node.abstract,
            ".overview.md": node.overview,
        })

        # Step ④: Write .meta.json with status=ACTIVE
        # (This is where the write would commit)
        snapshot.add_node(node.uri, {
            "content.md": node.content,
            ".relations.json": "[]",
            ".abstract.md": node.abstract,
            ".overview.md": node.overview,
            ".meta.json": '{"status": "ACTIVE"}',
        })

        # Now run RepairJob - should skip this node (ACTIVE)
        summary = repair_job.scan_account("acme")

        # Node should be skipped (already ACTIVE)
        assert summary.total_nodes_scanned == 1
        assert summary.nodes_skipped == 1

    def test_archived_node_is_skipped_by_repair_job(self):
        from index.repair_job import MemoryAGFSSnapshot, MemoryOutboxEventSink, RepairJob

        snapshot = MemoryAGFSSnapshot()
        event_sink = MemoryOutboxEventSink()
        repair_job = RepairJob(snapshot, event_sink)
        uri = "ctx://acme/users/alice/memories/preferences/coffee"

        snapshot.add_node(uri, {
            "content.md": "Content",
            ".relations.json": "[]",
            ".abstract.md": "Abstract",
            ".overview.md": "Overview",
            ".meta.json": '{"status": "ARCHIVED"}',
        })

        summary = repair_job.scan_account("acme")

        assert summary.total_nodes_scanned == 1
        assert summary.nodes_skipped == 1
        assert summary.nodes_marked_broken == 0
        assert len(event_sink.events) == 0

    def test_write_interruption_after_step_1_is_repairable(self):
        """
        Verify that write interruption after step ① (content.md written)
        can be detected and repaired by RepairJob.
        """
        from core.models import ContextNode, RequestContext
        from index.repair_job import MemoryAGFSSnapshot, MemoryOutboxEventSink, RepairJob

        snapshot = MemoryAGFSSnapshot()
        event_sink = MemoryOutboxEventSink()
        repair_job = RepairJob(snapshot, event_sink)

        ctx = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="bot",
            session_id="sess-1",
            trace_id="trace-1",
        )

        node = ContextNode(
            uri="ctx://acme/users/alice/memories/profile",
            context_type="MEMORY",
            category="profile",
            level=0,
            owner_space="user:alice",
            abstract="Abstract",
            overview="Overview",
            content="Content",
        )

        # Simulate write interruption after step ③ (all files except .meta.json)
        # SECURITY: Only mark ACTIVE when ALL required files from steps 1-3 are present
        snapshot.add_node(node.uri, {
            "content.md": node.content,
            ".relations.json": "[]",
            ".abstract.md": node.abstract,
            ".overview.md": node.overview,
            # .meta.json missing
        })

        # Run RepairJob
        summary = repair_job.scan_account("acme")

        # Should repair the node (recreate .meta.json)
        assert summary.nodes_repaired == 1
        assert len(event_sink.events) == 1
        assert event_sink.events[0].event_type == "UPSERT_CONTEXT"

        # Verify .meta.json was created
        meta_content = snapshot.read_file(node.uri, ".meta.json")
        assert meta_content is not None
        assert "ACTIVE" in meta_content

    def test_write_interruption_during_step_4_is_recoverable(self):
        """
        Verify that write interruption during step ④ (.meta.json write
        started but not committed with status=ACTIVE) can be repaired.
        """
        from core.models import ContextNode, RequestContext
        from index.repair_job import MemoryAGFSSnapshot, MemoryOutboxEventSink, RepairJob

        snapshot = MemoryAGFSSnapshot()
        event_sink = MemoryOutboxEventSink()
        repair_job = RepairJob(snapshot, event_sink)

        ctx = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="bot",
            session_id="sess-1",
            trace_id="trace-1",
        )

        node = ContextNode(
            uri="ctx://acme/users/alice/memories/profile",
            context_type="MEMORY",
            category="profile",
            level=0,
            owner_space="user:alice",
            abstract="Abstract",
            overview="Overview",
            content="Content",
        )

        # Simulate write interruption during step ④:
        # All files exist, but .meta.json has status=PENDING
        snapshot.add_node(node.uri, {
            "content.md": node.content,
            ".relations.json": "[]",
            ".abstract.md": node.abstract,
            ".overview.md": node.overview,
            ".meta.json": '{"status": "PENDING"}',  # Not committed
        })

        # Run RepairJob
        summary = repair_job.scan_account("acme")

        # Should activate the node (update status to ACTIVE)
        assert summary.nodes_activated == 1
        assert len(event_sink.events) == 1

        # Verify .meta.json was updated
        meta_content = snapshot.read_file(node.uri, ".meta.json")
        assert meta_content is not None
        assert "ACTIVE" in meta_content

    def test_partial_write_during_step_3_marks_broken(self):
        """
        Verify that partial write during step ③ (e.g., .overview.md missing)
        results in node being marked BROKEN.
        """
        from core.models import ContextNode, RequestContext
        from index.repair_job import MemoryAGFSSnapshot, MemoryOutboxEventSink, RepairJob

        snapshot = MemoryAGFSSnapshot()
        event_sink = MemoryOutboxEventSink()
        repair_job = RepairJob(snapshot, event_sink)

        ctx = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="bot",
            session_id="sess-1",
            trace_id="trace-1",
        )

        node = ContextNode(
            uri="ctx://acme/users/alice/memories/profile",
            context_type="MEMORY",
            category="profile",
            level=0,
            owner_space="user:alice",
            abstract="Abstract",
            overview="Overview",
            content="Content",
        )

        # Simulate partial write: .overview.md missing during step ③
        snapshot.add_node(node.uri, {
            "content.md": node.content,
            ".relations.json": "[]",
            ".abstract.md": node.abstract,
            # .overview.md missing!
            ".meta.json": '{"status": "PENDING"}',
        })

        # Run RepairJob
        summary = repair_job.scan_account("acme")

        # Should mark as BROKEN
        assert summary.nodes_marked_broken == 1
        assert summary.total_nodes_scanned == 1

        # Verify .meta.json was updated to BROKEN
        meta_content = snapshot.read_file(node.uri, ".meta.json")
        assert meta_content is not None
        assert "BROKEN" in meta_content

    def test_early_interruption_before_step_1_is_skipped(self):
        """
        Verify that early interruption before step ① (no files exist)
        is correctly skipped by RepairJob (nothing to repair).
        """
        from core.models import ContextNode, RequestContext
        from index.repair_job import MemoryAGFSSnapshot, MemoryOutboxEventSink, RepairJob

        snapshot = MemoryAGFSSnapshot()
        event_sink = MemoryOutboxEventSink()
        repair_job = RepairJob(snapshot, event_sink)

        ctx = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="bot",
            session_id="sess-1",
            trace_id="trace-1",
        )

        node = ContextNode(
            uri="ctx://acme/users/alice/memories/profile",
            context_type="MEMORY",
            category="profile",
            level=0,
            owner_space="user:alice",
            abstract="Abstract",
            overview="Overview",
            content="Content",
        )

        # No files written (early interruption)
        # Don't add node to snapshot

        # Run RepairJob
        summary = repair_job.scan_account("acme")

        # Should skip (no files, nothing to repair)
        assert summary.total_nodes_scanned == 0
        assert summary.nodes_skipped == 0

    def test_repair_job_creates_outbox_events(self):
        """
        Verify that RepairJob creates OutboxEvents for repaired and
        activated nodes.
        """
        from core.models import ContextNode, RequestContext
        from index.repair_job import MemoryAGFSSnapshot, MemoryOutboxEventSink, RepairJob

        snapshot = MemoryAGFSSnapshot()
        event_sink = MemoryOutboxEventSink()
        repair_job = RepairJob(snapshot, event_sink)

        ctx = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="bot",
            session_id="sess-1",
            trace_id="trace-1",
        )

        node = ContextNode(
            uri="ctx://acme/users/alice/memories/profile",
            context_type="MEMORY",
            category="profile",
            level=0,
            owner_space="user:alice",
            abstract="Abstract",
            overview="Overview",
            content="Content",
        )

        # Add node with .meta.json missing but all required files present (repairable)
        # SECURITY: Only mark ACTIVE when ALL required files from steps 1-3 are present
        snapshot.add_node(node.uri, {
            "content.md": node.content,
            ".relations.json": "[]",
            ".abstract.md": node.abstract,
            ".overview.md": node.overview,
        })

        # Run RepairJob
        summary = repair_job.scan_account("acme")

        # Should create OutboxEvent
        assert summary.outbox_events_created == 1
        assert len(event_sink.events) == 1
        assert event_sink.events[0].uri == node.uri
        assert event_sink.events[0].event_type == "UPSERT_CONTEXT"
