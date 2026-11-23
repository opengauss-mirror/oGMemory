# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Bridge durable SessionState into rolling SessionWindowState."""

from __future__ import annotations

from datetime import datetime, timezone

from session.models import SessionWindowState
from session.session_state import SessionState


def apply_session_state_bridge(
    window_state: SessionWindowState,
    session_state: SessionState,
    session_id: str,
    *,
    turn_count: int = 0,
    force: bool = False,
) -> bool:
    version = session_state.get_version(session_id)
    if not force and version == window_state.session_state_version:
        return False

    task_state = session_state.get_task_state(session_id)
    has_state = bool(
        task_state.objective
        or task_state.current_stage
        or task_state.next_step
        or task_state.blockers
        or session_state.get_commitments(session_id, status=None)
    )
    if not has_state and version == 0:
        return False

    if task_state.objective and task_state.current_stage:
        active_task = f"{task_state.objective} - {task_state.current_stage}"
    else:
        active_task = task_state.objective or ""
    window_state.active_task = active_task

    window_state.open_loops = [
        loop.content for loop in session_state.get_open_loops(session_id)
    ]
    window_state.confirmed_constraints = session_state.get_confirmed_constraints(session_id)
    window_state.recent_decisions = session_state.get_recent_decisions(session_id)
    window_state.uncertainties = list(task_state.blockers)
    window_state.session_state_version = version
    window_state.session_state_synced_at = datetime.now(timezone.utc).isoformat()
    window_state.session_state_sync_turn_count = turn_count
    return True
