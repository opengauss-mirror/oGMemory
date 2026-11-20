"""MemoryService lifecycle callbacks used by a registered migration participant."""
import time

from core.models import RequestContext, Role
from service.migration.maintenance import MaintenanceError


class ServiceLifecycle:
    def __init__(self, service, *, timeout=30):
        self.service = service
        self.timeout = timeout
        self.restart_outbox = False
        self.restart_code_sync = False

    def pause(self):
        service = self.service
        deadline = time.monotonic() + self.timeout
        def remaining():
            return max(0, deadline - time.monotonic())
        self.restart_outbox = self.restart_outbox or service._outbox_thread is not None
        space_worker = getattr(service, '_space_worker', None)
        if space_worker is not None and not space_worker.stop(timeout=remaining()):
            return False
        if space_worker is not None:
            service._space_worker = None
        service._outbox_stop.set()
        if service._outbox_listener is not None:
            service._outbox_listener.stop()
        if not service.stop_skill_sedimentation(timeout=remaining()):
            return False
        service._extraction_retry.close(remaining())
        for thread in (service._outbox_thread, service._extraction_retry._thread):
            if thread is not None:
                thread.join(remaining())
                if thread.is_alive():
                    return False
        if service._scene_manager is not None and not service._scene_manager.stop(timeout=remaining()):
            return False
        if service._code_workspace_sync_service is not None:
            sync = service._code_workspace_sync_service
            self.restart_code_sync = self.restart_code_sync or getattr(sync, '_scheduler', None) is not None
            sync.shutdown(wait=True)
        if service._lexical_backfill is not None and not service._lexical_backfill.close(timeout=remaining()):
            return False
        if time.monotonic() > deadline:
            return False
        if service._tool_result_io_runner is not None and not service._tool_result_io_runner.wait_idle(remaining()):
            return False
        content = getattr(service, '_space_content_store', None)
        if content is not None and not content.close(timeout=remaining()):
            return False
        fts = getattr(service, "_agfs_fts", None)
        if fts is not None and not fts.close(remaining()):
            return False
        self._save_sessions()
        # Close SDK clients before the independent migration reader is opened.
        # Reassigning a Python field alone does not invalidate native caches.
        vector = service._vector_index
        if service._projection_index is not None and service._lexical_index is not None:
            if not service._projection_index.close(timeout=remaining()):
                return False
        if vector is not None:
            release = getattr(vector, 'release_for_migration', None) or getattr(vector, 'close', None)
            if callable(release):
                release()
        return time.monotonic() <= deadline

    def _save_sessions(self):
        service = self.service
        manager = service._session_mgr
        if manager is None:
            return
        with manager._lock:
            sessions = list(manager._sessions.items())
        for session_id, buffer in sessions:
            if (buffer.active_operation_count or buffer.extraction_active_count or buffer.commit_in_progress
                    or buffer.dispose_in_progress or buffer.cleaning):
                raise MaintenanceError('session still has an active business operation')
            fallback = RequestContext(service._default_account_id, service._default_user_id,
                                      service._default_agent_id, session_id, 'migration', Role.ROOT)
            ctx = manager._session_state_context(session_id, buffer.meta, fallback)
            with service._session_mutation_scope(ctx, refresh=False):
                if not manager.save_session_state(session_id, ctx):
                    raise MaintenanceError('session checkpoint could not be persisted')

    def refresh(self):
        service = self.service
        active = service._admission.store.status()
        if active is None:
            raise MaintenanceError('cache refresh requires active maintenance')
        original_session = service._session_mgr
        original_retry = service._extraction_retry
        original_discovery = service._extraction_discovery
        # Identity/authentication clients are deliberately retained. All business
        # clients and session caches will be opened lazily from restored storage.
        for name in (
            '_agfs_fts', '_migration_context_fs', '_read_api', '_write_api', '_session_mgr', '_archive_store', '_trace_event_store',
            '_vector_index', '_projection_index', '_lexical_index', '_lexical_backfill',
            '_graph_component_cache', '_bm25_index', '_outbox_listener', '_outbox_thread',
            '_skill_sedimentation_service', '_skill_sedimentation_worker', '_skill_sedimentation_thread',
            '_code_resource_lifecycle_service', '_code_workspace_sync_service', '_code_retrieval_service',
            '_code_file_edit_service', '_tool_output_filter_service', '_extraction_discovery',
            '_scene_manager', '_scene_ranker', '_scene_source', '_scene_service',
            '_tool_result_store', '_tool_result_externalizer', '_tool_result_turn_ledger',
            '_tool_result_io_runner', '_tool_result_governance',
            '_space_worker', '_space_repository', '_space_content_store',
        ):
            setattr(service, name, None)
        service._archive_store_failed = False
        from service.extraction_retry import ExtractionRetryWorker
        if active['modified']:
            service._extraction_retry = ExtractionRetryWorker(service)
        else:
            # A cancelled/failed preflight must not discard accepted in-memory
            # messages or pending extraction work when no import changed data.
            service._session_mgr = original_session
            service._extraction_retry = original_retry
            service._extraction_discovery = original_discovery

    def resume(self):
        service = self.service
        service._outbox_stop.clear()
        service._extraction_retry._stop.clear()
        if service._extraction_discovery is not None:
            service._extraction_retry.start_discovery(service._extraction_discovery)
        if service._cfg.tool_result_externalization_enabled:
            service._initialize_tool_result_externalization(service._configured_tool_result_store)
        if getattr(service._cfg, 'shared_spaces_enabled', False):
            service._start_space_worker()
        if service._cfg.skill_sedimentation_enabled:
            service._start_skill_sedimentation_worker()
        service.start_extraction_recovery()
        if self.restart_code_sync:
            service._get_code_workspace_sync_service().start()
            self.restart_code_sync = False
        if self.restart_outbox:
            service._start_outbox_worker()
        self.restart_outbox = False
