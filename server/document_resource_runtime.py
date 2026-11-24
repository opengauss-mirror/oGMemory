"""Compose resource storage with the shared account control plane."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, replace
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from core.document_backend import DocumentBackendError
from core.knowledge_base import (
    KnowledgeBase,
    KnowledgeBaseContext,
    KnowledgeBaseGrant,
    user_generation,
)
from core.models import Role
from fs.sql_adapter.pool import SharedConnectionPool
from providers.resource_store.ragflow import RagFlowError, RagFlowResourceAdapter
from providers.resource_store.ragflow_backend import RagFlowDocumentBackend
from providers.unified_config import ExternalKnowledgeBaseConfig, OgMemConfig
from server.api_keys import APIKeyManager
from server.audit import AuditService
from server.auth import AuthenticationError, AuthService, ControlPlaneDisabledError
from server.control_plane_store import ControlPlaneStore
from server.document_resource_service import DocumentResourceService
from server.document_resource_store import (
    DocumentResourceCommitError,
    DocumentResourceStoreError,
    SQLDocumentResourceStore,
)
from server.sql_control_plane_store import SQLControlPlaneStore
from server.tenant_admin import TenantAdminService

logger = logging.getLogger(__name__)


class DocumentResourceRuntime:
    def __init__(
        self,
        config: OgMemConfig,
        pool: SharedConnectionPool,
        control_store: Any,
    ) -> None:
        self._config = config
        self._pool = pool
        self._keys = APIKeyManager(control_store)
        self._auth = AuthService(config, self._keys)
        self._accounts = TenantAdminService(self._keys, control_store, AuditService(control_store))
        self._bindings: SQLDocumentResourceStore | None = None
        self._lock = Lock()

    def authenticate(self, headers: dict) -> KnowledgeBaseContext:
        if not self._auth.role_control_active():
            raise ControlPlaneDisabledError("Document resources require enabled role control")
        identity = self._auth.resolve_identity(headers, include_user_generation=True)
        if identity is None:
            raise AuthenticationError("API key is required")
        ctx = self._auth.build_request_context(
            identity,
            account_id=self._config.account_id,
            user_id=self._config.user_id,
            agent_id="",
            session_id="",
        )
        generation = user_generation(identity.user_created_at)
        if identity.registered_user and generation is None:
            raise AuthenticationError("registered user has no active identity generation")
        return KnowledgeBaseContext(
            **asdict(ctx if generation is not None else replace(ctx, user_id="")),
            user_generation=generation,
        )

    def _external_config(self) -> ExternalKnowledgeBaseConfig:
        config = self._config.knowledge_base
        if config.mode == "local":
            raise ControlPlaneDisabledError(
                "Local knowledge base document operations are unavailable"
            )
        external = config.external
        if external is None or external.provider != "ragflow":
            raise ControlPlaneDisabledError("External knowledge base provider is unavailable")
        return external

    def scope(self, ctx: KnowledgeBaseContext) -> tuple[str, list[dict[str, str]]]:
        self._external_config()
        if ctx.role == Role.ROOT:
            account_ids = [
                row["account_id"]
                for row in self._accounts.list_accounts(ctx)["accounts"]
                if row.get("status", "active") == "active"
            ]
        else:
            try:
                self._accounts.require_active_account(ctx, ctx.account_id, admin_only=False)
            except FileNotFoundError:
                raise PermissionError(
                    "document resources are unavailable for this account"
                ) from None
            account_ids = [ctx.account_id]
        if not account_ids:
            raise PermissionError("document resources are unavailable for this account")
        account_id = ctx.account_id if ctx.account_id in account_ids else account_ids[0]
        return account_id, [{"account_id": value} for value in account_ids]

    def _resource_store(self, ctx: KnowledgeBaseContext) -> SQLDocumentResourceStore:
        self._accounts.require_active_account(ctx, ctx.account_id, admin_only=False)
        external = self._external_config()
        with self._lock:
            if self._bindings is None:
                self._bindings = SQLDocumentResourceStore(self._pool)
            bindings = self._bindings
        namespace = external.account_namespaces.get(ctx.account_id)
        if (
            namespace
            and bindings.catalog.get(ctx.account_id, self._default_id(ctx.account_id)) is None
        ):
            bindings.catalog.ensure_default(
                KnowledgeBase(
                    ctx.account_id,
                    self._default_id(ctx.account_id),
                    "默认知识库",
                    "shared",
                    None,
                    RagFlowDocumentBackend.name,
                    namespace,
                    is_default=True,
                )
            )
        return bindings

    @staticmethod
    def _default_id(account_id: str) -> UUID:
        return uuid5(NAMESPACE_URL, f"ogmemory:knowledge-base:default:{account_id}")

    @staticmethod
    def _public(
        base: KnowledgeBase, ctx: KnowledgeBaseContext, members: dict[str, KnowledgeBaseGrant]
    ) -> dict:
        permissions = base.permissions(ctx, members)
        payload = {
            "id": str(base.id),
            "name": base.name,
            "visibility": base.visibility,
            "owner_user_id": base.owner_user_id,
            "is_default": base.is_default,
            "permissions": permissions,
        }
        if permissions["manage"]:
            payload["members"] = [
                {"user_id": user, "permission": grant.permission}
                for user, grant in sorted(members.items())
            ]
        return payload

    def list_knowledge_bases(self, ctx: KnowledgeBaseContext) -> dict:
        entries = self._resource_store(ctx).catalog.list(ctx.account_id)
        items = [
            self._public(base, ctx, members)
            for base, members in entries
            if base.permissions(ctx, members)["read"]
        ]
        can_create_shared = ctx.role in {Role.ROOT, Role.ADMIN}
        return {
            "items": items,
            "default_knowledge_base_id": next(
                (item["id"] for item in items if item["is_default"]), None
            ),
            "permissions": {
                "create": bool(ctx.user_generation) or can_create_shared,
                "create_private": bool(ctx.user_generation),
                "create_shared": can_create_shared,
            },
        }

    def create_knowledge_base(self, ctx: KnowledgeBaseContext, body: dict) -> dict:
        store = self._resource_store(ctx)
        if set(body) - {"name", "visibility", "members"}:
            raise ValueError("unsupported knowledge base field")
        name = _name(body.get("name"))
        visibility = body.get("visibility")
        if visibility == "private":
            if not ctx.user_id or not ctx.user_generation:
                raise PermissionError(
                    "a registered user key is required for a private knowledge base"
                )
            owner = ctx.user_id
            owner_generation = ctx.user_generation
        elif visibility == "shared":
            if ctx.role not in {Role.ROOT, Role.ADMIN}:
                raise PermissionError("account administrator required for a shared knowledge base")
            owner = None
            owner_generation = None
        else:
            raise ValueError("visibility must be private or shared")
        members = self._members(ctx, body.get("members", []), visibility)
        identifier = uuid4()
        with self._adapter() as adapter:
            namespace = adapter.create_dataset(f"ogmemory-{identifier.hex}")
        base = KnowledgeBase(
            ctx.account_id,
            identifier,
            name,
            visibility,
            owner,
            RagFlowDocumentBackend.name,
            namespace,
            owner_user_generation=owner_generation,
        )
        try:
            store.catalog.add(base, members)
        except DocumentResourceStoreError as exc:
            try:
                stored = self._recover_create(
                    store, base, commit_uncertain=isinstance(exc, DocumentResourceCommitError)
                )
                if stored is not None and (stored[0].backend, stored[0].namespace) == (
                    base.backend, base.namespace
                ):
                    return self._public(stored[0], ctx, stored[1])
            except (DocumentResourceStoreError, DocumentBackendError, FileExistsError):
                logger.error(
                    "knowledge base creation cleanup pending id=%s namespace=%s", identifier, namespace
                )
            raise
        return self._public(base, ctx, members)

    def _recover_create(
        self, store: SQLDocumentResourceStore, base: KnowledgeBase, *, commit_uncertain: bool
    ) -> tuple[KnowledgeBase, dict[str, KnowledgeBaseGrant]] | None:
        if commit_uncertain:
            with store.lock_knowledge_base(base):
                stored = store.catalog.get(base.account_id, base.id)
                if stored is not None:
                    return stored
                used = store.catalog.uses_namespace(base.backend, base.namespace)
        else:
            try:
                used = store.catalog.uses_namespace(base.backend, base.namespace)
            except DocumentResourceStoreError:
                used = False
        if not used and base.namespace not in self._external_config().account_namespaces.values():
            with self._adapter() as adapter:
                adapter.delete_dataset_if_empty(base.namespace, expected_name=f"ogmemory-{base.id.hex}")
        return None

    def update_knowledge_base(
        self, ctx: KnowledgeBaseContext, knowledge_base_id: UUID, body: dict
    ) -> dict:
        if not body or set(body) - {"name", "members"}:
            raise ValueError("supply name or members to update")
        store = self._resource_store(ctx)
        base, _ = self._authorized(store, ctx, knowledge_base_id, "manage")
        name = _name(body["name"]) if "name" in body else None
        grants = self._members(ctx, body["members"], base.visibility) if "members" in body else None
        with store.lock_knowledge_base(base):
            base, members = self._authorized(store, ctx, knowledge_base_id, "manage")
            updated = replace(base, name=name) if name is not None else base
            store.catalog.update(updated, grants)
        return self._public(updated, ctx, grants if grants is not None else members)

    def delete_knowledge_base(self, ctx: KnowledgeBaseContext, knowledge_base_id: UUID) -> None:
        store = self._resource_store(ctx)
        base, _ = self._authorized(store, ctx, knowledge_base_id, "manage")
        with store.lock_knowledge_base(base):
            base, _ = self._authorized(store, ctx, knowledge_base_id, "manage")
            if base.is_default:
                raise FileExistsError("the default knowledge base is managed by configuration")
            if store.has_documents(base.account_id, base.backend, base.namespace):
                raise FileExistsError("knowledge base contains documents")
            with self._adapter() as adapter:
                adapter.delete_dataset_if_empty(base.namespace)
            store.catalog.remove(base)

    def _member_generations(
        self, ctx: KnowledgeBaseContext, *, include_admins: bool = False
    ) -> dict[str, str]:
        users = self._accounts.list_users(ctx, ctx.account_id)["users"]
        return {
            row["user_id"]: generation
            for row in users
            if row.get("status", "active") == "active"
            and (include_admins or row.get("role") != "admin")
            and (generation := user_generation(row.get("created_at"))) is not None
        }

    def _members(
        self, ctx: KnowledgeBaseContext, value: Any, visibility: str
    ) -> dict[str, KnowledgeBaseGrant]:
        if not isinstance(value, list) or len(value) > 100:
            raise ValueError("members must be a list of up to 100 grants")
        if not value:
            return {}
        if visibility != "shared":
            raise ValueError("only shared knowledge bases support member grants")
        active_users = self._member_generations(ctx, include_admins=True)
        members: dict[str, KnowledgeBaseGrant] = {}
        for entry in value:
            if not isinstance(entry, dict) or set(entry) != {"user_id", "permission"}:
                raise ValueError("each member must provide user_id and permission")
            user, grant = entry["user_id"], entry["permission"]
            generation = active_users.get(user) if isinstance(user, str) else None
            if generation is None:
                raise ValueError("member must be an active user in the account")
            if user in members or grant not in ("read", "write"):
                raise ValueError("member grants require a unique user and read or write permission")
            members[user] = KnowledgeBaseGrant(grant, generation)
        return members

    @staticmethod
    def _authorized(
        store: SQLDocumentResourceStore,
        ctx: KnowledgeBaseContext,
        identifier: UUID,
        permission: str,
    ) -> tuple[KnowledgeBase, dict[str, KnowledgeBaseGrant]]:
        entry = store.catalog.get(ctx.account_id, identifier)
        if entry is None:
            raise FileNotFoundError("knowledge base not found")
        base, members = entry
        permissions = base.permissions(ctx, members)
        if not permissions["read"]:
            raise FileNotFoundError("knowledge base not found")
        if not permissions[permission]:
            raise PermissionError("knowledge base access denied")
        if base.backend != RagFlowDocumentBackend.name:
            raise ControlPlaneDisabledError("knowledge base provider is unavailable")
        return base, members

    @contextmanager
    def _adapter(self) -> Iterator[RagFlowResourceAdapter]:
        external = self._external_config()
        try:
            with RagFlowResourceAdapter(
                base_url=external.base_url, api_key=external.api_key
            ) as adapter:
                yield adapter
        except RagFlowError:
            raise DocumentBackendError("knowledge base backend operation failed") from None

    @contextmanager
    def open_documents(
        self,
        ctx: KnowledgeBaseContext,
        knowledge_base_id: UUID | None = None,
        *,
        write: bool = False,
    ) -> Iterator[DocumentResourceService]:
        bindings = self._resource_store(ctx)
        identifier = knowledge_base_id or self._default_id(ctx.account_id)
        permission = "write" if write else "read"
        base, _ = self._authorized(bindings, ctx, identifier, permission)
        with bindings.lock_knowledge_base(base) if write else nullcontext():
            if write:
                base, _ = self._authorized(bindings, ctx, identifier, permission)
            with self._adapter() as adapter:
                yield DocumentResourceService(
                    bindings, RagFlowDocumentBackend(adapter, base.namespace), ctx.account_id
                )


def _name(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not 1 <= len(value.strip()) <= 128
        or not value.strip().isprintable()
    ):
        raise ValueError("knowledge base name must contain 1 to 128 printable characters")
    return value.strip()


def create_document_resource_runtime(config: OgMemConfig) -> DocumentResourceRuntime:
    if config.knowledge_base.mode == "local":
        raise ControlPlaneDisabledError("Local knowledge base document operations are unavailable")
    if not config.sql_connection_string:
        raise ControlPlaneDisabledError("Document resources require SQL connection settings")
    pool = SharedConnectionPool(
        config.sql_connection_string,
        pool_size=config.sql_pool_size,
        dialect=config.sql_dialect or "postgresql",
    )
    control_store: SQLControlPlaneStore | ControlPlaneStore
    if config.storage_backend == "sql" or config.shared_spaces_enabled:
        control_store = SQLControlPlaneStore(pool=pool, control_plane_only=True)
    else:
        try:
            from fs.agfs_adapter.client import create_agfs_client
        except ImportError:
            control_store = ControlPlaneStore(
                config.agfs_mount_prefix, local_root=str(Path.cwd() / ".ogmem_control")
            )
        else:
            control_store = ControlPlaneStore(
                config.agfs_mount_prefix, client=create_agfs_client(config)
            )
    return DocumentResourceRuntime(config, pool, control_store)
