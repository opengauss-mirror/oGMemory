"""Database-dialect contract for SQL-backed stores."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

POSTGRESQL = "postgresql"
OPENGAUSS = "opengauss"


class SQLDialect(ABC):
    """Own database-specific SQL and capabilities used by storage adapters."""

    name: str
    supports_batch_claim: bool = True
    lock_timeout_parameter: str | None = "lock_timeout"

    def prepare_schema(self, schema_sql: str) -> str:
        return schema_sql

    def validate_connection(self, cursor: Any) -> None:
        return None

    @abstractmethod
    def ensure_workspace_root(self, cursor: Any, account_id: str, root_id: str) -> None: ...

    @abstractmethod
    def ensure_extraction_work_account(self, cursor: Any, account_id: str) -> None: ...

    @abstractmethod
    def write_context_node(
        self, cursor: Any, values: tuple[Any, ...], *, create_only: bool
    ) -> bool: ...

    @abstractmethod
    def rewrite_moved_relations(
        self,
        cursor: Any,
        *,
        from_uri: str,
        to_uri: str,
        from_prefix: str,
        to_prefix: str,
        json_wrapper: Callable[[Any], Any],
    ) -> None: ...

    @abstractmethod
    def table_exists(self, cursor: Any, table_name: str) -> bool: ...

    @abstractmethod
    def insert_session_archive(self, cursor: Any, values: tuple[Any, ...]) -> None: ...

    @abstractmethod
    def upsert_relation_edge(self, cursor: Any, values: tuple[Any, ...]) -> None: ...

    @abstractmethod
    def insert_tool_result_artifact(self, cursor: Any, values: tuple[Any, ...]) -> bool: ...

    @abstractmethod
    def insert_audit_log(self, cursor: Any, values: tuple[Any, ...]) -> None: ...

    @abstractmethod
    def notify_outbox(self, cursor: Any, channel: str, payload: str) -> None: ...

    @abstractmethod
    def insert_trace_event(
        self, cursor: Any, values: tuple[Any, ...]
    ) -> tuple[bool, Any | None]: ...

    @abstractmethod
    def upsert_trace_session(self, cursor: Any, values: tuple[Any, ...]) -> str | None: ...

    @abstractmethod
    def ensure_skill_candidate(self, cursor: Any, values: tuple[Any, ...]) -> None: ...

    @abstractmethod
    def ensure_skill_candidate_evidence(self, cursor: Any, values: tuple[Any, ...]) -> None: ...

    @abstractmethod
    def touch_skill_work_account(self, cursor: Any, account_id: str) -> None: ...

    @abstractmethod
    def insert_skill_example_evidence(self, cursor: Any, values: tuple[Any, ...]) -> None: ...


def normalize_sql_dialect(value: str | SQLDialect | None) -> str:
    if isinstance(value, SQLDialect):
        return value.name
    dialect = str(value or POSTGRESQL).strip().casefold()
    if dialect == "postgres":
        dialect = POSTGRESQL
    if dialect not in {POSTGRESQL, OPENGAUSS, "auto"}:
        raise ValueError("sql dialect must be postgresql or opengauss, or auto")
    return dialect


def resolve_sql_dialect(value: str | SQLDialect | None) -> SQLDialect:
    if isinstance(value, SQLDialect):
        return value
    dialect = normalize_sql_dialect(value)
    if dialect == "auto":
        raise ValueError("auto SQL dialect requires a database connection")
    if dialect == OPENGAUSS:
        from fs.sql_adapter.opengauss_dialect import OpenGaussDialect

        return OpenGaussDialect()
    from fs.sql_adapter.postgresql_dialect import PostgreSQLDialect

    return PostgreSQLDialect()
