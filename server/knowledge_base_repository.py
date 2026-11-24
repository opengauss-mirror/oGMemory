"""SQL catalog for knowledge base ownership, provider binding and grants."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from hashlib import sha256
from typing import Any
from uuid import UUID

from core.knowledge_base import KnowledgeBase, KnowledgeBaseGrant

_SCHEMA = """
CREATE TABLE IF NOT EXISTS knowledge_bases (
    account_id VARCHAR(128) NOT NULL,
    knowledge_base_id VARCHAR(36) NOT NULL,
    name VARCHAR(128) NOT NULL,
    visibility VARCHAR(8) NOT NULL,
    owner_user_id VARCHAR(128),
    owner_user_generation VARCHAR(64),
    backend VARCHAR(64) NOT NULL,
    namespace VARCHAR(128) NOT NULL,
    is_default BOOLEAN NOT NULL DEFAULT FALSE,
    PRIMARY KEY (account_id, knowledge_base_id),
    UNIQUE (backend, namespace),
    CHECK ((visibility = 'private' AND owner_user_id IS NOT NULL
            AND owner_user_generation IS NOT NULL AND NOT is_default)
        OR (visibility = 'shared' AND owner_user_id IS NULL AND owner_user_generation IS NULL))
);
CREATE TABLE IF NOT EXISTS knowledge_base_members (
    account_id VARCHAR(128) NOT NULL,
    knowledge_base_id VARCHAR(36) NOT NULL,
    user_id VARCHAR(128) NOT NULL,
    user_generation VARCHAR(64) NOT NULL,
    permission VARCHAR(8) NOT NULL CHECK (permission IN ('read', 'write')),
    PRIMARY KEY (account_id, knowledge_base_id, user_id),
    FOREIGN KEY (account_id, knowledge_base_id)
        REFERENCES knowledge_bases(account_id, knowledge_base_id) ON DELETE CASCADE
);
"""


def knowledge_base_lock_id(base: KnowledgeBase) -> int:
    key = f"ogmemory:knowledge-base:{base.account_id}\0{base.id}".encode()
    return int.from_bytes(sha256(key).digest()[:8], "big", signed=True)


class KnowledgeBaseRepository:
    def __init__(self, transaction: Callable[[], AbstractContextManager[Any]]) -> None:
        self._transaction = transaction

    def ensure_schema(self, cur: Any) -> None:
        cur.execute(_SCHEMA)

    def ensure_default(self, base: KnowledgeBase) -> None:
        with self._transaction() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(%s)", (knowledge_base_lock_id(base),))
            cur.execute(
                "SELECT backend, namespace FROM knowledge_bases "
                "WHERE account_id = %s AND knowledge_base_id = %s",
                (base.account_id, str(base.id)),
            )
            row = cur.fetchone()
            if row is None:
                self._insert(cur, base)
            elif tuple(row) != (base.backend, base.namespace):
                raise FileExistsError("default knowledge base backend binding has changed")

    def get(
        self, account_id: str, knowledge_base_id: UUID
    ) -> tuple[KnowledgeBase, dict[str, KnowledgeBaseGrant]] | None:
        entries = self._read(account_id, knowledge_base_id)
        return entries[0] if entries else None

    def list(self, account_id: str) -> list[tuple[KnowledgeBase, dict[str, KnowledgeBaseGrant]]]:
        return self._read(account_id)

    def add(self, base: KnowledgeBase, members: dict[str, KnowledgeBaseGrant]) -> None:
        with self._transaction() as cur:
            self._insert(cur, base)
            self._add_members(cur, base, members)

    def update(self, base: KnowledgeBase, members: dict[str, KnowledgeBaseGrant] | None) -> None:
        key = (base.account_id, str(base.id))
        with self._transaction() as cur:
            cur.execute(
                "UPDATE knowledge_bases SET name = %s WHERE account_id = %s AND knowledge_base_id = %s",
                (base.name, *key),
            )
            if members is not None:
                cur.execute(
                    "DELETE FROM knowledge_base_members WHERE account_id = %s AND knowledge_base_id = %s",
                    key,
                )
                self._add_members(cur, base, members)

    def remove(self, base: KnowledgeBase) -> None:
        with self._transaction() as cur:
            cur.execute(
                "DELETE FROM knowledge_bases WHERE account_id = %s AND knowledge_base_id = %s AND NOT is_default",
                (base.account_id, str(base.id)),
            )

    def _read(
        self, account_id: str, knowledge_base_id: UUID | None = None
    ) -> list[tuple[KnowledgeBase, dict[str, KnowledgeBaseGrant]]]:
        selection = " AND b.knowledge_base_id = %s" if knowledge_base_id is not None else ""
        params = (
            (account_id, str(knowledge_base_id)) if knowledge_base_id is not None else (account_id,)
        )
        with self._transaction() as cur:
            cur.execute(
                f"""
                SELECT b.knowledge_base_id, b.name, b.visibility, b.owner_user_id,
                       b.backend, b.namespace, b.is_default, b.owner_user_generation,
                       m.user_id, m.permission, m.user_generation
                FROM knowledge_bases b LEFT JOIN knowledge_base_members m
                    ON b.account_id = m.account_id AND b.knowledge_base_id = m.knowledge_base_id
                WHERE b.account_id = %s{selection}
                ORDER BY b.is_default DESC, b.name, b.knowledge_base_id, m.user_id
                """,
                params,
            )
            rows = cur.fetchall()
        bases: dict[str, tuple[KnowledgeBase, dict[str, KnowledgeBaseGrant]]] = {}
        for (
            identifier,
            name,
            visibility,
            owner,
            backend,
            namespace,
            is_default,
            generation,
            user,
            grant,
            user_generation,
        ) in rows:
            if identifier not in bases:
                bases[identifier] = (
                    KnowledgeBase(
                        account_id,
                        UUID(identifier),
                        name,
                        visibility,
                        owner,
                        backend,
                        namespace,
                        is_default,
                        generation,
                    ),
                    {},
                )
            if user is not None:
                bases[identifier][1][user] = KnowledgeBaseGrant(grant, user_generation)
        return list(bases.values())

    @staticmethod
    def _insert(cur: Any, base: KnowledgeBase) -> None:
        cur.execute(
            """
            INSERT INTO knowledge_bases
                (account_id, knowledge_base_id, name, visibility, owner_user_id, backend, namespace,
                 is_default, owner_user_generation)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                base.account_id,
                str(base.id),
                base.name,
                base.visibility,
                base.owner_user_id,
                base.backend,
                base.namespace,
                base.is_default,
                base.owner_user_generation,
            ),
        )

    @staticmethod
    def _add_members(cur: Any, base: KnowledgeBase, members: dict[str, KnowledgeBaseGrant]) -> None:
        if members:
            cur.executemany(
                "INSERT INTO knowledge_base_members "
                "(account_id, knowledge_base_id, user_id, permission, user_generation) "
                "VALUES (%s, %s, %s, %s, %s)",
                [
                    (base.account_id, str(base.id), user, grant.permission, grant.user_generation)
                    for user, grant in members.items()
                ],
            )
