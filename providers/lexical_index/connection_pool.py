# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Connection pool owned only by SQL lexical projections."""

from __future__ import annotations

import threading
import time


class LexicalConnectionPool:
    """Small bounded pool with connection/query timeouts and drainable close."""

    def __init__(
        self,
        connection_string: str,
        pool_size: int,
        connect_fn,
        *,
        connect_timeout_seconds: int | None = None,
        session_statement_timeout_ms: int | None = None,
    ) -> None:
        if pool_size < 1:
            raise ValueError("pool_size must be positive")
        self.connection_string = connection_string
        self.connect_timeout_seconds = connect_timeout_seconds
        self.session_statement_timeout_ms = session_statement_timeout_ms
        self._pool_size = pool_size
        self._connect = connect_fn
        self._idle: list = []
        self._active = 0
        self._borrowed: set[int] = set()
        self._closing = False
        self._condition = threading.Condition()
        self._slots = threading.Semaphore(pool_size)

    @staticmethod
    def _close(conn) -> None:
        if getattr(conn, "closed", 1) == 0:
            conn.close()

    def _alive(self, conn) -> bool:
        if getattr(conn, "closed", 1) != 0:
            return False
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
            conn.rollback()
            return True
        except Exception:
            self._close(conn)
            return False

    def _new_connection(self):
        kwargs = {}
        if self.connect_timeout_seconds is not None:
            kwargs["connect_timeout"] = self.connect_timeout_seconds
        conn = self._connect(self.connection_string, **kwargs)
        if self.session_statement_timeout_ms is not None:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT set_config(%s, %s, false)",
                        ("statement_timeout", str(self.session_statement_timeout_ms)),
                    )
                conn.commit()
            except Exception:
                self._close(conn)
                raise
        return conn

    def get_connection(self, timeout: float | None = 30.0):
        with self._condition:
            if self._closing:
                raise RuntimeError("LexicalConnectionPool is closing")
        if not self._slots.acquire(timeout=timeout):
            raise RuntimeError("timed out waiting for a lexical SQL connection")
        conn = None
        active_incremented = False
        try:
            with self._condition:
                if self._closing:
                    raise RuntimeError("LexicalConnectionPool is closing")
                if self._idle:
                    conn = self._idle.pop()
                self._active += 1
                active_incremented = True
            if conn is None or not self._alive(conn):
                conn = self._new_connection()
            with self._condition:
                self._borrowed.add(id(conn))
            return conn
        except Exception:
            with self._condition:
                if active_incremented:
                    self._active -= 1
                self._condition.notify_all()
            self._slots.release()
            raise

    def return_connection(self, conn) -> None:
        with self._condition:
            if id(conn) not in self._borrowed:
                self._close(conn)
                return
            self._borrowed.remove(id(conn))
            self._active -= 1
            if self._closing or getattr(conn, "closed", 1) != 0:
                self._close(conn)
            else:
                self._idle.append(conn)
            self._condition.notify_all()
        self._slots.release()

    def close_all(self, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + max(0.0, timeout)
        with self._condition:
            self._closing = True
            while self._idle:
                self._close(self._idle.pop())
            while self._active:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True
