"""Deadline-bound SQL snapshot I/O; import and COPY retain normal connections."""
import time

from psycopg2 import sql

from fs.sql_adapter.pool import _deadline_connection, wait_for_postgres


class SnapshotConnection:
    def __init__(self, dsn, deadline):
        self.deadline = deadline
        self._context = _deadline_connection(dsn, self.deadline)
        self.raw = self._context.__enter__()
        try:
            with self.cursor() as cur:
                cur.execute('BEGIN READ ONLY')
        except BaseException:
            self.close()
            raise

    def __getattr__(self, name):
        return getattr(self.raw, name)

    def cursor(self, name=None):
        return SnapshotCursor(self, name)

    def wait(self):
        try:
            wait_for_postgres(self.raw, self.deadline)
        except BaseException:
            self.close()
            raise

    def commit(self):
        with self.cursor() as cur:
            cur.execute('COMMIT')
            cur.execute('BEGIN READ ONLY')

    def rollback(self):
        if not self.raw.closed:
            with self.cursor() as cur:
                cur.execute('ROLLBACK')
                cur.execute('BEGIN READ ONLY')

    def close(self):
        self.raw.close()


class SnapshotCursor:
    def __init__(self, connection, name):
        self.connection, self.name = connection, name
        self.raw = connection.raw.cursor()
        self.itersize = 256
        self.declared = False

    def __getattr__(self, name):
        return getattr(self.raw, name)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        # Closing the connection discards server cursors on errors/timeouts.
        if self.declared and not self.connection.raw.closed and args[0] is None:
            self._execute(sql.SQL('CLOSE {}').format(sql.Identifier(self.name)))
        self.raw.close()

    def _execute(self, query, params=None):
        if time.monotonic() >= self.connection.deadline:
            self.connection.close()
            raise TimeoutError('SQL snapshot deadline exceeded')
        self.raw.execute(query, params)
        self.connection.wait()

    def execute(self, query, params=None):
        if self.name:
            query = sql.SQL('DECLARE {} NO SCROLL CURSOR FOR ').format(sql.Identifier(self.name)).as_string(
                self.connection.raw).encode() + self.raw.mogrify(query, params)
            self._execute(query)
            self.declared = True
        else:
            self._execute(query, params)

    def __iter__(self):
        if not self.name:
            yield from self.raw
            return
        while True:
            self._execute(sql.SQL('FETCH FORWARD {} FROM {}').format(
                sql.Literal(self.itersize), sql.Identifier(self.name)))
            rows = self.raw.fetchall()
            if not rows:
                return
            yield from rows
