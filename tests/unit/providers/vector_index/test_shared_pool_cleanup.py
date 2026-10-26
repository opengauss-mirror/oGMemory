from types import SimpleNamespace
from unittest.mock import Mock

from psycopg2.extensions import TRANSACTION_STATUS_INTRANS

from fs.sql_adapter import pool as pool_module
from fs.sql_adapter.pool import SharedConnectionPool
from providers.vector_index.opengauss_index import OpenGaussVectorIndex


def test_vector_pool_replaces_connection_after_cleanup_failure(monkeypatch):
    broken = Mock(closed=0, info=SimpleNamespace(transaction_status=TRANSACTION_STATUS_INTRANS))
    broken.rollback.side_effect = RuntimeError("connection lost during rollback")
    broken.close.side_effect = lambda: setattr(broken, "closed", 1)
    replacement = Mock(closed=0)
    monkeypatch.setattr(pool_module.psycopg2, "connect", Mock(side_effect=[broken, replacement]))
    pool = SharedConnectionPool("dbname=vector", pool_size=1)
    index = OpenGaussVectorIndex.__new__(OpenGaussVectorIndex)
    index._init_pool(pool, "", 1)
    borrowed = index._get_connection()
    index._return_connection(borrowed)
    assert broken.closed == 1
    fresh = pool.get_connection(timeout=0.1)
    assert fresh is replacement
    pool.return_connection(fresh)
    pool.close_all()
