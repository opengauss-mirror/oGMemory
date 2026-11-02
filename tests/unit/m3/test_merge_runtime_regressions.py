"""Runtime contracts lost in the AntTrail merge."""

import struct
import wave
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from fs.sql_adapter.dialect import resolve_sql_dialect
from providers.vector_index.opengauss_index import OpenGaussVectorIndex


def test_shared_vector_pool_is_borrowed_returned_and_not_closed(monkeypatch):
    pool = MagicMock()
    pool.connection_string = "postgresql://vector-db/vectors"
    pool.sql_dialect = resolve_sql_dialect("opengauss")
    conn = pool.get_connection.return_value
    monkeypatch.setattr("providers.vector_index.opengauss_index.OPENGAUSS_AVAILABLE", True)
    index = OpenGaussVectorIndex(pool=pool, dimension=2)
    assert index._connection_string == pool.connection_string
    pool.get_connection.assert_called_once()
    pool.return_connection.assert_called_once_with(conn)
    index.close()
    conn.close.assert_not_called()
    pool.close_all.assert_not_called()


def test_vector_initialization_failure_is_not_swallowed(monkeypatch):
    pool = MagicMock()
    pool.connection_string = "postgresql://vector-db/vectors"
    pool.sql_dialect = resolve_sql_dialect("opengauss")
    conn = pool.get_connection.return_value
    conn.cursor.side_effect = RuntimeError("database unavailable")
    monkeypatch.setattr("providers.vector_index.opengauss_index.OPENGAUSS_AVAILABLE", True)
    with pytest.raises(RuntimeError, match="initialization failed"):
        OpenGaussVectorIndex(pool=pool, dimension=2)
    pool.return_connection.assert_called_once_with(conn)
    conn.rollback.assert_called()


def test_vector_hits_keep_graph_provenance_and_custom_metadata():
    hit = OpenGaussVectorIndex._row_to_hit({
        "uri": "ctx://m3bench/episode",
        "node_id": "graph-node",
        "owner_space": "users/current",
        "filters": {"owner_space": "users/stale"},
        "metadata": {"extracted_node_id": "episode", "custom_field": "preserved"},
    })
    assert hit.owner_space == "users/current"
    assert hit.metadata == {
        "node_id": "graph-node", "extracted_node_id": "episode",
        "custom_field": "preserved",
    }


def test_provider_config_uses_registered_vector_factory(monkeypatch):
    from providers.config import ProviderConfig
    from providers.vector_index import factory

    monkeypatch.setattr(factory, "_PROVIDERS", dict(factory._PROVIDERS))
    sentinel = object()
    factory.register_vector_provider("merge-test", lambda config: sentinel)
    assert ProviderConfig(vector_db_type=" MERGE-TEST ").create_vector_index() is sentinel


def test_provider_config_passes_shared_pool_without_requiring_second_dsn(monkeypatch):
    from providers.config import ProviderConfig

    constructor = MagicMock()
    monkeypatch.setattr(
        "providers.vector_index.opengauss_index.OpenGaussVectorIndex", constructor,
    )
    pool = MagicMock()
    ProviderConfig(vector_db_type="opengauss").create_vector_index(pool=pool)
    assert constructor.call_args.kwargs["pool"] is pool


@pytest.mark.parametrize("streaming_header", [False, True])
def test_diarization_of_empty_pcm_does_not_call_model(tmp_path, streaming_header):
    from extraction.operators.builtin.speaker_diarization import SpeakerDiarizationExtractor

    path = tmp_path / "empty.wav"
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"")
    if streaming_header:
        payload = bytearray(path.read_bytes())
        struct.pack_into("<I", payload, 4, 0xFFFFFFFF)
        struct.pack_into("<I", payload, 40, 0xFFFFFFFF)
        path.write_bytes(payload)
    operator = SpeakerDiarizationExtractor(backend="mock")
    operator._backend = MagicMock()
    node = SimpleNamespace(node_id="empty-audio", metadata={"source_path": str(path)})
    result = operator.extract(node, SimpleNamespace())
    assert not result.error
    assert result.extracted_content["turns"] == []
    assert result.extracted_content["num_speakers"] == 0
    operator._backend.diarize.assert_not_called()
