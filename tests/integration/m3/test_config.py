"""Tests for M3 config loading from ogmemory.yaml."""

from __future__ import annotations

from pathlib import Path

from m3.config import load_m3_runtime_config


def test_load_m3_runtime_config_reads_m3_block_without_env(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    cfg_path = tmp_path / "ogmemory.yaml"
    cfg_path.write_text(
        """
llm:
  provider: openai
  api_key: top-key
  base_url: https://llm.example/v1
  model: gpt-test
embedding:
  provider: mock
  model: top-embedding
vector_db:
  dimension: 12
  connection_string: host=localhost dbname=vectors
storage:
  connection_string: host=localhost dbname=test
m3:
  episodic_llm:
    provider: minimax
    api_key: episodic-key
    base_url: https://api.minimaxi.com/v1
    model: MiniMax-M3
    temperature: 0.2
    max_tokens: 777
  qa_llm:
    provider: mock
    model: qa-model
  embedding:
    provider: mock
    model: m3-embedding
    dimension: 8
""".strip(),
        encoding="utf-8",
    )

    cfg = load_m3_runtime_config(str(cfg_path))
    assert cfg.episodic_llm.provider == "minimax"
    assert cfg.episodic_llm.api_key == "episodic-key"
    assert cfg.episodic_llm.model == "MiniMax-M3"
    assert cfg.episodic_llm.max_tokens == 777
    assert cfg.qa_llm.provider == "mock"
    assert cfg.qa_llm.model == "qa-model"
    assert cfg.embedding_provider == "mock"
    assert cfg.embedding_model == "m3-embedding"
    assert cfg.embedding_dimension == 8
    assert cfg.graph_connection_string == "host=localhost dbname=test"


def test_load_m3_runtime_config_env_placeholders_require_export(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    cfg_path = tmp_path / "ogmemory.yaml"
    cfg_path.write_text(
        """
m3:
  episodic_llm:
    provider: minimax
    api_key: "${MINIMAX_API_KEY}"
    model: MiniMax-M3
""".strip(),
        encoding="utf-8",
    )
    cfg = load_m3_runtime_config(str(cfg_path))
    assert cfg.episodic_llm.api_key is None

    monkeypatch.setenv("MINIMAX_API_KEY", "exported-key")
    cfg = load_m3_runtime_config(str(cfg_path))
    assert cfg.episodic_llm.api_key == "exported-key"


def test_vector_index_connection_string_inherits_vector_db(monkeypatch, tmp_path: Path):
    """The M3 vector index has no own DSN key — it inherits vector_db's.

    Regression test for the config-surface simplification: a
    ``m3.vector_index.connection_string`` key must be ignored (not
    read), and the effective DSN must always equal
    ``vector_db.connection_string``.
    """
    cfg_path = tmp_path / "ogmemory.yaml"
    cfg_path.write_text(
        """
vector_db:
  connection_string: host=primary dbname=graph
storage:
  connection_string: host=primary dbname=storage
m3:
  vector_index:
    backend: pgvector
    connection_string: host=stale dbname=should-be-ignored
    table_name: custom_m3_table
    pool_size: 3
""".strip(),
        encoding="utf-8",
    )
    cfg = load_m3_runtime_config(str(cfg_path))
    # DSN inherited from vector_db, not the stale m3.vector_index key.
    assert cfg.vector_index_connection_string == "host=primary dbname=graph"
    assert cfg.graph_connection_string == "host=primary dbname=storage"
    # The M3-specific knobs are still honoured.
    assert cfg.vector_index_backend == "pgvector"
    assert cfg.vector_index_table_name == "custom_m3_table"
    assert cfg.vector_index_pool_size == 3
