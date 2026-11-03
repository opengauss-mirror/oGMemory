"""Tests for V1 auth and sharing config in OgMemConfig."""

from __future__ import annotations

import json

import pytest

from providers.unified_config import OgMemConfig


def test_explicit_model_overrides_take_priority_over_yaml(monkeypatch):
    from providers import unified_config as uc

    monkeypatch.setattr(
        uc,
        "_load_yaml",
        lambda _path: {
            "llm": {
                "provider": "mock",
                "api_key": "yaml-llm-key",
                "base_url": "https://yaml-llm.example/v1",
                "model": "yaml-llm-model",
            },
            "embedding": {
                "provider": "st",
                "api_key": "yaml-embedding-key",
                "base_url": "https://yaml-embedding.example/v1",
                "model": "yaml-embedding-model",
            },
        },
    )
    monkeypatch.setenv(
        "OGMEM_MODEL_CONFIG_OVERRIDES",
        json.dumps(
            {
                "llmProvider": "openai",
                "llmApiKey": "plugin-llm-key",
                "llmBaseUrl": "https://plugin-llm.example/v1",
                "llmModel": "plugin-llm-model",
                "llmReasoningEffort": "max",
                "llmIsReasoningModel": True,
                "llmExtraBody": {"thinking": {"type": "enabled"}},
                "embeddingProvider": "openai",
                "embeddingApiKey": "plugin-embedding-key",
                "embeddingBaseUrl": "https://plugin-embedding.example/v1",
                "embeddingModel": "plugin-embedding-model",
            }
        ),
    )

    cfg = OgMemConfig.load("ignored.yaml")

    assert cfg.provider == "openai"
    assert cfg.openai_api_key == "plugin-llm-key"
    assert cfg.openai_base_url == "https://plugin-llm.example/v1"
    assert cfg.openai_llm_model == "plugin-llm-model"
    assert cfg.llm_reasoning_effort == "max"
    assert cfg.llm_is_reasoning_model is True
    assert cfg.llm_extra_body == {"thinking": {"type": "enabled"}}
    assert cfg.embedding_provider == "openai"
    assert cfg.openai_embedding_api_key == "plugin-embedding-key"
    assert cfg.openai_embedding_base_url == "https://plugin-embedding.example/v1"
    assert cfg.openai_embedding_model == "plugin-embedding-model"


def test_load_reads_v1_auth_and_sharing_from_env(monkeypatch):
    monkeypatch.setenv("OG_ROLE_CONTROL_ENABLED", "true")
    monkeypatch.setenv("OG_ROOT_API_KEY", "root-key")
    monkeypatch.setenv("OG_ADMIN_API_KEYS", "admin-a,admin-b")
    monkeypatch.setenv("OG_AGENT_SHARED_MODE", "user")
    monkeypatch.setenv("OG_AGENT_SHARED_LIST", "agent-1,agent-2")
    monkeypatch.setenv("OG_HTTP_IP_ALLOWLIST", "127.0.0.1,10.0.0.0/8")
    monkeypatch.setenv("OG_HTTP_IP_ALLOWLIST_TRUST_PROXY", "true")

    cfg = OgMemConfig.load("/tmp/non-existent-ogmem.yaml")

    assert cfg.role_control_enabled is True
    assert cfg.root_api_key == "root-key"
    assert cfg.admin_api_keys == ["admin-a", "admin-b"]
    assert cfg.agent_shared_mode == "user"
    assert cfg.agent_shared_list == ["agent-1", "agent-2"]
    assert cfg.http_ip_allowlist == ["127.0.0.1", "10.0.0.0/8"]
    assert cfg.http_ip_allowlist_trust_proxy is True


def test_load_uses_safe_defaults_for_v1_auth_and_sharing():
    cfg = OgMemConfig.load("/tmp/non-existent-ogmem.yaml")

    assert cfg.role_control_enabled is False
    assert cfg.root_api_key is None
    assert cfg.admin_api_keys == []
    assert cfg.agent_shared_mode == "off"
    assert cfg.agent_shared_list == []
    assert cfg.http_ip_allowlist == []
    assert cfg.http_ip_allowlist_trust_proxy is False
    assert cfg.llm_reasoning_effort is None
    assert cfg.llm_is_reasoning_model is None


def test_tool_result_search_deadline_defaults_to_two_seconds():
    cfg = OgMemConfig.load("/tmp/non-existent-ogmem.yaml")

    assert cfg.tool_result_search_deadline_ms == 2_000


def test_tool_result_search_deadline_reads_yaml_and_env(monkeypatch):
    from providers import unified_config as uc

    monkeypatch.setenv("OGMEM_TOOL_RESULT_SEARCH_DEADLINE_MS", "2500")
    monkeypatch.setattr(
        uc,
        "_load_yaml",
        lambda _path: {
            "tool_result_externalization": {"search_deadline_ms": 3000},
        },
    )
    assert OgMemConfig.load("ignored.yaml").tool_result_search_deadline_ms == 3_000

    monkeypatch.setattr(
        uc,
        "_load_yaml",
        lambda _path: {"tool_result_externalization": {}},
    )
    assert OgMemConfig.load("ignored.yaml").tool_result_search_deadline_ms == 2_500


def test_load_reads_reasoning_configuration_from_yaml(monkeypatch):
    from providers import unified_config as uc

    monkeypatch.setattr(uc, "_load_yaml", lambda _path: {"llm": {
        "reasoning_effort": "xhigh",
        "extra_body": {"thinking": {"type": "enabled"}},
    }})

    cfg = OgMemConfig.load("ignored.yaml")

    assert cfg.llm_reasoning_effort == "xhigh"
    assert cfg.llm_extra_body == {"thinking": {"type": "enabled"}}


def test_load_reads_reasoning_configuration_from_env(monkeypatch):
    monkeypatch.setenv("LLM_REASONING_EFFORT", "max")
    monkeypatch.setenv("LLM_IS_REASONING_MODEL", "true")
    monkeypatch.setenv("LLM_EXTRA_BODY", '{"enable_thinking":true}')

    cfg = OgMemConfig.load("/tmp/non-existent-ogmem.yaml")

    assert cfg.llm_reasoning_effort == "max"
    assert cfg.llm_is_reasoning_model is True
    assert cfg.llm_extra_body == {"enable_thinking": True}


def test_service_threads_supports_new_config_and_workers_fallback(monkeypatch):
    from providers import unified_config as uc

    monkeypatch.setattr(
        uc,
        "_load_yaml",
        lambda _path: {"service": {"workers": 3}},
    )
    assert OgMemConfig.load("ignored.yaml").threads == 3

    monkeypatch.setattr(
        uc,
        "_load_yaml",
        lambda _path: {"service": {"workers": 1, "threads": 6}},
    )
    cfg = OgMemConfig.load("ignored.yaml")
    assert cfg.workers == 1
    assert cfg.threads == 6

    monkeypatch.setenv("OGMEM_THREADS", "5")
    monkeypatch.setattr(uc, "_load_yaml", lambda _path: {"service": {}})
    assert OgMemConfig.load("ignored.yaml").threads == 5


def test_embedding_max_input_chars_loads_yaml_then_env(monkeypatch):
    from providers import unified_config as uc

    monkeypatch.setenv("OGMEM_EMBEDDING_MAX_INPUT_CHARS", "7000")
    monkeypatch.setattr(
        uc,
        "_load_yaml",
        lambda _path: {"embedding": {"max_input_chars": 8_000}},
    )
    assert OgMemConfig.load("ignored.yaml").embedding_max_input_chars == 8_000

    monkeypatch.setattr(uc, "_load_yaml", lambda _path: {"embedding": {}})
    assert OgMemConfig.load("ignored.yaml").embedding_max_input_chars == 7_000


def test_embedding_max_input_chars_defaults_to_legacy_limit():
    assert OgMemConfig().embedding_max_input_chars == 30_000


def test_embedding_batch_size_loads_yaml_then_env(monkeypatch):
    from providers import unified_config as uc

    monkeypatch.setenv("OGMEM_EMBEDDING_BATCH_SIZE", "4")
    monkeypatch.setattr(uc, "_load_yaml", lambda _: {"embedding": {"batch_size": 8}})
    assert OgMemConfig.load("ignored.yaml").embedding_batch_size == 8
    monkeypatch.setattr(uc, "_load_yaml", lambda _: {})
    assert OgMemConfig.load("ignored.yaml").embedding_batch_size == 4


@pytest.mark.parametrize("value", [0, -1, True, 1.5, 2049])
def test_embedding_batch_size_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="embedding_batch_size"):
        OgMemConfig(embedding_batch_size=value)


def test_embedding_max_input_chars_rejects_non_positive_values():
    with pytest.raises(ValueError, match="embedding_max_input_chars must be positive"):
        OgMemConfig(embedding_max_input_chars=0)


@pytest.mark.parametrize("value", [True, False, 1.5])
def test_embedding_max_input_chars_rejects_non_integer_values(value):
    with pytest.raises(ValueError, match="embedding_max_input_chars must be a positive integer"):
        OgMemConfig(embedding_max_input_chars=value)


def test_embedding_max_input_chars_accepts_positive_integer():
    config = OgMemConfig(embedding_max_input_chars=4_096)

    assert config.embedding_max_input_chars == 4_096


def test_code_edit_capability_defaults_off_and_loads_independently(monkeypatch):
    from providers import unified_config as uc

    defaults = OgMemConfig()
    assert defaults.mcp_enable_code_edit_tools is False

    monkeypatch.setattr(
        uc,
        "_load_yaml",
        lambda _path: {
            "mcp": {
                "enable_code_edit_tools": True,
                "enable_write_tools": False,
            },
        },
    )
    yaml_cfg = OgMemConfig.load("ignored.yaml")
    assert yaml_cfg.mcp_enable_code_edit_tools is True
    assert yaml_cfg.mcp_enable_write_tools is False

    monkeypatch.setattr(uc, "_load_yaml", lambda _path: {})
    monkeypatch.setenv("OGMEM_MCP_ENABLE_CODE_EDIT_TOOLS", "true")
    monkeypatch.setenv("OGMEM_MCP_ENABLE_WRITE_TOOLS", "false")
    env_cfg = OgMemConfig.load("ignored.yaml")
    assert env_cfg.mcp_enable_code_edit_tools is True
    assert env_cfg.mcp_enable_write_tools is False


def test_tool_output_filter_config_loads_canonical_env_and_validates_budgets(monkeypatch):
    from providers import unified_config as uc

    default_cfg = OgMemConfig()
    assert default_cfg.tool_output_filter_enabled is False
    assert default_cfg.tool_output_filter_output_max_chars == 50_000
    assert default_cfg.tool_output_recovery_max_chars == 50_000

    monkeypatch.setattr(
        uc,
        "_load_yaml",
        lambda _path: {
            "features": {
                "tool_output_filter": {
                    "enabled": True,
                    "adaptive_enabled": False,
                    "read_min_chars": 111,
                    "read_max_lines": 22,
                    "bash_min_chars": 333,
                    "bash_min_lines": 44,
                    "test_max_lines": 55,
                    "diff_max_lines": 66,
                    "error_max_lines": 77,
                    "generic_max_lines": 88,
                    "output_max_chars": 44_444,
                    "recovery_max_chars": 33_333,
                },
                # Flat forms are legacy compatibility inputs. The canonical
                # nested settings above must win when both are present.
                "tool_output_filter_enabled": False,
                "tool_output_filter_read_min_chars": 999,
                "tool_output_filter_output_max_chars": 55_555,
                "tool_output_recovery_max_chars": 66_666,
                "tool_output": {"recovery_max_chars": 77_777},
            }
        },
    )
    cfg = OgMemConfig.load("ignored.yaml")
    assert cfg.tool_output_filter_enabled is True
    assert cfg.tool_output_filter_adaptive_enabled is False
    assert cfg.tool_output_filter_read_min_chars == 111
    assert cfg.tool_output_filter_read_max_lines == 22
    assert cfg.tool_output_filter_bash_min_chars == 333
    assert cfg.tool_output_filter_bash_min_lines == 44
    assert cfg.tool_output_filter_test_max_lines == 55
    assert cfg.tool_output_filter_diff_max_lines == 66
    assert cfg.tool_output_filter_error_max_lines == 77
    assert cfg.tool_output_filter_generic_max_lines == 88
    assert cfg.tool_output_filter_output_max_chars == 44_444
    assert cfg.tool_output_recovery_max_chars == 33_333

    monkeypatch.setattr(uc, "_load_yaml", lambda _path: {})
    monkeypatch.setenv("OGMEM_TOOL_OUTPUT_FILTER_READ_MIN_CHARS", "456")
    monkeypatch.setenv("OGMEM_TOOL_OUTPUT_FILTER_READ_MAX_LINES", "57")
    monkeypatch.setenv("OGMEM_TOOL_OUTPUT_FILTER_BASH_MIN_CHARS", "5678")
    monkeypatch.setenv("OGMEM_TOOL_OUTPUT_FILTER_BASH_MIN_LINES", "89")
    monkeypatch.setenv("OGMEM_TOOL_OUTPUT_FILTER_RECOVERY_MAX_CHARS", "70000")
    env_cfg = OgMemConfig.load("ignored.yaml")
    assert env_cfg.tool_output_filter_read_min_chars == 456
    assert env_cfg.tool_output_filter_read_max_lines == 57
    assert env_cfg.tool_output_filter_bash_min_chars == 5678
    assert env_cfg.tool_output_filter_bash_min_lines == 89
    assert env_cfg.tool_output_recovery_max_chars == 70000

    monkeypatch.setenv("OGMEM_TOOL_OUTPUT_FILTER_OUTPUT_MAX_CHARS", "46000")
    monkeypatch.setattr(
        uc,
        "_load_yaml",
        lambda _path: {
            "features": {
                "tool_output_filter": {
                    "read_min_chars": 789,
                    "output_max_chars": 45_000,
                    "recovery_max_chars": 40_000,
                }
            }
        },
    )
    yaml_cfg = OgMemConfig.load("ignored.yaml")
    assert yaml_cfg.tool_output_filter_read_min_chars == 789
    assert yaml_cfg.tool_output_filter_read_max_lines == 57
    assert yaml_cfg.tool_output_filter_output_max_chars == 45_000
    assert yaml_cfg.tool_output_recovery_max_chars == 40_000

    with pytest.raises(ValueError, match="read_max_lines must be a positive integer"):
        OgMemConfig(tool_output_filter_read_max_lines=True)
    with pytest.raises(ValueError, match="diff_max_lines must not exceed 10000"):
        OgMemConfig(tool_output_filter_diff_max_lines=10001)


@pytest.mark.parametrize("invalid", [[], "", False, 0])
def test_tool_output_filter_nested_config_requires_a_mapping(monkeypatch, invalid):
    from providers import unified_config as uc

    monkeypatch.setattr(
        uc,
        "_load_yaml",
        lambda _path: {"features": {"tool_output_filter": invalid}},
    )

    with pytest.raises(ValueError, match="features.tool_output_filter must be a mapping"):
        OgMemConfig.load("ignored.yaml")


def test_tool_output_filter_caps_reject_unsafe_or_conflicting_values(monkeypatch):
    from providers import unified_config as uc

    with pytest.raises(ValueError, match="output_max_chars must be at least 256"):
        OgMemConfig(
            tool_output_filter_read_min_chars=1,
            tool_output_filter_bash_min_chars=1,
            tool_output_filter_output_max_chars=255,
        )
    with pytest.raises(ValueError, match="at least the read and bash min chars"):
        OgMemConfig(
            tool_output_filter_read_min_chars=300,
            tool_output_filter_bash_min_chars=400,
            tool_output_filter_output_max_chars=399,
        )
    with pytest.raises(
        ValueError,
        match="tool_output_recovery_max_chars must be a positive integer",
    ):
        OgMemConfig(tool_output_recovery_max_chars=True)

    monkeypatch.setattr(
        uc,
        "_load_yaml",
        lambda _path: {"features": {"tool_output_filter": {"output_max_chars": True}}},
    )
    with pytest.raises(
        ValueError,
        match="OGMEM_TOOL_OUTPUT_FILTER_OUTPUT_MAX_CHARS must be an integer",
    ):
        OgMemConfig.load("ignored.yaml")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("read_min_chars", 1.5),
        ("test_max_lines", 2.5),
        ("output_max_chars", 512.5),
        ("recovery_max_chars", 1024.5),
    ],
)
def test_tool_output_filter_yaml_rejects_non_integer_numbers(monkeypatch, field, value):
    from providers import unified_config as uc

    monkeypatch.setattr(
        uc,
        "_load_yaml",
        lambda _path: {"features": {"tool_output_filter": {field: value}}},
    )

    with pytest.raises(ValueError, match="must be an integer"):
        OgMemConfig.load("ignored.yaml")


def test_code_sync_defaults_disabled_and_loads_nested_yaml(monkeypatch):
    from providers import unified_config as uc

    default_cfg = OgMemConfig()
    assert default_cfg.code_sync_enabled is False
    assert default_cfg.code_resources_max_files == 5000
    assert default_cfg.code_workspace_allowed_roots == []
    assert default_cfg.code_sync_default_trigger_mode == "hybrid"
    assert default_cfg.code_sync_default_wait_mode == "accepted"
    assert default_cfg.code_sync_reconcile_interval_seconds == 300

    monkeypatch.setattr(
        uc,
        "_load_yaml",
        lambda _path: {
            "features": {
                "code_resources_enabled": True,
                "code_resources_max_files": 41,
                "code_workspace_allowed_roots": ["/srv/ogmemory/workspaces"],
                "code_sync": {
                    "enabled": True,
                    "default_trigger_mode": "external_event",
                    "default_wait_mode": "filesystem_complete",
                    "reconcile_interval_seconds": 45,
                    "watch_interval_seconds": 3,
                    "run_timeout_seconds": 12,
                    "failure_backoff_seconds": 8,
                    "max_pending_roots": 7,
                    "debounce_seconds": 1,
                },
            }
        },
    )

    cfg = OgMemConfig.load("ignored.yaml")
    assert cfg.code_sync_enabled is True
    assert cfg.code_resources_max_files == 41
    assert cfg.code_workspace_allowed_roots == ["/srv/ogmemory/workspaces"]
    assert cfg.code_sync_default_trigger_mode == "external_event"
    assert cfg.code_sync_default_wait_mode == "filesystem_complete"
    assert cfg.code_sync_reconcile_interval_seconds == 45
    assert cfg.code_sync_watch_interval_seconds == 3
    assert cfg.code_sync_run_timeout_seconds == 12
    assert cfg.code_sync_failure_backoff_seconds == 8
    assert cfg.code_sync_max_pending_roots == 7
    assert cfg.code_sync_debounce_seconds == 1


def test_code_sync_env_and_validation_fail_closed(monkeypatch):
    from providers import unified_config as uc

    monkeypatch.setattr(uc, "_load_yaml", lambda _path: {})
    monkeypatch.setenv("OGMEM_CODE_RESOURCES_ENABLED", "true")
    monkeypatch.setenv("OGMEM_CODE_RESOURCES_MAX_FILES", "73")
    monkeypatch.setenv("OGMEM_CODE_WORKSPACE_ALLOWED_ROOTS", "/srv/one,/srv/two")
    monkeypatch.setenv("OGMEM_CODE_SYNC_ENABLED", "true")
    monkeypatch.setenv("OGMEM_CODE_SYNC_DEFAULT_TRIGGER_MODE", "manual")
    monkeypatch.setenv("OGMEM_CODE_SYNC_DEFAULT_WAIT_MODE", "index_visible")
    monkeypatch.setenv("OGMEM_CODE_SYNC_RUN_TIMEOUT_SECONDS", "17")

    cfg = OgMemConfig.load("ignored.yaml")
    assert cfg.code_sync_enabled is True
    assert cfg.code_resources_max_files == 73
    assert cfg.code_workspace_allowed_roots == ["/srv/one", "/srv/two"]
    assert cfg.code_sync_default_trigger_mode == "manual"
    assert cfg.code_sync_default_wait_mode == "index_visible"
    assert cfg.code_sync_run_timeout_seconds == 17

    with pytest.raises(ValueError, match="requires code_resources_enabled"):
        OgMemConfig(code_sync_enabled=True)
    with pytest.raises(ValueError, match="default_trigger_mode is invalid"):
        OgMemConfig(code_sync_default_trigger_mode="watcher")
    with pytest.raises(ValueError, match="default_wait_mode is invalid"):
        OgMemConfig(code_sync_default_wait_mode="forever")
    with pytest.raises(ValueError, match="code_resources_max_files must be positive"):
        OgMemConfig(code_resources_max_files=0)
    with pytest.raises(ValueError, match="code_sync_run_timeout_seconds must be positive"):
        OgMemConfig(code_sync_run_timeout_seconds=0)


@pytest.mark.parametrize(
    "value",
    ["/srv/ogmemory/workspaces", ["/srv/ogmemory/workspaces", ""]],
)
def test_code_workspace_allowed_roots_requires_non_empty_string_list(value):
    with pytest.raises(ValueError, match="code_workspace_allowed_roots must be a list"):
        OgMemConfig(code_workspace_allowed_roots=value)


@pytest.mark.parametrize(
    "field",
    [
        "code_resources_max_files",
        "code_sync_reconcile_interval_seconds",
        "code_sync_watch_interval_seconds",
        "code_sync_run_timeout_seconds",
        "code_sync_failure_backoff_seconds",
        "code_sync_max_pending_roots",
        "code_sync_debounce_seconds",
    ],
)
def test_code_sync_integer_settings_reject_booleans(field):
    with pytest.raises(ValueError, match=f"{field} must be a positive integer"):
        OgMemConfig(**{field: True})


def test_code_sync_yaml_integer_settings_reject_booleans(monkeypatch):
    from providers import unified_config as uc

    monkeypatch.setattr(
        uc,
        "_load_yaml",
        lambda _path: {"features": {"code_resources_max_files": True}},
    )

    with pytest.raises(ValueError, match="OGMEM_CODE_RESOURCES_MAX_FILES must be an integer"):
        OgMemConfig.load("ignored.yaml")


def test_code_retrieval_mode_defaults_full_and_loads_nested_candidate_limits(monkeypatch):
    from providers import unified_config as uc

    default_cfg = OgMemConfig()
    assert default_cfg.code_retrieval_mode == "full"
    assert default_cfg.code_retrieval_candidate_discovery_pool_limit == 300
    assert default_cfg.code_retrieval_candidate_file_limit == 80
    assert default_cfg.code_retrieval_candidate_snippet_limit == 500
    assert default_cfg.code_retrieval_candidate_semantic_limit == 16
    assert default_cfg.code_retrieval_candidate_embedding_chars == 4096
    assert default_cfg.code_retrieval_candidate_embedding_batch_texts == 8
    assert default_cfg.code_retrieval_candidate_embedding_batch_tokens == 8000
    assert default_cfg.code_retrieval_candidate_read_bytes == 64 * 1024 * 1024
    assert default_cfg.code_retrieval_candidate_discovery_deadline_ms == 15_000
    assert default_cfg.code_retrieval_candidate_embedding_deadline_ms == 330_000
    assert default_cfg.code_retrieval_candidate_request_deadline_ms == 360_000

    monkeypatch.setattr(
        uc,
        "_load_yaml",
        lambda _path: {
            "features": {
                "code_retrieval": {
                    "mode": "candidate_local",
                    "candidate_discovery_pool_limit": 90,
                    "candidate_file_limit": 12,
                    "candidate_snippet_limit": 24,
                    "candidate_semantic_limit": 7,
                    "candidate_embedding_chars": 1024,
                    "candidate_embedding_batch_texts": 3,
                    "candidate_embedding_batch_tokens": 2000,
                    "candidate_read_bytes": 4096,
                    "candidate_discovery_deadline_ms": 1000,
                    "candidate_embedding_deadline_ms": 2000,
                    "candidate_request_deadline_ms": 4000,
                }
            }
        },
    )

    cfg = OgMemConfig.load("ignored.yaml")

    assert cfg.code_retrieval_mode == "candidate_local"
    assert cfg.code_retrieval_candidate_discovery_pool_limit == 90
    assert cfg.code_retrieval_candidate_file_limit == 12
    assert cfg.code_retrieval_candidate_snippet_limit == 24
    assert cfg.code_retrieval_candidate_semantic_limit == 7
    assert cfg.code_retrieval_candidate_embedding_chars == 1024
    assert cfg.code_retrieval_candidate_embedding_batch_texts == 3
    assert cfg.code_retrieval_candidate_embedding_batch_tokens == 2000
    assert cfg.code_retrieval_candidate_read_bytes == 4096
    assert cfg.code_retrieval_candidate_discovery_deadline_ms == 1000
    assert cfg.code_retrieval_candidate_embedding_deadline_ms == 2000
    assert cfg.code_retrieval_candidate_request_deadline_ms == 4000


def test_code_retrieval_mode_env_and_explicit_config_path(monkeypatch, tmp_path):
    from providers import unified_config as uc

    monkeypatch.setattr(uc, "_load_yaml", lambda _path: {})
    monkeypatch.setenv("OGMEM_CODE_RETRIEVAL_MODE", "candidate_local")
    assert OgMemConfig.load("ignored.yaml").code_retrieval_mode == "candidate_local"

    monkeypatch.delenv("OGMEM_CODE_RETRIEVAL_MODE")
    config_path = tmp_path / "candidate-local.yaml"
    config_path.write_text(
        "features:\n  code_retrieval:\n    mode: candidate_local\n",
        encoding="utf-8",
    )
    monkeypatch.undo()
    assert OgMemConfig.load(str(config_path)).code_retrieval_mode == "candidate_local"


@pytest.mark.parametrize(
    "raw",
    (
        {"code_retrieval_mode": "candidate_local"},
        {"features": {"code_retrieval_mode": "candidate_local"}},
    ),
)
def test_code_retrieval_mode_loads_legacy_flat_yaml_forms(monkeypatch, raw):
    from providers import unified_config as uc

    monkeypatch.setattr(uc, "_load_yaml", lambda _path: raw)

    assert OgMemConfig.load("ignored.yaml").code_retrieval_mode == "candidate_local"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"code_retrieval_mode": "hybrid"}, "code_retrieval_mode is invalid"),
        (
            {
                "code_retrieval_candidate_file_limit": 80,
                "code_retrieval_candidate_discovery_pool_limit": 79,
            },
            "file_limit cannot exceed discovery pool limit",
        ),
        (
            {
                "code_retrieval_candidate_semantic_limit": 16,
                "code_retrieval_candidate_snippet_limit": 15,
            },
            "semantic_limit cannot exceed snippet limit",
        ),
        (
            {
                "code_retrieval_mode": "candidate_local",
                "code_retrieval_candidate_embedding_chars": 4096,
                "embedding_max_input_chars": 4095,
            },
            "candidate_embedding_chars cannot exceed embedding_max_input_chars",
        ),
        (
            {
                "code_retrieval_candidate_discovery_deadline_ms": 15_000,
                "code_retrieval_candidate_embedding_deadline_ms": 330_000,
                "code_retrieval_candidate_request_deadline_ms": 344_999,
            },
            "discovery and embedding deadlines exceed request deadline",
        ),
        (
            {"code_retrieval_candidate_discovery_deadline_ms": 15_001},
            "code_retrieval_candidate_discovery_deadline_ms must not exceed 15000",
        ),
        (
            {"code_retrieval_candidate_discovery_pool_limit": 301},
            "code_retrieval_candidate_discovery_pool_limit must not exceed 300",
        ),
        (
            {"code_retrieval_candidate_file_limit": 81},
            "code_retrieval_candidate_file_limit must not exceed 80",
        ),
        (
            {"code_retrieval_candidate_snippet_limit": 501},
            "code_retrieval_candidate_snippet_limit must not exceed 500",
        ),
        (
            {"code_retrieval_candidate_semantic_limit": 17},
            "code_retrieval_candidate_semantic_limit must not exceed 16",
        ),
        (
            {"code_retrieval_candidate_embedding_chars": 4097},
            "code_retrieval_candidate_embedding_chars must not exceed 4096",
        ),
        (
            {"code_retrieval_candidate_embedding_batch_texts": 9},
            "code_retrieval_candidate_embedding_batch_texts must not exceed 8",
        ),
        (
            {"code_retrieval_candidate_embedding_batch_tokens": 8001},
            "code_retrieval_candidate_embedding_batch_tokens must not exceed 8000",
        ),
        (
            {"code_retrieval_candidate_read_bytes": 64 * 1024 * 1024 + 1},
            "code_retrieval_candidate_read_bytes must not exceed 67108864",
        ),
        (
            {"code_retrieval_candidate_embedding_deadline_ms": 330_001},
            "code_retrieval_candidate_embedding_deadline_ms must not exceed 330000",
        ),
        (
            {"code_retrieval_candidate_request_deadline_ms": 360_001},
            "code_retrieval_candidate_request_deadline_ms must not exceed 360000",
        ),
    ],
)
def test_code_retrieval_candidate_config_rejects_invalid_relationships(kwargs, message):
    with pytest.raises(ValueError, match=message):
        OgMemConfig(**kwargs)


def test_full_mode_preserves_existing_smaller_provider_input_cap():
    full = OgMemConfig(embedding_max_input_chars=1024)

    assert full.code_retrieval_mode == "full"
    assert full.embedding_max_input_chars == 1024
    with pytest.raises(ValueError, match="candidate_embedding_chars cannot exceed embedding_max_input_chars"):
        OgMemConfig(
            code_retrieval_mode="candidate_local",
            embedding_max_input_chars=1024,
        )


def test_code_retrieval_candidate_yaml_rejects_non_integer_values(monkeypatch):
    from providers import unified_config as uc

    monkeypatch.setattr(
        uc,
        "_load_yaml",
        lambda _path: {
            "features": {"code_retrieval": {"candidate_file_limit": 1.5}}
        },
    )

    with pytest.raises(ValueError, match="OGMEM_CODE_RETRIEVAL_CANDIDATE_FILE_LIMIT must be an integer"):
        OgMemConfig.load("ignored.yaml")


def test_trace_turn_limit_reads_storage_yaml_and_env(monkeypatch):
    from providers import unified_config as uc

    monkeypatch.setenv("OGMEM_TRACE_MAX_TURN_EVENTS", "64")
    monkeypatch.setattr(
        uc,
        "_load_yaml",
        lambda _path: {"storage": {"trace_max_turn_events": 256}},
    )
    assert OgMemConfig.load("ignored.yaml").trace_max_turn_events == 256

    monkeypatch.setattr(uc, "_load_yaml", lambda _path: {"storage": {}})
    assert OgMemConfig.load("ignored.yaml").trace_max_turn_events == 64


def test_sql_dialect_reads_storage_yaml_and_env(monkeypatch):
    from providers import unified_config as uc

    monkeypatch.setenv("SQL_DIALECT", "opengauss")
    monkeypatch.setattr(
        uc,
        "_load_yaml",
        lambda _path: {"storage": {"dialect": "postgres"}},
    )
    assert OgMemConfig.load("ignored.yaml").sql_dialect == "postgresql"

    monkeypatch.setattr(uc, "_load_yaml", lambda _path: {"storage": {}})
    assert OgMemConfig.load("ignored.yaml").sql_dialect == "opengauss"


def test_sql_dialect_rejects_unknown_value():
    with pytest.raises(ValueError, match="postgresql or opengauss"):
        OgMemConfig(sql_dialect="mysql")


@pytest.mark.parametrize("invalid", [0, 1, 1025])
def test_trace_turn_limit_rejects_invalid_bounds(invalid):
    with pytest.raises(ValueError, match="between 2 and 1024"):
        OgMemConfig(trace_max_turn_events=invalid)


def test_skill_sedimentation_defaults_off_and_loads_nested_config(monkeypatch):
    from providers import unified_config as uc

    monkeypatch.setattr(
        uc,
        "_load_yaml",
        lambda _path: {
            "storage": {"backend": "sql", "connection_string": "postgresql://unused"},
            "skills": {"sedimentation": {
                "enabled": True, "min_evidence_turns": 7,
                "min_distinct_sessions": 4, "semantic_update_threshold": .94,
                "semantic_hold_threshold": .81, "candidate_limit": 25,
                "catalog_scan_limit": 2500,
            }},
        },
    )
    cfg = OgMemConfig.load("ignored.yaml")
    assert cfg.skill_sedimentation_enabled is True
    assert cfg.skill_min_evidence_turns == 7
    assert cfg.skill_min_distinct_sessions == 4
    assert cfg.skill_semantic_update_threshold == .94
    assert cfg.skill_candidate_limit == 25
    assert cfg.skill_catalog_scan_limit == 2500
    default_cfg = OgMemConfig()
    assert default_cfg.skill_sedimentation_enabled is False
    assert default_cfg.skill_min_evidence_turns == 1
    assert default_cfg.skill_min_distinct_sessions == 1


def test_skill_catalog_scan_limit_cannot_be_below_page_size():
    with pytest.raises(ValueError, match="cannot be below"):
        OgMemConfig(skill_candidate_limit=200, skill_catalog_scan_limit=100)


def test_enabled_skill_sedimentation_accepts_agfs_backend():
    cfg = OgMemConfig(
        skill_sedimentation_enabled=True, storage_backend="agfs", workers=1,
    )

    assert cfg.storage_backend == "agfs"


def test_enabled_skill_sedimentation_rejects_unknown_backend():
    with pytest.raises(ValueError, match="storage.backend=agfs or sql"):
        OgMemConfig(skill_sedimentation_enabled=True, storage_backend="agsf")


def test_enabled_agfs_skill_sedimentation_requires_one_service_worker():
    with pytest.raises(ValueError, match="service.workers=1"):
        OgMemConfig(skill_sedimentation_enabled=True, storage_backend="agfs", workers=2)


def test_enabled_skill_sedimentation_requires_sql_connection_string():
    with pytest.raises(ValueError, match="storage.sql.connection_string"):
        OgMemConfig(skill_sedimentation_enabled=True, storage_backend="sql")


def test_hybrid_retrieval_requires_sql_canonical_storage():
    with pytest.raises(ValueError, match="storage.backend=sql"):
        OgMemConfig(retrieval_hybrid_enabled=True, storage_backend="agfs")


def test_hybrid_retrieval_requires_sql_canonical_connection():
    with pytest.raises(ValueError, match="storage.connection_string"):
        OgMemConfig(retrieval_hybrid_enabled=True, storage_backend="sql")


def test_dump_summary_redacts_nested_lexical_connection_string():
    cfg = OgMemConfig(
        retrieval_lexical_options={
            "connection_string": "postgresql://lex:secret@db/lexical",
            "table_name": "lexical_index",
        }
    )

    summary = cfg.dump_summary()

    assert summary["retrieval_lexical_options"]["connection_string"] == "<configured>"
    assert "secret" not in repr(summary)


def test_dump_summary_preserves_non_secret_token_limits():
    cfg = OgMemConfig(llm_max_tokens=1234, compact_prepare_token_ttl=456)

    summary = cfg.dump_summary()

    assert summary["llm_max_tokens"] == 1234
    assert summary["compact_prepare_token_ttl"] == 456


def test_enabled_skill_sedimentation_rejects_lease_shorter_than_external_call_budget():
    with pytest.raises(ValueError, match="at least 180"):
        OgMemConfig(
            skill_sedimentation_enabled=True,
            storage_backend="sql",
            sql_connection_string="postgresql://unused",
            skill_lease_seconds=179,
        )


def test_enabled_skill_sedimentation_requires_room_for_pending_support():
    with pytest.raises(ValueError, match="max_evidence_turns must be at least 2"):
        OgMemConfig(
            skill_sedimentation_enabled=True,
            storage_backend="sql",
            sql_connection_string="postgresql://unused",
            skill_max_evidence_turns=1,
        )


def test_skill_max_trace_chars_matches_compactor_minimum():
    with pytest.raises(ValueError, match="max_trace_chars must be at least 1024"):
        OgMemConfig(skill_max_trace_chars=1023)


def test_compact_prepare_token_ttl_defaults_to_300():
    cfg = OgMemConfig.load("/tmp/non-existent-ogmem.yaml")

    assert cfg.compact_prepare_token_ttl == 300


def test_compact_prepare_token_ttl_reads_yaml_and_env(monkeypatch):
    from providers import unified_config

    monkeypatch.setenv("OGMEM_COMPACT_PREPARE_TOKEN_TTL", "123")
    monkeypatch.setattr(
        unified_config,
        "_load_yaml",
        lambda _path: {"memory": {"compact_prepare_token_ttl": 456}},
    )

    cfg = OgMemConfig.load("unused.yaml")
    assert cfg.compact_prepare_token_ttl == 456

    monkeypatch.setattr(unified_config, "_load_yaml", lambda _path: {"memory": {}})
    cfg = OgMemConfig.load("unused.yaml")
    assert cfg.compact_prepare_token_ttl == 123


def test_rolling_compress_enabled_defaults_to_true():
    cfg = OgMemConfig.load("/tmp/non-existent-ogmem.yaml")

    assert cfg.rolling_compress_enabled is True


def test_rolling_compress_enabled_can_be_disabled_by_env(monkeypatch):
    monkeypatch.setenv("OGMEM_ROLLING_COMPRESS_ENABLED", "false")

    cfg = OgMemConfig.load("/tmp/non-existent-ogmem.yaml")

    assert cfg.rolling_compress_enabled is False


def test_rolling_compress_enabled_yaml_overrides_env(monkeypatch):
    from providers import unified_config

    monkeypatch.setenv("OGMEM_ROLLING_COMPRESS_ENABLED", "false")
    monkeypatch.setattr(
        unified_config,
        "_load_yaml",
        lambda _path: {"memory": {"rolling_compress_enabled": True}},
    )

    cfg = OgMemConfig.load("unused.yaml")

    assert cfg.rolling_compress_enabled is True


def test_archive_merge_prd_defaults():
    cfg = OgMemConfig()

    assert cfg.archive_max_count == 10
    assert cfg.archive_merge_threshold == 10
    assert not hasattr(cfg, "archive_auto_merge_enabled")
    assert not hasattr(cfg, "archive_merge_max_messages")


def test_archive_merge_reads_prd_yaml_and_env(monkeypatch):
    monkeypatch.setenv("OGMEM_ARCHIVE_MAX_COUNT", "7")
    monkeypatch.setenv("OGMEM_ARCHIVE_MERGE_THRESHOLD", "6")

    cfg = OgMemConfig.load(config_path="missing.yaml")

    assert cfg.archive_max_count == 7
    assert cfg.archive_merge_threshold == 6


def test_rolling_fallback_defaults_disabled_and_env_override(monkeypatch):
    assert OgMemConfig().rolling_compress_fallback_enabled is False

    monkeypatch.setenv("OGMEM_ROLLING_COMPRESS_FALLBACK_ENABLED", "true")
    cfg = OgMemConfig.load(config_path="missing.yaml")

    assert cfg.rolling_compress_fallback_enabled is True


def test_compact_gap_completion_defaults():
    cfg = OgMemConfig()

    assert cfg.prefetch_enabled is False
    assert cfg.prefetch_top_k == 5
    assert cfg.session_state_bridge_enabled is True
    assert cfg.session_state_sync_interval_turns == 1
    assert cfg.topic_detection_enabled is False
    assert cfg.compression_quality_enabled is False
    assert cfg.compression_quality_persist_metadata is False
    assert cfg.retrieval_candidate_pool_enabled is False
    assert cfg.retrieval_result_ranker_strategy == "semantic"
    assert cfg.retrieval_global_result_rerank_enabled is False
    assert cfg.retrieval_owner_scope_filter_enabled is False


def test_compact_gap_completion_reads_yaml_and_env(monkeypatch):
    from providers import unified_config as uc

    monkeypatch.setattr(
        uc,
        "_load_yaml",
        lambda _path: {
            "memory": {
                "prefetch_enabled": True,
                "prefetch_top_k": 9,
                "session_state_bridge_enabled": False,
                "session_state_sync_interval_turns": 4,
                "topic_detection_enabled": True,
                "compression_quality_enabled": True,
                "compression_quality_persist_metadata": True,
            }
        },
    )
    cfg = OgMemConfig.load("ignored.yaml")
    assert cfg.prefetch_enabled is True
    assert cfg.prefetch_top_k == 9
    assert cfg.session_state_bridge_enabled is False
    assert cfg.session_state_sync_interval_turns == 4
    assert cfg.topic_detection_enabled is True
    assert cfg.compression_quality_enabled is True
    assert cfg.compression_quality_persist_metadata is True

    monkeypatch.setenv("OGMEM_PREFETCH_ENABLED", "false")
    monkeypatch.setenv("OGMEM_PREFETCH_TOP_K", "3")
    monkeypatch.setenv("OGMEM_SESSION_STATE_BRIDGE_ENABLED", "true")
    monkeypatch.setenv("OGMEM_SESSION_STATE_SYNC_INTERVAL_TURNS", "2")
    monkeypatch.setenv("OGMEM_TOPIC_DETECTION_ENABLED", "false")
    monkeypatch.setenv("OGMEM_COMPRESSION_QUALITY_ENABLED", "false")
    monkeypatch.setenv("OGMEM_COMPRESSION_QUALITY_PERSIST_METADATA", "false")
    monkeypatch.setattr(uc, "_load_yaml", lambda _path: {"memory": {}})
    cfg = OgMemConfig.load("ignored.yaml")
    assert cfg.prefetch_enabled is False
    assert cfg.prefetch_top_k == 3
    assert cfg.session_state_bridge_enabled is True
    assert cfg.session_state_sync_interval_turns == 2
    assert cfg.topic_detection_enabled is False
    assert cfg.compression_quality_enabled is False
    assert cfg.compression_quality_persist_metadata is False


def test_retrieval_config_reads_yaml_and_env(monkeypatch):
    from providers import unified_config as uc

    monkeypatch.setattr(
        uc,
        "_load_yaml",
        lambda _path: {
            "storage": {
                "backend": "sql",
                "connection_string": "postgresql://unused",
            },
            "retrieval": {
                "max_top_k": 60,
                "default_top_k": 15,
                "global_search_topk": 45,
                "candidate_pool_enabled": True,
                "candidate_pool_multiplier": 4,
                "min_candidate_pool_size": 20,
                "result_ranker_strategy": "metadata_aware",
                "global_result_rerank_enabled": True,
                "owner_scope_filter_enabled": True,
                "hybrid_enabled": True,
                "vector_top_k": 120,
                "lexical_top_k": 40,
                "rrf_k": 55,
                "semantic_weight": 1.5,
                "lexical_weight": 0.5,
                "lexical": {
                    "provider": "opengauss_native",
                    "options": {"table_name": "lexical_test"},
                },
            }
        },
    )
    cfg = OgMemConfig.load("ignored.yaml")
    assert cfg.retrieval_max_top_k == 60
    assert cfg.retrieval_default_top_k == 15
    assert cfg.retrieval_global_search_topk == 45
    assert cfg.retrieval_candidate_pool_enabled is True
    assert cfg.retrieval_candidate_pool_multiplier == 4
    assert cfg.retrieval_min_candidate_pool_size == 20
    assert cfg.retrieval_result_ranker_strategy == "metadata_aware"
    assert cfg.retrieval_global_result_rerank_enabled is True
    assert cfg.retrieval_owner_scope_filter_enabled is True
    assert cfg.retrieval_hybrid_enabled is True
    assert cfg.retrieval_vector_top_k == 120
    assert cfg.retrieval_lexical_top_k == 40
    assert cfg.retrieval_rrf_k == 55
    assert cfg.retrieval_semantic_weight == 1.5
    assert cfg.retrieval_lexical_weight == 0.5
    assert cfg.retrieval_lexical_options == {"table_name": "lexical_test"}

    monkeypatch.setenv("OGMEM_RETRIEVAL_CANDIDATE_POOL_ENABLED", "false")
    monkeypatch.setenv("OGMEM_RETRIEVAL_RESULT_RANKER_STRATEGY", "semantic")
    monkeypatch.setenv("OGMEM_RETRIEVAL_GLOBAL_RESULT_RERANK_ENABLED", "false")
    monkeypatch.setenv("OGMEM_RETRIEVAL_OWNER_SCOPE_FILTER_ENABLED", "false")
    monkeypatch.setattr(uc, "_load_yaml", lambda _path: {"retrieval": {}})
    cfg = OgMemConfig.load("ignored.yaml")
    assert cfg.retrieval_candidate_pool_enabled is False
    assert cfg.retrieval_result_ranker_strategy == "semantic"
    assert cfg.retrieval_global_result_rerank_enabled is False
    assert cfg.retrieval_owner_scope_filter_enabled is False


def test_hybrid_retrieval_accepts_custom_lexical_provider():
    cfg = OgMemConfig(
        storage_backend="sql",
        sql_connection_string="postgresql://unused",
        retrieval_hybrid_enabled=True,
        retrieval_lexical_provider="custom",
    )

    assert cfg.retrieval_lexical_provider == "custom"


@pytest.mark.parametrize(
    "override",
    [
        {"retrieval_vector_top_k": 10_001},
        {"retrieval_lexical_top_k": 201},
        {"retrieval_lexical_top_k": 0},
        {"retrieval_rrf_k": 0},
        {"retrieval_semantic_weight": float("nan")},
        {"retrieval_lexical_weight": float("inf")},
        {"retrieval_semantic_weight": float("-inf")},
        {"retrieval_lexical_weight": -0.1},
        {"retrieval_semantic_weight": 0, "retrieval_lexical_weight": 0},
        {"retrieval_backfill_page_size": 0},
        {"retrieval_backfill_page_size": 10_001},
    ],
)
def test_hybrid_retrieval_rejects_unbounded_or_non_finite_tuning(override):
    with pytest.raises(ValueError):
        OgMemConfig(**override)


def test_disabled_tool_result_settings_do_not_reject_startup():
    cfg = OgMemConfig(
        tool_result_externalization_enabled=False,
        tool_result_single_chars=0,
        tool_result_stub_summary_turn_chars=0,
        tool_result_search_chunk_bytes=1,
        tool_result_search_overlap_bytes=2,
    )

    assert cfg.tool_result_externalization_enabled is False


def test_enabled_tool_result_settings_are_still_validated():
    with pytest.raises(ValueError, match="tool_result_single_chars must be positive"):
        OgMemConfig(
            tool_result_externalization_enabled=True,
            tool_result_single_chars=0,
        )


def test_tool_result_summary_reuses_synopsis_budget_and_loads_turn_budget(monkeypatch):
    from providers import unified_config as uc

    monkeypatch.setattr(
        uc,
        "_load_yaml",
        lambda _path: {"tool_result_externalization": {
            "enabled": True,
            "synopsis_max_chars": 1234,
            "stub_summary_turn_chars": 5678,
        }},
    )
    cfg = OgMemConfig.load("ignored.yaml")
    assert cfg.tool_result_synopsis_max_chars == 1234
    assert cfg.tool_result_stub_summary_turn_chars == 5678

    monkeypatch.setattr(uc, "_load_yaml", lambda _path: {})
    monkeypatch.setenv("OGMEM_TOOL_RESULT_SYNOPSIS_MAX_CHARS", "2222")
    monkeypatch.setenv("OGMEM_TOOL_RESULT_STUB_SUMMARY_TURN_CHARS", "7777")
    cfg = OgMemConfig.load("ignored.yaml")
    assert cfg.tool_result_synopsis_max_chars == 2222
    assert cfg.tool_result_stub_summary_turn_chars == 7777


def test_tool_result_first_view_defaults_enabled_and_loads_yaml_or_env(monkeypatch):
    from providers import unified_config as uc

    assert OgMemConfig().tool_result_first_view_enabled is True

    monkeypatch.setenv("OGMEM_TOOL_RESULT_FIRST_VIEW_ENABLED", "false")
    monkeypatch.setattr(uc, "_load_yaml", lambda _path: {})
    assert OgMemConfig.load("ignored.yaml").tool_result_first_view_enabled is False

    monkeypatch.delenv("OGMEM_TOOL_RESULT_FIRST_VIEW_ENABLED")
    monkeypatch.setattr(
        uc,
        "_load_yaml",
        lambda _path: {
            "tool_result_externalization": {"first_view_enabled": True},
        },
    )
    assert OgMemConfig.load("ignored.yaml").tool_result_first_view_enabled is True


def test_tool_result_semantic_turn_mode_loads_from_env(monkeypatch):
    monkeypatch.setenv("OGMEM_TOOL_RESULT_SEMANTIC_TURN_MODE", "user_message")

    cfg = OgMemConfig.load("/tmp/non-existent-ogmem.yaml")

    assert cfg.tool_result_semantic_turn_mode == "user_message"


def test_code_active_retrieval_config_reads_nested_values_and_rejects_invalid_limits(monkeypatch):
    from providers import unified_config as uc

    monkeypatch.setattr(
        uc,
        "_load_yaml",
        lambda _path: {
            "features": {
                "code_retrieval": {
                    "fusion_mode": "weighted_sum",
                    "semantic_weight": 0.5,
                    "bm25_weight": 0.2,
                    "symbolic_weight": 0.1,
                    "graph_weight": 0.2,
                    "candidate_limit": 12,
                    "result_limit": 4,
                    "snippet_chars": 400,
                    "batch_chars": 1200,
                }
            }
        },
    )
    cfg = OgMemConfig.load("ignored.yaml")
    assert cfg.code_retrieval_fusion_mode == "weighted_sum"
    assert cfg.code_retrieval_candidate_limit == 12
    assert cfg.code_retrieval_result_limit == 4
    assert cfg.code_retrieval_snippet_chars == 400
    assert cfg.code_retrieval_batch_chars == 1200

    with pytest.raises(ValueError, match="cannot exceed candidate limit"):
        OgMemConfig(code_retrieval_candidate_limit=2, code_retrieval_result_limit=3)
    with pytest.raises(ValueError, match="invalid"):
        OgMemConfig(code_retrieval_fusion_mode="unsupported")


def test_code_active_retrieval_config_reads_environment_and_fails_closed_weights(monkeypatch):
    from providers import unified_config as uc

    monkeypatch.setattr(uc, "_load_yaml", lambda _path: {})
    monkeypatch.setenv("OGMEM_CODE_RETRIEVAL_ENABLED", "true")
    monkeypatch.setenv("OGMEM_CODE_RETRIEVAL_FUSION_MODE", "sum")
    monkeypatch.setenv("OGMEM_CODE_RETRIEVAL_SEMANTIC_WEIGHT", "0.6")
    monkeypatch.setenv("OGMEM_CODE_RETRIEVAL_CANDIDATE_LIMIT", "16")
    monkeypatch.setenv("OGMEM_CODE_RETRIEVAL_RESULT_LIMIT", "5")
    monkeypatch.setenv("OGMEM_CODE_RETRIEVAL_SNIPPET_CHARS", "300")
    monkeypatch.setenv("OGMEM_CODE_RETRIEVAL_BATCH_CHARS", "900")

    cfg = OgMemConfig.load("ignored.yaml")

    assert cfg.code_retrieval_enabled is True
    assert cfg.code_retrieval_fusion_mode == "sum"
    assert cfg.code_retrieval_semantic_weight == pytest.approx(0.6)
    assert cfg.code_retrieval_candidate_limit == 16
    assert cfg.code_retrieval_result_limit == 5
    assert cfg.code_retrieval_snippet_chars == 300
    assert cfg.code_retrieval_batch_chars == 900

    with pytest.raises(ValueError, match="finite"):
        OgMemConfig(code_retrieval_semantic_weight=float("nan"))
    with pytest.raises(ValueError, match="non-negative"):
        OgMemConfig(code_retrieval_bm25_weight=-0.1)
    with pytest.raises(ValueError, match="non-zero"):
        OgMemConfig(
            code_retrieval_enabled=True,
            code_retrieval_semantic_enabled=False,
            code_retrieval_bm25_enabled=False,
            code_retrieval_symbolic_enabled=False,
            code_retrieval_graph_enabled=False,
        )


def test_code_active_retrieval_config_rejects_boolean_numeric_yaml_values(monkeypatch):
    from providers import unified_config as uc

    monkeypatch.setattr(
        uc,
        "_load_yaml",
        lambda _path: {"features": {"code_retrieval": {"semantic_weight": True}}},
    )

    with pytest.raises(ValueError, match="OGMEM_CODE_RETRIEVAL_SEMANTIC_WEIGHT must be a number"):
        OgMemConfig.load("ignored.yaml")


def _clear_db_env(monkeypatch):
    """Clear both the current and legacy DB env names so tests are hermetic."""
    for name in (
        "SQL_CONNECTION_STRING",
        "STORAGE_DB_CONNECTION_STRING",
        "STORAGE_DB_POOL_SIZE",
        "OPENGAUSS_CONNECTION_STRING",
        "OPENGAUSS_DIMENSION",
        "OPENGAUSS_TABLE_NAME",
        "OPENGAUSS_POOL_SIZE",
        "VECTOR_DB_CONNECTION_STRING",
        "VECTOR_DB_DIMENSION",
        "VECTOR_DB_TABLE_NAME",
        "VECTOR_DB_POOL_SIZE",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("vector_source", ["yaml", "VECTOR_DB_CONNECTION_STRING", "OPENGAUSS_CONNECTION_STRING"])
def test_sql_storage_dsn_falls_back_to_vector_dsn(monkeypatch, vector_source):
    from providers import unified_config

    _clear_db_env(monkeypatch)
    dsn = "host=vector dbname=og"
    data = {"vector_db": {"connection_string": dsn}} if vector_source == "yaml" else {}
    if vector_source != "yaml":
        monkeypatch.setenv(vector_source, dsn)
    monkeypatch.setattr(unified_config, "_load_yaml", lambda _path: data)
    cfg = OgMemConfig.load("unused.yaml")
    assert cfg.opengauss_connection_string == dsn
    assert cfg.sql_connection_string == dsn


def test_sql_storage_dsn_env_takes_precedence_over_vector_fallback(monkeypatch):
    """STORAGE_DB_CONNECTION_STRING env wins over the vector DSN fallback."""
    from providers import unified_config

    _clear_db_env(monkeypatch)
    monkeypatch.setenv("STORAGE_DB_CONNECTION_STRING", "host=env dbname=og")
    monkeypatch.setattr(
        unified_config,
        "_load_yaml",
        lambda _path: {"vector_db": {"connection_string": "host=vector dbname=og"}},
    )

    cfg = OgMemConfig.load("unused.yaml")

    assert cfg.sql_connection_string == "host=env dbname=og"


def test_explicit_sql_storage_yaml_dsn_is_not_replaced_by_vector_fallback(monkeypatch):
    from providers import unified_config

    _clear_db_env(monkeypatch)
    monkeypatch.setattr(
        unified_config,
        "_load_yaml",
        lambda _path: {
            "storage": {"backend": "sql", "connection_string": "host=storage dbname=og"},
            "vector_db": {"connection_string": "host=vector dbname=og"},
        },
    )

    assert OgMemConfig.load("unused.yaml").sql_connection_string == "host=storage dbname=og"


def test_sql_storage_dsn_legacy_env_alias_still_works(monkeypatch):
    """SQL_CONNECTION_STRING (legacy alias) still resolves when the new
    STORAGE_DB_CONNECTION_STRING is unset."""
    from providers import unified_config

    _clear_db_env(monkeypatch)
    monkeypatch.setenv("SQL_CONNECTION_STRING", "host=legacy dbname=og")
    monkeypatch.setattr(
        unified_config,
        "_load_yaml",
        lambda _path: {"vector_db": {"connection_string": "host=vector dbname=og"}},
    )

    cfg = OgMemConfig.load("unused.yaml")

    assert cfg.sql_connection_string == "host=legacy dbname=og"


def test_vector_db_env_vars_renamed_with_legacy_alias(monkeypatch):
    """VECTOR_DB_* is the canonical env surface; OPENGAUSS_* remains a
    legacy alias, and the new name wins when both are set."""
    from providers import unified_config

    _clear_db_env(monkeypatch)
    monkeypatch.setattr(
        unified_config,
        "_load_yaml",
        lambda _path: {},
    )

    # Legacy alias alone still works.
    monkeypatch.setenv("OPENGAUSS_CONNECTION_STRING", "host=legacy dbname=vec")
    monkeypatch.setenv("OPENGAUSS_DIMENSION", "768")
    cfg = OgMemConfig.load("unused.yaml")
    assert cfg.opengauss_connection_string == "host=legacy dbname=vec"
    assert cfg.opengauss_dimension == 768

    # The new name wins over the legacy alias.
    monkeypatch.setenv("VECTOR_DB_CONNECTION_STRING", "host=new dbname=vec")
    monkeypatch.setenv("VECTOR_DB_TABLE_NAME", "m3_vectors")
    cfg = OgMemConfig.load("unused.yaml")
    assert cfg.opengauss_connection_string == "host=new dbname=vec"
    assert cfg.opengauss_table_name == "m3_vectors"
    # Untouched legacy value still applies for fields without a new-name
    # counterpart set.
    assert cfg.opengauss_dimension == 768
