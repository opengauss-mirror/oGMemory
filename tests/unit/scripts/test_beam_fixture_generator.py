"""Native Dev fixture files, no Docker launch or external model calls."""
import json
import shutil
import stat
import subprocess

import pytest
import yaml

from providers.unified_config import OgMemConfig
from scripts import create_beam_fixture as fixture
from scripts.beam_eval_safety import json_digest, validate_agent_config
from scripts.run_beam_eval import load_setup


def arguments(tmp_path):
    return ["--directory", str(tmp_path / "fixture"), "--parquet", str(tmp_path / "data.parquet"),
            "--run-id", "beam-dev-test-1234", "--embedding-dimension", "1536"]


@pytest.fixture
def credentials(monkeypatch):
    env = dict(BEAM_LLM_API_KEY="private-extraction-key", BEAM_LLM_BASE_URL="https://example.test/v1",
        BEAM_LLM_MODEL="extraction-model", BEAM_EMBEDDING_API_KEY="private-embedding-key",
        BEAM_EMBEDDING_BASE_URL="https://embed.test/v1", BEAM_EMBEDDING_MODEL="embedding-model",
        BEAM_QA_API_KEY="private-qa-key", BEAM_QA_BASE_URL="https://qa.test/v1",
        BEAM_QA_MODEL="answer-model", BEAM_AGENT_TOKEN="private-gateway-token",
        BEAM_MEMORY_API_KEY="private-memory-key", BEAM_POSTGRES_PASSWORD="private-db-password")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return env


def test_default_plan_needs_no_credentials_and_writes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(fixture, "build_files", lambda *_: pytest.fail("Credentials were read"))
    fixture.main(arguments(tmp_path))
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "planned" and report["docker_calls"] == report["provider_calls"] == 0
    assert report["extraction_max_tokens"] == 32768
    assert OgMemConfig().llm_max_tokens == 4096
    assert not (tmp_path / "fixture").exists()


def test_beam_output_headroom_preserves_other_dev_defaults_and_isolation(tmp_path, credentials, capsys):
    fixture.main(arguments(tmp_path) + ["--execute"])
    root = tmp_path / "fixture"
    report_text = capsys.readouterr().out
    assert not any(secret in report_text for key, secret in credentials.items() if "KEY" in key or "PASSWORD" in key or "TOKEN" in key)
    c = OgMemConfig.load(str(root / "memory.private.yaml"))
    defaults = OgMemConfig()
    assert c.llm_temperature == defaults.llm_temperature == 0.7
    assert c.llm_max_tokens == 32768
    assert defaults.llm_max_tokens == OgMemConfig().llm_max_tokens == 4096
    assert c.after_turn_threshold == defaults.after_turn_threshold == 400
    assert c.source_occurrence_reconciliation_enabled == defaults.source_occurrence_reconciliation_enabled
    assert c.retrieval_result_ranker_strategy == defaults.retrieval_result_ranker_strategy
    assert c.decision is None and c.graph_backend == "none"
    assert not c.enable_graph_expansion and not c.graph_supplement_enabled
    assert not c.scene_enabled and not c.scene_generation_enabled and not c.scene_retrieval_enabled
    assert not c.enable_cache and not c.mcp_enabled
    assert c.http_ip_allowlist == ["127.0.0.1", "::1"] and c.role_control_enabled
    assert c.sql_connection_string == c.opengauss_connection_string
    assert c.opengauss_dimension == 1536 and c.openai_llm_model == "extraction-model"
    setup = load_setup(root / "setup.json")
    assert setup["variant"] == "baseline" and not (root / "results").exists()
    agent = json.loads((root / "agent-state/openclaw.json").read_text())
    validate_agent_config(agent, setup["account"], setup["user"], setup["memory_url"])
    assert agent["agents"]["defaults"]["model"]["primary"] == "beam-qa/answer-model"
    assert "temperature" not in agent["agents"]["defaults"]
    assert agent["models"]["providers"]["beam-qa"]["models"][0]["maxTokens"] == 8192
    assert list((root / "agent-state/workspace").iterdir()) == []
    compose = yaml.safe_load((root / "compose.private.yaml").read_text())
    services = compose["services"]
    assert services["postgres"]["ports"] == ["127.0.0.1:25432:5432"]
    assert "network_mode" not in services["postgres"]
    assert services["memory"]["network_mode"] == services["agent"]["network_mode"] == "host"
    assert services["agent"]["volumes"] == [str(root / "agent-state") + ":/home/node/.openclaw"]
    assert services["memory"]["build"]["context"] == str(fixture.ROOT)
    assert services["memory"]["entrypoint"] == ["/bin/sh", "/opt/ogmem/scripts/beam_sql_entrypoint.sh"]
    for file in [root / "memory.private.yaml", root / "compose.private.yaml", root / "agent-state/openclaw.json"]:
        assert stat.S_IMODE(file.stat().st_mode) == 0o600


def test_missing_credentials_fail_before_creating_anything(tmp_path, monkeypatch):
    for key in list(__import__("os").environ):
        if key.startswith("BEAM_"):
            monkeypatch.delenv(key)
    with pytest.raises(ValueError, match="Set credentials"):
        fixture.main(arguments(tmp_path) + ["--execute"])
    assert not (tmp_path / "fixture").exists()


def test_refuses_existing_fixture_without_overwrite(tmp_path, credentials):
    root = tmp_path / "fixture"
    root.mkdir()
    (root / "existing").write_text("retained")
    with pytest.raises(ValueError, match="overwrite"):
        fixture.main(arguments(tmp_path) + ["--execute"])
    assert (root / "existing").read_text() == "retained"


@pytest.mark.parametrize("options", [
    ["--run-id", "unsafe;sql"], ["--directory", str(fixture.ROOT / "results")],
    ["--directory", str(fixture.ROOT.parent)], ["--sql-port", "20800"],
    ["--agent-port", "65536"], ["--temperature", "nan"], ["--max-tokens", "0"],
    ["--embedding-dimension", "0"], ["--conversation-index", "-1"],
])
def test_invalid_fixture_inputs_fail_before_writes(tmp_path, options):
    with pytest.raises(ValueError):
        fixture.main(arguments(tmp_path) + options)
    assert not (tmp_path / "fixture").exists()


def test_explicit_sampling_changes_are_recorded_not_silently_selected(tmp_path, credentials, capsys):
    fixture.main(arguments(tmp_path) + ["--temperature", "0", "--max-tokens", "32768", "--thinking-disabled", "--execute"])
    c = OgMemConfig.load(str(tmp_path / "fixture/memory.private.yaml"))
    assert c.llm_temperature == 0 and c.llm_max_tokens == 32768
    assert c.llm_extra_body == {"thinking": {"type": "disabled"}}
    report = json.loads(capsys.readouterr().out)
    assert report["extraction_max_tokens"] == 32768


def test_explicit_native_output_cap_is_supported_without_mutating_defaults(tmp_path, credentials, capsys):
    fixture.main(arguments(tmp_path) + ["--max-tokens", "4096", "--execute"])
    c = OgMemConfig.load(str(tmp_path / "fixture/memory.private.yaml"))
    report = json.loads(capsys.readouterr().out)
    assert c.llm_max_tokens == report["extraction_max_tokens"] == 4096
    assert OgMemConfig().llm_max_tokens == 4096
    assert c.llm_temperature == 0.7


@pytest.mark.parametrize("temperature", ["0", "0.7"])
def test_graph_profile_uses_only_native_feature_switches_and_records_models(tmp_path, credentials, capsys, temperature):
    fixture.main(arguments(tmp_path) + ["--profile", "graph", "--temperature", temperature, "--execute"])
    root = tmp_path / "fixture"
    cfg = OgMemConfig.load(str(root / "memory.private.yaml"))
    defaults = OgMemConfig()
    assert cfg.graph_backend == "sql" and cfg.enable_graph_expansion and cfg.graph_supplement_enabled
    assert not cfg.scene_enabled and not cfg.scene_generation_enabled and not cfg.scene_retrieval_enabled
    assert cfg.graph_expand_top_k == defaults.graph_expand_top_k == 5
    assert cfg.graph_expand_limit == defaults.graph_expand_limit == 3
    assert cfg.graph_edge_similarity_threshold == defaults.graph_edge_similarity_threshold
    assert cfg.source_occurrence_reconciliation_enabled == defaults.source_occurrence_reconciliation_enabled
    assert cfg.after_turn_threshold == defaults.after_turn_threshold == 400
    setup = load_setup(root / "setup.json")
    assert setup["variant"] == setup["fixture_profile"] == "graph"
    manifest = setup["benchmark_settings"]
    assert setup["native_policy"]["after_turn_threshold"] == 400
    assert setup["native_policy"]["recall_score_gap_ratio"] == 0.10
    assert setup["native_policy"]["extraction_temperature_arguments"] == [None, 0]
    assert setup["native_policy"]["extraction_effective_temperatures"] == [float(temperature), 0]
    assert all(value > 0 for value in setup["native_policy"]["routing_keyword_counts"].values())
    assert manifest["native_extraction_temperature_arguments"] == [None, 0]
    assert manifest["native_extraction_effective_temperatures"] == [float(temperature), 0]
    assert manifest["embedding_model"] == "embedding-model" and manifest["embedding_dimension"] == 1536
    assert manifest["extraction_model"] == "extraction-model" and manifest["qa_model"] == "answer-model"
    assert manifest["routing"] == "native keyword matching; unchanged"
    compose = yaml.safe_load((root / "compose.private.yaml").read_text())
    env = compose["services"]["memory"]["environment"]
    assert env["BEAM_COMPOSE_CAPTURE"] == "1"
    assert "BEAM_ROUTING_MODE" not in env and "BEAM_JSON_FALLBACK_INPUT_CHARS" not in env
    assert all("CONFLICT" not in k for k in env)
    assert not any(secret in json.dumps(setup) for k, secret in credentials.items() if any(s in k for s in ("KEY", "TOKEN", "PASSWORD")))


def test_only_baseline_and_graph_profiles_are_supported(tmp_path):
    with pytest.raises(SystemExit):
        fixture.main(arguments(tmp_path) + ["--profile", "scored-graph"])


def test_beam_sql_entrypoint_has_one_native_index_owner_and_no_policy_patch():
    text = (fixture.ROOT / "scripts/beam_sql_entrypoint.sh").read_text()
    assert "server.app:app" in text and "--workers 1" in text
    assert "run_index_service" not in text and "standalone_runtime" not in text
    assert "BEAM_ROUTING" not in text and "qualification" not in text


def test_profile_changes_only_explicit_graph_flags_in_memory_config(tmp_path, credentials, capsys):
    configs = {}
    for profile in ("baseline", "graph"):
        options = arguments(tmp_path) + ["--directory", str(tmp_path / profile), "--profile", profile, "--execute"]
        fixture.main(options)
        configs[profile] = yaml.safe_load((tmp_path / profile / "memory.private.yaml").read_text())
    assert configs["baseline"].pop("graph") == dict(backend="none", enable_expansion=False, supplement_enabled=False)
    assert configs["graph"].pop("graph") == dict(backend="sql", enable_expansion=True, supplement_enabled=True)
    assert configs["baseline"] == configs["graph"]


def test_fixture_does_not_rewrite_native_templates_or_keyword_patterns(tmp_path, credentials, capsys):
    from scripts.beam_eval_safety import source_hashes
    from retrieval import query_planner
    before = source_hashes(fixture.ROOT)
    patterns = {k: tuple(getattr(query_planner, k)) for k in (
        "_MEMORY_PATTERNS", "_SKILL_PATTERNS", "_RESOURCE_PATTERNS")}
    fixture.main(arguments(tmp_path) + ["--profile", "graph", "--execute"])
    assert source_hashes(fixture.ROOT) == before
    assert patterns == {k: tuple(getattr(query_planner, k)) for k in patterns}


@pytest.mark.skipif(shutil.which("node") is None, reason="Node required for native renderer smoke")
def test_current_native_agent_renderer_preserves_generated_config(tmp_path, credentials):
    fixture.main(arguments(tmp_path) + ["--execute"])
    path = tmp_path / "fixture/agent-state/openclaw.json"
    before = json.loads(path.read_text())
    completed = subprocess.run(["node", str(fixture.ROOT / "docker/configure-openclaw.cjs"),
        str(fixture.ROOT / "config/openclaw/ogmem.template.json"), str(path)],
        env={"PATH": __import__("os").environ.get("PATH", "")}, capture_output=True, text=True, timeout=20)
    assert completed.returncode == 0, completed.stderr
    assert json_digest(json.loads(path.read_text())) == json_digest(before)
