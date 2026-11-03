from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tests" / "e2e"))

from eval import build_parser  # noqa: E402
from eval import validate_dataset  # noqa: E402
from run_mcp_locomo import (  # noqa: E402
    _copy_auth_state,
    _evaluation_config,
    _load_model_config,
    _start_gateway,
    _validate_mcp_user_scope,
)


def test_runner_accepts_stage_and_dataset_path(tmp_path: Path) -> None:
    dataset = tmp_path / "locomo.json"
    dataset.write_text("[]", encoding="utf-8")

    args = build_parser().parse_args(["ingest", str(dataset)])

    assert args.mode == "ingest"
    assert args.dataset == dataset
    assert args.output_dir is None
    assert args.sample is None


def test_dataset_validation_accepts_locomo_shape(tmp_path: Path) -> None:
    dataset = tmp_path / "locomo.json"
    payload = [
        {
            "sample_id": "s0",
            "conversation": {
                "speaker_a": "A",
                "speaker_b": "B",
                "session_1": [{"speaker": "A", "text": "Hello"}],
            },
            "qa": [
                {"question": "When?", "answer": 2022},
                {"question": "Adversarial question", "category": 5},
            ],
        }
    ]
    dataset.write_text(json.dumps(payload), encoding="utf-8")

    assert validate_dataset(dataset) == payload


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda item: item.pop("sample_id"), "item at index 0: sample_id"),
        (
            lambda item: item["conversation"].pop("speaker_a"),
            "item at index 0: conversation.speaker_a",
        ),
        (
            lambda item: item["conversation"].pop("session_1"),
            "item at index 0: conversation must include at least one session_N array",
        ),
        (
            lambda item: item["conversation"]["session_1"][0].pop("speaker"),
            "conversation.session_1 message at index 0: speaker",
        ),
        (
            lambda item: item["conversation"]["session_1"][0].pop("text"),
            "conversation.session_1 message at index 0: text",
        ),
        (
            lambda item: item["qa"][0].pop("question"),
            "item at index 0, QA at index 0: question",
        ),
        (
            lambda item: item["qa"][0].update(answer=[]),
            "item at index 0, QA at index 0: answer",
        ),
        (
            lambda item: item["qa"][0].update(answer=""),
            "item at index 0, QA at index 0: answer must not be empty",
        ),
        (lambda item: item.update(qa=[]), "item at index 0: qa must include"),
    ],
)
def test_dataset_validation_reports_required_field(tmp_path: Path, mutation, message: str) -> None:
    dataset = tmp_path / "locomo.json"
    item = {
        "sample_id": "s0",
        "conversation": {
            "speaker_a": "A",
            "speaker_b": "B",
            "session_1": [{"speaker": "A", "text": "Hello"}],
        },
        "qa": [{"question": "Who spoke?", "answer": "A"}],
    }
    mutation(item)
    dataset.write_text(json.dumps([item]), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        validate_dataset(dataset)


def test_gateway_starts_with_isolated_state(tmp_path: Path) -> None:
    proc = Mock()
    proc.poll.return_value = None
    runtime_home = tmp_path / "runtime-home"
    log_path = tmp_path / "gateway.log"
    auth_source_dir = tmp_path / "source-agent"
    auth_source_dir.mkdir()
    config = {
        "gateway": {"port": 23456, "auth": {"token": "test-token"}},
        "agents": {"list": [{"id": "main", "agentDir": str(auth_source_dir)}]},
        "models": {
            "providers": {"stored-provider": {"baseUrl": "https://configured.example.invalid/v1"}}
        },
    }
    (auth_source_dir / "auth-profiles.json").write_text(
        json.dumps({"version": 1, "profiles": {"provider:default": {"type": "api_key"}}}),
        encoding="utf-8",
    )
    (auth_source_dir / "models.json").write_text(
        json.dumps(
            {
                "providers": {
                    "stored-provider": {
                        "baseUrl": "https://example.invalid/v1",
                        "api": "openai-completions",
                        "apiKey": "stored-key",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with (
        patch.dict(os.environ, {"HOME": "original-home", "USERPROFILE": "original-profile"}),
        patch("run_mcp_locomo.subprocess.Popen", return_value=proc) as popen,
        patch("run_mcp_locomo.requests.get", return_value=Mock(ok=True)),
    ):
        resolved = _load_model_config(config, auth_source_dir)
        _start_gateway(resolved, runtime_home, log_path, auth_source_dir)
        returned_proc, env = _start_gateway(
            resolved, runtime_home, log_path, auth_source_dir, command="custom-openclaw"
        )

    state_dir = runtime_home / ".openclaw"
    assert returned_proc is proc
    assert env["OPENCLAW_STATE_DIR"] == str(state_dir)
    assert env["OPENCLAW_CONFIG_PATH"] == str(state_dir / "openclaw.json")
    assert env["HOME"] == "original-home"
    assert env["USERPROFILE"] == "original-profile"
    runtime_config = json.loads((state_dir / "openclaw.json").read_text(encoding="utf-8"))
    assert runtime_config["agents"]["list"][0]["agentDir"] == str(
        state_dir / "agents" / "main" / "agent"
    )
    assert runtime_config["agents"]["defaults"]["workspace"] == str(state_dir / "workspace")
    assert runtime_config["models"]["providers"]["stored-provider"] == {
        "baseUrl": "https://configured.example.invalid/v1",
        "api": "openai-completions",
        "apiKey": "stored-key",
    }
    assert config["agents"]["list"][0]["agentDir"] == str(auth_source_dir)
    assert (state_dir / "agents" / "main" / "agent" / "auth-profiles.json").is_file()
    assert popen.call_count == 2
    assert popen.call_args.args[0] == ["custom-openclaw", "gateway", "--port", "23456"]


def test_auth_copy_exports_only_auth_tables(tmp_path: Path) -> None:
    source = tmp_path / "source-agent"
    destination = tmp_path / "destination-agent"
    source.mkdir()
    with sqlite3.connect(source / "openclaw-agent.sqlite") as connection:
        connection.execute("CREATE TABLE auth_profile_store (store_key TEXT, store_json TEXT)")
        connection.execute("CREATE TABLE auth_profile_state (state_key TEXT, state_json TEXT)")
        connection.execute("CREATE TABLE memory_index_meta (key TEXT, value TEXT)")
        connection.execute(
            "INSERT INTO auth_profile_store VALUES (?, ?)",
            ("primary", json.dumps({"version": 1, "profiles": {"provider:default": {}}})),
        )
        connection.execute(
            "INSERT INTO auth_profile_state VALUES (?, ?)",
            ("primary", json.dumps({"version": 1, "lastGood": {}})),
        )
        connection.execute("INSERT INTO memory_index_meta VALUES ('private', 'do-not-copy')")

    copied = _copy_auth_state(source, destination)

    assert copied == ["auth-profiles.json", "auth-state.json"]
    assert (
        json.loads((destination / "auth-profiles.json").read_text(encoding="utf-8"))["version"] == 1
    )
    assert not (destination / "openclaw-agent.sqlite").exists()


def test_eval_config_isolates_mcp_tools_and_competing_memory_plugins() -> None:
    original = {
        "gateway": {"auth": {"mode": "password", "password": "old-password"}},
        "mcp": {
            "servers": {
                "ogmemory": {"url": "http://old/mcp"},
                "openviking": {"url": "http://127.0.0.1:1933/mcp"},
            }
        },
        "plugins": {
            "slots": {
                "memory": "existing-memory-plugin",
                "contextEngine": "openviking",
            },
            "entries": {
                "existing-memory-plugin": {"enabled": True},
                "third-party-mem0": {"enabled": True},
            },
        },
        "tools": {"profile": "full", "alsoAllow": ["openviking__search"]},
    }

    configured = _evaluation_config(
        original,
        mcp_url="http://127.0.0.1:8091/mcp",
        headers={"X-API-Key": "root-key", "X-Account-ID": "acct-demo"},
        user_id="run-conv-3",
        memory_agent_id="run-agent",
        gateway_token="isolated-token",
    )

    server = configured["mcp"]["servers"]["ogmemory"]
    assert server["headers"]["X-User-ID"] == "run-conv-3"
    assert server["headers"]["X-Agent-ID"] == "run-agent"
    assert server["timeout"] == 1200
    assert server["toolFilter"] == {"include": ["ogm_*", "resources_*"]}
    assert set(configured["mcp"]["servers"]) == {"ogmemory"}
    assert configured["gateway"]["auth"] == {"mode": "token", "token": "isolated-token"}
    assert configured["gateway"]["http"]["endpoints"]["responses"]["enabled"] is True
    assert configured["plugins"]["slots"]["memory"] == "none"
    assert configured["plugins"]["slots"]["contextEngine"] == "legacy"
    assert configured["plugins"]["entries"]["existing-memory-plugin"]["enabled"] is False
    assert configured["plugins"]["entries"]["third-party-mem0"]["enabled"] is False
    assert configured["tools"] == {
        "profile": "minimal",
        "alsoAllow": [
            "ogmemory__ogm_query_memory",
            "ogmemory__ogm_read_memory",
            "ogmemory__ogm_write_memory",
            "ogmemory__resources_list",
            "ogmemory__resources_read",
        ],
        "deny": ["session_status"],
    }
    assert original["plugins"]["entries"]["existing-memory-plugin"]["enabled"] is True
    assert original["gateway"]["auth"]["mode"] == "password"


def test_eval_config_preserves_openclaw_answer_model() -> None:
    original = {
        "agents": {"defaults": {"model": {"primary": "tested/answer-model"}}},
        "models": {"providers": {"tested": {"api": "openai-responses"}}},
        "mcp": {"servers": {}},
        "plugins": {},
        "tools": {},
    }

    configured = _evaluation_config(
        original,
        mcp_url="http://127.0.0.1:8091/mcp",
        headers={"X-API-Key": "root-key", "X-Account-ID": "acct-demo"},
        user_id="run-conv-3",
        memory_agent_id="run-agent",
        gateway_token="isolated-token",
    )

    assert configured["agents"] == original["agents"]
    assert configured["models"] == original["models"]


def test_mcp_user_scope_rejects_member_key() -> None:
    response = Mock(status_code=403)

    with patch("run_mcp_locomo.requests.get", return_value=response):
        with pytest.raises(RuntimeError, match="requires an Admin or Root"):
            _validate_mcp_user_scope(
                "http://127.0.0.1:8091/mcp",
                {"X-API-Key": "member-key", "X-Account-ID": "acct-1"},
            )


def test_mcp_user_scope_allows_disabled_role_control() -> None:
    response = Mock(status_code=503)
    response.json.return_value = {"error": "Admin API is unavailable when role control is disabled"}

    with patch("run_mcp_locomo.requests.get", return_value=response):
        _validate_mcp_user_scope(
            "http://127.0.0.1:8091/mcp",
            {"X-API-Key": "root-key", "X-Account-ID": "acct-1"},
        )
