from __future__ import annotations

from types import SimpleNamespace

import pytest

from cli.commands import onboard
from cli.lib.deploy_helpers import generate_quickstart_env
from cli.lib.onboard_config import existing_defaults
from cli.main import build_parser


def test_onboard_parser_supports_local_and_embedding_options():
    args = build_parser().parse_args(
        [
            "onboard",
            "--non-interactive",
            "--mode",
            "local",
            "--embedding-base-url",
            "https://embed.example/v1",
            "--embedding-api-key",
            "embed-key",
            "--embedding-dimension",
            "1536",
        ]
    )

    assert args.mode == "local"
    assert args.embedding_base_url == "https://embed.example/v1"
    assert args.embedding_api_key == "embed-key"
    assert args.embedding_dimension == 1536


def test_provider_change_uses_new_embedding_dimension_default(monkeypatch):
    monkeypatch.setattr(onboard, "_print_header", lambda *args: None)
    args = build_parser().parse_args([
        "onboard", "--non-interactive", "--mode", "local",
        "--provider", "zhipu", "--api-key", "key",
        "--embedding-provider", "zhipu",
    ])
    existing_defaults(args, {
        "embedding": {"provider": "openai", "model": "text-embedding-ada-002"},
        "vector_db": {"dimension": 1536},
    })

    embedding = onboard._step3_embedding(args, {
        "provider": "zhipu", "api_key": "key",
        "base_url": "https://open.bigmodel.cn/api/paas/v4", "model": "glm-4-flash",
    })

    assert embedding["model"] == "embedding-3"
    assert embedding["dimension"] == 2048


def test_noninteractive_model_change_requires_dimension():
    args = build_parser().parse_args([
        "onboard", "--non-interactive", "--mode", "local",
        "--provider", "openai", "--api-key", "key",
        "--embedding-model", "custom-embedding",
    ])

    with pytest.raises(ValueError, match="--embedding-dimension"):
        existing_defaults(args, {
            "embedding": {"provider": "openai", "model": "text-embedding-ada-002"},
            "vector_db": {"dimension": 1536},
        })


def test_step4_defaults_to_opengauss(monkeypatch):
    monkeypatch.setattr(onboard, "_print_header", lambda *args: None)
    monkeypatch.setattr("cli.lib.db_setup.check_postgresql", lambda conn: True)
    monkeypatch.setattr("cli.lib.db_setup.check_opengauss_compatibility", lambda conn: True)
    args = SimpleNamespace(
        non_interactive=True,
        vector_db=None,
        storage_backend="sql",
        sql_dialect="opengauss",
        db_connection="host=127.0.0.1 port=5432 dbname=ogmemory",
    )

    vdb, db = onboard._step4_database(args)

    assert vdb["type"] == "opengauss"
    assert vdb["connection_string"] == args.db_connection
    assert db["backend"] == "sql"
    assert db["reachable"] is True


def test_interactive_mock_llm_does_not_prompt_for_endpoint(monkeypatch):
    monkeypatch.setattr(onboard, "_print_header", lambda *args: None)
    monkeypatch.setattr(onboard, "select", lambda *args, **kwargs: 4)

    def unexpected(*args, **kwargs):
        raise AssertionError("mock provider must not request credentials or an endpoint")

    monkeypatch.setattr(onboard, "input_masked", unexpected)
    monkeypatch.setattr(onboard, "input_validated", unexpected)

    config = onboard._step2_llm(SimpleNamespace(non_interactive=False))
    assert config == {"provider": "mock", "api_key": "", "base_url": "", "model": "mock"}


def test_interactive_chroma_agfs_does_not_prompt_for_sql(monkeypatch):
    monkeypatch.setattr(onboard, "_print_header", lambda *args: None)
    choices = []

    def choose(*args, **kwargs):
        choices.append(args[0])
        return 1

    monkeypatch.setattr(onboard, "select", choose)
    args = SimpleNamespace(
        non_interactive=False,
        mode="plugin",
        vector_db=None,
        storage_backend="agfs",
        sql_dialect=None,
        db_connection=None,
    )

    vdb, db = onboard._step4_database(args)

    assert choices == ["Vector database:"]
    assert vdb == {"type": "chroma"}
    assert db["backend"] == "agfs"



def test_install_deps_uses_valid_extras_and_reports_failure(monkeypatch):
    calls = []

    class Result:
        returncode = 1

    monkeypatch.setattr(onboard.shutil, "which", lambda name: None)
    monkeypatch.setattr(onboard, "confirm", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        onboard.subprocess,
        "run",
        lambda command, **kwargs: calls.append(command) or Result(),
    )
    monkeypatch.setattr(onboard, "_print_header", lambda *args: None)

    assert onboard._step_install_deps(
        SimpleNamespace(), {"provider": "mock"}, {"type": "chroma"}
    ) is False
    assert calls == [[onboard.sys.executable, "-m", "pip", "install", "-e", ".[chroma]"]]


def test_generate_quickstart_env_uses_shell_quoting(tmp_path):
    path = generate_quickstart_env(
        {"provider": "openai", "api_key": "key with 'quote", "base_url": "https://llm", "model": "m"},
        {"provider": "openai", "api_key": "embed", "base_url": "https://embed", "model": "e", "dimension": 1024},
        instance_name="locomo",
        output_path=tmp_path / "ogmemory.env",
    )

    text = path.read_text()
    assert "export INSTANCE_NAME=locomo" in text
    assert """export LLM_API_KEY='key with '"'"'quote'""" in text
    assert "EMBEDDING_BASE_URL=https://embed" in text


def test_step4_skips_host_database_check_for_docker(monkeypatch):
    monkeypatch.setattr(onboard, "_print_header", lambda *args: None)
    def unexpected(*args, **kwargs):
        raise AssertionError("docker quickstart should check its own database")
    monkeypatch.setattr("cli.lib.db_setup.check_postgresql", unexpected)

    args = SimpleNamespace(
        non_interactive=True,
        mode="docker",
        vector_db=None,
        storage_backend="sql",
        sql_dialect="opengauss",
        db_connection=None,
    )
    vdb, db = onboard._step4_database(args)

    assert vdb["type"] == "opengauss"
    assert db["reachable"] is True


def test_docker_deployment_failure_is_reported(monkeypatch, tmp_path):
    monkeypatch.setattr(onboard, "_print_header", lambda *args: None)
    monkeypatch.setattr(onboard, "confirm", lambda *args, **kwargs: True)
    monkeypatch.setattr(onboard, "_step1_check", lambda args: True)
    monkeypatch.setattr(onboard, "_step2_llm", lambda args: {"provider": "mock", "model": "mock", "api_key": "", "base_url": ""})
    monkeypatch.setattr(onboard, "_step3_embedding", lambda args, llm: {"provider": "mock", "model": "mock", "api_key": "", "base_url": "", "dimension": 2})
    monkeypatch.setattr(onboard, "_step4_database", lambda args: ({"type": "opengauss"}, {"backend": "sql", "reachable": True}))
    monkeypatch.setattr(onboard, "_step5_target", lambda args: "docker")
    monkeypatch.setattr(onboard, "_step_docker_setup", lambda *args: {"env_path": str(tmp_path / "env"), "deploy_failed": True})

    args = SimpleNamespace(non_interactive=True, mode="docker", port=18090)
    assert onboard.run(args) == 1


@pytest.mark.parametrize("mode,backend,expect_agfs", [
    ("local", "sql", False), ("headless", "sql", False),
    ("docker", "sql", False), ("local", "agfs", True),
    (None, None, True),
])
def test_environment_check_scopes_agfs(monkeypatch, capsys, mode, backend, expect_agfs):
    from cli.commands import check
    calls = []
    monkeypatch.setattr(check.shutil, "which", lambda _: None)
    monkeypatch.setattr(check.importlib, "import_module", lambda _: object())
    monkeypatch.setattr(check, "check_agfs", lambda: calls.append(True))
    check.run(SimpleNamespace(mode=mode, storage_backend=backend))
    output = capsys.readouterr().out
    assert bool(calls) == expect_agfs
    assert ("Section 2: AGFS Service" in output) == expect_agfs


@pytest.mark.parametrize("compatible,schema_ok,expected_exit", [
    (False, True, 1), (True, False, 1), (True, True, 0),
])
def test_local_database_failure_stops_before_install_or_config(
    monkeypatch, tmp_path, compatible, schema_ok, expected_exit,
):
    from cli.lib import db_setup
    calls = []
    monkeypatch.delenv("OGMEM_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("cli.lib.process.find_project_root", lambda: tmp_path)
    monkeypatch.setattr(onboard, "_step1_check", lambda args: True)
    monkeypatch.setattr(onboard, "_step2_llm", lambda args: {"provider": "mock", "model": "mock"})
    monkeypatch.setattr(onboard, "_step3_embedding", lambda *args: {"provider": "mock", "model": "mock", "dimension": 8})
    monkeypatch.setattr(onboard, "select", lambda *args, **kwargs: 0)
    monkeypatch.setattr(onboard, "confirm", lambda prompt, **kwargs: prompt == "Initialize schema?")
    monkeypatch.setattr(db_setup, "check_postgresql", lambda conn: True)
    monkeypatch.setattr(db_setup, "check_opengauss_compatibility", lambda conn: compatible)
    monkeypatch.setattr(db_setup, "init_schema", lambda *args: calls.append("schema") or schema_ok)
    monkeypatch.setattr(onboard, "_step_install_deps", lambda *args: calls.append("install") or True)
    monkeypatch.setattr("cli.lib.onboard_config.save_config", lambda *args, **kwargs: calls.append("config") or True)
    args = build_parser().parse_args(["onboard", "--mode", "local", "--db-connection", "test-dsn"])
    assert onboard.run(args) == expected_exit
    if expected_exit:
        assert "install" not in calls and "config" not in calls
    else:
        assert calls == ["schema", "install", "config"]
    if not compatible:
        assert "schema" not in calls


@pytest.mark.parametrize("backend", ["opengauss", "milvus"])
@pytest.mark.parametrize("non_interactive", [False, True])
def test_docker_passes_selected_vector_to_deployment(monkeypatch, backend, non_interactive):
    args = build_parser().parse_args(["onboard", "--mode", "docker", "--vector-db", backend])
    args.non_interactive = non_interactive
    monkeypatch.setattr(onboard, "_step5_target", lambda args: "docker")
    monkeypatch.setattr(onboard, "_step1_check", lambda args: True)
    monkeypatch.setattr(onboard, "_step2_llm", lambda args: {"provider": "openai", "model": "test"})
    monkeypatch.setattr(onboard, "_step3_embedding", lambda *args: {"dimension": 3})
    monkeypatch.setattr(onboard, "_step4_database", lambda args: ({"type": backend}, {}))
    received = []

    def deploy(args, llm, embedding, vdb):
        received.append(vdb)
        return {"env_path": "/tmp/test.env"}

    monkeypatch.setattr(onboard, "_step_docker_setup", deploy)
    assert onboard.run(args) == 0
    assert received == [{"type": backend}]


def test_docker_interactive_vector_menu_offers_supported_providers(monkeypatch):
    args = build_parser().parse_args(["onboard", "--mode", "docker"])
    monkeypatch.setattr(onboard, "_print_header", lambda *args: None)
    monkeypatch.setattr(onboard, "input_validated", lambda *args: "host=db")
    monkeypatch.setattr(onboard, "confirm", lambda *args, **kwargs: True)
    seen = []

    def choose(prompt, options, **kwargs):
        if prompt == "Vector database:":
            seen.append(options)
        return 0

    monkeypatch.setattr(onboard, "select", choose)
    vdb, _ = onboard._step4_database(args)
    assert vdb["type"] == "opengauss"
    assert seen == [["openGauss", "Milvus"]]


@pytest.mark.parametrize("mode", ["local", "headless"])
def test_local_vector_choices_keep_chroma_and_milvus(mode):
    args = build_parser().parse_args(["onboard", "--mode", mode, "--vector-db", "milvus"])
    assert onboard._vector_db_choices(args) == ["opengauss", "chroma", "milvus"]


def test_milvus_defaults_survive_wizard_rerun_and_explicit_changes(monkeypatch):
    from cli.lib.onboard_config import merge_config

    args = build_parser().parse_args(["onboard", "--mode", "docker", "--non-interactive"])
    raw = {"vector_db": {"type": "milvus", "milvus_uri": "http://existing:19530",
                         "milvus_collection": "existing", "milvus_db_name": "tenantdb",
                         "milvus_timeout": 9}}
    existing_defaults(args, raw)
    monkeypatch.setattr(onboard, "_print_header", lambda *args: None)
    vdb, _ = onboard._step4_database(args)
    assert vdb == raw["vector_db"]
    changed = {**vdb, "milvus_uri": "http://new:19530", "milvus_collection": "new"}
    assert merge_config(raw, {"vector_db": changed})["vector_db"] == changed


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_invalid_milvus_timeout_is_rejected_before_deployment(value):
    args = build_parser().parse_args(["onboard", "--mode", "docker", "--non-interactive",
                                     "--vector-db", "milvus", "--milvus-timeout", value])
    with pytest.raises(ValueError, match="finite and positive"):
        onboard._step4_database(args)


@pytest.mark.parametrize("non_interactive", [False, True])
def test_docker_rejects_chroma_before_database_prompts(monkeypatch, non_interactive):
    args = build_parser().parse_args(["onboard", "--mode", "docker", "--vector-db", "chroma"])
    args.non_interactive = non_interactive
    monkeypatch.setattr(onboard, "select", lambda *a, **kw: pytest.fail("unexpected prompt"))
    with pytest.raises(ValueError, match="Chroma requires local/headless"):
        onboard._step4_database(args)


def test_docker_environment_rejects_chroma(tmp_path):
    with pytest.raises(ValueError, match="Unsupported Docker vector provider"):
        generate_quickstart_env({}, {}, vector_config={"type": "chroma"},
                                output_path=tmp_path / "unsupported.env")
    assert not (tmp_path / "unsupported.env").exists()
