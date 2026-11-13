"""Exercise the onboarding/quickstart boundary without a real deployment."""
from __future__ import annotations

import os
import re
import shlex
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from cli.commands import onboard
from cli.lib import deploy_helpers
from cli.main import build_parser

ROOT = Path(__file__).resolve().parents[3]
HAS_BASH = os.name != "nt" and shutil.which("bash")


def model_config(provider="openai"):
    return {"provider": provider, "base_url": "https://model.example.test/v1",
            "api_key": "dummy-key", "model": "dummy-model", "dimension": 1024}


def test_generated_env_is_private_and_restricts_existing_file(tmp_path):
    envfile = tmp_path / "ogmemory.env"
    envfile.write_text("old contents")
    envfile.chmod(0o644)

    deploy_helpers.generate_quickstart_env(
        model_config(), model_config(), output_path=envfile,
    )

    assert stat.S_IMODE(envfile.stat().st_mode) == 0o600
    assert "export LLM_API_KEY=dummy-key" in envfile.read_text()


@pytest.mark.skipif(not HAS_BASH, reason="POSIX shell required")
def test_printed_deferred_command_matches_immediate_execution(tmp_path, monkeypatch, capsys):
    root = tmp_path / "project 'quoted' $(touch UNEXPECTED)"
    script = root / "deploy/user-quickstart/deploy.sh"
    script.parent.mkdir(parents=True)
    script.write_text(
        '#!/bin/bash\nset -eu\n'
        'printf "%s\\n" "$OGMEMORY_ENV_FILE" "$OGMEMORY_STACK_ROOT" '
        '"$OGMEMORY_RUNTIME_ENV_FILE" > "$REVIEW_CAPTURE"\n'
    )
    capture = tmp_path / "environment.txt"
    monkeypatch.setenv("REVIEW_CAPTURE", str(capture))
    # Conflicting inherited settings must be overridden by both launch paths.
    for name in ("OGMEMORY_ENV_FILE", "OGMEMORY_STACK_ROOT", "OGMEMORY_RUNTIME_ENV_FILE"):
        monkeypatch.setenv(name, "/wrong/inherited/path")
    monkeypatch.setattr("cli.lib.process.find_project_root", lambda: root)
    monkeypatch.setattr(deploy_helpers, "get_project_root", lambda: root)
    monkeypatch.setattr(onboard, "confirm", lambda *args, **kwargs: False)
    args = build_parser().parse_args(["onboard", "--mode", "docker"])
    result = onboard._step_docker_setup(args, model_config(), model_config(), {"type": "opengauss"})
    output = re.sub(r"\x1b\[[0-9;]*m", "", capsys.readouterr().out)
    command = output.split("Skipped. Run: ", 1)[1].splitlines()[0]
    subprocess.run(["bash", "-c", command], cwd=tmp_path, check=True)
    deferred = capture.read_text().splitlines()
    state = root / ".ogmem_data/onboard-docker"
    assert deploy_helpers.deploy_quickstart(
        env_file=Path(result["env_path"]), stack_root=state,
        runtime_env_file=state / "ogmemory-runtime.env",
    ) == 0
    assert capture.read_text().splitlines() == deferred == [
        str(state / "ogmemory.env"), str(state), str(state / "ogmemory-runtime.env"),
    ]
    assert not (tmp_path / "UNEXPECTED").exists()


@pytest.mark.skipif(not HAS_BASH, reason="POSIX shell required")
@pytest.mark.parametrize("llm_provider,embedding_provider,adapter", [
    ("openai", "openai", "openai"), ("volcengine", "openai", "openai"),
    ("volcengine", "volcengine", "volcengine"),
    ("dashscope", "dashscope", "openai"), ("zhipu", "zhipu", "openai"),
])
def test_supported_providers_pass_actual_quickstart_validation(tmp_path, llm_provider, embedding_provider, adapter):
    envfile = deploy_helpers.generate_quickstart_env(
        model_config(llm_provider), model_config(embedding_provider), output_path=tmp_path / "model.env",
    )
    values = subprocess.check_output([
        "bash", "-c", 'source "$1"; printf "%s\\n" "$EMBEDDING_PROVIDER" '
        '"$EMBEDDING_BASE_URL" "$EMBEDDING_API_KEY" "$EMBEDDING_MODEL"', "verify", str(envfile),
    ], text=True).splitlines()
    assert values == [adapter, "https://model.example.test/v1", "dummy-key", "dummy-model"]
    shims = tmp_path / "bin"
    shims.mkdir()
    for name, body in {
        "docker": 'test "$1" = info || { echo UNEXPECTED_DOCKER_ACTION >&2; exit 97; }\n',
        "uname": 'echo aarch64\n',
        "ss": 'exit 0\n',
        "curl": 'echo MODEL_PREFLIGHT_REACHED >&2\nexit 98\n',
    }.items():
        path = shims / name
        path.write_text("#!/bin/sh\n" + body)
        path.chmod(0o700)
    completed = subprocess.run(
        ["bash", str(ROOT / "deploy/user-quickstart/deploy.sh")],
        env={"PATH": f"{shims}:/usr/bin:/bin", "HOME": str(tmp_path),
             "OGMEMORY_ENV_FILE": str(envfile), "LANG": "C.UTF-8"},
        capture_output=True, text=True, timeout=10,
    )
    # Stop at the first model request; reaching it proves the real shell's
    # provider checks accepted the generated config. No network is used.
    assert completed.returncode == 1
    assert "MODEL_PREFLIGHT_REACHED" in completed.stderr
    assert "UNEXPECTED_DOCKER_ACTION" not in completed.stderr
    assert "EMBEDDING_PROVIDER 只支持" not in completed.stderr


@pytest.mark.skipif(not HAS_BASH, reason="POSIX shell required")
def test_runtime_env_escapes_custom_paths(tmp_path):
    state = tmp_path / "state 'quoted'"
    state.mkdir()
    envfile = deploy_helpers.generate_quickstart_env(
        model_config(), model_config(), output_path=state / "ogmemory.env",
    )
    runtime = state / "ogmemory-runtime.env"
    shims = tmp_path / "bin"
    shims.mkdir()
    scripts = {
        "uname": "echo aarch64\n",
        "ss": "exit 0\n",
        "curl": '''
out=''
endpoint=''
while [ "$#" -gt 0 ]; do
  case "$1" in
    -o) out="$2"; shift 2 ;;
    *) endpoint="$1"; shift ;;
  esac
done
case "$endpoint" in
  *chat/completions) printf '%s' '{"choices":[]}' > "$out" ;;
  *) printf '%s' '{"data":[{"embedding":[0]}]}' > "$out" ;;
esac
printf 200
''',
        "docker": '''
case "${1-}" in
  info) [ "${2-}" = --format ] && printf '/tmp\\n'; exit 0 ;;
  container) exit 1 ;;
  ps) exit 0 ;;
  image) exit 1 ;;
  pull) exit 98 ;;
  *) exit 1 ;;
esac
''',
    }
    for name, body in scripts.items():
        script = shims / name
        script.write_text("#!/bin/sh\n" + body)
        script.chmod(0o700)

    completed = subprocess.run(
        ["bash", str(ROOT / "deploy/user-quickstart/deploy.sh")],
        env={
            "PATH": f"{shims}:/usr/bin:/bin",
            "HOME": str(tmp_path),
            "OGMEMORY_ENV_FILE": str(envfile),
            "OGMEMORY_STACK_ROOT": str(state),
            "OGMEMORY_RUNTIME_ENV_FILE": str(runtime),
            "LANG": "C.UTF-8",
        },
        capture_output=True, text=True, timeout=10,
    )

    assert completed.returncode == 1
    subprocess.run(["bash", "-n", str(runtime)], check=True)
    values = subprocess.check_output([
        "bash", "-c", 'source "$1"; printf "%s\\n" "$STACK_DIR"',
        "verify", str(runtime),
    ], text=True).splitlines()
    assert values == [str(state / "ogmemory-ogmem")]


@pytest.mark.parametrize("field,provider", [
    ("llm", "mock"), ("embedding", "mock"), ("embedding", "st"), ("embedding", "unknown"),
])
def test_unsupported_provider_fails_before_writing_or_launching(tmp_path, monkeypatch, capsys, field, provider):
    monkeypatch.setattr("cli.lib.process.find_project_root", lambda: tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid providers must not prompt for or start deployment")
    monkeypatch.setattr(onboard, "confirm", forbidden)
    monkeypatch.setattr(deploy_helpers, "deploy_quickstart", forbidden)
    args = build_parser().parse_args(["onboard", "--mode", "docker", "--non-interactive"])
    llm, embedding = model_config(), model_config()
    (llm if field == "llm" else embedding)["provider"] = provider
    assert onboard._step_docker_setup(args, llm, embedding, {"type": "opengauss"}) == {"deploy_failed": True}
    assert "does not support" in capsys.readouterr().out
    assert not (tmp_path / ".ogmem_data").exists()


def test_docker_menus_offer_only_supported_providers(monkeypatch):
    menus = []
    def choose(label, options, **kwargs):
        menus.append((label, options))
        assert "mock" not in options and "st" not in options
        return options.index("dashscope")
    monkeypatch.setattr(onboard, "select", choose)
    monkeypatch.setattr(onboard, "confirm", lambda *args, **kwargs: False)
    monkeypatch.setattr(onboard, "input_masked", lambda *args: "dummy")
    monkeypatch.setattr(onboard, "input_validated", lambda label, default: default)
    args = build_parser().parse_args(["onboard", "--mode", "docker"])
    llm = onboard._step2_llm(args)
    embedding = onboard._step3_embedding(args, llm)
    assert llm["provider"] == embedding["provider"] == "dashscope"
    assert [label for label, _ in menus] == ["LLM provider:", "Embedding provider:"]


@pytest.mark.parametrize("readme", ["README.md", "README_zh.md"])
def test_documented_docker_command_consumes_all_explicit_model_settings(monkeypatch, readme):
    text = (ROOT / readme).read_text(encoding="utf-8")
    block = next(block for block in re.findall(r"```bash\n(.*?)```", text, re.S)
                 if "ogmem onboard" in block and "--mode docker" in block)
    values = {
        "LLM_API_KEY": "explicit-llm-key", "LLM_BASE_URL": "https://private-llm.example.test/v1",
        "LLM_MODEL": "custom-llm", "EMBEDDING_API_KEY": "explicit-embedding-key",
        "EMBEDDING_BASE_URL": "https://private-embedding.example.test/v1", "EMBEDDING_MODEL": "custom-embedding",
    }
    for key, value in values.items():
        block = block.replace("$" + key, value)
    command = shlex.split(block.replace("\\\n", ""))
    args = build_parser().parse_args(command[1:])
    def forbidden(*args, **kwargs):
        raise AssertionError("fully configured README example must not prompt")
    for name in ("select", "confirm", "input_masked", "input_validated"):
        monkeypatch.setattr(onboard, name, forbidden)
    assert args.non_interactive
    llm = onboard._step2_llm(args)
    embedding = onboard._step3_embedding(args, llm)
    assert llm == {"provider": "openai", "api_key": values["LLM_API_KEY"],
                   "base_url": values["LLM_BASE_URL"], "model": values["LLM_MODEL"]}
    assert embedding == {"provider": "openai", "api_key": values["EMBEDDING_API_KEY"],
                         "base_url": values["EMBEDDING_BASE_URL"], "model": values["EMBEDDING_MODEL"], "dimension": 1024}


@pytest.mark.parametrize("backend", ["opengauss", "milvus"])
def test_selected_backend_and_milvus_parameters_reach_environment(tmp_path, backend, monkeypatch):
    monkeypatch.setenv("MILVUS_TOKEN", "dummy-token")
    envfile = deploy_helpers.generate_quickstart_env(
        model_config(), model_config(), output_path=tmp_path / "backend.env",
        vector_config={"type": backend, "milvus_uri": "http://milvus.example:19530",
                       "milvus_collection": "selected", "milvus_db_name": "testdb"},
    )
    env = subprocess.check_output([
        "bash", "-c", 'source "$1"; printf "%s\\n" "$VECTOR_DB_TYPE" "$MILVUS_URI" '
        '"$MILVUS_COLLECTION" "$MILVUS_DB_NAME" "$MILVUS_TOKEN"', "verify", str(envfile),
    ], text=True).splitlines()
    assert env == [backend, "http://milvus.example:19530", "selected", "testdb", "dummy-token"]
    assert stat.S_IMODE(envfile.stat().st_mode) == 0o600


@pytest.mark.skipif(not HAS_BASH, reason="POSIX shell required")
@pytest.mark.parametrize("backend,probe_failure", [
    ("opengauss", False), ("milvus", False), ("milvus", True),
])
@pytest.mark.parametrize("grpc_override", [None, "", "custom-bypass.example"])
def test_real_quickstart_renders_selected_backend(tmp_path, backend, probe_failure, grpc_override):
    from string import Template

    import yaml

    envfile = deploy_helpers.generate_quickstart_env(
        model_config(), model_config(), output_path=tmp_path / "runtime.env",
        vector_config={"type": backend, "milvus_uri": "http://milvus.example:19530"},
    )
    capture = tmp_path / "docker.log"
    shims = tmp_path / "bin"
    shims.mkdir()
    scripts = {
        "uname": "echo aarch64\n", "ss": "exit 0\n",
        "curl": '''
out=''
while [ "$#" -gt 0 ]; do
  case "$1" in -o) out="$2"; shift 2 ;; *) shift ;; esac
done
if [ -n "$out" ]; then printf '%s' '{"choices":[],"data":[{"embedding":[0]}]}' > "$out"; fi
printf 200
''',
        "docker": '''
printf '%s\\n' "$*" >> "$CAPTURE"
case "$1" in
  info) [ "${2-}" = --format ] && echo /tmp; exit 0 ;;
  container) exit 1 ;;
  ps) exit 0 ;;
  run) case "$*" in *scripts.check_milvus*)
    if [ "$PREFLIGHT_FAIL" = 1 ]; then echo PREFLIGHT_DENIED >&2; exit 97; fi ;;
    esac ;;
  exec) case "$*" in *'python -c'*) echo VALIDATION_STOP >&2; exit 98 ;; esac ;;
esac
exit 0
''',
    }
    for name, body in scripts.items():
        p = shims / name
        p.write_text("#!/bin/sh\n" + body)
        p.chmod(0o700)
    completed = subprocess.run(
        ["bash", str(ROOT / "deploy/user-quickstart/deploy.sh")],
        env={"PATH": f"{shims}:/usr/bin:/bin", "HOME": str(tmp_path),
             "OGMEMORY_ENV_FILE": str(envfile), "OGMEMORY_STACK_ROOT": str(tmp_path),
             "OGMEM_IMAGE": "test-current-image", "CAPTURE": str(capture), "LANG": "C.UTF-8",
             "PREFLIGHT_FAIL": "1" if probe_failure else "0",
             **({"MILVUS_NO_GRPC_PROXY": grpc_override} if grpc_override is not None else {})},
        capture_output=True, text=True, timeout=10,
    )
    commands = capture.read_text()
    if backend == "milvus":
        probe = next(line for line in commands.splitlines() if "scripts.check_milvus" in line)
        assert "--network ogmem_ogmemory_net" in probe
        assert "--add-host host.docker.internal:host-gateway" in probe
        assert "MILVUS_URI=http://milvus.example:19530" in probe
        assert "VECTOR_DB_DIMENSION=1024" in probe
        assert "-e MILVUS_TOKEN" in probe
        bypass = "milvus.example" if grpc_override is None else grpc_override
        assert f"no_grpc_proxy={bypass}" in shlex.split(probe)
    if probe_failure:
        assert completed.returncode == 1
        assert "未替换现有容器" in completed.stderr
        assert "rm -f" not in commands
        assert "run -d" not in commands
        return
    assert "VALIDATION_STOP" in completed.stderr, completed.stderr
    if backend == "milvus":
        assert commands.index("scripts.check_milvus") < commands.index("rm -f")
        assert commands.count(f"-e no_grpc_proxy={bypass} ") == 2
    text = (tmp_path / "ogmemory-ogmem/ogmemory.yaml").read_text()
    config = yaml.safe_load(Template(text).safe_substitute({
        "VECTOR_DB_TYPE": backend, "MILVUS_URI": "http://milvus.example:19530",
        "MILVUS_COLLECTION": "contextengine", "MILVUS_DB_NAME": "default", "MILVUS_TIMEOUT": "30",
    }))
    assert config["vector_db"]["type"] == backend
    assert config["vector_db"]["milvus_uri"] == "http://milvus.example:19530"
    commands = capture.read_text()
    assert f"VECTOR_DB_TYPE={backend}" in commands
    assert "/vector:/data/vector" not in commands
    assert ("--name ogmem_opengauss" in commands) == (backend == "opengauss")
    if backend != "opengauss":
        assert "--entrypoint python test-current-image" in commands


@pytest.mark.skipif(not HAS_BASH, reason="POSIX shell required")
@pytest.mark.parametrize("budget", ["0", "86401", "1.5"])
def test_invalid_readiness_budget_fails_before_deployment(tmp_path, budget):
    envfile = deploy_helpers.generate_quickstart_env(
        model_config(), model_config(), output_path=tmp_path / "runtime.env",
    )
    result = subprocess.run(
        ["bash", str(ROOT / "deploy/user-quickstart/deploy.sh")],
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path),
             "OGMEMORY_ENV_FILE": str(envfile), "OGMEM_DEPLOY_WAIT_SECONDS": budget},
        capture_output=True, text=True, timeout=5,
    )
    assert result.returncode == 1
    assert "OGMEM_DEPLOY_WAIT_SECONDS 必须是 1-86400" in result.stderr
    assert not (tmp_path / "ogmemory-ogmem").exists()
