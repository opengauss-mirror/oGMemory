"""CV YAML must reach the HTTP client without wrapper/profile defaults masking it."""

import os
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from m3.config import load_cv_service_config
from m3.detect_faces import _build_backend, parse_args
from m3.pipeline import load_pipeline_spec
from m3.pipeline.spec import StageSpec
from m3.pipeline.stages import detect_faces_stage


@pytest.fixture(autouse=True)
def clear_cv_environment(monkeypatch):
    for key in ("OGMEM_CONFIG", "OGMEM_CV_SERVICE_URL", "OGMEM_CV_AUTH_TOKEN", "CV_SERVICE_URL"):
        monkeypatch.delenv(key, raising=False)


def write_config(tmp_path, block):
    path = tmp_path / "ogmemory.yaml"
    path.write_text(yaml.safe_dump({"cv_service": block}))
    return str(path)


def test_missing_block_uses_compose_defaults(tmp_path):
    path = tmp_path / "old-config.yaml"
    path.write_text("{}")
    config = load_cv_service_config(str(path))
    assert config.base_url == "http://127.0.0.1:8081"
    assert config.timeout_seconds == 30.0
    assert config.auth_token is None


def test_external_service_reaches_http_backend(tmp_path):
    path = write_config(tmp_path, {
        "base_url": "https://gpu.example/cv/",
        "timeout_seconds": 75,
        "auth_token": "test-secret",
    })
    backend = _build_backend(parse_args(["--config", path, "--frames-manifest", "unused"]))
    assert backend._base_url == "https://gpu.example/cv"
    assert backend._timeout == 75
    assert backend._static_headers["Authorization"] == "Bearer test-secret"
    assert "test-secret" not in repr(load_cv_service_config(path))


def test_env_then_cli_override_yaml(tmp_path, monkeypatch):
    path = write_config(tmp_path, {
        "base_url": "http://yaml.example", "timeout_seconds": 75, "auth_token": "yaml-token",
    })
    monkeypatch.setenv("OGMEM_CV_SERVICE_URL", "http://env.example")
    monkeypatch.setenv("OGMEM_CV_AUTH_TOKEN", "env-token")
    config = load_cv_service_config(path)
    assert config.base_url == "http://env.example"
    assert config.auth_token == "env-token"
    assert config.timeout_seconds == 75
    backend = _build_backend(parse_args([
        "--config", path, "--frames-manifest", "unused",
        "--cv-service-url", "http://cli.example",
        "--cv-timeout", "12", "--cv-auth-token", "cli-token",
    ]))
    assert backend._base_url == "http://cli.example"
    assert backend._timeout == 12
    assert backend._static_headers["Authorization"] == "Bearer cli-token"


def test_yaml_environment_references_and_config_selection(tmp_path, monkeypatch):
    path = write_config(tmp_path, {
        "base_url": "${MY_CV_URL}", "timeout_seconds": "${MY_CV_TIMEOUT}",
        "auth_token": "${MY_CV_TOKEN}",
    })
    for key in ("MY_CV_URL", "MY_CV_TIMEOUT", "MY_CV_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OGMEM_CONFIG", path)
    assert load_cv_service_config().base_url == "http://127.0.0.1:8081"
    assert load_cv_service_config().timeout_seconds == 30
    monkeypatch.setenv("MY_CV_URL", "http://remote.example:8081")
    monkeypatch.setenv("MY_CV_TIMEOUT", "60")
    monkeypatch.setenv("MY_CV_TOKEN", "placeholder-secret")
    config = load_cv_service_config()
    assert config.base_url == "http://remote.example:8081"
    assert config.timeout_seconds == 60
    assert config.auth_token == "placeholder-secret"


@pytest.mark.parametrize("block", [
    [], "http://wrong-place",
    {"base_url": 123}, {"base_url": "gpu:8081"}, {"base_url": "file:///tmp/image"},
    {"base_url": "http://gpu:invalid"},
    {"timeout_seconds": 0}, {"timeout_seconds": -1}, {"timeout_seconds": True},
    {"timeout_seconds": ".nan"}, {"timeout_seconds": float("inf")},
    {"timeout_seconds": "invalid"}, {"auth_token": 123},
])
def test_invalid_settings_fail_early(tmp_path, block):
    with pytest.raises(ValueError, match="cv_service"):
        load_cv_service_config(write_config(tmp_path, block))


def test_mock_does_not_require_cv_configuration(tmp_path):
    path = write_config(tmp_path, {"base_url": "invalid"})
    backend = _build_backend(parse_args([
        "--config", path, "--frames-manifest", "unused", "--backend", "mock",
    ]))
    assert backend.name == "mock"


@pytest.mark.parametrize("explicit", [False, True])
def test_pipeline_forwards_selected_config_and_explicit_overrides(tmp_path, monkeypatch, explicit):
    path = write_config(tmp_path, {
        "base_url": "http://yaml.example", "timeout_seconds": 75, "auth_token": "yaml-token",
    })
    output = tmp_path / "faces.jsonl"
    output.write_text("")
    ctx = SimpleNamespace(
        config_path=path, artifact=lambda name: output, graph_store="memory",
        connection_string=None, account_id="test", graph=Mock(), op_service=Mock(),
    )
    backends = []

    def run(args, **kwargs):
        assert args.config == path
        backends.append(_build_backend(args))
        return 0

    monkeypatch.setattr("m3.detect_faces.run", run)
    params = {} if not explicit else {
        "cv_service_url": "http://stage.example", "cv_timeout": 15,
        "cv_auth_token": "stage-token",
    }
    result = detect_faces_stage(ctx, StageSpec("detect_faces", "m3.detect_faces", params=params))
    assert result.ok
    backend = backends[0]
    assert backend._base_url == ("http://stage.example" if explicit else "http://yaml.example")
    assert backend._timeout == (15 if explicit else 75)
    assert backend._static_headers["Authorization"] == (
        "Bearer stage-token" if explicit else "Bearer yaml-token"
    )


@pytest.mark.parametrize("profile", [
    "video_episodic", "video_episodic_qa", "video_episodic_faces", "video_episodic_full",
])
def test_default_profiles_and_cli_do_not_mask_yaml(profile):
    from m3.run_video_pipeline import _legacy_set_overrides, parse_args as parse_pipeline_args

    spec = load_pipeline_spec(f"config/m3/pipelines/{profile}.yaml")
    stage = next(s for s in spec.stages if s.id == "detect_faces")
    assert "cv_timeout" not in stage.params
    args = parse_pipeline_args(["--video", "unused.mp4"])
    assert not any(".cv_timeout=" in value for value in _legacy_set_overrides(args))
    args = parse_pipeline_args(["--video", "unused.mp4", "--cv-timeout", "12"])
    assert "detect_faces.params.cv_timeout=12.0" in _legacy_set_overrides(args)


@pytest.mark.parametrize("override", [None, "http://explicit.example:8081"])
def test_wrapper_only_overrides_cv_when_requested(tmp_path, override):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    uv = bindir / "uv"
    uv.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    uv.chmod(0o755)
    env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}", RUN_ROOT=str(tmp_path / "run"))
    if override:
        env["CV_SERVICE_URL"] = override
    result = subprocess.run(
        ["bash", "scripts/run_m3_random10.sh"], env=env, capture_output=True, text=True, check=True,
    )
    options = [line for line in result.stdout.splitlines() if ".cv_service_url=" in line]
    assert options == ([] if override is None else [f"detect_faces.params.cv_service_url={override}"])
