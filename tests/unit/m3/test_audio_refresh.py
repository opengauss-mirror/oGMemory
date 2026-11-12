"""Refresh configuration, stage failures and shell wrapper compatibility."""

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from m3.audio_cache import signature
from m3.pipeline import stages
from m3.pipeline.context import PipelineContext
from m3.pipeline.spec import StageSpec
from m3.voice_asr import parse_args


def test_signature_includes_effective_endpoint_not_credentials(tmp_path, monkeypatch):
    path = tmp_path / "a.wav"
    path.write_bytes(b"audio")
    audio = SimpleNamespace(node_id="audio", metadata={"source_path": str(path)})
    args = parse_args(["--backend", "openai_http"])
    monkeypatch.setenv("OGMEM_ASR_BASE_URL", "http://first/v1")
    first = signature(audio, args, "voice_asr", "acct")
    monkeypatch.setenv("OGMEM_ASR_BASE_URL", "http://second/v1")
    second = signature(audio, args, "voice_asr", "acct")
    assert first and second and first != second
    args.api_key = "not-written-to-the-receipt"
    assert signature(audio, args, "voice_asr", "acct") == second
    assert signature(audio, args, "voice_asr", "other-account") != second
    args.base_url = "http://first/v1/"
    assert signature(audio, args, "voice_asr", "acct") == first


@pytest.mark.parametrize("kind", ["voice_asr", "voice_segmentation", "speaker_embedding"])
@pytest.mark.parametrize("failed,segments", [(0, 1), (1, 0), (1, 1), (2, 0)])
def test_stage_preserves_failures_and_valid_silence(tmp_path, monkeypatch, kind, failed, segments):
    import importlib

    module = importlib.import_module("m3." + kind)

    # With attempted=10 the tolerance floor max(1, int(10*0.15)) = 1:
    # a single failed clip is tolerated (the stage stays ok); two
    # failures exceed the ratio and fail the stage.
    def driver(args, **kwargs):
        path = Path(args.output_jsonl)
        path.write_text("{}\n" * segments)
        path.with_suffix(".summary.json").write_text(
            json.dumps(
                {
                    "attempted_audio": 10,
                    "completed_audio": 10 - failed,
                    "cached_audio": 1,
                    "executed_audio": 9,
                    "failed_audio": failed,
                }
            )
        )
        return 2 if failed > 1 else 0

    monkeypatch.setattr(module, "run_" + kind, driver)
    ctx = PipelineContext(
        video=tmp_path / "video.mp4",
        video_id="v",
        seconds=0,
        work_dir=tmp_path,
        config_path=tmp_path / "config.yaml",
        graph_store="memory",
        connection_string=None,
        account_id="acct",
        graph=MagicMock(),
        op_service=MagicMock(),
        artifacts={},
        variables={},
    )
    result = getattr(stages, kind + "_stage")(
        ctx,
        StageSpec(
            id=kind,
            type="m3." + kind,
            params={"output_jsonl": str(tmp_path / "out.jsonl")},
        ),
    )
    tolerated = failed <= 1
    assert result.ok is tolerated
    assert result.counts["cached_audio"] == 1
    assert bool(result.errors) is (not tolerated)


@pytest.mark.parametrize(
    "mode,expected", [(None, "true"), ("refresh", "true"), ("cached", "false")]
)
def test_wrapper_preserves_default_and_limits_cached_mode(tmp_path, mode, expected):
    # Substitute only uv: no models, config loading or experiment writes.
    bindir = tmp_path / "bin"
    bindir.mkdir()
    uv = bindir / "uv"
    uv.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    uv.chmod(0o755)
    env = dict(
        os.environ, PATH=str(bindir) + ":" + os.environ["PATH"], RUN_ROOT=str(tmp_path / "run")
    )
    env.pop("ASR_VAD_MODE", None)
    if mode:
        env["ASR_VAD_MODE"] = mode
    result = subprocess.run(
        ["bash", "scripts/run_m3_random10.sh", "--videos", "bedroom_03", "--force"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    args = result.stdout.splitlines()
    for kind in ("voice_asr", "voice_segmentation"):
        assert f"{kind}.params.force={expected}" in args
    assert "speaker_diarization.params.force=true" in args
    assert "speaker_embedding.params.force=true" in args
    assert "--force" in args
    assert args[args.index("--videos") + 1] == "bedroom_03"


def test_wrapper_rejects_unknown_mode(tmp_path):
    env = dict(os.environ, ASR_VAD_MODE="typo", RUN_ROOT=str(tmp_path / "run"))
    result = subprocess.run(
        ["bash", "scripts/run_m3_random10.sh"], env=env, capture_output=True, text=True
    )
    assert result.returncode == 2
    assert "ASR_VAD_MODE" in result.stderr
    assert not (tmp_path / "run").exists()


def test_embedding_probe_defers_to_dependency_validation():
    result = stages._probe_speaker_embedding(MagicMock(), MagicMock(), MagicMock())
    assert not result.complete
    assert "diarization dependency" in result.reason
