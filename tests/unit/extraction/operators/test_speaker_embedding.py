"""Tests for the ``speaker_embedding`` operator + backends.

Three layers:

* Pure aggregation helpers — ``_aggregate_turns_by_speaker``,
  ``_aggregate_total_duration``.
* Backends — ``MockSpeakerEmbeddingBackend`` deterministic stub.
* Operator — ``SpeakerEmbeddingExtractor.extract(audio, ctx,
  diarization_extracted_node=...)`` with the mock backend;
  asserts the parent ``ExtractedNode.extracted_content`` shape
  + the ``SPEAKER_EMBEDDING_MATERIALIZATION`` child payload
  shape.

All tests run in-process — no speechbrain / torch download.
"""

from __future__ import annotations

import math
import os
import struct
import tempfile
import wave
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from core.graph_enums import RawDataType
from core.graph_models import create_extracted_node, create_raw_node
from core.models import RequestContext
from extraction.operators.builtin.embedding_backends import (
    MockSpeakerEmbeddingBackend,
    SpeakerEmbedding,
    SpeakerEmbeddingBackendError,
    SpeakerEmbeddingResult,
    SpeechBrainECAPABackend,
    build_speaker_embedding_backend,
)
from extraction.operators.builtin.speaker_embedding import (
    SpeakerEmbeddingExtractor,
    _aggregate_turns_by_speaker,
    _aggregate_total_duration,
)
from extraction.operators.materialize import SPEAKER_EMBEDDING_MATERIALIZATION


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_sine_wav(path: Path, *, duration_sec: float = 1.0) -> None:
    n = int(round(16000 * duration_sec))
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        frames = bytearray()
        for i in range(n):
            sample = int(0.5 * 32767 * math.sin(2 * math.pi * 440 * i / 16000))
            frames.extend(struct.pack("<h", max(-32767, min(32767, sample))))
        w.writeframes(bytes(frames))


def _audio_node(path: Path) -> Any:
    return create_raw_node(
        raw_type=RawDataType.AUDIO,
        content=b"",
        account_id="acct_test",
        metadata={
            "format": "wav", "sample_rate": 16000, "channels": 1,
            "source_path": str(path),
            "video_id": "vid_test", "clip_id": 0,
        },
        node_id="audio_emb_test".ljust(32, "x")[:32],
    )


def _make_diar_node(
    audio_id: str, turns: list[dict],
) -> Any:
    return create_extracted_node(
        source_raw_id=audio_id,
        extractor_type="speaker_diarization",
        extraction_schema="speaker_diarization_v1",
        account_id="acct_test",
        extracted_content={
            "num_speakers": len({t["speaker_id"] for t in turns}),
            "turns": turns,
            "source_audio_node_id": audio_id,
        },
        node_id="diar_test".ljust(32, "x")[:32],
    )


@pytest.fixture
def req_ctx() -> RequestContext:
    return RequestContext(
        account_id="acct_test",
        user_id="u_test", agent_id="a_test",
        session_id="s_test", trace_id="t_test",
    )


# ---------------------------------------------------------------------------
# Pure aggregation helpers
# ---------------------------------------------------------------------------

def test_aggregate_turns_by_speaker_groups_correctly():
    turns = [
        {"start_sec": 0.0, "end_sec": 1.0, "speaker_id": "A"},
        {"start_sec": 2.0, "end_sec": 3.0, "speaker_id": "B"},
        {"start_sec": 1.0, "end_sec": 2.0, "speaker_id": "A"},
        {"start_sec": 4.0, "end_sec": 5.0, "speaker_id": "A"},
    ]
    grouped = _aggregate_turns_by_speaker(turns)
    assert sorted(grouped.keys()) == ["A", "B"]
    assert len(grouped["A"]) == 3
    assert len(grouped["B"]) == 1


def test_aggregate_turns_by_speaker_skips_empty_speaker_id():
    turns = [
        {"start_sec": 0.0, "end_sec": 1.0, "speaker_id": ""},
        {"start_sec": 1.0, "end_sec": 2.0, "speaker_id": "A"},
    ]
    grouped = _aggregate_turns_by_speaker(turns)
    assert "A" in grouped
    assert "" not in grouped


def test_aggregate_total_duration_sums_differences():
    turns = [
        {"start_sec": 0.0, "end_sec": 1.5, "speaker_id": "A"},
        {"start_sec": 2.0, "end_sec": 4.0, "speaker_id": "A"},
    ]
    assert _aggregate_total_duration(turns) == pytest.approx(3.5)


def test_aggregate_total_duration_handles_empty():
    assert _aggregate_total_duration([]) == 0.0


# ---------------------------------------------------------------------------
# Mock backend
# ---------------------------------------------------------------------------

def test_mock_backend_returns_unit_vector():
    backend = MockSpeakerEmbeddingBackend(embedding_dim=192)
    emb = backend.embed(
        "/tmp/a.wav",
        start_sec=0.0, end_sec=1.0, speaker_id="A",
    )
    assert len(emb) == 192
    norm = math.sqrt(sum(x * x for x in emb))
    assert norm == pytest.approx(1.0, abs=1e-6)


def test_mock_backend_deterministic_for_same_inputs():
    backend = MockSpeakerEmbeddingBackend()
    e1 = backend.embed("/tmp/a.wav", start_sec=0.0, end_sec=1.0, speaker_id="A")
    e2 = backend.embed("/tmp/a.wav", start_sec=0.0, end_sec=1.0, speaker_id="A")
    assert e1 == e2


def test_mock_backend_differs_on_inputs():
    backend = MockSpeakerEmbeddingBackend()
    e1 = backend.embed("/tmp/a.wav", start_sec=0.0, end_sec=1.0, speaker_id="A")
    e2 = backend.embed("/tmp/a.wav", start_sec=0.0, end_sec=1.0, speaker_id="B")
    e3 = backend.embed("/tmp/a.wav", start_sec=0.5, end_sec=1.0, speaker_id="A")
    e4 = backend.embed("/tmp/b.wav", start_sec=0.0, end_sec=1.0, speaker_id="A")
    assert e1 != e2
    assert e1 != e3
    assert e1 != e4


def test_mock_backend_custom_dim():
    backend = MockSpeakerEmbeddingBackend(embedding_dim=64)
    emb = backend.embed(
        "/tmp/a.wav",
        start_sec=0.0, end_sec=1.0, speaker_id="A",
    )
    assert len(emb) == 64


def test_mock_backend_fail_on_path_substring():
    backend = MockSpeakerEmbeddingBackend(fail_on_path_substring="bad")
    with pytest.raises(SpeakerEmbeddingBackendError):
        backend.embed("/tmp/bad.wav", start_sec=0.0, end_sec=1.0, speaker_id="A")


# ---------------------------------------------------------------------------
# SpeechBrain backend (without loading)
# ---------------------------------------------------------------------------

def test_speechbrain_backend_name_and_dim():
    backend = SpeechBrainECAPABackend()
    assert backend.name == "speechbrain"
    assert backend.embedding_dim == 192


def test_speechbrain_backend_missing_dep_message():
    backend = SpeechBrainECAPABackend()
    with patch.dict(
        "sys.modules",
        {"torch": None, "torchaudio": None,
         "speechbrain": None,
         "speechbrain.inference": None,
         "speechbrain.inference.speaker": None},
    ):
        with pytest.raises(SpeakerEmbeddingBackendError) as excinfo:
            backend._load()
    assert "speechbrain" in str(excinfo.value).lower() or "torch" in str(excinfo.value).lower()


# ---------------------------------------------------------------------------
# build_speaker_embedding_backend factory
# ---------------------------------------------------------------------------

def test_build_speaker_embedding_backend_default_is_speechbrain():
    backend = build_speaker_embedding_backend("speechbrain")
    assert isinstance(backend, SpeechBrainECAPABackend)


def test_build_speaker_embedding_backend_aliases():
    assert isinstance(
        build_speaker_embedding_backend("ecapa"),
        SpeechBrainECAPABackend,
    )
    assert isinstance(
        build_speaker_embedding_backend("ecapa-tdnn"),
        SpeechBrainECAPABackend,
    )
    assert isinstance(
        build_speaker_embedding_backend("mock"),
        MockSpeakerEmbeddingBackend,
    )


def test_build_speaker_embedding_backend_unknown_raises():
    with pytest.raises(SpeakerEmbeddingBackendError):
        build_speaker_embedding_backend("nonexistent")


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------

def test_speaker_embedding_operator_happy_path(
    tmp_path: Path, req_ctx,
):
    wav = tmp_path / "x.wav"
    _write_sine_wav(wav, duration_sec=10.0)
    audio = _audio_node(wav)
    diar_node = _make_diar_node(audio.node_id, [
        {"start_sec": 0.0, "end_sec": 5.0, "speaker_id": "SPEAKER_00"},
        {"start_sec": 5.0, "end_sec": 10.0, "speaker_id": "SPEAKER_01"},
    ])
    extractor = SpeakerEmbeddingExtractor(
        backend="mock", embedding_dim=192,
    )
    result = extractor.extract(
        audio, req_ctx, diarization_extracted_node=diar_node,
    )
    assert result.error is None
    assert result.extracted_content["embedding_dim"] == 192
    assert len(result.extracted_content["speakers"]) == 2
    for s in result.extracted_content["speakers"]:
        assert len(s["speaker_embedding"]) == 192
        # L2-normalised.
        norm = math.sqrt(sum(x * x for x in s["speaker_embedding"]))
        assert norm == pytest.approx(1.0, abs=1e-6)


def test_speaker_embedding_operator_no_diarization(
    tmp_path: Path, req_ctx,
):
    wav = tmp_path / "x.wav"
    _write_sine_wav(wav)
    audio = _audio_node(wav)
    extractor = SpeakerEmbeddingExtractor(backend="mock")
    result = extractor.extract(audio, req_ctx, diarization_extracted_node=None)
    assert result.error is not None
    assert result.extracted_content["speakers"] == []


def test_speaker_embedding_operator_missing_path(
    tmp_path: Path, req_ctx,
):
    audio = _audio_node(tmp_path / "does_not_exist.wav")
    extractor = SpeakerEmbeddingExtractor(backend="mock")
    result = extractor.extract(audio, req_ctx, diarization_extracted_node=None)
    assert result.error
    assert result.extracted_content["speakers"] == []
    assert result.confidence == 0.0


def test_speaker_embedding_operator_backend_failure_continues(
    tmp_path: Path, req_ctx,
):
    wav = tmp_path / "x.wav"
    _write_sine_wav(wav)
    audio = _audio_node(wav)
    diar_node = _make_diar_node(audio.node_id, [
        {"start_sec": 0.0, "end_sec": 1.0, "speaker_id": "A"},
        {"start_sec": 2.0, "end_sec": 3.0, "speaker_id": "B"},
    ])
    extractor = SpeakerEmbeddingExtractor(backend="mock")
    # Fail on every embed call. The operator catches per-speaker
    # errors and continues, ending with confidence < 1.
    with patch.object(
        extractor._backend, "embed",
        side_effect=SpeakerEmbeddingBackendError("mock", "boom"),
    ):
        result = extractor.extract(
            audio, req_ctx, diarization_extracted_node=diar_node,
        )
    assert result.confidence == 0.0
    assert result.extracted_content["speakers"] == []
    assert "per_speaker_error" in result.extracted_content


def test_speaker_embedding_operator_metadata_fields():
    extractor = SpeakerEmbeddingExtractor(backend="mock")
    assert extractor.extractor_type == "speaker_embedding"
    assert extractor.schema_name == "speaker_embedding_v1"
    assert extractor.supported_types == [RawDataType.AUDIO]
    info = extractor.info
    assert info.materialization is SPEAKER_EMBEDDING_MATERIALIZATION


# ---------------------------------------------------------------------------
# Materialization policy
# ---------------------------------------------------------------------------

def test_speaker_embedding_materialization_child_id_deterministic():
    item = {"speaker_id": "SPEAKER_00", "speaker_embedding": [0.0] * 192}
    cid_a = SPEAKER_EMBEDDING_MATERIALIZATION.child_id_fn(
        item, "audio_xyz", {"source_path": "/tmp/x.wav"},
    )
    cid_b = SPEAKER_EMBEDDING_MATERIALIZATION.child_id_fn(
        item, "audio_xyz", {"source_path": "/tmp/x.wav"},
    )
    assert cid_a == cid_b
    assert len(cid_a) == 32
    # Different speaker → different id
    item_b = {"speaker_id": "SPEAKER_01", "speaker_embedding": [0.0] * 192}
    assert cid_a != SPEAKER_EMBEDDING_MATERIALIZATION.child_id_fn(
        item_b, "audio_xyz", {"source_path": "/tmp/x.wav"},
    )


def test_speaker_embedding_materialization_payload_shape():
    item = {
        "speaker_id": "SPEAKER_00",
        "speaker_embedding": [0.1, 0.2, 0.3],
        "embedding_dim": 192,
        "model": "speechbrain",
        "num_turns": 4,
        "total_duration_sec": 12.5,
    }
    payload = SPEAKER_EMBEDDING_MATERIALIZATION.child_payload_fn(
        item, "audio_xyz", {"source_path": "/tmp/x.wav", "video_id": "v1", "clip_id": 7},
    )
    assert payload["speaker_id"] == "SPEAKER_00"
    assert payload["speaker_embedding"] == [0.1, 0.2, 0.3]
    assert payload["embedding_dim"] == 192
    assert payload["model"] == "speechbrain"
    assert payload["num_turns"] == 4
    assert payload["total_duration_sec"] == 12.5
    assert payload["source_audio_node_id"] == "audio_xyz"
    assert payload["source_video_id"] == "v1"
    assert payload["source_clip_id"] == 7


def test_speaker_embedding_materialization_edge_props():
    item = {"speaker_id": "SPEAKER_00", "speaker_embedding": []}
    props = SPEAKER_EMBEDDING_MATERIALIZATION.edge_properties_fn(
        item, "audio_xyz", None,
    )
    assert props["relation"] == "speaker_embedding"
    assert props["speaker_id"] == "SPEAKER_00"


# ---------------------------------------------------------------------------
# Lazy import contract
# ---------------------------------------------------------------------------

def test_builtin_import_succeeds_without_speechbrain():
    from extraction.operators import builtin
    assert hasattr(builtin, "SpeakerEmbeddingExtractor")

@pytest.mark.parametrize('start,end', [
    (14.172, 14.189), (15.843, 15.860), (27.014, 27.031),
    (20.281, 20.298), (8.553, 8.570),
])
def test_real_backend_short_audio_is_typed_skip_without_loading_model(start, end):
    from types import SimpleNamespace

    import numpy as np

    from extraction.operators.builtin.embedding_backends import SpeakerEmbeddingTooShortError

    backend = SpeechBrainECAPABackend()
    with patch.dict('sys.modules', {
        'torch': SimpleNamespace(),
        'torchaudio': SimpleNamespace(load=lambda _: (np.zeros((1, 480000)), 16000)),
    }), patch.object(backend, '_load', side_effect=AssertionError('model loaded')):
        with pytest.raises(SpeakerEmbeddingTooShortError) as exc:
            backend.embed('fake.wav', start_sec=start, end_sec=end)
    assert exc.value.duration_sec == pytest.approx(0.017)


def test_out_of_bounds_audio_is_hard_error_not_short_skip():
    from types import SimpleNamespace

    import numpy as np

    from extraction.operators.builtin.embedding_backends import SpeakerEmbeddingTooShortError

    with patch.dict('sys.modules', {
        'torch': SimpleNamespace(),
        'torchaudio': SimpleNamespace(load=lambda _: (np.zeros((1, 16000)), 16000)),
    }):
        with pytest.raises(SpeakerEmbeddingBackendError) as exc:
            SpeechBrainECAPABackend().embed('fake.wav', start_sec=2, end_sec=3)
    assert not isinstance(exc.value, SpeakerEmbeddingTooShortError)


def test_exact_100ms_is_not_skipped():
    from types import SimpleNamespace

    import numpy as np

    backend = SpeechBrainECAPABackend()
    with patch.dict('sys.modules', {
        'torch': SimpleNamespace(),
        'torchaudio': SimpleNamespace(load=lambda _: (np.zeros((1, 16000)), 16000)),
    }), patch.object(backend, '_load', side_effect=RuntimeError('reached model')):
        with pytest.raises(RuntimeError, match='reached model'):
            backend.embed('fake.wav', start_sec=0, end_sec=0.1)


class TestSpeechBrainSavedirFallback:
    """savedir must resolve to a writable location even when HOME=/

    The m3-runner container ships HOME=/ — the legacy ~/.cache default
    resolved to /.cache and every ECAPA embed failed with EACCES
    (observed: 71/79 clips failed before HOME was overridden manually).
    """

    def test_savedir_explicit_param_wins(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        monkeypatch.delenv("HF_HOME", raising=False)
        backend = SpeechBrainECAPABackend(savedir="/explicit/dir")
        assert backend._savedir == "/explicit/dir"

    def test_savedir_default_uses_home_cache_when_writable(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        monkeypatch.delenv("HF_HOME", raising=False)
        backend = SpeechBrainECAPABackend()
        assert backend._savedir == str(
            tmp_path / ".cache" / "speechbrain" / "speechbrain_spkrec-ecapa-voxceleb"
        )

    def test_savedir_falls_back_when_home_is_root(self, monkeypatch, tmp_path):
        # HOME=/ -> ~/.cache probes /, which is not writable for non-root.
        monkeypatch.setenv("HOME", "/")
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
        backend = SpeechBrainECAPABackend()
        assert backend._savedir == str(
            tmp_path / "hf" / "speechbrain" / "speechbrain_spkrec-ecapa-voxceleb"
        )

    def test_savedir_prefers_xdg_cache_home_over_hf_home(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", "/")
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
        monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
        backend = SpeechBrainECAPABackend()
        assert backend._savedir == str(
            tmp_path / "xdg" / "speechbrain" / "speechbrain_spkrec-ecapa-voxceleb"
        )

    def test_savedir_tempdir_fallback_when_home_is_root(self, monkeypatch, tmp_path):
        # The real m3-runner scenario: HOME=/, no XDG/HF overrides — the
        # legacy /.cache path is unwritable, so the tempdir candidate wins.
        monkeypatch.setenv("HOME", "/")
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        monkeypatch.delenv("HF_HOME", raising=False)
        backend = SpeechBrainECAPABackend()
        assert backend._savedir == os.path.join(
            tempfile.gettempdir(), "speechbrain",
            "speechbrain_spkrec-ecapa-voxceleb",
        )
