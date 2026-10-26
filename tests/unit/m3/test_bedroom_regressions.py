"""Offline regressions from bedroom_03; no credentials or model calls."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.graph_models import create_extracted_node
from core.llm_output import final_text
from core.temporal import boundary_transcript, within_scope
from extraction.operators.builtin.episodic_memory_generator import _normalise_episodic_response
from m3.control_adapter import M3TaskContext, OGMemoryControlAdapter, SearchConfig
from m3.pipeline.context import StageResult
from m3.search import EpisodicSearcher
from m3.temporal import annotation_scope
from providers.llm import MockLLM
from providers.llm.openai_llm import OpenAILLM
from service.graph_service import ClipScope, GraphService
from tests.unit.m3.test_retrieval_core import Embedder, ctx, graph_search, record


@pytest.mark.parametrize("text,expected", [
    ('<think>{"name":"wrong"}</think> Milk.', "Milk."),
    ("<THINK>\nprivate\n</THINK>Black album.", "Black album."),
    ("<think>unfinished", ""),
    ("prefix <think>unfinished", ""),
    ("<think>done</think><think>unfinished", ""),
    ("private prefix</think>Final", "Final"),
    (None, ""),
])
def test_final_output_drops_reasoning(text, expected):
    assert final_text(text) == expected


@pytest.mark.parametrize("wrapper", ["{}", "```json\n{}\n```", "Here is the result:\n{}"])
def test_parser_ignores_reasoning_json_and_handles_string_braces(wrapper):
    expected = {"description": 'A bag with a "}" label is placed in a wardrobe.',
                "characters": [], "entities": [], "events": []}
    content = '<think>{"name":"Person 1","description":"wrong"}</think>\n'
    content += wrapper.format(json.dumps(expected))
    assert OpenAILLM._extract_json_from_text(content) == expected


def test_reasoning_only_json_is_not_repaired_into_an_answer():
    with pytest.raises(json.JSONDecodeError):
        OpenAILLM._extract_json_from_text('<think>{"description":"not final"}')


def test_multimodal_reasoning_prefix_does_not_trigger_second_api_call():
    provider = object.__new__(OpenAILLM)
    provider._client = Mock()
    provider._model, provider._temperature, provider._max_tokens = "test", 0, 100
    provider._is_reasoning_model_override = None
    provider._reasoning_effort = None
    provider._extra_body = None
    provider._track_usage = Mock()
    provider._client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content='<think>{"name":"fragment"}</think>```json\n{"description":"Final scene."}\n```'
        ))]
    )
    result = provider._complete_multimodal_structured([], {"type": "object"})
    assert result == {"description": "Final scene."}
    provider._client.chat.completions.create.assert_called_once()


@pytest.mark.parametrize("value", [
    {"name": "Person", "description": "Wearing glasses"},
    {"name": "door", "type": "object"}, {},
    {"description": 123}, {"description": "   "},
    {"description": "A room.", "characters": "not a list"},
])
def test_invalid_memory_is_not_silently_accepted(value):
    with pytest.raises(ValueError, match="Invalid episodic"):
        _normalise_episodic_response(value)


def content(clip=5, start=150, duration=30.13401):
    return {
        "source_video_id": "v", "source_clip_id": clip,
        "source_start_sec": start, "source_duration_sec": duration,
        "description": "FUTURE_SUMMARY", "characters": ["FUTURE_PERSON"],
        "embedding": [1, 0], "transcript_search_text": "FUTURE_TOKEN",
        "speech_transcript": [
            {"start_sec": 10.9, "end_sec": 18.6, "text": "the black album",
             "speaker_id": "FUTURE_ID", "extra": "FUTURE_EXTRA"},
            {"start_sec": 18.98, "end_sec": 27.2, "text": "FUTURE_SPEECH"},
        ],
    }


@pytest.mark.parametrize("clip,start,end,previous,timestamp", [
    (5, 150, 18.6, 4, "02:51"),
    (19, 570, 6.96, 18, "09:39"),
    (61, 1830, 1.87, 60, "30:37"),
])
def test_real_question_boundaries_rescue_only_completed_speech(clip, start, end, previous, timestamp):
    data = content(clip, start)
    data["speech_transcript"] = [{"start_sec": 0, "end_sec": end, "text": "safe"}]
    bound, cutoff = annotation_scope(previous, timestamp)
    assert not within_scope(data, "v", bound, cutoff)
    projected = boundary_transcript(data, cutoff)
    assert projected["speech_transcript"][0]["text"] == "safe"
    assert within_scope(projected, "v", bound, cutoff)


def test_projection_is_whitelisted_nonmutating_and_rejects_malformed_segments():
    data = content()
    data["speech_transcript"] += [
        None, {"start_sec": -1, "end_sec": 2, "text": "bad"},
        {"start_sec": 2, "end_sec": 1, "text": "bad"},
        {"start_sec": 0, "end_sec": float("nan"), "text": "bad"},
        {"start_sec": 0, "end_sec": 1, "text": {}},
    ]
    projected = boundary_transcript(data, 171)
    assert len(projected["speech_transcript"]) == 1
    assert "FUTURE" not in json.dumps(projected)
    projected["speech_transcript"][0]["text"] = "changed"
    assert data["speech_transcript"][0]["text"] == "the black album"
    assert boundary_transcript(data, 150) == {}
    assert boundary_transcript(data, 181) == {}
    assert boundary_transcript({**data, "source_start_sec": None}, 171) == {}


def boundary_service(data=None, account="m3bench", extractor="episodic_memory_generator"):
    node = create_extracted_node(
        source_raw_id="raw", extractor_type=extractor, extraction_schema="m3",
        account_id=account, extracted_content=data if data is not None else content(),
    )
    store, index = Mock(), Mock()
    store.get_node.return_value = node
    index.get_records_at_time.return_value = [SimpleNamespace(metadata={"extracted_node_id": node.node_id})]
    return GraphService(store, vector_index=index), node


def test_live_graph_boundary_projection_does_not_use_vector_or_lexical_scores():
    graph, node = boundary_service()
    hits = graph.get_boundary_transcript(ctx(), scope=ClipScope("v", 5, 171),
                                         extractor_types=("episodic_memory_generator",))
    assert len(hits) == 1
    assert "FUTURE" not in json.dumps(hits[0].node.extracted_content)
    assert node.extracted_content["description"] == "FUTURE_SUMMARY"
    assert hits[0].score == 0
    kwargs = graph._vector_index.get_records_at_time.call_args.kwargs
    assert kwargs["timestamp"] == 171
    assert kwargs["filters"]["account_id"] == "m3bench"
    assert "before_sec" not in kwargs["filters"]
    graph._vector_index.search_by_vector.assert_not_called()
    graph._vector_index.search_by_transcript.assert_not_called()


@pytest.mark.parametrize("change", ["account", "video", "clip", "type", "deleted"])
def test_live_boundary_rechecks_scope_and_deleted_nodes(change):
    data = content()
    if change == "video":
        data["source_video_id"] = "other"
    if change == "clip":
        data["source_clip_id"] = 6  # Explicit upper bound is still respected.
    graph, _ = boundary_service(data, account="other" if change == "account" else "m3bench",
                               extractor="other" if change == "type" else "episodic_memory_generator")
    if change == "deleted":
        graph._store.get_node.return_value = None
    assert graph.get_boundary_transcript(ctx(), scope=ClipScope("v", 5, 171),
                                         extractor_types=("episodic_memory_generator",)) == []


def test_search_exposes_boundary_even_when_other_branches_disabled(offline_tokenizer):
    graph, _ = boundary_service()
    graph.search_by_embedding = Mock(return_value=[])
    result = graph_search(graph, SearchConfig(video_id="v", before_clip=5, before_sec=171))
    assert not result["hits"] and not result["context_hits"]
    assert result["boundary_hits"][0]["speech_transcript"][0]["text"] == "the black album"
    assert "FUTURE" not in json.dumps(result)
    assert result["boundary_hits"][0]["retrieval_sources"] == ["boundary_transcript"]


@pytest.fixture
def offline_tokenizer(monkeypatch):
    monkeypatch.setattr("retrieval.bm25_index._tokenize", lambda text: text.lower().split())


def test_direct_search_also_projects_boundary():
    rec = record("n", 5, "FUTURE_TEXT", [1, 0], start=150, duration=30.13401)
    rec["metadata"].update(content())
    searcher = EpisodicSearcher.from_records([rec], embedder=Embedder())
    result = searcher.search("anything", video_id="v", before_clip=5, before_sec=171)
    assert len(result) == 1
    assert result[0].metadata["evidence_role"] == "boundary_transcript"
    assert "the black album" in result[0].text
    assert "FUTURE" not in json.dumps({"text": result[0].text, "meta": result[0].metadata})


def test_adapter_normalizes_free_text_and_rejects_reasoning_only_answer_tool():
    adapter = OGMemoryControlAdapter(graph=Mock(), embedder=Embedder(), llm=MockLLM())
    assert adapter._parse_output("<think>private</think>Milk", ctx()) == ["Milk"]
    assert adapter._parse_output("<think>private", ctx()) is None
    assert "error" in adapter._execute_tool("Answer", {"final_answer": "<think>private</think>"}, ctx())


@pytest.mark.parametrize("ok,warnings,status", [
    (True, [], "success"), (True, ["unavailable"], "degraded"), (False, [], "failed"),
])
def test_stage_status_distinguishes_degraded(ok, warnings, status):
    assert StageResult("s", "type", ok, warnings=warnings).to_dict()["status"] == status


def test_real_extracted_node_resume_uses_its_embedding_not_missing_metadata():
    from m3.pipeline.stages import _classify_extracted_node
    _, node = boundary_service()
    node.mark_completed(content(), 1)
    assert _classify_extracted_node(node, require_embeddings=True) is None
    node.extracted_content.pop("embedding")
    assert _classify_extracted_node(node, require_embeddings=True) == "missing_embedding"


def diarization_graph():
    graph = Mock()
    graph.list_raw_nodes.return_value = [
        SimpleNamespace(node_id=f"audio{i}", metadata={"video_id": "v", "clip_id": i})
        for i in range(2)
    ]
    graph.list_extracted_nodes.return_value = []
    return graph


def test_diarization_missing_token_preflight_never_calls_extractor(tmp_path, monkeypatch, caplog):
    from m3.speaker_diarization import parse_args, run_speaker_diarization
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    extract = Mock(side_effect=AssertionError("must not invoke unavailable model"))
    monkeypatch.setattr(
        "extraction.operators.builtin.speaker_diarization.SpeakerDiarizationExtractor.extract", extract,
    )
    output = tmp_path / "turns.jsonl"
    assert run_speaker_diarization(parse_args(["--output-jsonl", str(output)]), diarization_graph()) == 2
    summary = json.loads(output.with_suffix(".summary.json").read_text())
    assert summary["status"] == "failed"
    assert summary["failed_audio_nodes"] == 2 and summary["turns"] == 0
    assert summary["preflight_error"] == "missing_huggingface_token"
    extract.assert_not_called()
    assert caplog.text.count("Speaker diarization unavailable:") == 1


def test_diarization_partial_failures_are_not_reported_as_success(tmp_path, monkeypatch):
    from m3.speaker_diarization import parse_args, run_speaker_diarization
    turn = {"speaker_id": "s0", "start_sec": 0, "end_sec": 1}
    extraction = SimpleNamespace(extracted_content={"turns": [turn], "num_speakers": 1})
    monkeypatch.setattr(
        "extraction.operators.builtin.speaker_diarization.SpeakerDiarizationExtractor.extract",
        Mock(return_value=extraction),
    )
    service = Mock()
    service.persist_extraction.side_effect = [
        {"error": "backend failed"}, {
            "materialized_children": [{"child_node_id": "turn1"}],
            "extracted_node_id": "parent1",
        },
    ]
    monkeypatch.setattr("service.operator_service.OperatorService", Mock(return_value=service))
    output = tmp_path / "turns.jsonl"
    args = parse_args(["--output-jsonl", str(output), "--backend", "mock"])
    graph = diarization_graph()
    graph.get_extracted_node.side_effect = lambda node_id, ctx: SimpleNamespace(
        node_id=node_id, source_raw_id="audio1", confidence=1.0,
        extracted_content={"child_node_ids": ["turn1"], "child_count": 1},
    )
    assert run_speaker_diarization(args, graph) == 2
    summary = json.loads(output.with_suffix(".summary.json").read_text())
    assert summary["status"] == "failed"
    assert summary["failed_audio_nodes"] == 1 and summary["turns"] == 1


def test_boundary_counts_as_evidence_for_no_gain_detection(offline_tokenizer):
    graph, node = boundary_service()
    graph.search_by_embedding = Mock(return_value=[])
    adapter = OGMemoryControlAdapter(graph=graph, embedder=Embedder(), llm=MockLLM())
    task = M3TaskContext(question="Which album?", video_id="v", before_clip=5, before_sec=171)
    adapter._build_messages(ctx(), task)
    result = adapter._execute_tool("Search", {"query": "album"}, ctx())
    assert result["boundary_hits"]
    assert task.search_history[0]["boundary_node_ids"] == [node.node_id]
    assert task.search_history[0]["new_evidence_count"] == 1


def test_diarization_missing_dependency_preflight(tmp_path, monkeypatch):
    from m3.speaker_diarization import parse_args, run_speaker_diarization
    monkeypatch.setenv("HF_TOKEN", "test-placeholder")
    monkeypatch.setattr("m3.speaker_diarization.importlib.util.find_spec", lambda name: None)
    extract = Mock(side_effect=AssertionError("must not invoke unavailable model"))
    monkeypatch.setattr(
        "extraction.operators.builtin.speaker_diarization.SpeakerDiarizationExtractor.extract", extract,
    )
    output = tmp_path / "turns.jsonl"
    assert run_speaker_diarization(parse_args(["--output-jsonl", str(output)]), diarization_graph()) == 2
    summary = json.loads(output.with_suffix(".summary.json").read_text())
    assert summary["preflight_error"] == "missing_pyannote_dependency"
    extract.assert_not_called()


def test_stage_preserves_diarization_diagnostics(tmp_path, monkeypatch):
    from m3.pipeline.stages import speaker_diarization_stage
    output = tmp_path / "speaker_turns.jsonl"
    output.write_text("")
    diagnostics = {"status": "failed", "failed_audio_nodes": 2, "turns": 0}
    output.with_suffix(".summary.json").write_text(json.dumps(diagnostics))
    monkeypatch.setattr("m3.speaker_diarization.run_speaker_diarization", lambda *a, **kw: 2)
    context = SimpleNamespace(
        work_dir=tmp_path, graph_store="memory", connection_string=None,
        graph=Mock(), video_id="bedroom_03", account_id="m3-bench",
    )
    stage = SimpleNamespace(id="speaker_diarization", type="speaker_diarization", params={})
    result = speaker_diarization_stage(context, stage)
    assert result.to_dict()["status"] == "failed"
    assert not result.ok
    assert result.counts["failed_audio_nodes"] == 2
    assert result.outputs["diarization_diagnostics"] == diagnostics
    assert result.errors[0]["type"] == "AudioExtractionError"


def test_answer_tool_accepts_only_normalized_final_text():
    adapter = OGMemoryControlAdapter(graph=Mock(), embedder=Embedder(), llm=MockLLM())
    adapter._build_messages(ctx(), M3TaskContext(question="Which album?", video_id="v"))
    result = adapter._execute_tool("Answer", {"final_answer": "<think>private</think>Black album."}, ctx())
    assert result["status"] == "answered"
    assert adapter._parse_output("ignored", ctx()) == ["Black album."]


def test_index_preserves_structured_asr_with_prebuilt_text(tmp_path):
    from m3.index_episodic import build_index
    segment = {"video_id": "v", "clip_id": 5, "start_sec": 10.9,
               "end_sec": 18.6, "text": "the black album"}
    manifest = tmp_path / "asr.jsonl"
    manifest.write_text(json.dumps(segment) + "\n")
    rec = {"video_id": "v", "clip_id": 5, "source_start_sec": 150,
           "source_duration_sec": 30, "description": "FUTURE_SUMMARY"}
    result = build_index([rec], embedder=Embedder(), asr_manifest=manifest)[0]
    assert result["metadata"]["speech_transcript"][0]["text"] == "the black album"
    projected = boundary_transcript(result["metadata"], 171)
    assert projected["speech_transcript"][0]["text"] == "the black album"
    assert "FUTURE" not in json.dumps(projected)


def test_index_uses_manifest_embedded_transcript_without_asr_file():
    from m3.index_episodic import build_index
    rec = {"video_id": "v", "clip_id": 0, "description": "A room.",
           "speech_transcript": [{"start_sec": 0, "end_sec": 1, "text": "spoken fact"}],
           "characters": [{"name": "Person"}], "events": ["event"], "entities": ["bag"]}
    result = build_index([rec], embedder=Embedder())[0]
    assert "spoken fact" in result["text"]
    for key in ("speech_transcript", "characters", "events", "entities"):
        assert result["metadata"][key] == rec[key]


@pytest.mark.parametrize("cached_last", [False, True])
def test_fatal_diarization_failure_stops_uncached_audio(tmp_path, monkeypatch, cached_last):
    from m3.speaker_diarization import parse_args, run_speaker_diarization
    extract = Mock(return_value=SimpleNamespace(
        extracted_content={"backend_unavailable": True},
        error="403 GatedRepo: pyannote/speaker-diarization-community-1",
    ))
    monkeypatch.setattr(
        "extraction.operators.builtin.speaker_diarization.SpeakerDiarizationExtractor.extract",
        extract,
    )
    persist = Mock()
    monkeypatch.setattr("service.operator_service.OperatorService", persist)
    output = tmp_path / "turns.jsonl"
    args = parse_args(["--output-jsonl", str(output), "--backend", "mock"])
    graph = diarization_graph()
    if cached_last:
        graph.list_extracted_nodes.return_value = [SimpleNamespace(
            source_raw_id="audio1", node_id="cached-turn", extraction_status="completed",
            extracted_content={"speaker_id": "S1", "start_sec": 0, "end_sec": 1},
        )]
    assert run_speaker_diarization(args, graph) == 2
    extract.assert_called_once()
    persist.return_value.persist_extraction.assert_not_called()
    summary = json.loads(output.with_suffix(".summary.json").read_text())
    # An existence-only legacy turn must not bypass dependency validation.
    assert summary["failed_audio_nodes"] == 2
    assert summary["resumed_turns"] == 0
    assert "community-1" in summary["preflight_error"]


def test_face_stage_zero_embeddings_is_degraded(tmp_path, monkeypatch):
    from m3.pipeline.stages import face_clustering_stage
    def run_clustering(args, **kwargs):
        assert args.account_id == "experiment-account"
        assert args.cache_dir == str(tmp_path.parent / "_cache" / "face_clustering")
        return 0
    monkeypatch.setattr("m3.face_clustering.run_face_clustering", run_clustering)
    context = SimpleNamespace(work_dir=tmp_path, graph_store="memory",
                              connection_string=None, graph=Mock(),
                              account_id="experiment-account")
    stage = SimpleNamespace(id="face_clustering", type="face_clustering", params={})
    result = face_clustering_stage(context, stage)
    assert result.to_dict()["status"] == "degraded"
    assert "Haar" in result.warnings[0]


@pytest.mark.parametrize("nested", [False, True])
def test_http_face_embeddings_survive_materialization(nested):
    from extraction.operators.backends.http_backend import HTTPBackend
    from extraction.operators.materialize import _face_payload
    backend = object.__new__(HTTPBackend)
    metadata = {"embedding": [1.0, 0.0], "detector": "test"}
    item = {"label": "face", "confidence": 0.9, "bbox": [0, 0, 1, 1],
            **({"metadata": metadata} if nested else metadata)}
    detection = backend._parse_response({"detections": [item]})[0]
    payload = _face_payload({"metadata": detection.metadata, "bbox": detection.bbox,
                             "confidence": detection.confidence}, "frame", {})
    assert payload["embedding"] == [1.0, 0.0]


@pytest.mark.parametrize("message,fatal", [
    ("403 GatedRepo: pyannote/speaker-diarization-community-1", True),
    ("401 Unauthorized", True),
    ("temporary connection timeout", False),
])
def test_pyannote_access_failure_is_fatal_but_network_timeout_is_not(monkeypatch, message, fatal):
    import sys
    from extraction.operators.builtin.diarization_backends import (
        PyannoteDiarizationBackend, DiarizationBackendError,
    )
    pipeline = Mock()
    pipeline.from_pretrained.side_effect = RuntimeError(message)
    monkeypatch.setitem(sys.modules, "torch", Mock())
    monkeypatch.setitem(sys.modules, "pyannote", Mock())
    monkeypatch.setitem(sys.modules, "pyannote.audio", SimpleNamespace(Pipeline=pipeline))
    backend = PyannoteDiarizationBackend(hf_token="test-placeholder")
    with pytest.raises(DiarizationBackendError) as error:
        backend._load()
    assert error.value.fatal is fatal
    assert message in str(error.value)


@pytest.mark.parametrize("vector", [[float("nan")], [float("inf")], [0, 0]])
def test_face_clustering_rejects_invalid_embeddings(vector):
    from m3.face_clustering import _face_records_from_graph
    graph = Mock()
    graph.list_extracted_nodes.return_value = [
        SimpleNamespace(extracted_content={"embedding": vector})
    ]
    assert _face_records_from_graph(graph, ctx()) == []


def test_index_prefers_diarized_transcript_over_earlier_asr_export(tmp_path):
    from m3.index_episodic import build_index
    segment = {"start_sec": 0, "end_sec": 1, "text": "hello", "speaker_id": "S1"}
    rec = {"video_id": "v", "clip_id": 0, "description": "A room.",
           "speech_transcript": [segment]}
    manifest = tmp_path / "asr.jsonl"
    manifest.write_text(json.dumps({**segment, "speaker_id": None,
                                    "video_id": "v", "clip_id": 0}))
    result = build_index([rec], embedder=Embedder(), asr_manifest=manifest)[0]
    assert result["metadata"]["speech_transcript"] == [segment]


def test_spectral_voice_clusters_are_not_reported_as_verified_identity(tmp_path, monkeypatch):
    from m3.pipeline.stages import voice_clustering_stage
    (tmp_path / "voice_clusters.jsonl").write_text(json.dumps({
        "anchor_node_id": "one-cluster", "embedding_kind": "spectral",
    }) + "\n")
    monkeypatch.setattr("m3.voice_clustering.run_voice_clustering", lambda *a, **kw: 0)
    context = SimpleNamespace(work_dir=tmp_path, graph_store="memory",
                              connection_string=None, graph=Mock(), account_id="acct")
    stage = SimpleNamespace(id="voice_clustering", type="voice_clustering", params={})
    result = voice_clustering_stage(context, stage)
    assert result.to_dict()["status"] == "degraded"
    assert "verified speakers" in result.warnings[0]
