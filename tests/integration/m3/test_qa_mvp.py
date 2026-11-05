"""Tests for M3 episodic index/search/QA MVP."""

from __future__ import annotations

import json
from pathlib import Path

from m3.index_episodic import (
    build_index,
    build_retrieval_text,
    index_record_from_episodic,
)
from m3.qa import answer_question, iter_annotation_questions, parse_args as parse_qa_args, run_qa
from m3.search import EpisodicSearcher
from providers.embedder import MockEmbedder
from providers.llm import MockLLM


def _episodic_record(video_id="living_room_06", clip_id=0, description="A mug of coffee is on the table."):
    return {
        "video_id": video_id,
        "clip_id": clip_id,
        "video_raw_node_id": f"raw-{video_id}-{clip_id}",
        "extracted_node_id": f"ext-{video_id}-{clip_id}",
        "description": description,
        "characters": [{"name": "robot", "role": "observer"}],
        "events": [{"type": "observation", "summary": "The camera sees the table."}],
        "entities": [{"name": "coffee", "type": "drink"}],
        "frame_count": 2,
        "has_audio": False,
    }


class TestIndexEpisodic:
    def test_build_retrieval_text_includes_structured_fields(self):
        text = build_retrieval_text(_episodic_record())
        assert "Description:" in text
        assert "mug of coffee" in text
        assert "Events:" in text
        assert "Entities:" in text
        assert "Characters:" in text

    def test_transcript_block_first_when_asr_segments_present(self):
        """Default transcript-first ordering: speech must lead retrieval text.

        Regression: pre-quickwin, the transcript was appended LAST so the
        embedder weighted the (often paraphrased) LLM description 5-10×
        higher than the verbatim speech. Speech-anchored questions like
        "put them in an album" then ranked below the relevant clip.
        """
        asr_segments = [
            {"start_sec": 0.85, "end_sec": 2.13, "text": "put our photos in the album"},
            {"start_sec": 4.78, "end_sec": 7.12, "text": "Okay, my lovely bag",
             "speaker_id": "S1"},
        ]
        text = build_retrieval_text(
            _episodic_record(description="Two women talk about their photos."),
            asr_segments=asr_segments,
        )
        # Transcript block leads; description comes after.
        assert text.startswith("Speech (verbatim):")
        assert text.find("put our photos in the album") < text.find("Description:")
        # Speaker labels surface when diarization ran.
        assert "S1: Okay, my lovely bag" in text

    def test_no_transcript_falls_back_to_description(self):
        """Vision-only path keeps Description as the first section."""
        text = build_retrieval_text(_episodic_record())
        assert text.startswith("Description:")

    def test_transcript_first_false_restores_legacy_ordering(self):
        """Opt-out path: pass transcript_first=False to keep the pre-quickwin
        description-first ordering (useful for A/B comparisons)."""
        asr_segments = [
            {"start_sec": 0.85, "end_sec": 2.13, "text": "put our photos in the album"},
        ]
        text = build_retrieval_text(
            _episodic_record(description="Two women talk about their photos."),
            asr_segments=asr_segments,
            transcript_first=False,
        )
        assert text.startswith("Description:")
        assert "put our photos in the album" in text

    def test_build_index_persists_transcript_first_text(self, tmp_path):
        """build_index must keep the transcript-first text on the record.

        The embedding is computed from the retrieval text; this test
        pins that the JSONL record's ``text`` field reflects the new
        ordering, so the embedder and downstream consumers see it.
        """
        embedder = MockEmbedder(dimension=8)
        asr_segments = [
            {"start_sec": 0.85, "end_sec": 2.13, "text": "put our photos in the album"},
        ]
        rec = _episodic_record(
            video_id="v", clip_id=7,
            description="Two women talk about their photos.",
        )
        # No ASR manifest → no transcript → description leads.
        index_no_asr = build_index([rec], embedder=embedder, asr_manifest=None)
        assert index_no_asr[0]["text"].startswith("Description:")

        # Build a voice_asr.jsonl manifest for (v, 7) and rebuild.
        manifest = tmp_path / "voice_asr.jsonl"
        manifest.write_text(json.dumps({
            "video_id": "v", "clip_id": 7,
            "start_sec": asr_segments[0]["start_sec"],
            "end_sec": asr_segments[0]["end_sec"],
            "text": asr_segments[0]["text"],
        }) + "\n", encoding="utf-8")
        index_with_asr = build_index(
            [rec], embedder=embedder, asr_manifest=manifest,
        )
        assert index_with_asr[0]["text"].startswith("Speech (verbatim):")
        assert "put our photos in the album" in index_with_asr[0]["text"]

    def test_index_record_shape(self):
        rec = index_record_from_episodic(_episodic_record(), [0.1, 0.2], include_embedding=True)
        assert rec["video_id"] == "living_room_06"
        assert rec["clip_id"] == 0
        assert rec["embedding"] == [0.1, 0.2]
        assert rec["uri"].startswith("m3://living_room_06")
        assert rec["source"] == "episodic_memory_generator"

    def test_build_index_skips_empty_records(self):
        embedder = MockEmbedder(dimension=8)
        records = [_episodic_record(), {"video_id": "v", "clip_id": 1}]
        index = build_index(records, embedder=embedder)
        # The sparse second record still has Source text, so both are kept.
        assert len(index) == 2
        assert all(len(r["embedding"]) == 8 for r in index)


class TestSearch:
    def test_search_filters_by_video_and_before_clip(self):
        embedder = MockEmbedder(dimension=8)
        records = [
            _episodic_record("living_room_06", 0, "Coffee is on the table."),
            _episodic_record("living_room_06", 5, "A future clip shows tea."),
            _episodic_record("kitchen_22", 0, "An apple is on the counter."),
        ]
        # The EpisodicSearcher no longer takes a raw records list —
        # it wraps a VectorIndex. ``from_records`` builds an
        # in-memory index inline so unit tests don't need to stand
        # up a real vector store. The records shape is unchanged
        # from the legacy JSONL output (same key names), so this
        # migration is one-line per call site.
        searcher = EpisodicSearcher.from_records(build_index(records, embedder=embedder), embedder=embedder)
        hits = searcher.search(
            "What drink is on the table?",
            video_id="living_room_06",
            before_clip=0,
            top_k=5,
        )
        assert len(hits) == 1
        assert hits[0].video_id == "living_room_06"
        assert hits[0].clip_id == 0
        assert "Coffee" in hits[0].text

    def test_include_neighbors_adds_adjacent_clip(self):
        embedder = MockEmbedder(dimension=8)
        records = [
            _episodic_record("v", 0, "The robot enters the kitchen."),
            _episodic_record("v", 1, "The robot sees a red apple."),
        ]
        searcher = EpisodicSearcher.from_records(build_index(records, embedder=embedder), embedder=embedder)
        hits = searcher.search("red apple", video_id="v", top_k=2, include_neighbors=1)
        assert {h.clip_id for h in hits} == {0, 1}


class TestQAMVP:
    def test_iter_annotation_questions_normalizes_fixture(self):
        annotations = {
            "vid": {
                "video_path": "data/videos/robot/vid.mp4",
                "qa_list": [{"question": "Q?", "answer": "A", "type": ["T"], "before_clip": 1}],
            }
        }
        qs = list(iter_annotation_questions(annotations))
        assert qs[0]["video_id"] == "vid"
        assert qs[0]["question"] == "Q?"
        assert qs[0]["answer"] == "A"
        assert qs[0]["before_clip"] == 1

    def test_answer_question_uses_llm_schema(self):
        llm = MockLLM()
        llm._fixtures = {"coffee": {"answer": "Coffee.", "reasoning": "Evidence mentions coffee.", "confidence": 0.9}}
        evidence = []
        result = answer_question(
            llm=llm,
            question="What drink is on the table? coffee",
            evidence=evidence,
            video_id="living_room_06",
        )
        assert result["answer"] == "Coffee."
        assert result["confidence"] == 0.9

    def test_run_qa_writes_jsonl(self, tmp_path: Path, monkeypatch):
        annotations = {
            "living_room_06": {
                "video_path": "data/videos/robot/living_room_06.mp4",
                "qa_list": [{
                    "question": "What drink is on the table? coffee",
                    "answer": "Coffee.",
                    "question_id": "q1",
                    "type": ["Object Recognition"],
                    "before_clip": 0,
                }],
            }
        }
        ann_path = tmp_path / "annotations.json"
        ann_path.write_text(json.dumps(annotations), encoding="utf-8")

        embedder = MockEmbedder(dimension=8)
        index = build_index([_episodic_record()], embedder=embedder)
        searcher = EpisodicSearcher.from_records(index, embedder=embedder)
        monkeypatch.setattr("m3.qa._build_qa_vector_index", lambda args: searcher._index)

        out = tmp_path / "qa.jsonl"
        args = parse_qa_args([
            "--annotations", str(ann_path),
            "--output-jsonl", str(out),
            "--embedder", "mock",
            "--embedding-dimension", "8",
            "--llm", "mock",
        ])
        rc = run_qa(args)
        assert rc == 0
        rows = [json.loads(line) for line in out.read_text().splitlines() if line]
        assert len(rows) == 1
        assert rows[0]["question_id"] == "q1"
        assert rows[0]["predicted_answer"]
        assert rows[0]["evidence"]
