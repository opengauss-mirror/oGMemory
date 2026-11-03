"""Vector-only retrieval with an empty NLP resource cache."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("segments", [None, [], [{"text": " \t\n"}]])
def test_vector_evidence_preserves_scores_and_provenance_offline(tmp_path, segments):
    result = subprocess.run(
        [sys.executable, "-c", """
import json
import sys
import nltk
from tests.unit.m3.test_transcript_rerank import _make_hit
from m3.transcript_rerank import rerank_with_transcript_bm25

nltk.data.path = [sys.argv[1]]
nltk.download = lambda *args, **kwargs: False
segments = json.loads(sys.argv[2])
hits = [_make_hit(score=0.9, clip_id=10, transcript_segments=segments),
        _make_hit(score=0.7, clip_id=20, transcript_segments=segments)]
result = rerank_with_transcript_bm25("photos album", hits)
assert result == hits
print(json.dumps([[hit.video_id, hit.clip_id, hit.score] for hit in result]))
""", str(tmp_path), json.dumps(segments)],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [["vid", 10, 0.9], ["vid", 20, 0.7]]
