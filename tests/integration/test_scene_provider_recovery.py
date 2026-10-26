"""Provider failure recovery through the scene worker and decision engine."""

import httpx
import pytest
from openai import OpenAI

from providers.llm.openai_llm import OpenAILLM
from providers.token_tracker import TokenTracker
from scene.decision import SceneDecisionEngine
from scene.jobs import InMemorySceneJobStore, SceneWorker
from scene.sql_store import SQLSceneJobStore
from tests.integration.test_scene_sql_persistence import scene_sql as scene_sql
from tests.unit.scene.test_decision_preflight import JSONModel
from tests.unit.scene.test_engine_refresh import source_case


@pytest.mark.parametrize("durable", [False, True], ids=["memory", "sql"])
@pytest.mark.parametrize("status", [500, 503, 429])
def test_provider_failure_recovers_to_published_scene(request, durable, status):
    pool = request.getfixturevalue("scene_sql")[0] if durable else None
    jobs = SQLSceneJobStore(pool) if durable else InMemorySceneJobStore()
    source, nodes, _, proposal = source_case()
    uris = [node.uri for node in nodes]
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(status, json={
            "error": {"message": "provider request req_1429 temporarily unavailable"},
        })

    with OpenAI(api_key="test", base_url="http://provider.test/v1", max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(respond))) as client:
        provider = OpenAILLM.__new__(OpenAILLM)
        provider._client = client
        provider.token_tracker = TokenTracker(model="test")

        class RecoveringModel(JSONModel):
            def complete_json(self, prompt, schema):
                if not calls:
                    provider._create_completion(model="test", messages=[{"role": "user", "content": prompt}])
                return super().complete_json(prompt, schema)

        model = RecoveringModel({"status": "READY"}, proposal.model_dump(), {"decision": "ACCEPT"})
        engine = SceneDecisionEngine(model)
        worker = SceneWorker(jobs, engine, source, retry_delay_seconds=0)
        jobs.enqueue("acct", "user:alice", uris, token_count=3)
        first = worker.process_next("worker")
        assert len(calls) == 1
        assert first.job.failure_count == 1
        if status == 429:
            assert first.job.status == "FAILED"
            assert first.halted and worker.halted
            assert jobs.summary("acct", "user:alice")["last_job"]["reason"] == "provider rate limit (429)"
            jobs.enqueue("acct", "user:alice", uris, token_count=3)
            worker = SceneWorker(jobs, engine, source)
        else:
            assert (first.job.status, first.halted) == ("PENDING", False)
        completed = worker.process_next("recovered-worker")
        assert completed.job.status == "SUCCEEDED"
        assert completed.decision.action == "CREATE"
        assert source.repository.snapshot.revision == 1
        assert source.repository.snapshot.scenes[0].cluster.members
        assert len(model.requests) == 3
        assert len(calls) == 1
        if status != 429:
            assert completed.job.job_id == first.job.job_id
            assert (completed.job.attempt, completed.job.failure_count) == (2, 1)
