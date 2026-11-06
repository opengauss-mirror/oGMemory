import csv
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from types import ModuleType

import pytest
import requests


def test_deepagents_entry_uses_per_question_sessions_and_blocks_qa_writes(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "e2e"))
    from run_deepagents_locomo import main

    dataset = tmp_path / "data.json"
    dataset.write_text(json.dumps([
        {
            "sample_id": "fixture",
            "sessions": [{"session_id": "original", "message": "Source dialogue."}],
            "questions": [
                {"question": "First question?", "answer": "Private reference 1"},
                {"question": "Second question?", "answer": "Private reference 2"},
            ],
        },
    ]))
    output = tmp_path / "result"
    calls = []
    lifecycle = []
    factory_module = ModuleType("fixture_deepagents_host")

    class Host:
        def __init__(self, args):
            self.args = args

        def prepare(self):
            lifecycle.append("prepare")

        def invoke(self, prompt, key):
            calls.append((prompt, key))
            with requests.Session() as client:
                client.trust_env = False
                response = client.post(
                    self.args.ogmemory_url + "/api/v1/after_turn",
                    json={"sessionId": key},
                    timeout=2,
                )
            assert response.json() == {"ok": True, "readOnly": True}
            return "Fixture answer", {}

        def close(self):
            lifecycle.append("close")

    factory_module.create = Host
    monkeypatch.setitem(sys.modules, factory_module.__name__, factory_module)
    assert main([
        "qa", str(dataset), "--sample", "0",
        "--ogmemory-url", "http://127.0.0.1:1", "--parallel", "1",
        "--output-dir", str(output), "--host-factory", "fixture_deepagents_host:create",
    ]) == 0

    assert lifecycle == ["prepare", "close"]
    assert [prompt for prompt, _ in calls] == [
        "Answer the question directly: First question?",
        "Answer the question directly: Second question?",
    ]
    assert len({key for _, key in calls}) == 2
    with (output / "qa_results.csv").open() as stream:
        assert [row["response"] for row in csv.DictReader(stream)] == ["Fixture answer"] * 2
    observations = json.loads((output / "qa-write-observations.json").read_text())
    assert observations == {"accepted": [], "intercepted": ["/api/v1/after_turn"] * 2}
    manifest = json.loads((output / "qa-manifest.json").read_text())
    assert manifest["adapter"]["host_factory"] == "fixture_deepagents_host:create"


def test_verified_ingest_saves_host_readiness_evidence(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "e2e"))
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    from run_deepagents_locomo import main

    dataset = tmp_path / "data.json"
    dataset.write_text(json.dumps([{
        "sample_id": "fixture",
        "sessions": [{"session_id": "source", "message": "Original dialogue."}],
        "questions": [{"question": "What happened?", "answer": "Something."}],
    }]))
    output = tmp_path / "result"
    settled = []
    requests_seen = []

    class MemoryService(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests_seen.append((self.path, payload["sessionId"]))
            if self.path == "/api/v1/after_turn":
                result = {"ok": True, "status": "accumulating"}
            else:
                assert self.path == "/api/v1/call/wait_until_idle"
                result = {"ok": True, "idle": True, "outbox": {"pending": 0}}
            body = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    class Host:
        def __init__(self, args):
            self.args = args

        def prepare(self):
            pass

        def invoke(self, prompt, key):
            response = requests.post(
                self.args.ogmemory_url + "/api/v1/after_turn",
                json={
                    "sessionId": "native-source",
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=2,
            )
            response.raise_for_status()
            return "Received", {}

        def settle(self, sid):
            settled.append(sid)
            return {"session_id": sid, "recall_verified": True}

        def close(self):
            pass

    factory_module = ModuleType("fixture_deepagents_ingest_host")
    factory_module.create = Host
    factory_module.alternate = Host
    monkeypatch.setitem(sys.modules, factory_module.__name__, factory_module)
    with ThreadingHTTPServer(("127.0.0.1", 0), MemoryService) as server:
        thread = Thread(target=server.serve_forever)
        thread.start()
        try:
            args = [
                "ingest", str(dataset), "--all-samples", "--verify-writes",
                "--ogmemory-url", f"http://127.0.0.1:{server.server_port}",
                "--ingest-record", str(tmp_path / "ingest-record.json"),
                "--output-dir", str(output), "--run-id", "same-run",
                "--host-factory",
            ]
            assert main([*args, "fixture_deepagents_ingest_host:create"]) == 0
            with pytest.raises(RuntimeError, match="Ingest adapter changed"):
                main([*args, "fixture_deepagents_ingest_host:alternate"])
        finally:
            server.shutdown()
            thread.join()

    assert requests_seen == [
        ("/api/v1/after_turn", "native-source"),
        ("/api/v1/call/wait_until_idle", "native-source"),
    ]
    assert settled == ["native-source"]
    evidence = json.loads((output / "source-completion.json").read_text())
    assert evidence == {"sources": [{"session_id": "native-source", "recall_verified": True}]}
