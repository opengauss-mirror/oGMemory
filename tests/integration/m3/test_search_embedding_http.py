"""Legacy M3 retrieval through the OpenAI embedding HTTP transport."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from jsonschema import ValidationError, validate

from m3.search import EpisodicSearcher
from providers.embedder.openai_embedder import OpenAIEmbedder


@pytest.fixture
def embedding_endpoint():
    exchanges = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            try:
                validate(request["input"], {"type": "array", "minItems": 1,
                         "items": {"type": "string", "minLength": 1}})
            except ValidationError:
                status = 400
                response = {"error": {"message": "input must contain nonempty strings",
                                     "type": "invalid_request_error", "param": "input"}}
            else:
                status = 200
                response = {"object": "list", "model": request["model"],
                            "data": [{"object": "embedding", "index": i,
                                      "embedding": [1.0, 0.0, 0.0, 0.0]}
                                     for i, _ in enumerate(request["input"])],
                            "usage": {"prompt_tokens": 4, "total_tokens": 4}}
            exchanges.append({"path": self.path, "request": request,
                              "status": status, "response": response})
            print("EMBEDDING_HTTP " + json.dumps(exchanges[-1]), flush=True)
            body = json.dumps(response).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", exchanges
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("source", ["empty", "audit", "embedded"])
@pytest.mark.parametrize("published_dimension", [True, False])
def test_legacy_search_uses_valid_embedding_requests(
    tmp_path, embedding_endpoint, source, published_dimension,
):
    url, exchanges = embedding_endpoint
    provider = OpenAIEmbedder(base_url=url, api_key="local-contract", model="contract-model", dimension=4)

    class ProtocolEmbedder:
        def embed_texts(self, texts):
            return provider.embed_texts(texts)

    embedder = provider if published_dimension else ProtocolEmbedder()
    record = {"video_id": "coffee", "clip_id": 0, "text": "A mug of coffee is on the table.",
              "source_extracted_node_id": "coffee-0"}
    if source == "embedded":
        record["embedding"] = [1.0, 0.0, 0.0, 0.0]
    path = tmp_path / "episodic_index.jsonl"
    path.write_text("" if source == "empty" else json.dumps(record) + "\n", encoding="utf-8")
    print("SEARCH_REQUEST " + json.dumps({"source": source, "published_dimension": published_dimension,
                                         "records": [] if source == "empty" else [record]}))
    try:
        searcher = EpisodicSearcher.from_index_path(path, embedder=embedder)
        hits = searcher.search("coffee", video_id="coffee", top_k=1)
        print("SEARCH_RESPONSE " + json.dumps([hit.to_dict() for hit in hits]))
        assert [hit.source_extracted_node_id for hit in hits] == ([] if source == "empty" else ["coffee-0"])
        expected_calls = 1 if published_dimension or source == "embedded" else 2
        assert len(exchanges) == expected_calls
        assert all(item["path"] == "/v1/embeddings" and item["status"] == 200 for item in exchanges)
        assert exchanges[-1]["request"]["input"] == ["coffee"]
    finally:
        provider._client.close()
