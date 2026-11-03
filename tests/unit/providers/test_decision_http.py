import io
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from unittest.mock import Mock

import pytest

import core.operation_deadline as deadline
from core.errors import OperationDeadlineExceededError
from providers.decision import DecisionConfig, DecisionError, JevDecisionClient

QUESTION = {"q": {"type": "noul", "instructions": "Relevant?"}}
RESPONSE = b'{"answers":{"q":{"type":"noul","noul":0.8}}}'


def client(url="http://example.invalid", **kwargs):
    return JevDecisionClient(DecisionConfig(base_url=url, model="test", **kwargs))


@pytest.mark.parametrize("remaining,configured,expected", [(1, 30, 1), (10, 2, 2), (None, 7, 7)])
def test_timeout_is_bounded(monkeypatch, remaining, configured, expected):
    monkeypatch.setattr(deadline, "monotonic", lambda: 100.0)
    send = Mock(return_value=io.BytesIO(RESPONSE))
    monkeypatch.setattr("providers.decision.client._open_once", send)
    with deadline.operation_deadline(remaining):
        client(timeout_seconds=configured).evaluate({}, QUESTION)
    assert send.call_args.kwargs["timeout"] == expected


@pytest.mark.parametrize("expired_before_request", [False, True])
def test_deadline_prevents_or_classifies_network_failure(monkeypatch, expired_before_request):
    clock = [100.0]
    monkeypatch.setattr(deadline, "monotonic", lambda: clock[0])

    def fail(*args, **kwargs):
        clock[0] = 102.0
        raise TimeoutError("synthetic timeout")

    send = Mock(side_effect=fail)
    monkeypatch.setattr("providers.decision.client._open_once", send)
    with deadline.operation_deadline(1), pytest.raises(OperationDeadlineExceededError):
        if expired_before_request:
            clock[0] = 102.0
        client().evaluate({}, QUESTION)
    assert send.call_count == (0 if expired_before_request else 1)


@pytest.fixture
def serve():
    servers = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.server.received.append((self.path, self.headers.get("Authorization")))
            location = self.server.location if self.path == "/v1/systemone" else None
            self.send_response(self.server.redirect_status if location else 200)
            if location:
                self.send_header("Location", location)
            self.send_header("Content-Length", str(0 if location else len(RESPONSE)))
            self.end_headers()
            if not location:
                self.wfile.write(RESPONSE)

        def do_POST(self):
            self.do_GET()

        def log_message(self, *args):
            pass

    def start():
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.received, server.location = [], None
        thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        thread.start()
        servers.append((server, thread))
        return server

    yield start
    for server, thread in reversed(servers):
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


@pytest.mark.parametrize(
    "status,cross_origin", [(code, True) for code in (301, 302, 303, 307, 308)] + [(303, False)]
)
def test_redirect_is_not_replayed_or_forwarded(serve, status, cross_origin):
    source = serve()
    target = serve() if cross_origin else source
    source.location = f"http://127.0.0.1:{target.server_port}/target"
    source.redirect_status = status
    url = f"http://127.0.0.1:{source.server_port}"
    with pytest.raises(DecisionError) as caught:
        client(url, api_key="synthetic-key").evaluate({}, QUESTION)
    assert source.received == [("/v1/systemone", "Bearer synthetic-key")]
    if cross_origin:
        assert target.received == []
    assert caught.value.status_code == status and "synthetic-key" not in str(caught.value)
