import json
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def enabled(value):
    return str(value).lower().strip() in {"1", "true", "yes", "on"}


class Client:
    def __init__(self):
        location = Path(os.environ.get("OG_MEMORY_ZCODE_CONFIG") or
                        Path.home() / ".ogmemory/zcode-plugin.json")
        config = json.loads(location.read_text(encoding="utf-8-sig")) if location.exists() else {}

        def setting(key, name, default=None):
            return os.environ.get(name, config.get(key, default))

        self.url = setting("url", "OG_MEMORY_URL", "http://127.0.0.1:8090").rstrip("/")
        url = urlsplit(self.url)
        if (not url.hostname or url.username or url.password or url.query or url.fragment
                or url.scheme not in {"http", "https"}
                or (url.scheme == "http" and url.hostname not in {"localhost", "127.0.0.1", "::1"})):
            raise ValueError("oGMemory requires HTTPS outside loopback")
        self.identity = {
            "accountId": setting("account_id", "OG_MEMORY_ACCOUNT_ID", "acct-demo"),
            "userId": setting("user_id", "OG_MEMORY_USER_ID", "u-zcode"),
            "agentId": setting("agent_id", "OG_MEMORY_AGENT_ID", "zcode"),
        }
        self.key = setting("api_key", "OG_AUTH_API_KEY", "")
        self.trace = enabled(setting("auto_trace", "OGMEM_AUTO_TRACE", True))
        self.trace_content = enabled(setting("trace_capture_content", "OGMEM_TRACE_CAPTURE_CONTENT", True))
        self.capture = enabled(setting("auto_capture", "OGMEM_AUTO_CAPTURE", True))
        self.recall = enabled(setting("auto_recall", "OGMEM_AUTO_RECALL", True))
        self.force_extract = enabled(setting("force_extract_on_stop", "OGMEM_FORCE_EXTRACT_ON_STOP", False))

    def request(self, path, body=None, timeout=3):
        headers = {"Content-Type": "application/json", "Accept": "application/json",
                   "X-Account-ID": self.identity["accountId"], "X-User-ID": self.identity["userId"],
                   "X-Agent-ID": self.identity["agentId"]}
        if self.key:
            headers["X-API-Key"] = self.key
        encoded = json.dumps(body, ensure_ascii=False).encode() if body is not None else None
        req = Request(self.url + "/api/v1/" + path, data=encoded, headers=headers)
        try:
            response = build_opener(NoRedirect()).open(req, timeout=timeout)
        except HTTPError as exc:
            response = exc
        with response:
            raw = response.read(1024 * 1024)
            try:
                payload = json.loads(raw)
            except ValueError:
                payload = {}
            return response.code, payload if isinstance(payload, dict) else {}, response.headers.get("Retry-After")
