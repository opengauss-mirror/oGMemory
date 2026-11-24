"""Exercise document resources on an oGMemory server backed by real RAGFlow."""

import json
import os
import sys
from time import monotonic, sleep
from uuid import UUID, uuid4

import httpx

DOCUMENTS = "/api/v1/resources/documents"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def request(
    client: httpx.Client, method: str, path: str, status: int, stage: str, **kwargs
) -> httpx.Response:
    try:
        response = client.request(method, path, **kwargs)
    except httpx.HTTPError:
        raise RuntimeError(f"{stage}: request failed") from None
    require(
        response.status_code == status, f"{stage}: HTTP {response.status_code}, expected {status}"
    )
    return response


def check_document(client: httpx.Client, path: str, uri: str, content: bytes, stage: str) -> dict:
    started = monotonic()
    deadline = started + 120
    while monotonic() < deadline:
        status = request(client, "GET", path, 200, f"{stage} status").json()
        state = status.get("index_state")
        if state == "ready":
            break
        require(state != "failed", f"{stage}: indexing failed")
        sleep(2)
    else:
        raise RuntimeError(f"{stage}: indexing timed out")
    chunks = status.get("chunk_count")
    require(type(chunks) is int and chunks > 0, f"{stage}: missing indexed chunks")
    download = request(client, "GET", f"{path}/content", 200, f"{stage} download")
    require(download.content == content, f"{stage}: downloaded bytes differ")

    query = content.decode("utf-8").strip()
    result = request(
        client,
        "POST",
        f"{DOCUMENTS}/search",
        200,
        f"{stage} search",
        json={"query": query, "limit": 10},
    ).json()
    hits = result.get("hits")
    require(
        isinstance(hits, list)
        and any(
            hit.get("uri") == uri
            and isinstance(hit.get("content"), str)
            and query in hit["content"]
            and isinstance(hit.get("chunk_id"), str)
            and bool(hit["chunk_id"])
            and type(hit.get("rank")) is int
            and hit["rank"] > 0
            for hit in hits
            if isinstance(hit, dict)
        ),
        f"{stage}: expected document was absent from search results",
    )
    return {
        "index_state": state,
        "chunk_count": chunks,
        "verified_seconds": round(monotonic() - started, 2),
    }


def run(client: httpx.Client) -> None:
    marker = uuid4().hex
    original = f"OrbitGate acceptance {marker} requires two reviewers.\n".encode()
    replacement = f"OrbitGate acceptance {marker} requires three reviewers.\n".encode()
    started = monotonic()
    created = request(
        client,
        "POST",
        DOCUMENTS,
        201,
        "upload",
        files={"file": (f"smoke-{marker}.txt", original, "text/plain")},
    ).json()
    resource_id = str(UUID(created["id"]))
    path = f"{DOCUMENTS}/{resource_id}"
    checked = False
    try:
        uri = created["uri"]
        require(
            uri.endswith(f"/resources/doc-{UUID(resource_id).hex}"),
            "upload: unexpected resource URI",
        )
        initial = check_document(client, path, uri, original, "initial")
        updated = request(
            client,
            "PUT",
            path,
            200,
            "replacement",
            files={"file": (f"smoke-{marker}-replacement.txt", replacement, "text/plain")},
        ).json()
        require(
            updated.get("id") == resource_id and updated.get("uri") == uri,
            "replacement: resource identity changed",
        )
        replaced = check_document(client, path, uri, replacement, "replacement")
        checked = True
    finally:
        try:
            request(client, "DELETE", path, 204, "cleanup")
        except RuntimeError as exc:
            if checked:
                raise
            print(f"{exc}; resource_id={resource_id}", file=sys.stderr)
    request(client, "DELETE", path, 204, "repeat cleanup")
    print(
        json.dumps(
            {
                "resource_id": resource_id,
                "uploaded": initial,
                "replaced": replaced,
                "delete_status": 204,
                "repeat_delete_status": 204,
                "total_seconds": round(monotonic() - started, 2),
            }
        )
    )


def main() -> int:
    base_url = os.getenv("OGMEM_BASE_URL")
    api_key = os.getenv("OGMEM_API_KEY")
    if not base_url or not api_key:
        print("Set OGMEM_BASE_URL and OGMEM_API_KEY", file=sys.stderr)
        return 2
    try:
        with httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        ) as client:
            run(client)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except (AttributeError, KeyError, TypeError, ValueError, httpx.HTTPError, httpx.InvalidURL):
        print("Smoke test received an invalid configuration or response", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
