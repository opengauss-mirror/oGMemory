"""Map document resources to the RAGFlow HTTP API."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from threading import Lock
from typing import Any

import httpx

from core.resource_models import DocumentResource, ResourceChunk

_REMOTE_ID = re.compile(r"[A-Za-z0-9_-]+")
_DATASET_LOCKS = tuple(Lock() for _ in range(64))
_LEGACY_RUN_STATUS = {
    "0": "UNSTART",
    "1": "RUNNING",
    "2": "STOPPED",
    "3": "COMPLETED",
    "4": "FAILED",
    "5": "SCHEDULED",
    "UNSTART": "UNSTART",
    "RUNNING": "RUNNING",
    "CANCEL": "STOPPED",
    "DONE": "COMPLETED",
    "FAIL": "FAILED",
    "SCHEDULE": "SCHEDULED",
}


@dataclass(frozen=True)
class RagFlowDocumentRef:
    resource: DocumentResource
    dataset_id: str
    document_id: str

    def __post_init__(self) -> None:
        if not _REMOTE_ID.fullmatch(self.dataset_id) or not _REMOTE_ID.fullmatch(self.document_id):
            raise ValueError("invalid RAGFlow dataset or document ID")


@dataclass(frozen=True)
class RagFlowDocumentStatus:
    ingestion_status: str
    chunk_count: int


class RagFlowError(RuntimeError):
    def __init__(self, message: str, *, http_status: int | None = None, code: int | None = None):
        super().__init__(message)
        self.http_status = http_status
        self.code = code


class RagFlowDocumentNotFoundError(RagFlowError):
    pass


class RagFlowResourceAdapter:
    """Translate operations on trusted dataset and document references."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout: float = 10.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            transport=transport,
        )

    def __enter__(self) -> RagFlowResourceAdapter:
        return self

    def __exit__(self, *_exc: object) -> None:
        self._client.close()

    def create_dataset(self, name: str) -> str:
        data = self._json_request(
            "POST", "api/v1/datasets", json={"name": name, "permission": "me"}
        )
        dataset_id = data.get("id") if isinstance(data, dict) else None
        if not isinstance(dataset_id, str) or not _REMOTE_ID.fullmatch(dataset_id):
            raise RagFlowError("unexpected RAGFlow dataset identity")
        return dataset_id

    def delete_dataset_if_empty(self, dataset_id: str) -> None:
        """Delete an exclusively managed, private dataset after checking it is empty."""
        if not _REMOTE_ID.fullmatch(dataset_id):
            raise ValueError("invalid RAGFlow dataset ID")
        with _DATASET_LOCKS[hash(dataset_id) % len(_DATASET_LOCKS)]:
            self._delete_dataset_if_empty(dataset_id)

    def _delete_dataset_if_empty(self, dataset_id: str) -> None:
        path = f"api/v1/datasets/{dataset_id}"
        try:
            dataset = self._json_request("GET", path)
            if not isinstance(dataset, dict) or dataset.get("id") != dataset_id:
                raise RagFlowError("unexpected RAGFlow dataset identity")
            permission = dataset.get("permission")
            count, size, connectors = (
                dataset.get("document_count"),
                dataset.get("size"),
                dataset.get("connectors"),
            )
            if (
                permission not in ("me", "team")
                or type(count) is not int
                or type(size) is not int
                or not isinstance(connectors, list)
                or count < 0
                or size < 0
            ):
                raise RagFlowError("unexpected RAGFlow dataset metadata")
            if permission != "me" or count or size or connectors:
                raise FileExistsError(
                    "RAGFlow dataset must be private, empty and without connectors"
                )

            documents = self._json_request("GET", path + "/documents", params={"page_size": 1})
            total = documents.get("total") if isinstance(documents, dict) else None
            rows = documents.get("docs") if isinstance(documents, dict) else None
            if type(total) is not int or total < 0 or not isinstance(rows, list):
                raise RagFlowError("unexpected RAGFlow document list")
            if total or rows:
                raise FileExistsError("RAGFlow dataset must be empty")

            result = self._json_request(
                "DELETE", "api/v1/datasets", json={"ids": [dataset_id], "delete_all": False}
            )
            succeeded = result.get("success_count") if isinstance(result, dict) else None
            if type(succeeded) is not int or succeeded != 1:
                raise RagFlowError("RAGFlow did not confirm dataset deletion")
        except RagFlowError:
            if (
                self._json_request(
                    "GET", "api/v1/datasets", params={"ids": dataset_id, "page_size": 1}
                )
                != []
            ):
                raise

    def upload(
        self, resource: DocumentResource, content: bytes, *, dataset_id: str
    ) -> RagFlowDocumentRef:
        if not _REMOTE_ID.fullmatch(dataset_id):
            raise ValueError("invalid RAGFlow dataset ID")
        with _DATASET_LOCKS[hash(dataset_id) % len(_DATASET_LOCKS)]:
            data = self._json_request(
                "POST",
                f"api/v1/datasets/{dataset_id}/documents",
                files={"file": (resource.filename, content, resource.media_type)},
            )
        if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict):
            raise RagFlowError("unexpected RAGFlow upload result")
        row = data[0]
        document_id = row.get("id")
        if (
            row.get("dataset_id") != dataset_id
            or not isinstance(document_id, str)
            or not _REMOTE_ID.fullmatch(document_id)
        ):
            raise RagFlowError("unexpected RAGFlow document identity")
        return RagFlowDocumentRef(resource, dataset_id, document_id)

    def start_indexing(self, ref: RagFlowDocumentRef) -> None:
        self._json_request(
            "POST",
            f"api/v1/datasets/{ref.dataset_id}/chunks",
            json={"document_ids": [ref.document_id]},
        )

    def get_status(self, ref: RagFlowDocumentRef) -> RagFlowDocumentStatus:
        data = self._json_request(
            "GET",
            f"api/v1/datasets/{ref.dataset_id}/documents",
            params={"ids": ref.document_id, "page_size": 1},
        )
        rows = data.get("docs") if isinstance(data, dict) else None
        if rows == []:
            raise RagFlowDocumentNotFoundError("RAGFlow document was not found")
        if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
            raise RagFlowError("expected one RAGFlow document in status response")
        row = rows[0]
        if row.get("id") != ref.document_id or row.get("dataset_id") != ref.dataset_id:
            raise RagFlowError("unexpected RAGFlow document identity")
        status, count = row.get("ingestion_status"), row.get("chunk_count")
        if status is None and isinstance(run_status := row.get("run"), str):
            status = _LEGACY_RUN_STATUS.get(run_status)
        if not isinstance(status, str) or not status or type(count) is not int or count < 0:
            raise RagFlowError("unexpected RAGFlow ingestion status")
        return RagFlowDocumentStatus(status, count)

    def download(self, ref: RagFlowDocumentRef) -> bytes:
        response = self._request(
            "GET", f"api/v1/datasets/{ref.dataset_id}/documents/{ref.document_id}"
        )
        disposition = response.headers.get("content-disposition", "").split(";", 1)[0]
        if disposition.strip().lower() != "attachment":
            raise RagFlowError("expected a RAGFlow document attachment")
        return response.content

    def delete(self, ref: RagFlowDocumentRef) -> int:
        data = self._json_request(
            "DELETE",
            f"api/v1/datasets/{ref.dataset_id}/documents",
            json={"ids": [ref.document_id], "delete_all": False},
        )
        deleted = data.get("deleted") if isinstance(data, dict) else None
        if type(deleted) is not int or deleted not in {0, 1}:
            raise RagFlowError("unexpected RAGFlow deletion result")
        return deleted

    def search(
        self, query: str, documents: Sequence[RagFlowDocumentRef], *, limit: int = 10
    ) -> list[ResourceChunk]:
        if not query.strip() or limit < 1:
            raise ValueError("search requires a query and a positive limit")
        resources: dict[tuple[str, str], DocumentResource] = {}
        for ref in documents:
            if (
                resources.setdefault((ref.dataset_id, ref.document_id), ref.resource)
                != ref.resource
            ):
                raise ValueError("a RAGFlow document maps to multiple resources")
        if not resources:
            return []
        data = self._json_request(
            "POST",
            "api/v1/retrieval",
            json={
                "question": query,
                "dataset_ids": list(dict.fromkeys(dataset_id for dataset_id, _ in resources)),
                "document_ids": list(dict.fromkeys(document_id for _, document_id in resources)),
                "page_size": limit,
                "include_knowledge_compilation": False,
            },
        )
        rows = data.get("chunks") if isinstance(data, dict) else None
        if not isinstance(rows, list):
            raise RagFlowError("unexpected RAGFlow retrieval result")
        hits: list[ResourceChunk] = []
        for row in rows[:limit]:
            if not isinstance(row, dict):
                raise RagFlowError("unexpected RAGFlow chunk")
            dataset_id, document_id = row.get("dataset_id"), row.get("document_id")
            if not isinstance(dataset_id, str) or not isinstance(document_id, str):
                raise RagFlowError("unexpected RAGFlow chunk document identity")
            resource = resources.get((dataset_id, document_id))
            if resource is None:
                continue
            chunk_id, content = row.get("id"), row.get("content")
            if not isinstance(chunk_id, str) or not chunk_id or not isinstance(content, str):
                raise RagFlowError("unexpected RAGFlow chunk fields")
            hits.append(ResourceChunk(resource.uri, chunk_id, content, len(hits) + 1))
        return hits

    def _json_request(self, method: str, path: str, **kwargs: Any) -> Any:
        response = self._request(method, path, **kwargs)
        try:
            payload = response.json()
        except ValueError:
            raise RagFlowError("invalid RAGFlow JSON response") from None
        code = payload.get("code") if isinstance(payload, dict) else None
        if type(code) is not int:
            raise RagFlowError("missing RAGFlow response code")
        if code:
            raise RagFlowError("RAGFlow API request failed", code=code)
        return payload.get("data")

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.RequestError:
            raise RagFlowError(f"RAGFlow {method} transport failed") from None
        if not response.is_success:
            raise RagFlowError("RAGFlow HTTP request failed", http_status=response.status_code)
        return response
