import json
from uuid import UUID

import httpx
import pytest

from core.resource_models import DocumentResource, ResourceChunk
from providers.resource_store.ragflow import (
    RagFlowDocumentRef,
    RagFlowDocumentStatus,
    RagFlowResourceAdapter,
)


def test_ragflow_create_and_delete_empty_dataset():
    dataset_id = "cde3db5fe62d11f09be10242ac170003"
    collection_path = "/api/v1/datasets"
    dataset_path = f"{collection_path}/{dataset_id}"
    operations = []
    datasets = {}

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer ragflow-test-key"
        method, path = request.method, request.url.path
        operations.append((method, path))
        if (method, path) == ("POST", collection_path):
            assert request.content
            assert json.loads(request.content) == {
                "name": "ogmemory-maintenance",
                "permission": "me",
            }
            data = {
                "id": dataset_id,
                "name": "ogmemory-maintenance",
                "permission": "me",
                "document_count": 0,
                "size": 0,
                "connectors": [],
            }
            datasets[dataset_id] = data
        elif (method, path) == ("GET", dataset_path):
            if dataset_id not in datasets:
                return httpx.Response(200, json={"code": 102, "message": "dataset is absent"})
            data = datasets[dataset_id]
        elif (method, path) == ("GET", dataset_path + "/documents"):
            assert request.url.params["page_size"] == "1"
            data = {"total": 0, "docs": []}
        elif (method, path) == ("GET", collection_path):
            assert request.url.params["ids"] == dataset_id
            data = list(datasets.values())
        else:
            assert (method, path) == ("DELETE", collection_path)
            assert json.loads(request.content) == {"ids": [dataset_id], "delete_all": False}
            del datasets[dataset_id]
            data = {"success_count": 1}
        return httpx.Response(200, json={"code": 0, "data": data})

    with RagFlowResourceAdapter(
        base_url="https://ragflow.example.com",
        api_key="ragflow-test-key",
        transport=httpx.MockTransport(handle),
    ) as adapter:
        assert adapter.create_dataset("ogmemory-maintenance") == dataset_id
        adapter.delete_dataset_if_empty(dataset_id)
        adapter.delete_dataset_if_empty(dataset_id)

    assert operations == [
        ("POST", collection_path),
        ("GET", dataset_path),
        ("GET", dataset_path + "/documents"),
        ("DELETE", collection_path),
        ("GET", dataset_path),
        ("GET", collection_path),
    ]


@pytest.mark.parametrize("legacy", [False, True], ids=["v1", "v0.27.2"])
def test_ragflow_document_lifecycle(legacy):
    dataset_id = "cde3db5fe62d11f09be10242ac170003"
    document_id = "e4077414e62d11f0ab310242ac170005"
    chunk_id = "a8ad91840a559091"
    text = "The maintenance window opens at 02:00 UTC."
    content = text.encode()
    resource = DocumentResource(
        account_id="acme",
        resource_id=UUID("1baed8f4-6e2d-4b84-ac28-ac91269a9580"),
        filename="maintenance.txt",
        media_type="text/plain",
    )
    calls = []
    documents_path = f"/api/v1/datasets/{dataset_id}/documents"
    chunks_path = f"/api/v1/datasets/{dataset_id}/chunks"
    status_field = "run" if legacy else "ingestion_status"
    finished_status = "DONE" if legacy else "COMPLETED"

    def handle(request: httpx.Request) -> httpx.Response:
        operation = (request.method, request.url.path)
        calls.append(operation)
        assert request.headers["authorization"] == "Bearer ragflow-test-key"

        if operation == ("POST", documents_path):
            assert request.headers["content-type"].startswith("multipart/form-data;")
            assert b'name="file"; filename="maintenance.txt"' in request.content
            assert b"Content-Type: text/plain" in request.content
            assert content in request.content
            data = [
                {
                    "id": document_id,
                    "dataset_id": dataset_id,
                    "name": resource.filename,
                    status_field: "UNSTART",
                }
            ]
        elif operation == ("POST", chunks_path):
            assert json.loads(request.content) == {"document_ids": [document_id]}
            return httpx.Response(200, json={"code": 0})
        elif operation == ("GET", documents_path):
            assert request.url.params["ids"] == document_id
            data = {
                "total": 1,
                "docs": [
                    {
                        "id": document_id,
                        "dataset_id": dataset_id,
                        status_field: finished_status,
                        "chunk_count": 1,
                    }
                ],
            }
        elif operation == ("POST", "/api/v1/retrieval"):
            assert json.loads(request.content) == {
                "question": "When does maintenance open?",
                "dataset_ids": [dataset_id],
                "document_ids": [document_id],
                "page_size": 3,
                "include_knowledge_compilation": False,
            }
            data = {
                "total": 1,
                "chunks": [
                    {
                        "id": chunk_id,
                        "content": text,
                        "document_id": document_id,
                        "dataset_id": dataset_id,
                    }
                ],
            }
        elif operation == ("GET", f"{documents_path}/{document_id}"):
            return httpx.Response(
                200,
                headers={
                    "Content-Disposition": 'attachment; filename="maintenance.txt"',
                    "Content-Type": "text/plain",
                },
                content=content,
            )
        else:
            assert operation == ("DELETE", documents_path)
            assert json.loads(request.content) == {"ids": [document_id], "delete_all": False}
            data = {"deleted": 1}
        return httpx.Response(200, json={"code": 0, "data": data})

    with RagFlowResourceAdapter(
        base_url="https://ragflow.example.com",
        api_key="ragflow-test-key",
        transport=httpx.MockTransport(handle),
    ) as adapter:
        ref = adapter.upload(resource, content, dataset_id=dataset_id)
        assert ref == RagFlowDocumentRef(resource, dataset_id, document_id)
        adapter.start_indexing(ref)
        assert adapter.get_status(ref) == RagFlowDocumentStatus("COMPLETED", 1)
        assert adapter.search("When does maintenance open?", [ref], limit=3) == [
            ResourceChunk(resource.uri, chunk_id, text, 1)
        ]
        assert adapter.download(ref) == content
        assert adapter.delete(ref) == 1

    assert calls == [
        ("POST", documents_path),
        ("POST", chunks_path),
        ("GET", documents_path),
        ("POST", "/api/v1/retrieval"),
        ("GET", f"{documents_path}/{document_id}"),
        ("DELETE", documents_path),
    ]


def test_ragflow_go_legacy_scheduled_status():
    resource = DocumentResource(
        account_id="acme",
        resource_id=UUID("79c58e44-404f-4502-9de0-09f082900559"),
        filename="scheduled.txt",
        media_type="text/plain",
    )
    ref = RagFlowDocumentRef(resource, "acme-dataset", "scheduled-document")

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/v1/datasets/acme-dataset/documents"
        assert request.url.params["ids"] == ref.document_id
        return httpx.Response(
            200,
            json={
                "code": 0,
                "data": {
                    "total": 1,
                    "docs": [
                        {
                            "id": ref.document_id,
                            "dataset_id": ref.dataset_id,
                            "run": "5",
                            "chunk_count": 0,
                        }
                    ],
                },
            },
        )

    with RagFlowResourceAdapter(
        base_url="https://ragflow.example.com",
        api_key="ragflow-test-key",
        transport=httpx.MockTransport(handle),
    ) as adapter:
        assert adapter.get_status(ref) == RagFlowDocumentStatus("SCHEDULED", 0)


def test_ragflow_reads_existing_document_by_reference():
    content = b"An existing RAGFlow document."
    resource = DocumentResource(
        account_id="acme",
        resource_id=UUID("79c58e44-404f-4502-9de0-09f082900559"),
        filename="existing.txt",
        media_type="text/plain",
    )
    ref = RagFlowDocumentRef(resource, "existing-dataset", "existing-document")

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/v1/datasets/existing-dataset/documents/existing-document"
        return httpx.Response(
            200,
            headers={"Content-Disposition": 'attachment; filename="existing.txt"'},
            content=content,
        )

    with RagFlowResourceAdapter(
        base_url="https://ragflow.example.com",
        api_key="ragflow-test-key",
        transport=httpx.MockTransport(handle),
    ) as adapter:
        assert adapter.download(ref) == content
