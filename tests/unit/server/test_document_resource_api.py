import json
from contextlib import nullcontext
from dataclasses import replace
from io import BytesIO
from uuid import UUID

import httpx
import pytest
from flask import Flask

from providers.resource_store.ragflow import RagFlowResourceAdapter
from providers.unified_config import OgMemConfig
from server import document_resource_runtime, document_resource_service
from server.api_keys import APIKeyManager
from server.control_plane_store import ControlPlaneStore
from server.document_resource_api import create_document_resource_blueprint
from server.document_resource_runtime import DocumentResourceRuntime
from server.document_resource_store import DocumentBinding


@pytest.mark.parametrize("use_yaml", [False, True], ids=["env", "yaml-overrides-env"])
def test_registered_accounts_use_their_document_namespaces(monkeypatch, tmp_path, use_yaml):
    account_id = "acme"
    other_account_id = "other"
    env_namespaces = {account_id: "acme-env", other_account_id: "other-env"}
    namespaces = {account_id: "acme-yaml", other_account_id: "other-yaml"}
    env_base_url, env_key = "https://ragflow-env.example", "ragflow-env-key"
    base_url, token = "https://ragflow-yaml.example", "ragflow-yaml-key"
    config_path = tmp_path / "config.yaml"
    if use_yaml:
        config_path.write_text(
            "knowledge_base:\n"
            "  mode: external\n"
            "  external:\n"
            "    provider: ragflow\n"
            f"    base_url: {base_url}\n"
            f"    api_key: {token}\n"
            f"    account_namespaces: {json.dumps(namespaces)}\n",
            encoding="utf-8",
        )
    else:
        namespaces = env_namespaces
        base_url, token = env_base_url, env_key
        config_path.write_text("{}\n", encoding="utf-8")

    monkeypatch.setenv("OGMEM_KNOWLEDGE_BASE_MODE", "external")
    monkeypatch.setenv("OG_ROLE_CONTROL_ENABLED", "true")
    monkeypatch.setenv("OG_ROOT_API_KEY", "root-key")
    monkeypatch.setenv("OGMEM_KNOWLEDGE_BASE_EXTERNAL_PROVIDER", "ragflow")
    monkeypatch.setenv("OGMEM_KNOWLEDGE_BASE_EXTERNAL_BASE_URL", env_base_url)
    monkeypatch.setenv("OGMEM_KNOWLEDGE_BASE_EXTERNAL_API_KEY", env_key)
    monkeypatch.setenv(
        "OGMEM_KNOWLEDGE_BASE_EXTERNAL_ACCOUNT_NAMESPACES", json.dumps(env_namespaces)
    )
    config = OgMemConfig.load(str(config_path))

    dataset_id, other_dataset_id = namespaces[account_id], namespaces[other_account_id]
    documents_path = f"/api/v1/datasets/{dataset_id}/documents"
    parse_path = f"/api/v1/datasets/{dataset_id}/chunks"
    other_documents_path = f"/api/v1/datasets/{other_dataset_id}/documents"
    other_parse_path = f"/api/v1/datasets/{other_dataset_id}/chunks"
    url = "/api/v1/resources/documents"
    pool = object()
    bindings = {}
    bases = {}
    first_content = b"Maintenance starts at 02:00 UTC."
    updated_content = b"Maintenance starts at 03:00 UTC."
    other_content = b"Other account maintenance starts at 04:00 UTC."
    checklist_content = b"Maintenance readiness checklist."
    ids = iter([UUID(int=3), UUID(int=2), UUID(int=1)])
    monkeypatch.setattr(document_resource_service, "uuid4", lambda: next(ids))
    uploads = iter(
        [
            (documents_path, dataset_id, "first.txt", first_content, "doc-one"),
            (other_documents_path, other_dataset_id, "other.txt", other_content, "doc-other"),
            (documents_path, dataset_id, "updated.txt", updated_content, "doc-two"),
            (documents_path, dataset_id, "checklist.txt", checklist_content, "doc-checklist"),
        ]
    )
    calls = []

    class Catalog:
        def get(self, account_id, id):
            return bases.get((account_id, id))

        def list(self, account_id):
            return [row for (account, _), row in bases.items() if account == account_id]

        def ensure_default(self, base):
            bases.setdefault((base.account_id, base.id), (base, {}))

        def update(self, base, members):
            key = (base.account_id, base.id)
            bases[key] = (base, bases[key][1] if members is None else members)

    class DocumentStore:
        def __init__(self, connection_pool):
            assert connection_pool is pool
            self.catalog = Catalog()

        def lock_knowledge_base(self, base):
            return nullcontext()

        def add(self, ref):
            bindings[(ref.resource.account_id, ref.resource.resource_id)] = DocumentBinding(ref)

        def get(self, account_id, resource_id):
            return bindings.get((account_id, resource_id))

        def list_active(self, account_id, backend, namespace, *, object_ids=None):
            return [
                binding.ref
                for binding in bindings.values()
                if not binding.deleted
                and (object_ids is None or binding.ref.object_id in object_ids)
                and (binding.ref.resource.account_id, binding.ref.backend, binding.ref.namespace)
                == (account_id, backend, namespace)
            ]

        def list_page(self, account_id, backend, namespace, *, limit, offset):
            resources = [ref.resource for ref in self.list_active(account_id, backend, namespace)]
            resources.sort(key=lambda resource: resource.resource_id)
            return resources[offset : offset + limit], len(resources)

        def replace(self, current, updated):
            key = (updated.resource.account_id, updated.resource.resource_id)
            assert bindings[key] == current
            bindings[key] = DocumentBinding(updated, current.ref.object_id)
            return True

        def clear_retired(self, binding):
            resource = binding.ref.resource
            bindings[(resource.account_id, resource.resource_id)] = replace(
                binding, retired_object_id=None
            )

        def mark_deleted(self, binding):
            resource = binding.ref.resource
            bindings[(resource.account_id, resource.resource_id)] = replace(binding, deleted=True)
            return True

        def remove(self, binding):
            resource = binding.ref.resource
            del bindings[(resource.account_id, resource.resource_id)]

    control_store = ControlPlaneStore(mount_prefix="", local_root=str(tmp_path))
    key_manager = APIKeyManager(control_store)
    acme_key = key_manager.create_account(account_id, "alice")
    reader_key = key_manager.register_user(account_id, "reader")
    other_key = key_manager.create_account(other_account_id, "bob")
    key_manager.create_account("unmapped", "charlie")
    runtime = DocumentResourceRuntime(config, pool, control_store)

    def handle(provider_request):
        assert provider_request.headers["authorization"] == f"Bearer {token}"
        assert str(provider_request.url).startswith(base_url + "/")
        operation = (provider_request.method, provider_request.url.path)
        if operation in {("POST", documents_path), ("POST", other_documents_path)}:
            path, namespace, filename, content, document_id = next(uploads)
            assert operation == ("POST", path)
            assert f'filename="{filename}"'.encode() in provider_request.content
            assert content in provider_request.content
            payload = {"data": [{"id": document_id, "dataset_id": namespace, "name": filename}]}
        elif operation in {("POST", parse_path), ("POST", other_parse_path)}:
            (document_id,) = json.loads(provider_request.content)["document_ids"]
            payload = {}
        elif operation == ("GET", documents_path):
            document_id = provider_request.url.params["ids"]
            assert document_id == "doc-two"
            document = {"id": document_id, "dataset_id": dataset_id, "run": "DONE", "chunk_count": 1}
            payload = {"data": {"total": 1, "docs": [document]}}
        elif operation == ("GET", f"{documents_path}/doc-two"):
            calls.append((*operation, "doc-two"))
            return httpx.Response(
                200,
                headers={"Content-Disposition": 'attachment; filename="updated.txt"'},
                content=updated_content,
            )
        elif operation == ("POST", "/api/v1/retrieval"):
            query = json.loads(provider_request.content)
            assert query == {
                "question": "When does maintenance start?",
                "dataset_ids": [dataset_id],
                "page_size": 64,
                "rerank_candidates_count": 64,
                "include_knowledge_compilation": False,
            }
            document_id = "doc-two"
            chunk = {"dataset_id": dataset_id, "document_id": document_id, "id": "chunk-two",
                     "content": updated_content.decode()}
            payload = {"data": {"chunks": [{**chunk, "document_id": "unbound"}, chunk]}}
        else:
            assert operation == ("DELETE", documents_path)
            (document_id,) = json.loads(provider_request.content)["ids"]
            payload = {"data": {"deleted": 1}}
        calls.append((*operation, document_id))
        return httpx.Response(200, json={"code": 0, **payload})

    transport = httpx.MockTransport(handle)
    monkeypatch.setattr(document_resource_runtime, "SQLDocumentResourceStore", DocumentStore)
    monkeypatch.setattr(
        document_resource_runtime,
        "RagFlowResourceAdapter",
        lambda **kwargs: RagFlowResourceAdapter(**kwargs, transport=transport),
    )

    app = Flask(__name__)
    app.testing = True
    app.register_blueprint(
        create_document_resource_blueprint(lambda: runtime, lambda exc: pytest.fail(str(exc)))
    )
    client = app.test_client()
    headers = {"X-API-Key": acme_key}

    def json_result(response, expected=200):
        assert response.status_code == expected
        return response.get_json()

    def upload(path, content, filename, *, method="POST", auth=headers):
        upload_body = {"file": (BytesIO(content), filename, "text/plain")}
        return client.open(path, method=method, headers=auth, data=upload_body)

    assert json_result(client.get(f"{url}/scope", headers=headers)) == {
        "account_id": account_id, "accounts": [{"account_id": account_id}],
    }
    root_headers = {"X-API-Key": "root-key", "X-Account-ID": other_account_id}
    assert json_result(client.get(f"{url}/scope", headers=root_headers)) == {
        "account_id": other_account_id,
        "accounts": [{"account_id": account_id}, {"account_id": other_account_id}, {"account_id": "unmapped"}],
    }

    catalog_url = "/api/v1/knowledge-bases"
    catalog = json_result(client.get(catalog_url, headers=headers))
    base_id = catalog["default_knowledge_base_id"]
    assert catalog["items"][0]["id"] == base_id
    assert catalog["items"][0]["is_default"] is True

    result = json_result(upload(url, first_content, "first.txt"), 201)
    resource_id = UUID(result["id"])
    assert result == {
        "id": str(resource_id), "uri": f"ctx://{account_id}/resources/doc-{resource_id.hex}",
        "filename": "first.txt", "media_type": "text/plain", "indexing_submitted": True,
    }
    key = (account_id, resource_id)
    assert bindings[key].ref.object_id == "doc-one"

    other_headers = {"Authorization": f"Bearer {other_key}"}
    other_result = json_result(upload(url, other_content, "other.txt", auth=other_headers), 201)
    other_resource_id = UUID(other_result["id"])
    assert other_result["uri"] == f"ctx://{other_account_id}/resources/doc-{other_resource_id.hex}"
    other_binding_key = (other_account_id, other_resource_id)
    assert bindings[other_binding_key].ref.namespace == other_dataset_id
    assert bindings[other_binding_key].ref.object_id == "doc-other"

    path = f"{url}/{resource_id}"
    updated = json_result(upload(path, updated_content, "updated.txt", method="PUT"))
    assert updated == {**result, "filename": "updated.txt", "cleanup_pending": False}
    assert bindings[key].ref.object_id == "doc-two"
    assert bindings[key].retired_object_id is None
    updated_item = {field: updated[field] for field in ("id", "uri", "filename", "media_type")}
    assert json_result(client.get(path, headers=headers)) == {
        **updated_item, "index_state": "ready", "chunk_count": 1,
    }
    download = client.get(f"{path}/content", headers=headers)
    assert download.status_code == 200
    assert download.mimetype == "text/plain"
    assert download.headers["Content-Disposition"] == "attachment; filename=updated.txt"
    assert download.data == updated_content

    grant = {"members": [{"user_id": "reader", "permission": "read"}]}
    assert client.patch(f"{catalog_url}/{base_id}", headers=headers, json=grant).status_code == 200
    reader_headers = {"X-API-Key": reader_key}
    reader_catalogue = json_result(client.get(catalog_url, headers=reader_headers))
    assert reader_catalogue["items"][0]["permissions"] == {"read": True, "write": False, "manage": False}
    search = client.post(f"{url}/search?knowledge_base_id={base_id}", headers=reader_headers,
                         json={"query": "When does maintenance start?", "limit": 3})
    assert json_result(search) == {"hits": [{
        "uri": result["uri"], "chunk_id": "chunk-two", "content": updated_content.decode(), "rank": 1,
    }]}

    checklist = json_result(upload(url, checklist_content, "checklist.txt"), 201)
    assert checklist.pop("indexing_submitted") is True
    for page, item in ((1, checklist), (2, updated_item)):
        listing = client.get(f"{url}?page={page}&page_size=1", headers=headers)
        assert json_result(listing) == {"items": [item], "page": page, "page_size": 1, "total": 2}
    assert other_result.pop("indexing_submitted") is True
    assert json_result(client.get(url, headers=other_headers)) == {
        "items": [other_result], "page": 1, "page_size": 20, "total": 1,
    }

    for _ in range(2):
        deleted = client.delete(path, headers=headers)
        assert deleted.status_code == 204 and deleted.data == b""
    assert client.delete(f"{url}/{checklist['id']}", headers=headers).status_code == 204
    assert list(bindings) == [other_binding_key]
    assert calls == [
        ("POST", documents_path, "doc-one"),
        ("POST", parse_path, "doc-one"),
        ("POST", other_documents_path, "doc-other"),
        ("POST", other_parse_path, "doc-other"),
        ("POST", documents_path, "doc-two"),
        ("POST", parse_path, "doc-two"),
        ("DELETE", documents_path, "doc-one"),
        ("GET", documents_path, "doc-two"),
        ("GET", f"{documents_path}/doc-two", "doc-two"),
        ("POST", "/api/v1/retrieval", "doc-two"),
        ("POST", documents_path, "doc-checklist"),
        ("POST", parse_path, "doc-checklist"),
        ("DELETE", documents_path, "doc-two"),
        ("DELETE", documents_path, "doc-checklist"),
    ]
