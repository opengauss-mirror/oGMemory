from contextlib import nullcontext

import pytest
from flask import Flask

from providers.unified_config import ExternalKnowledgeBaseConfig, KnowledgeBaseConfig, OgMemConfig
from server import document_resource_runtime
from server.api_keys import APIKeyManager
from server.control_plane_store import ControlPlaneStore
from server.document_resource_api import create_document_resource_blueprint
from server.document_resource_runtime import DocumentResourceRuntime


def test_manage_active_shared_knowledge_base_members(monkeypatch, tmp_path):
    class Catalog:
        def __init__(self):
            self.rows = {}

        def get(self, account_id, identifier):
            return self.rows.get((account_id, identifier))

        def list(self, account_id):
            return [row for (account, _), row in self.rows.items() if account == account_id]

        def ensure_default(self, base):
            self.rows.setdefault((base.account_id, base.id), (base, {}))

        def put_member(self, base, user_id, grant):
            self.rows[(base.account_id, base.id)][1][user_id] = grant

        def remove_member(self, base, user_id):
            self.rows[(base.account_id, base.id)][1].pop(user_id, None)

    class Store:
        def __init__(self, _pool):
            self.catalog = Catalog()

        def lock_knowledge_base(self, _base):
            return nullcontext()

    config = OgMemConfig(
        role_control_enabled=True,
        root_api_key="fixture-root",
        knowledge_base=KnowledgeBaseConfig(
            mode="external",
            external=ExternalKnowledgeBaseConfig(
                "ragflow",
                "https://ragflow.example.com",
                "provider-fixture-key",
                {"acme": "dataset"},
            ),
        ),
    )
    control = ControlPlaneStore(mount_prefix="", local_root=str(tmp_path))
    keys = APIKeyManager(control)
    manager = keys.create_account("acme", "manager")
    reader = keys.register_user("acme", "reader")
    keys.register_user("acme", "editor")
    keys.create_account("other", "other-admin")
    keys.register_user("other", "other-user")
    monkeypatch.setattr(document_resource_runtime, "SQLDocumentResourceStore", Store)
    runtime = DocumentResourceRuntime(config, object(), control)
    app = Flask(__name__)
    app.testing = True
    app.register_blueprint(
        create_document_resource_blueprint(lambda: runtime, lambda exc: pytest.fail(str(exc)))
    )
    client = app.test_client()
    admin_headers = {"X-API-Key": manager}
    catalogue = client.get("/api/v1/knowledge-bases", headers=admin_headers).get_json()
    endpoint = f"/api/v1/knowledge-bases/{catalogue['default_knowledge_base_id']}/members"

    candidates = client.get(f"{endpoint}?q=READ", headers=admin_headers)
    assert candidates.status_code == 200
    assert candidates.get_json() == {
        "members": [],
        "candidates": [{"user_id": "reader"}],
        "candidate_count": 1,
    }
    for user_id, permission in [("reader", "read"), ("editor", "write"), ("reader", "write")]:
        grant = client.put(
            f"{endpoint}/{user_id}", headers=admin_headers, json={"permission": permission}
        )
        assert grant.status_code == 200
        assert grant.get_json() == {"user_id": user_id, "permission": permission}
    members = client.get(endpoint, headers=admin_headers)
    assert members.status_code == 200
    assert members.get_json()["members"] == [
        {"user_id": "editor", "permission": "write"},
        {"user_id": "reader", "permission": "write"},
    ]
    reader_catalogue = client.get(
        "/api/v1/knowledge-bases", headers={"X-API-Key": reader}
    ).get_json()
    assert reader_catalogue["items"][0]["permissions"] == {
        "read": True,
        "write": True,
        "manage": False,
    }

    removal = client.delete(f"{endpoint}/editor", headers=admin_headers)
    assert removal.status_code == 204
    follow_up = client.get(endpoint, headers=admin_headers)
    assert follow_up.status_code == 200
    assert follow_up.get_json() == {
        "members": [{"user_id": "reader", "permission": "write"}],
        "candidates": [{"user_id": "editor"}],
        "candidate_count": 1,
    }
