"""Run a local shared-space QA service with real openGauss and AGFS stores."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.models import ContextNode, RequestContext, Role
from providers.embedder.mock_embedder import MockEmbedder
from providers.unified_config import OgMemConfig
from server.memory_service import MemoryService


class EvidenceModel:
    """Produce a deterministic plan from the real source snapshot."""

    def complete_json(self, prompt, schema):
        data = json.loads(prompt.split("\n", 1)[1])
        evidence = [source["content"] for source in data["sources"]]
        if any("QA_RETIRE" in content for content in evidence):
            return {"operations": [{"action": "ARCHIVE", "item_id": item["item_id"]}
                                   for item in data["existing"]]}
        operation = {"action": "ADD", "topic": "connection-pool",
                     "abstract": "连接池排障经验", "quality_score": 0.95,
                     "source_uris": [source["uri"] for source in data["sources"]]}
        if data["existing"]:
            operation.update(action="UPDATE", item_id=data["existing"][0]["item_id"])
        operation['content'] = "\n\n".join(evidence)
        return {"operations": [operation]}


class QAMemoryService(MemoryService):
    def get_llm(self):
        if self._llm is None:
            self._llm = EvidenceModel()
        return self._llm

    def _get_shared_embedder(self):
        if self._embedder is None:
            self._embedder = MockEmbedder(dimension=64)
        return self._embedder


def load_environment():
    return json.loads(Path(os.environ["OGMEM_SPACE_TEST_CONFIG"]).read_text(encoding="utf-8"))


def create_service(environment, account):
    return QAMemoryService(OgMemConfig(
        storage_backend="agfs", shared_spaces_enabled=True,
        sql_connection_string=environment["dsn"], sql_pool_size=10, sql_dialect="opengauss",
        agfs_base_url=environment["agfs_url"],
        agfs_mount_prefix=environment.get("agfs_mount_prefix", "/local"),
        vector_db_type="opengauss", opengauss_connection_string=environment["dsn"],
        opengauss_dimension=64, opengauss_table_name="shared_space_qa_vectors",
        opengauss_pool_size=5, embedding_provider="mock", index_interval=1,
        role_control_enabled=True, root_api_key=uuid4().hex, account_id=account,
        session_state_bridge_enabled=False, agent_shared_mode="account",
        directory_summary_enabled=False,
    ))


def register_members(service, account):
    manager = service.get_key_manager()
    manager.create_account(account, "admin")
    alice_key = manager.register_user(account, "alice")
    bob_key = manager.register_user(account, "bob")
    admin = RequestContext(account, "admin", "", "", uuid4().hex, Role.ADMIN)
    service.get_tenant_admin_service().create_agent(admin, account, "ops-agent", "alice")
    return {"account": account, "alice_key": alice_key, "bob_key": bob_key}


def write_source(service, account, content="检查长事务、连接泄漏，确认等待队列下降。"):
    ctx = RequestContext(account, "alice", "ops-agent", "", uuid4().hex)
    node = ContextNode(f"ctx://{account}/agents/ops-agent/memories/case/pool",
                       "MEMORY", "case", 2, "agent:ops-agent", "连接池耗尽",
                       "排查长事务和连接释放", content)
    service._get_context_fs().write_node(node, ctx)
    return node.uri


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credentials", type=Path, required=True,
                        help="Private JSON path for browser credentials; reused on restart")
    parser.add_argument("--port", type=int, default=18091)
    args = parser.parse_args()
    environment = load_environment()
    saved = json.loads(args.credentials.read_text()) if args.credentials.exists() else None
    account = saved["account"] if saved else "agfs-browser-" + uuid4().hex[:10]
    service = create_service(environment, account)
    try:
        if saved is None:
            saved = register_members(service, account)
            write_source(service, account)
        saved["url"] = f"http://127.0.0.1:{args.port}"
        args.credentials.parent.mkdir(parents=True, exist_ok=True)
        args.credentials.write_text(json.dumps(saved, indent=2) + "\n", encoding="utf-8")
        from werkzeug.serving import run_simple

        import server.app as public
        from server.management_app import create_app
        public._service = service
        print(json.dumps({"url": saved["url"], "account": account,
                          "credentials_file": str(args.credentials)}, ensure_ascii=False), flush=True)
        run_simple("127.0.0.1", args.port, create_app(), threaded=True)
    finally:
        service.shutdown(30)


if __name__ == "__main__":
    main()
