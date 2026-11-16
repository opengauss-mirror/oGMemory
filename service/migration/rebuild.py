"""Rebuild persistent projections without an LLM or a business worker."""

from __future__ import annotations

import json
import math
from itertools import islice

from core.models import RelationEdge, RequestContext, Role
from index.embedding_metadata import augment_with_embeddings
from index.index_record_builder import build_index_records, build_single_record
from service.migration.package import PackageError
from service.migration.records import archive_uri, identity, node_object


def validate_target_model(cfg, model_dimension):
    if model_dimension is None:
        return
    index_dimension = cfg.to_provider_config().embedding_dimension
    if model_dimension != index_dimension:
        raise PackageError(
            f"target embedding dimension {model_dimension} differs from "
            f"configured vector index dimension {index_dimension}; "
            "align the target model and index configuration before importing"
        )


def _sql_relation_endpoint(inv, uri, *, node_only=False):
    row = inv.db.execute(
        "SELECT kind,body FROM records WHERE kind IN ('nodes','archives') AND key=?",
        (uri.rstrip("/"),),
    ).fetchone()
    if row is None or (node_only and row[0] != "nodes"):
        raise PackageError("SQL graph projection requires node endpoints")
    record = json.loads(row[1])
    canonical = record["uri"] if row[0] == "nodes" else archive_uri(record)
    if uri != canonical:
        raise PackageError(
            f"SQL relation endpoint must exactly match stored URI: {uri} != {canonical}"
        )
    return record


def graph_endpoint_owners(inv, relation):
    return [
        _sql_relation_endpoint(inv, relation[key], node_only=True)["owner_space"]
        for key in ("from_uri", "to_uri")
    ]


def validate_relation_endpoints(cfg, inv):
    node_only = getattr(cfg, "graph_backend", None) == "sql"
    if node_only or cfg.storage_backend == "sql":
        for relation in inv.records("relations"):
            for key in ("from_uri", "to_uri"):
                _sql_relation_endpoint(inv, relation[key], node_only=node_only)


def records_for_node(r):
    if r["status"] != "ACTIVE" or r["metadata"].get("indexing_disabled") is True:
        return []
    node = node_object(r)
    if node.uri.endswith("/"):
        records = [
            build_single_record(
                node.uri,
                level,
                text,
                r["account_id"],
                node.owner_space,
                node.category,
                node.context_type,
            )
            for level, text in ((0, node.abstract), (1, node.overview))
            if text
        ]
        for record in records:
            record.metadata.update(
                node_type="directory",
                parent_uri=node.parent_uri,
                child_count=node.metadata.get("child_count", 0),
            )
            if record.level == 0:
                record.metadata["categories"] = node.metadata.get("categories", [])
        return records
    return build_index_records(node)


def batches(iterator, size=64):
    iterator = iter(iterator)
    while batch := list(islice(iterator, size)):
        yield batch


class Rebuilder:
    def __init__(self, cfg):
        self.cfg = cfg
        self.provider = cfg.to_provider_config()
        self.vector = self.provider.create_vector_index()
        self.embedder = None
        self.lexical = None
        self.pool = None
        try:
            if cfg.retrieval_hybrid_enabled:
                from providers.lexical_index import create_lexical_index

                self.lexical = create_lexical_index(cfg)
        except Exception:
            self.close()
            raise

    def close(self):
        for obj in (self.lexical, self.vector, self.embedder):
            close = getattr(obj, "close", None)
            if callable(close):
                close()
        if self.pool is not None:
            self.pool.close_all()

    def check_empty(self):
        check = getattr(self.vector, "check_migration_target", None)
        if not callable(check):
            raise PackageError("vector backend does not support persistent migration")
        try:
            check()
        except ValueError as exc:
            raise PackageError(str(exc)) from exc
        if self.lexical and self.lexical.health().get("records") != 0:
            raise PackageError("target lexical index is not empty")

    def _embed(self, texts):
        vectors = self.embedder.embed_texts(texts)
        if len(vectors) != len(texts) or any(
            len(v) != self.provider.embedding_dimension or not all(math.isfinite(x) for x in v)
            for v in vectors
        ):
            raise PackageError("embedding output count/dimension is invalid")
        return vectors

    def probe(self):
        self.embedder = self.provider.create_embedder()
        self._embed(["migration validation"])

    def _records(self, inv, scope=None):
        for r in inv.records("nodes", scope=scope):
            yield from records_for_node(r)

    def rebuild(self, inv, progress):
        count = 0
        for batch in batches(self._records(inv)):
            vectors = self._embed([r.text for r in batch])
            self.vector.upsert(augment_with_embeddings(batch, vectors, self.embedder))
            count += len(batch)
            progress(phase="reindexing", index_records=count)
        if self.lexical:
            from core.lexical import lexical_record_from_index_record

            scopes = inv.db.execute(
                "SELECT DISTINCT json_extract(body,'$.account_id'), "
                "json_extract(body,'$.owner_space') FROM records WHERE kind='nodes'"
            )
            for account, owner in scopes:
                rebuild_id = self.lexical.begin_rebuild()
                try:
                    success = self.lexical.replace_scopes(
                        rebuild_id=rebuild_id,
                        account_id=account,
                        owner_spaces=[owner],
                        records=(
                            lexical_record_from_index_record(r)
                            for r in self._records(inv, (account, owner))
                        ),
                    )
                    if not success:
                        raise PackageError("lexical rebuild was not published")
                finally:
                    self.lexical.abort_rebuild(rebuild_id)
        if self.cfg.graph_backend == "sql":
            self._graph(inv)
        return count

    def _graph(self, inv):
        from fs.sql_adapter.pool import SharedConnectionPool
        from providers.relation_store.sql_projection import SQLEdgeProjectionStore
        from providers.relation_store.sql_relation_store import SQLRelationStore

        self.pool = SharedConnectionPool(
            self.cfg.sql_connection_string, dialect=self.cfg.sql_dialect or "auto", pool_size=2
        )
        store = SQLRelationStore(
            pool=self.pool,
            edge_projection_profile_id=self.cfg.graph_edge_embedding_profile,
            edge_projection_representation=self.cfg.graph_projection_representation,
        )
        projector = SQLEdgeProjectionStore(pool=self.pool)
        profile = dict(
            profile_id=self.cfg.graph_edge_embedding_profile,
            model=getattr(self.embedder, "model", type(self.embedder).__name__),
            dimension=self.provider.embedding_dimension,
            representation_version=self.cfg.graph_projection_representation,
        )
        from providers.relation_store.projection_identity import FACTORIZED_REPRESENTATION_VERSION

        if self.cfg.graph_projection_representation == FACTORIZED_REPRESENTATION_VERSION:
            profile.update(
                max_input_chars=self.cfg.embedding_max_input_chars,
                multimodal=self.cfg.embedding_multimodal,
            )
        accounts = set()
        for r in inv.records("relations"):
            account, parts = identity(r["from_uri"])
            accounts.add(account)
            owners = graph_endpoint_owners(inv, r)
            ctx = RequestContext(
                account_id=account,
                user_id=parts[1] if parts[0] == "users" else "",
                agent_id=parts[1] if parts[0] == "agents" else "",
                session_id="",
                trace_id="migration",
                role=Role.ROOT,
                visible_owner_spaces=tuple(dict.fromkeys(owners)),
            )
            store.upsert_edges([RelationEdge(**r)], ctx)
        for account in accounts:
            while projector.drain_sync(account, self.embedder, profile):
                pass
        conn = self.pool.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM edge_projection_jobs WHERE status<>'COMPLETE' LIMIT 1")
                if cur.fetchone():
                    raise PackageError("relation projection rebuild incomplete")
                for account in accounts:
                    cur.execute("SELECT set_config('app.account_id', %s, true)", (account,))
                    cur.execute(
                        "UPDATE outbox_events SET status='DONE' WHERE event_type='PROJECT_GRAPH' AND account_id=%s",
                        (account,),
                    )
            conn.commit()
        finally:
            self.pool.return_connection(conn)

    def verify(self, inv):
        expected = 0
        for r in self._records(inv):
            actual = self.vector.get_record(r.id)
            if (
                actual is None
                or actual.text != r.text
                or any(actual.filters.get(k) != v for k, v in r.filters.items())
            ):
                raise PackageError(
                    "rebuilt index record is not queryable or differs from canonical data"
                )
            expected += 1
        if self.vector.count() != expected:
            raise PackageError("rebuilt index record count differs")
