"""Fresh-process public read/search checks for data created by MemoryWriteAPI."""

import json
import sys
from dataclasses import replace

from pyagfs import AGFSClient

from core.errors import AccessDeniedError
from core.models import RequestContext, RetrievalConfig
from fs.agfs_adapter.agfs_context_fs import AGFSContextFS
from fs.sql_adapter.sql_context_fs import SQLContextFS
from providers.unified_config import OgMemConfig
from retrieval.context_reader import ContextReader
from retrieval.pipeline import RetrievalPipeline
from retrieval.query_planner import QueryPlanner
from retrieval.result_ranker import ResultRanker
from retrieval.seed_retriever import SeedRetriever
from service.api import ReadAPI


def main():
    cfg = OgMemConfig(**json.load(open(sys.argv[1])))
    uri, content = sys.argv[2:]
    provider = cfg.to_provider_config()
    index, embedder = provider.create_vector_index(), provider.create_embedder()
    fs = (
        SQLContextFS(connection_string=cfg.sql_connection_string, dialect=cfg.sql_dialect)
        if cfg.storage_backend == "sql"
        else AGFSContextFS(AGFSClient(api_base_url=cfg.agfs_base_url), cfg.agfs_mount_prefix)
    )
    try:
        reader = ContextReader(fs, storage_backend=cfg.storage_backend)
        settings = RetrievalConfig(expand_hierarchy=False, owner_scope_filter_enabled=True)
        api = ReadAPI(
            RetrievalPipeline(
                QueryPlanner(settings),
                SeedRetriever(index, embedder, settings),
                None,
                ResultRanker(settings),
                settings,
                context_reader=reader,
            ),
            reader,
            settings,
        )
        ctx = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="main",
            session_id="consumer",
            trace_id="consumer",
        )
        assert api.read_memory(uri, ctx).content_excerpt == content
        result = api.search_memory(
            content, ctx, score_threshold=-1, use_scene=False, fill_content_for_top_k=10
        )
        assert len(result.hits) == 1
        assert result.hits[0].uri.removesuffix("/content.md") == uri
        assert result.hits[0].content_excerpt == content
        for hidden in (replace(ctx, account_id="other"), replace(ctx, user_id="bob")):
            assert not api.search_memory(content, hidden, score_threshold=-1, use_scene=False).hits
            try:
                api.read_memory(uri, hidden)
            except AccessDeniedError:
                pass
            else:
                raise AssertionError("cross-scope read was permitted")
        from core.models import WritePlan
        from retrieval.authorized_vector_search import AuthorizedVectorSearch
        from service.api import MemoryWriteAPI
        from service.relation_candidate_service import RelationCandidateService
        from tests.unit.commit.test_memory_identity import memory
        from tests.unit.server.test_extraction_batch_recovery import NoModel

        candidate = replace(memory("next", category="preference"), provenance_ids=[])
        written = MemoryWriteAPI(fs, NoModel(), directory_summary_enabled=False).write_memory(
            candidate, ctx
        )
        plan = WritePlan(action=written["action"], target_uri=written["target_uri"])
        recommendations, diagnostics = RelationCandidateService(
            AuthorizedVectorSearch(index, embedder, reader)
        ).recommend([candidate], [plan], ctx)
        assert [item.node_ref for item in recommendations] == [uri], diagnostics
        print("BUSINESS_CONSUMER_OK", cfg.storage_backend)
    finally:
        index.close()
        close = getattr(fs, "close", None)
        if close:
            close()


if __name__ == "__main__":
    main()
