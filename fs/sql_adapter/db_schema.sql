-- ContextEngine SQL Storage Schema
-- Single source of truth — all stores call ensure_schema() in schema.py
-- which executes this file.  Run standalone with: psql -f db_schema.sql
--
-- Every statement is idempotent (IF NOT EXISTS / DO $$ EXCEPTION).
-- NOTE: No explicit BEGIN/COMMIT — the Python caller (ensure_schema)
-- manages the transaction via psycopg2.  If running standalone with
-- psql, autocommit behaviour applies by default.

-- Main node storage (replaces flat files: content.md, .abstract.md, .overview.md, .meta.json)
CREATE TABLE IF NOT EXISTS context_nodes (
    uri          VARCHAR(512) PRIMARY KEY,
    account_id   VARCHAR(128) NOT NULL,
    owner_space  VARCHAR(128) NOT NULL,
    category     VARCHAR(64)  NOT NULL,
    context_type VARCHAR(32)  NOT NULL,
    level        INTEGER      NOT NULL DEFAULT 0,
    status       VARCHAR(16)  NOT NULL DEFAULT 'ACTIVE',
    content      TEXT         NOT NULL DEFAULT '',
    abstract     TEXT         NOT NULL DEFAULT '',
    overview     TEXT         NOT NULL DEFAULT '',
    relations    JSONB        NOT NULL DEFAULT '[]',
    metadata     JSONB        NOT NULL DEFAULT '{}',
    version      INTEGER      NOT NULL DEFAULT 1,
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cn_account  ON context_nodes (account_id);
CREATE INDEX IF NOT EXISTS idx_cn_owner   ON context_nodes (account_id, owner_space);
CREATE INDEX IF NOT EXISTS idx_cn_category ON context_nodes (account_id, category);
CREATE INDEX IF NOT EXISTS idx_cn_uri_prefix ON context_nodes (uri varchar_pattern_ops);

-- Outbox events (replaces .outbox/*.json files)
CREATE TABLE IF NOT EXISTS outbox_events (
    event_id      VARCHAR(36)  PRIMARY KEY,
    event_type    VARCHAR(32)  NOT NULL,
    uri           VARCHAR(512) NOT NULL,
    account_id    VARCHAR(128) NOT NULL,
    payload       JSONB        NOT NULL DEFAULT '{}',
    status        VARCHAR(16)  NOT NULL DEFAULT 'PENDING',
    retry_count   INTEGER      NOT NULL DEFAULT 0,
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    next_retry_at TIMESTAMPTZ,
    worker_id     VARCHAR(64),
    locked_at     TIMESTAMPTZ,
    seq           BIGSERIAL
);

CREATE INDEX IF NOT EXISTS idx_oe_pending ON outbox_events (account_id, status, next_retry_at)
    WHERE status IN ('PENDING', 'PROCESSING');

-- Ad-hoc migration: seq column for pre-existing tables.
-- No-op when CREATE TABLE above already included it.
-- NOTE: cannot use ALTER TABLE ADD BIGSERIAL directly (PG16 rejects it);
-- instead add the column as BIGINT, then bind a dedicated sequence.
DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'outbox_events' AND column_name = 'seq'
    ) THEN
        CREATE SEQUENCE IF NOT EXISTS outbox_events_seq_seq;
        ALTER TABLE outbox_events ADD COLUMN seq BIGINT NOT NULL DEFAULT nextval('outbox_events_seq_seq');
        ALTER SEQUENCE outbox_events_seq_seq OWNED BY outbox_events.seq;
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_oe_seq ON outbox_events (seq);
CREATE INDEX IF NOT EXISTS idx_oe_uri_unfinished
    ON outbox_events (account_id, uri, seq)
    WHERE status IN ('PENDING', 'PROCESSING');

-- Relation edges (replaces .relations.json per-node files)
CREATE TABLE IF NOT EXISTS relation_edges (
    from_uri      VARCHAR(512) NOT NULL,
    to_uri        VARCHAR(512) NOT NULL,
    relation_type VARCHAR(64)  NOT NULL,
    weight        DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    reason        TEXT         NOT NULL DEFAULT '',
    account_id    VARCHAR(128) NOT NULL,
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    PRIMARY KEY (from_uri, to_uri, relation_type)
);

CREATE INDEX IF NOT EXISTS idx_re_from ON relation_edges (from_uri);
CREATE INDEX IF NOT EXISTS idx_re_to   ON relation_edges (to_uri);

-- Keep canonical relation labels consistent with edge_projection_jobs.
-- CREATE TABLE IF NOT EXISTS alone does not widen existing installations.
-- Do not narrow databases that already use a wider or unbounded type.
DO $widen_relation_type$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = 'relation_edges'
          AND column_name = 'relation_type'
          AND data_type = 'character varying'
          AND character_maximum_length < 256
    ) THEN
        ALTER TABLE relation_edges ALTER COLUMN relation_type TYPE VARCHAR(256);
    END IF;
END $widen_relation_type$;

-- S8R-03P durable semantic-search projection for accepted relation edges.
-- JSONB keeps the disposable embedding projection backend-portable; canonical
-- relation identity and lifecycle remain owned by relation_edges.
ALTER TABLE relation_edges ADD COLUMN IF NOT EXISTS projection_edge_id VARCHAR(64);
ALTER TABLE relation_edges ADD COLUMN IF NOT EXISTS edge_projection JSONB;
CREATE UNIQUE INDEX IF NOT EXISTS idx_re_projection_edge_id
    ON relation_edges (account_id, projection_edge_id)
    WHERE projection_edge_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS edge_projection_jobs (
    account_id VARCHAR(128) NOT NULL,
    edge_id VARCHAR(64) NOT NULL,
    source_uri VARCHAR(512) NOT NULL,
    target_uri VARCHAR(512) NOT NULL,
    relation_type VARCHAR(256) NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    target_text TEXT NOT NULL DEFAULT '',
    source_revision INTEGER NOT NULL,
    target_revision INTEGER NOT NULL,
    representation_version VARCHAR(64) NOT NULL,
    embedding_profile VARCHAR(128) NOT NULL,
    status VARCHAR(16) NOT NULL DEFAULT 'PENDING'
        CHECK (status IN ('PENDING','PROCESSING','COMPLETE','FAILED')),
    attempt_count INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    claim_token VARCHAR(64),
    mutation_batch_key BIGINT,
    next_retry_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (account_id, edge_id)
);
CREATE INDEX IF NOT EXISTS idx_edge_projection_jobs_pending
    ON edge_projection_jobs (account_id, status, next_retry_at)
    WHERE status IN ('PENDING','FAILED');
ALTER TABLE edge_projection_jobs ADD COLUMN IF NOT EXISTS claim_token VARCHAR(64);
ALTER TABLE edge_projection_jobs ADD COLUMN IF NOT EXISTS mutation_batch_key BIGINT;
CREATE INDEX IF NOT EXISTS idx_edge_projection_jobs_source
    ON edge_projection_jobs (account_id, source_uri);
CREATE INDEX IF NOT EXISTS idx_edge_projection_jobs_target
    ON edge_projection_jobs (account_id, target_uri);
ALTER TABLE edge_projection_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE edge_projection_jobs FORCE ROW LEVEL SECURITY;
DO $$ BEGIN
    CREATE POLICY tenant_isolation ON edge_projection_jobs
        USING (account_id = current_setting('app.account_id', true))
        WITH CHECK (account_id = current_setting('app.account_id', true));
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

CREATE OR REPLACE FUNCTION ogmem_graph_projection_batch_key()
RETURNS BIGINT AS $$
BEGIN
    RETURN NULLIF(current_setting('app.graph_projection_batch_key'), '')::bigint;
EXCEPTION WHEN undefined_object THEN
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION invalidate_edge_projections_for_node_update()
RETURNS TRIGGER AS $$
DECLARE
    job RECORD;
    refreshed_edges TEXT[] := ARRAY[]::TEXT[];
    active_batch_key BIGINT := ogmem_graph_projection_batch_key();
BEGIN
    IF (OLD.version, OLD.content, OLD.abstract, OLD.overview)
       IS DISTINCT FROM
       (NEW.version, NEW.content, NEW.abstract, NEW.overview) THEN
        FOR job IN SELECT * FROM edge_projection_jobs
                    WHERE account_id = NEW.account_id
                      AND (source_uri = NEW.uri OR target_uri = NEW.uri)
                    ORDER BY edge_id FOR UPDATE
        LOOP
            -- Target-based encodings reuse vectors when the source version advances.
            -- A completed projection can follow a source-only revision in place.
            IF job.source_uri = NEW.uri AND job.target_uri <> NEW.uri
               AND job.status = 'COMPLETE' AND job.claim_token IS NULL
               AND job.source_revision = OLD.version
               AND job.representation_version IN (
                   'relation-target-v2', 'relation-target-factorized-full-v1'
               ) THEN
                UPDATE relation_edges
                   SET edge_projection = jsonb_set(
                           edge_projection, '{source_revision}', NEW.version::text::jsonb),
                       updated_at = NOW()
                 WHERE account_id = NEW.account_id AND projection_edge_id = job.edge_id
                   AND from_uri = job.source_uri AND to_uri = job.target_uri
                   AND relation_type = job.relation_type
                   AND edge_projection->>'edge_id' = job.edge_id
                   AND edge_projection->>'representation_version' = job.representation_version
                   AND edge_projection->>'embedding_profile' = job.embedding_profile
                   AND edge_projection->'source_revision' = job.source_revision::text::jsonb
                   AND edge_projection->'target_revision' = job.target_revision::text::jsonb;
                IF FOUND THEN
                    UPDATE edge_projection_jobs
                       SET source_revision = NEW.version, updated_at = NOW()
                     WHERE account_id = NEW.account_id AND edge_id = job.edge_id;
                    refreshed_edges := array_append(refreshed_edges, job.edge_id::text);
                    CONTINUE;
                END IF;
            END IF;
        UPDATE edge_projection_jobs
           SET source_revision = CASE WHEN source_uri = NEW.uri THEN NEW.version ELSE source_revision END,
               target_revision = CASE WHEN target_uri = NEW.uri THEN NEW.version ELSE target_revision END,
               target_text = CASE WHEN target_uri = NEW.uri
                                  THEN concat_ws(E'\n', NEW.abstract, NEW.overview, NEW.content)
                                  ELSE target_text END,
               status = 'PENDING', last_error = NULL,
               mutation_batch_key = active_batch_key,
               next_retry_at = CASE
                   WHEN status = 'PENDING' AND next_retry_at IS NOT NULL THEN next_retry_at
                   WHEN active_batch_key IS NULL THEN NULL
                   ELSE NOW() + INTERVAL '30 seconds'
               END,
               claim_token = NULL,
               updated_at = NOW()
         WHERE account_id = NEW.account_id
           AND edge_id = job.edge_id;
            -- Same transaction as invalidation: restart cannot lose discovery
            -- even after the ordinary node event was acknowledged.
            INSERT INTO outbox_events
                (event_id,event_type,uri,account_id,payload,status,retry_count)
            VALUES (md5(NEW.account_id || job.edge_id || clock_timestamp()::text || random()::text),
                    'PROJECT_GRAPH',NEW.uri,NEW.account_id,
                    ('{"edge_id":"' || job.edge_id || '"}')::jsonb,'PENDING',0);
        END LOOP;
        UPDATE relation_edges
           SET edge_projection = NULL, updated_at = NOW()
         WHERE account_id = NEW.account_id
           AND (from_uri = NEW.uri OR to_uri = NEW.uri)
           AND projection_edge_id IS NOT NULL
           AND NOT (projection_edge_id = ANY(refreshed_edges));
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
DROP TRIGGER IF EXISTS trg_invalidate_edge_projections ON context_nodes;
CREATE TRIGGER trg_invalidate_edge_projections
AFTER UPDATE OF version, content, abstract, overview ON context_nodes
FOR EACH ROW EXECUTE PROCEDURE invalidate_edge_projections_for_node_update();

-- Physical deletion must erase copied target text and revoke worker leases in
-- the same transaction. Recreating a URI at version 1 must not revive an old
-- relation/vector. Match the node -> jobs -> relations write lock order.
CREATE OR REPLACE FUNCTION remove_edge_projections_for_node_delete()
RETURNS TRIGGER AS $$
DECLARE
    job RECORD;
BEGIN
    FOR job IN SELECT edge_id FROM edge_projection_jobs
                WHERE account_id = OLD.account_id
                  AND (source_uri = OLD.uri OR target_uri = OLD.uri)
                ORDER BY edge_id FOR UPDATE
    LOOP
        DELETE FROM edge_projection_jobs
         WHERE account_id = OLD.account_id AND edge_id = job.edge_id;
    END LOOP;
    DELETE FROM relation_edges
     WHERE account_id = OLD.account_id
       AND (from_uri = OLD.uri OR to_uri = OLD.uri);
    RETURN OLD;
END;
$$ LANGUAGE plpgsql;
DROP TRIGGER IF EXISTS trg_remove_edge_projections ON context_nodes;
CREATE TRIGGER trg_remove_edge_projections
BEFORE DELETE ON context_nodes
FOR EACH ROW EXECUTE PROCEDURE remove_edge_projections_for_node_delete();

-- Session archives
CREATE TABLE IF NOT EXISTS session_archives (
    archive_id    VARCHAR(64)  NOT NULL,
    session_id    VARCHAR(128) NOT NULL,
    account_id    VARCHAR(128) NOT NULL,
    owner_user_id VARCHAR(128) NOT NULL DEFAULT '',
    owner_agent_id VARCHAR(128) NOT NULL DEFAULT '',
    abstract      TEXT         NOT NULL DEFAULT '',
    overview      TEXT         NOT NULL DEFAULT '',
    messages      JSONB        NOT NULL DEFAULT '[]',
    metadata      JSONB        NOT NULL DEFAULT '{}',
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    PRIMARY KEY (account_id, session_id, archive_id)
);

DO $$ BEGIN
    ALTER TABLE session_archives
        ADD COLUMN owner_user_id VARCHAR(128) NOT NULL DEFAULT '';
EXCEPTION WHEN duplicate_column THEN NULL;
END $$;
DO $$ BEGIN
    ALTER TABLE session_archives
        ADD COLUMN owner_agent_id VARCHAR(128) NOT NULL DEFAULT '';
EXCEPTION WHEN duplicate_column THEN NULL;
END $$;

CREATE INDEX IF NOT EXISTS idx_sa_account_session ON session_archives (account_id, session_id);
CREATE INDEX IF NOT EXISTS idx_sa_session         ON session_archives (session_id);

-- Dream recalls (for recall tracking and consolidation)
CREATE TABLE IF NOT EXISTS dream_recalls (
    id          BIGSERIAL    PRIMARY KEY,
    uri         TEXT         NOT NULL,
    account_id  TEXT         NOT NULL,
    owner_space TEXT         NOT NULL,
    query_text  TEXT,
    score       REAL,
    category    TEXT,
    recalled_at TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_dream_recalls_uri    ON dream_recalls (uri, account_id);
CREATE INDEX IF NOT EXISTS idx_dream_recalls_time   ON dream_recalls (recalled_at);

-- ============================================================
-- Row-Level Security: tenant isolation for Agent-facing tables
-- ============================================================
-- outbox_events is INTENTIONALLY EXCLUDED — the outbox worker
-- needs cross-tenant scan access to claim and dispatch events.
-- All RLS DDL runs inside the same transaction as CREATE TABLE
-- so psycopg2 cur.execute() in non-autocommit mode works reliably.

-- context_nodes
ALTER TABLE context_nodes ENABLE ROW LEVEL SECURITY;
ALTER TABLE context_nodes FORCE ROW LEVEL SECURITY;
DO $$ BEGIN
    CREATE POLICY tenant_isolation ON context_nodes
        USING (account_id = current_setting('app.account_id', true))
        WITH CHECK (account_id = current_setting('app.account_id', true));
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- Content-free recovery directory. The owner backfills existing accounts once;
-- the atomic migration restores tenant enforcement before becoming visible.
DO $$ BEGIN
    CREATE TABLE extraction_work_accounts (account_id VARCHAR(128) PRIMARY KEY);
    ALTER TABLE context_nodes NO FORCE ROW LEVEL SECURITY;
    INSERT INTO extraction_work_accounts (account_id)
        SELECT DISTINCT account_id FROM context_nodes WHERE category = 'state' AND status = 'ACTIVE';
    ALTER TABLE context_nodes FORCE ROW LEVEL SECURITY;
EXCEPTION WHEN duplicate_table THEN NULL;
END $$;

-- relation_edges
ALTER TABLE relation_edges ENABLE ROW LEVEL SECURITY;
ALTER TABLE relation_edges FORCE ROW LEVEL SECURITY;
DO $$ BEGIN
    CREATE POLICY tenant_isolation ON relation_edges
        USING (account_id = current_setting('app.account_id', true))
        WITH CHECK (account_id = current_setting('app.account_id', true));
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- session_archives
ALTER TABLE session_archives ENABLE ROW LEVEL SECURITY;
ALTER TABLE session_archives FORCE ROW LEVEL SECURITY;
DO $$ BEGIN
    CREATE POLICY tenant_isolation ON session_archives
        USING (account_id = current_setting('app.account_id', true))
        WITH CHECK (account_id = current_setting('app.account_id', true));
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- Complete logical tool outputs externalized from ordinary memory messages.
CREATE TABLE IF NOT EXISTS tool_result_artifacts (
    account_id       VARCHAR(128) NOT NULL,
    session_id       VARCHAR(128) NOT NULL,
    owner_user_id    VARCHAR(128) NOT NULL,
    owner_agent_id   VARCHAR(128) NOT NULL,
    artifact_id      VARCHAR(96)  NOT NULL,
    output_text      TEXT         NOT NULL,
    metadata         JSONB        NOT NULL,
    content_sha256   VARCHAR(64)  NOT NULL,
    created_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    PRIMARY KEY (account_id, session_id, owner_user_id, owner_agent_id, artifact_id)
);
CREATE INDEX IF NOT EXISTS idx_tool_result_artifacts_list
    ON tool_result_artifacts
       (account_id, session_id, owner_user_id, owner_agent_id, artifact_id DESC);
ALTER TABLE tool_result_artifacts ENABLE ROW LEVEL SECURITY;
ALTER TABLE tool_result_artifacts FORCE ROW LEVEL SECURITY;
DO $$ BEGIN
    CREATE POLICY tenant_isolation ON tool_result_artifacts
        USING (account_id = current_setting('app.account_id', true))
        WITH CHECK (account_id = current_setting('app.account_id', true));
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- dream_recalls
ALTER TABLE dream_recalls ENABLE ROW LEVEL SECURITY;
ALTER TABLE dream_recalls FORCE ROW LEVEL SECURITY;
DO $$ BEGIN
    CREATE POLICY dream_recalls_tenant ON dream_recalls
        USING (account_id = current_setting('app.account_id', true))
        WITH CHECK (account_id = current_setting('app.account_id', true));
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- ============================================================
-- Control-plane tables (auth, API keys, roles, audit logs)
-- ============================================================
-- No RLS — control-plane ops run server-side with full authority.
-- Tenant isolation is enforced by application layer.

CREATE TABLE IF NOT EXISTS cp_accounts (
    account_id   VARCHAR(128) PRIMARY KEY,
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    status       VARCHAR(16)  NOT NULL DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS cp_users (
    account_id   VARCHAR(128) NOT NULL,
    user_id      VARCHAR(128) NOT NULL,
    role         VARCHAR(16)  NOT NULL DEFAULT 'user',
    key          VARCHAR(64)  NOT NULL,
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    status       VARCHAR(16)  NOT NULL DEFAULT 'active',
    PRIMARY KEY (account_id, user_id)
);
CREATE INDEX IF NOT EXISTS idx_cp_users_key ON cp_users (key) WHERE status = 'active';

CREATE TABLE IF NOT EXISTS cp_agents (
    account_id     VARCHAR(128) NOT NULL,
    agent_id       VARCHAR(128) NOT NULL,
    owner_user_id  VARCHAR(128) NOT NULL,
    created_at     TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    PRIMARY KEY (account_id, agent_id)
);

CREATE TABLE IF NOT EXISTS cp_audit_logs (
    log_id      VARCHAR(36)  PRIMARY KEY,
    account_id  VARCHAR(128) NOT NULL,
    actor       VARCHAR(128) NOT NULL DEFAULT '',
    target      VARCHAR(256) NOT NULL DEFAULT '',
    action      VARCHAR(64)  NOT NULL DEFAULT '',
    timestamp   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    result      VARCHAR(32)  NOT NULL DEFAULT '',
    trace_id    VARCHAR(36)  NOT NULL DEFAULT '',
    details     JSONB        NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_cp_audit_account ON cp_audit_logs (account_id);

-- Protected workspace-sync registry. Canonical local paths stay in the JSON
-- state owned by the server control plane and never enter ContextFS/index
-- metadata. SQL row locks make lease and dirty-state transitions atomic across
-- application workers.
CREATE TABLE IF NOT EXISTS code_workspace_sync_roots (
    account_id       VARCHAR(128) NOT NULL,
    root_id          VARCHAR(128) NOT NULL,
    state            JSONB        NOT NULL DEFAULT '{}',
    lease_owner      VARCHAR(128),
    lease_expires_at TIMESTAMPTZ,
    created_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    PRIMARY KEY (account_id, root_id)
);
CREATE INDEX IF NOT EXISTS idx_workspace_sync_due
    ON code_workspace_sync_roots (account_id, lease_expires_at, updated_at);

-- ============================================================
-- Agent lifecycle traces
-- ============================================================
-- Hook events are stored independently from context_nodes so trace capture
-- never enters the retrieval, extraction, chunking, or vector-index paths.

CREATE SEQUENCE IF NOT EXISTS trace_event_index_seq;

CREATE TABLE IF NOT EXISTS trace_events (
    event_id       VARCHAR(64)  NOT NULL,
    uri            VARCHAR(512) PRIMARY KEY,
    account_id     VARCHAR(128) NOT NULL,
    owner_space    VARCHAR(128) NOT NULL,
    agent_id       VARCHAR(128) NOT NULL,
    session_id     VARCHAR(128) NOT NULL,
    hook_type      VARCHAR(64)  NOT NULL,
    event_index    BIGINT       NOT NULL DEFAULT nextval('trace_event_index_seq'),
    event_time     TIMESTAMPTZ  NOT NULL,
    content        JSONB        NOT NULL DEFAULT '{}',
    abstract       TEXT         NOT NULL DEFAULT '',
    overview       TEXT         NOT NULL DEFAULT '',
    created_at     TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    UNIQUE (account_id, event_id)
);
CREATE INDEX IF NOT EXISTS idx_trace_events_session
    ON trace_events (account_id, owner_space, agent_id, session_id, event_index);
CREATE INDEX IF NOT EXISTS idx_trace_events_batch
    ON trace_events (account_id, owner_space, agent_id, session_id,
                     (COALESCE(content->>'write_batch_id', '')), event_index);
CREATE INDEX IF NOT EXISTS idx_trace_events_hook_time
    ON trace_events (account_id, hook_type, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_trace_events_agent_time
    ON trace_events (account_id, agent_id, event_time DESC);

ALTER TABLE trace_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE trace_events FORCE ROW LEVEL SECURITY;
DO $$ BEGIN
    CREATE POLICY tenant_isolation ON trace_events
        USING (account_id = current_setting('app.account_id', true))
        WITH CHECK (account_id = current_setting('app.account_id', true));
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

CREATE TABLE IF NOT EXISTS trace_sessions (
    turn_id                VARCHAR(64)  NOT NULL,
    uri                    VARCHAR(512) PRIMARY KEY,
    account_id             VARCHAR(128) NOT NULL,
    owner_space            VARCHAR(128) NOT NULL,
    agent_id               VARCHAR(128) NOT NULL,
    session_id             VARCHAR(128) NOT NULL,
    started_at             TIMESTAMPTZ  NOT NULL,
    ended_at               TIMESTAMPTZ  NOT NULL,
    event_count            INTEGER      NOT NULL DEFAULT 0,
    abstract               TEXT         NOT NULL DEFAULT '',
    overview               TEXT         NOT NULL DEFAULT '',
    content                JSONB        NOT NULL DEFAULT '[]',
    metadata               JSONB        NOT NULL DEFAULT '{}',
    status                 VARCHAR(16)  NOT NULL DEFAULT 'finalized',
    stop_event_index       BIGINT       NOT NULL,
    prev_stop_event_index  BIGINT,
    created_at             TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at             TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    UNIQUE (account_id, turn_id)
);

CREATE INDEX IF NOT EXISTS idx_trace_sessions_session
    ON trace_sessions (account_id, session_id, stop_event_index);
CREATE INDEX IF NOT EXISTS idx_trace_sessions_time
    ON trace_sessions (account_id, ended_at DESC);

CREATE TABLE IF NOT EXISTS trace_session_events (
    account_id    VARCHAR(128) NOT NULL,
    evidence_uri  VARCHAR(512) NOT NULL,
    ordinal       BIGINT       NOT NULL,
    event         JSONB        NOT NULL DEFAULT '{}',
    PRIMARY KEY (account_id, evidence_uri, ordinal),
    FOREIGN KEY (evidence_uri) REFERENCES trace_sessions(uri) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS skill_candidates (
    candidate_id           VARCHAR(512) PRIMARY KEY,
    account_id             VARCHAR(128) NOT NULL,
    agent_id               VARCHAR(128) NOT NULL,
    capability_key         VARCHAR(128),
    state                  VARCHAR(16)  NOT NULL DEFAULT 'pending'
                           CHECK (state IN ('pending', 'processing', 'assessed', 'consumed', 'skipped', 'held')),
    assessment             JSONB        NOT NULL DEFAULT '{}',
    attempts               INTEGER      NOT NULL DEFAULT 0,
    compile_attempts       INTEGER      NOT NULL DEFAULT 0,
    next_attempt_at        TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    lease_owner            VARCHAR(128),
    lease_until            TIMESTAMPTZ,
    skill_uri              VARCHAR(512),
    consumed_at            TIMESTAMPTZ,
    error                  TEXT,
    created_at             TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at             TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    UNIQUE (account_id, candidate_id)
);

CREATE TABLE IF NOT EXISTS skill_candidate_evidence (
    account_id             VARCHAR(128) NOT NULL,
    candidate_id           VARCHAR(512) NOT NULL,
    evidence_uri           VARCHAR(512) NOT NULL,
    evidence_type          VARCHAR(32)  NOT NULL DEFAULT 'trace',
    session_id             VARCHAR(128) NOT NULL,
    created_at             TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    PRIMARY KEY (account_id, candidate_id, evidence_uri),
    FOREIGN KEY (account_id, candidate_id)
        REFERENCES skill_candidates(account_id, candidate_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_skill_candidates_pending
    ON skill_candidates (account_id, next_attempt_at, created_at)
    WHERE state IN ('pending', 'processing');
CREATE INDEX IF NOT EXISTS idx_skill_candidates_capability
    ON skill_candidates (
        account_id, agent_id, capability_key,
        next_attempt_at, created_at
    )
    WHERE state IN ('assessed', 'processing');
CREATE INDEX IF NOT EXISTS idx_skill_candidate_evidence_uri
    ON skill_candidate_evidence (account_id, evidence_uri);
CREATE INDEX IF NOT EXISTS idx_skill_candidate_evidence_session
    ON skill_candidate_evidence (account_id, session_id);

-- Durable support ledger for exact worked-example session accounting. Example
-- hashes prevent revised content from inheriting support from an older example
-- that used the same positional identifier.
CREATE TABLE IF NOT EXISTS skill_example_evidence (
    account_id             VARCHAR(128) NOT NULL,
    skill_uri              VARCHAR(512) NOT NULL,
    example_id             VARCHAR(64)  NOT NULL,
    example_hash           CHAR(64)     NOT NULL,
    evidence_uri           VARCHAR(512) NOT NULL,
    session_id             VARCHAR(128) NOT NULL,
    created_at             TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    PRIMARY KEY (
        account_id, skill_uri, example_id, example_hash, evidence_uri
    )
);

CREATE INDEX IF NOT EXISTS idx_skill_example_evidence_sessions
    ON skill_example_evidence (
        account_id, skill_uri, example_id, example_hash, session_id
    );

-- Durable scheduler queue. It contains tenant identifiers only; workers bind
-- tenant RLS before reading trace content. This is not a skill registry.
CREATE TABLE IF NOT EXISTS skill_work_accounts (
    account_id  VARCHAR(128) PRIMARY KEY,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

-- Trace length is no longer an eligibility signal. Requeue only candidates
-- rejected by the retired total-character guard; all other terminal decisions
-- remain untouched. Existing installations already FORCE RLS on this table,
-- so the table owner must temporarily regain its migration-only bypass. The
-- DO statement is atomic: an error restores the original FORCE setting.
DO $$ BEGIN
    ALTER TABLE skill_candidates NO FORCE ROW LEVEL SECURITY;
    UPDATE skill_candidates
    SET state = 'pending', error = NULL, next_attempt_at = NOW(), updated_at = NOW()
    WHERE state = 'skipped'
      AND error = 'trace exceeds configured assessment character budget';
    ALTER TABLE skill_candidates FORCE ROW LEVEL SECURITY;
END $$;

ALTER TABLE trace_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE trace_sessions FORCE ROW LEVEL SECURITY;
DO $$ BEGIN
    CREATE POLICY tenant_isolation ON trace_sessions
        USING (account_id = current_setting('app.account_id', true))
        WITH CHECK (account_id = current_setting('app.account_id', true));
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

ALTER TABLE trace_session_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE trace_session_events FORCE ROW LEVEL SECURITY;
DO $$ BEGIN
    CREATE POLICY tenant_isolation ON trace_session_events
        USING (account_id = current_setting('app.account_id', true))
        WITH CHECK (account_id = current_setting('app.account_id', true));
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

ALTER TABLE skill_candidates ENABLE ROW LEVEL SECURITY;
ALTER TABLE skill_candidates FORCE ROW LEVEL SECURITY;
DO $$ BEGIN
    CREATE POLICY tenant_isolation ON skill_candidates
        USING (account_id = current_setting('app.account_id', true))
        WITH CHECK (account_id = current_setting('app.account_id', true));
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

ALTER TABLE skill_candidate_evidence ENABLE ROW LEVEL SECURITY;
ALTER TABLE skill_candidate_evidence FORCE ROW LEVEL SECURITY;
DO $$ BEGIN
    CREATE POLICY tenant_isolation ON skill_candidate_evidence
        USING (account_id = current_setting('app.account_id', true))
        WITH CHECK (account_id = current_setting('app.account_id', true));
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

ALTER TABLE skill_example_evidence ENABLE ROW LEVEL SECURITY;
ALTER TABLE skill_example_evidence FORCE ROW LEVEL SECURITY;
DO $$ BEGIN
    CREATE POLICY tenant_isolation ON skill_example_evidence
        USING (account_id = current_setting('app.account_id', true))
        WITH CHECK (account_id = current_setting('app.account_id', true));
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;
