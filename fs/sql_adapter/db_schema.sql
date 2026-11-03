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
