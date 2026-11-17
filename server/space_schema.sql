ALTER TABLE cp_accounts ADD COLUMN IF NOT EXISTS authz_rev BIGINT NOT NULL DEFAULT 1;
ALTER TABLE cp_agents ADD COLUMN IF NOT EXISTS status VARCHAR(16) NOT NULL DEFAULT 'active';
CREATE UNIQUE INDEX IF NOT EXISTS uq_context_account_uri ON context_nodes(account_id,uri);

CREATE TABLE IF NOT EXISTS spaces (
    account_id VARCHAR(128) NOT NULL,
    space_id VARCHAR(64) NOT NULL,
    name VARCHAR(128) NOT NULL,
    owner_user_id VARCHAR(128) NOT NULL,
    state VARCHAR(16) NOT NULL DEFAULT 'ACTIVE',
    authz_rev BIGINT NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (account_id,space_id),
    FOREIGN KEY (account_id) REFERENCES cp_accounts(account_id),
    FOREIGN KEY (account_id,owner_user_id) REFERENCES cp_users(account_id,user_id),
    CHECK (state IN ('ACTIVE','ARCHIVED')),
    CHECK (authz_rev>0)
);
CREATE INDEX IF NOT EXISTS idx_spaces_list ON spaces(account_id,state,space_id);
ALTER TABLE spaces ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ;
DROP INDEX IF EXISTS uq_spaces_account_name;
CREATE UNIQUE INDEX uq_spaces_account_name ON spaces(account_id,btrim(name)) WHERE deleted_at IS NULL;

CREATE TABLE IF NOT EXISTS space_members (
    account_id VARCHAR(128) NOT NULL,
    space_id VARCHAR(64) NOT NULL,
    principal VARCHAR(134) NOT NULL,
    user_id VARCHAR(128),
    agent_id VARCHAR(128),
    role VARCHAR(16) NOT NULL DEFAULT 'member',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (account_id,space_id,principal),
    FOREIGN KEY (account_id,space_id) REFERENCES spaces(account_id,space_id),
    FOREIGN KEY (account_id,user_id) REFERENCES cp_users(account_id,user_id),
    FOREIGN KEY (account_id,agent_id) REFERENCES cp_agents(account_id,agent_id),
    CHECK ((user_id IS NOT NULL AND agent_id IS NULL AND principal='user:'||user_id) OR (agent_id IS NOT NULL AND user_id IS NULL AND principal='agent:'||agent_id)),
    CHECK (role IN ('manager','member')),
    CHECK (role<>'manager' OR user_id IS NOT NULL)
);
CREATE INDEX IF NOT EXISTS idx_members_principal ON space_members(account_id,principal,space_id);

CREATE TABLE IF NOT EXISTS space_source_bindings (
    account_id VARCHAR(128) NOT NULL,
    space_id VARCHAR(64) NOT NULL,
    binding_id VARCHAR(64) NOT NULL,
    principal VARCHAR(134) NOT NULL,
    memory_types JSONB NOT NULL DEFAULT '[]',
    authorized_by VARCHAR(128) NOT NULL,
    grant_rev BIGINT NOT NULL DEFAULT 1,
    state VARCHAR(16) NOT NULL DEFAULT 'ACTIVE',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (account_id,space_id,binding_id),
    FOREIGN KEY (account_id,space_id) REFERENCES spaces(account_id,space_id),
    FOREIGN KEY (account_id,authorized_by) REFERENCES cp_users(account_id,user_id),
    CHECK (state IN ('ACTIVE','PAUSED')),
    CHECK (grant_rev>0)
);
CREATE INDEX IF NOT EXISTS idx_bindings_source ON space_source_bindings(account_id,principal,state,space_id);
ALTER TABLE space_source_bindings ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ;
ALTER TABLE space_source_bindings DROP CONSTRAINT IF EXISTS space_source_bindings_account_id_space_id_principal_key;
CREATE UNIQUE INDEX IF NOT EXISTS uq_space_sources_principal ON space_source_bindings(account_id,space_id,principal) WHERE deleted_at IS NULL;

CREATE TABLE IF NOT EXISTS space_rules (
    account_id VARCHAR(128) NOT NULL,
    space_id VARCHAR(64) NOT NULL,
    rule_id VARCHAR(64) NOT NULL,
    name VARCHAR(128) NOT NULL,
    rule_rev BIGINT NOT NULL DEFAULT 1,
    state VARCHAR(16) NOT NULL DEFAULT 'ACTIVE',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (account_id,space_id,rule_id),
    FOREIGN KEY (account_id,space_id) REFERENCES spaces(account_id,space_id),
    CHECK (state IN ('ACTIVE','PAUSED','ARCHIVED')),
    CHECK (rule_rev>0)
);
ALTER TABLE space_rules ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ;

CREATE TABLE IF NOT EXISTS space_rule_revisions (
    account_id VARCHAR(128) NOT NULL,
    space_id VARCHAR(64) NOT NULL,
    name VARCHAR(128) NOT NULL,
    state VARCHAR(16) NOT NULL CHECK (state IN ('ACTIVE','PAUSED','ARCHIVED')),
    rule_id VARCHAR(64) NOT NULL,
    rule_rev BIGINT NOT NULL,
    spec JSONB NOT NULL,
    spec_hash CHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (account_id,space_id,rule_id,rule_rev),
    FOREIGN KEY (account_id,space_id,rule_id) REFERENCES space_rules(account_id,space_id,rule_id),
    CHECK (rule_rev>0)
);

CREATE TABLE IF NOT EXISTS space_rule_runtime (
    account_id VARCHAR(128) NOT NULL,
    space_id VARCHAR(64) NOT NULL,
    rule_id VARCHAR(64) NOT NULL,
    dirty_generation BIGINT NOT NULL DEFAULT 0,
    applied_generation BIGINT NOT NULL DEFAULT 0,
    next_run_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (account_id,space_id,rule_id),
    FOREIGN KEY (account_id,space_id,rule_id) REFERENCES space_rules(account_id,space_id,rule_id),
    CHECK (dirty_generation>=applied_generation AND applied_generation>=0)
);
CREATE INDEX IF NOT EXISTS idx_runtime_due ON space_rule_runtime(account_id,next_run_at,space_id,rule_id);

CREATE TABLE IF NOT EXISTS space_items (
    account_id VARCHAR(128) NOT NULL,
    space_id VARCHAR(64) NOT NULL,
    topic_key VARCHAR(128) NOT NULL,
    evidence_fingerprint CHAR(64) NOT NULL,
    item_id VARCHAR(64) NOT NULL,
    rule_id VARCHAR(64) NOT NULL,
    rule_rev BIGINT NOT NULL,
    uri VARCHAR(512) NOT NULL,
    kind VARCHAR(16) NOT NULL,
    state VARCHAR(16) NOT NULL DEFAULT 'ACTIVE',
    acl_mode VARCHAR(16) NOT NULL DEFAULT 'inherit',
    acl_origin VARCHAR(16) NOT NULL DEFAULT 'rule',
    quality_score DOUBLE PRECISION NOT NULL,
    last_used_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ NOT NULL,
    purge_after TIMESTAMPTZ,
    policy_snapshot JSONB NOT NULL,
    superseded_by VARCHAR(64),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (account_id,space_id,item_id),
    UNIQUE (account_id,uri),
    FOREIGN KEY (account_id,space_id,rule_id,rule_rev) REFERENCES space_rule_revisions(account_id,space_id,rule_id,rule_rev),
    FOREIGN KEY (account_id,uri) REFERENCES context_nodes(account_id,uri),
    CHECK (kind IN ('experience','skill')),
    CHECK (state IN ('ACTIVE','RECOMPUTING','SUPERSEDED','EXPIRED','ARCHIVED','PURGED')),
    CHECK (acl_mode IN ('inherit','restricted')),
    CHECK (acl_origin IN ('rule','manual')),
    CHECK (quality_score BETWEEN 0 AND 1)
);
CREATE INDEX IF NOT EXISTS idx_items_read ON space_items(account_id,space_id,state,item_id);
CREATE UNIQUE INDEX IF NOT EXISTS uq_items_current_evidence ON space_items(account_id,space_id,rule_id,rule_rev,topic_key,evidence_fingerprint)
WHERE state IN ('ACTIVE','RECOMPUTING');
DROP INDEX IF EXISTS uq_items_evidence;
CREATE INDEX IF NOT EXISTS idx_items_expire ON space_items(account_id,state,expires_at);

CREATE TABLE IF NOT EXISTS space_item_sources (
    account_id VARCHAR(128) NOT NULL,
    space_id VARCHAR(64) NOT NULL,
    item_id VARCHAR(64) NOT NULL,
    binding_id VARCHAR(64) NOT NULL,
    source_uri VARCHAR(512) NOT NULL,
    grant_rev BIGINT NOT NULL,
    source_content_rev BIGINT NOT NULL,
    source_hash CHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (account_id,space_id,item_id,binding_id,source_uri),
    FOREIGN KEY (account_id,space_id,item_id) REFERENCES space_items(account_id,space_id,item_id),
    FOREIGN KEY (account_id,space_id,binding_id) REFERENCES space_source_bindings(account_id,space_id,binding_id)
);
CREATE INDEX IF NOT EXISTS idx_sources_uri ON space_item_sources(account_id,source_uri,space_id,item_id);

CREATE TABLE IF NOT EXISTS space_read_acl (
    account_id VARCHAR(128) NOT NULL,
    space_id VARCHAR(64) NOT NULL,
    item_id VARCHAR(64) NOT NULL,
    principal VARCHAR(134) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (account_id,space_id,item_id,principal),
    FOREIGN KEY (account_id,space_id,item_id) REFERENCES space_items(account_id,space_id,item_id),
    FOREIGN KEY (account_id,space_id,principal) REFERENCES space_members(account_id,space_id,principal) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_read_acl_subject ON space_read_acl(account_id,space_id,principal,item_id);

CREATE TABLE IF NOT EXISTS space_jobs (
    account_id VARCHAR(128) NOT NULL,
    space_id VARCHAR(64) NOT NULL,
    job_id VARCHAR(64) NOT NULL,
    rule_id VARCHAR(64) NOT NULL,
    rule_rev BIGINT NOT NULL,
    operation VARCHAR(16) NOT NULL,
    generation BIGINT NOT NULL,
    state VARCHAR(16) NOT NULL DEFAULT 'QUEUED',
    snapshot JSONB NOT NULL,
    result JSONB NOT NULL DEFAULT '{}',
    claim_token VARCHAR(36),
    lease_until TIMESTAMPTZ,
    attempts INTEGER NOT NULL DEFAULT 0,
    next_run_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (account_id,space_id,job_id),
    UNIQUE (account_id,space_id,rule_id,rule_rev,operation,generation),
    FOREIGN KEY (account_id,space_id,rule_id,rule_rev) REFERENCES space_rule_revisions(account_id,space_id,rule_id,rule_rev),
    CHECK (operation IN ('GENERATE','RECOMPUTE','MAINTAIN','PURGE')),
    CHECK (state IN ('QUEUED','RUNNING','SUCCEEDED','SUPERSEDED','FAILED')),
    CHECK (attempts BETWEEN 0 AND 5)
);
CREATE INDEX IF NOT EXISTS idx_jobs_queue ON space_jobs(account_id,state,next_run_at,job_id);
CREATE INDEX IF NOT EXISTS idx_jobs_lease ON space_jobs(state,lease_until);

CREATE UNIQUE INDEX IF NOT EXISTS uq_jobs_active_rule ON space_jobs(account_id,space_id,rule_id)
 WHERE state IN ('QUEUED','RUNNING');
CREATE TABLE IF NOT EXISTS space_requests (
    account_id VARCHAR(128) NOT NULL,
    actor_id VARCHAR(128) NOT NULL,
    operation VARCHAR(256) NOT NULL,
    request_key VARCHAR(128) NOT NULL,
    request_hash CHAR(64) NOT NULL,
    response_body JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (account_id,actor_id,operation,request_key),
    FOREIGN KEY (account_id) REFERENCES cp_accounts(account_id)
);
CREATE INDEX IF NOT EXISTS idx_space_requests_expiry ON space_requests(expires_at);

DROP INDEX IF EXISTS idx_rules_creator;
ALTER TABLE space_rules DROP COLUMN IF EXISTS created_by;
ALTER TABLE space_rule_revisions DROP COLUMN IF EXISTS saved_by;
ALTER TABLE space_rule_runtime DROP COLUMN IF EXISTS source_cursor;
ALTER TABLE space_rule_runtime DROP COLUMN IF EXISTS effective_sources;
ALTER TABLE space_requests DROP COLUMN IF EXISTS response_status;

ALTER TABLE spaces ENABLE ROW LEVEL SECURITY;
ALTER TABLE spaces FORCE ROW LEVEL SECURITY;
DO $$ BEGIN
 CREATE POLICY tenant_isolation ON spaces
 USING (account_id = current_setting('app.account_id', true))
 WITH CHECK (account_id = current_setting('app.account_id', true));
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

ALTER TABLE space_members ENABLE ROW LEVEL SECURITY;
ALTER TABLE space_members FORCE ROW LEVEL SECURITY;
DO $$ BEGIN
 CREATE POLICY tenant_isolation ON space_members
 USING (account_id = current_setting('app.account_id', true))
 WITH CHECK (account_id = current_setting('app.account_id', true));
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

ALTER TABLE space_source_bindings ENABLE ROW LEVEL SECURITY;
ALTER TABLE space_source_bindings FORCE ROW LEVEL SECURITY;
DO $$ BEGIN
 CREATE POLICY tenant_isolation ON space_source_bindings
 USING (account_id = current_setting('app.account_id', true))
 WITH CHECK (account_id = current_setting('app.account_id', true));
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

ALTER TABLE space_rules ENABLE ROW LEVEL SECURITY;
ALTER TABLE space_rules FORCE ROW LEVEL SECURITY;
DO $$ BEGIN
 CREATE POLICY tenant_isolation ON space_rules
 USING (account_id = current_setting('app.account_id', true))
 WITH CHECK (account_id = current_setting('app.account_id', true));
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

ALTER TABLE space_rule_revisions ENABLE ROW LEVEL SECURITY;
ALTER TABLE space_rule_revisions FORCE ROW LEVEL SECURITY;
DO $$ BEGIN
 CREATE POLICY tenant_isolation ON space_rule_revisions
 USING (account_id = current_setting('app.account_id', true))
 WITH CHECK (account_id = current_setting('app.account_id', true));
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

ALTER TABLE space_rule_runtime ENABLE ROW LEVEL SECURITY;
ALTER TABLE space_rule_runtime FORCE ROW LEVEL SECURITY;
DO $$ BEGIN
 CREATE POLICY tenant_isolation ON space_rule_runtime
 USING (account_id = current_setting('app.account_id', true))
 WITH CHECK (account_id = current_setting('app.account_id', true));
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

ALTER TABLE space_items ENABLE ROW LEVEL SECURITY;
ALTER TABLE space_items FORCE ROW LEVEL SECURITY;
DO $$ BEGIN
 CREATE POLICY tenant_isolation ON space_items
 USING (account_id = current_setting('app.account_id', true))
 WITH CHECK (account_id = current_setting('app.account_id', true));
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

ALTER TABLE space_item_sources ENABLE ROW LEVEL SECURITY;
ALTER TABLE space_item_sources FORCE ROW LEVEL SECURITY;
DO $$ BEGIN
 CREATE POLICY tenant_isolation ON space_item_sources
 USING (account_id = current_setting('app.account_id', true))
 WITH CHECK (account_id = current_setting('app.account_id', true));
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

ALTER TABLE space_read_acl ENABLE ROW LEVEL SECURITY;
ALTER TABLE space_read_acl FORCE ROW LEVEL SECURITY;
DO $$ BEGIN
 CREATE POLICY tenant_isolation ON space_read_acl
 USING (account_id = current_setting('app.account_id', true))
 WITH CHECK (account_id = current_setting('app.account_id', true));
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

ALTER TABLE space_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE space_jobs FORCE ROW LEVEL SECURITY;
DO $$ BEGIN
 CREATE POLICY tenant_isolation ON space_jobs
 USING (account_id = current_setting('app.account_id', true))
 WITH CHECK (account_id = current_setting('app.account_id', true));
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

ALTER TABLE space_requests ENABLE ROW LEVEL SECURITY;
ALTER TABLE space_requests FORCE ROW LEVEL SECURITY;
DO $$ BEGIN
 CREATE POLICY tenant_isolation ON space_requests
 USING (account_id = current_setting('app.account_id', true))
 WITH CHECK (account_id = current_setting('app.account_id', true));
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

CREATE INDEX IF NOT EXISTS idx_space_content_text ON context_nodes USING GIN (to_tsvector('simple',abstract||' '||content)) WHERE owner_space LIKE 'space:%';

ALTER TABLE space_item_sources ADD COLUMN IF NOT EXISTS source_epoch CHAR(32) NOT NULL DEFAULT '';
ALTER TABLE space_items ADD COLUMN IF NOT EXISTS checked_rule_rev BIGINT;

CREATE OR REPLACE FUNCTION ogmemory_space_source_revision() RETURNS trigger AS $$
DECLARE
    source_epoch text;
BEGIN
    IF NEW.uri ~ '^ctx://[^/]+/(users|agents)/[^/]+/memories/' THEN
        IF TG_OP = 'UPDATE' THEN
            source_epoch := NULLIF(OLD.metadata->>'shared_source_epoch','');
            NEW.version := GREATEST(NEW.version, OLD.version);
            IF (NEW.uri, NEW.status, NEW.content, NEW.abstract, NEW.overview) IS DISTINCT FROM
               (OLD.uri, OLD.status, OLD.content, OLD.abstract, OLD.overview)
               OR (NEW.metadata->'agfs_content'->>'sha256') IS DISTINCT FROM
                  (OLD.metadata->'agfs_content'->>'sha256') THEN
                NEW.version := GREATEST(NEW.version, OLD.version + 1);
            END IF;
        END IF;
        NEW.metadata := NEW.metadata || json_build_object(
            'shared_source_epoch', COALESCE(source_epoch,md5(random()::text || clock_timestamp()::text || NEW.uri)),
            'version', NEW.version)::jsonb;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS ogmemory_space_source_revision ON context_nodes;
CREATE TRIGGER ogmemory_space_source_revision BEFORE INSERT OR UPDATE ON context_nodes
FOR EACH ROW EXECUTE PROCEDURE ogmemory_space_source_revision();

CREATE INDEX IF NOT EXISTS idx_cn_agfs_origin
ON context_nodes(account_id,(metadata->'agfs_content'->>'uri'))
WHERE (metadata->'agfs_content'->>'uri') IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_oe_agfs_reclaim
ON outbox_events(account_id,seq)
WHERE status='GC_PENDING' AND event_type IN ('RECLAIM_AGFS','RECLAIM_AGFS_STAGE');

CREATE OR REPLACE FUNCTION ogmemory_queue_agfs_cleanup() RETURNS trigger AS $$
DECLARE
    origin_uri text := OLD.metadata->'agfs_content'->>'uri';
BEGIN
    IF TG_OP = 'UPDATE' THEN
        IF origin_uri IS NOT DISTINCT FROM (NEW.metadata->'agfs_content'->>'uri') THEN
            RETURN NEW;
        END IF;
    END IF;
    IF left(origin_uri,6)='ctx://' AND split_part(origin_uri,'/',3)=OLD.account_id THEN
        INSERT INTO outbox_events(event_id,event_type,uri,account_id,status)
        VALUES(md5(random()::text || clock_timestamp()::text || OLD.uri),
               'RECLAIM_AGFS',origin_uri,OLD.account_id,'GC_PENDING');
    END IF;
    RETURN OLD;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS ogmemory_queue_agfs_cleanup ON context_nodes;
CREATE TRIGGER ogmemory_queue_agfs_cleanup AFTER DELETE OR UPDATE OF metadata ON context_nodes
FOR EACH ROW EXECUTE PROCEDURE ogmemory_queue_agfs_cleanup();

ALTER TABLE space_rule_runtime ADD COLUMN IF NOT EXISTS source_after TEXT NOT NULL DEFAULT '';
CREATE INDEX IF NOT EXISTS idx_jobs_completed_sources
ON space_jobs(account_id,space_id,rule_id,rule_rev,(result->>'source_fingerprint'))
WHERE state='SUCCEEDED';
CREATE INDEX IF NOT EXISTS idx_jobs_completed_rule
ON space_jobs(account_id,space_id,rule_id,rule_rev,generation DESC)
WHERE state='SUCCEEDED' AND operation IN ('GENERATE','RECOMPUTE');
