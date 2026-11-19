CREATE TABLE IF NOT EXISTS bot_session (
 account_id VARCHAR(128) NOT NULL, user_id VARCHAR(128) NOT NULL,
 session_id VARCHAR(64) NOT NULL, title VARCHAR(128) NOT NULL,
 scope JSONB NOT NULL DEFAULT '{"memory_scope":"user","resource_ids":[]}',
 agent_id VARCHAR(128) NOT NULL DEFAULT 'memorybot', session_generation BIGINT NOT NULL DEFAULT 1,
 next_seq BIGINT NOT NULL DEFAULT 1, revision BIGINT NOT NULL DEFAULT 1,
 archived BOOLEAN NOT NULL DEFAULT FALSE,
 created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
 updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
 PRIMARY KEY(account_id,session_id), UNIQUE(account_id,user_id,session_id),
 CHECK(next_seq>=1 AND revision>=1 AND session_generation>=1)
);
ALTER TABLE bot_session ADD COLUMN IF NOT EXISTS archived BOOLEAN NOT NULL DEFAULT FALSE;
CREATE INDEX IF NOT EXISTS bot_session_recent ON bot_session
 (account_id,user_id,archived,updated_at DESC,session_id DESC);
CREATE TABLE IF NOT EXISTS bot_schedule (
 account_id VARCHAR(128) NOT NULL, user_id VARCHAR(128) NOT NULL,
 schedule_id VARCHAR(64) NOT NULL, session_id VARCHAR(64) NOT NULL,
 name VARCHAR(128) NOT NULL, task_input JSONB NOT NULL, accepted_authz JSONB NOT NULL,
 rule JSONB NOT NULL, timezone VARCHAR(64) NOT NULL, condition JSONB,
 baseline JSONB NOT NULL DEFAULT '{}', state VARCHAR(16) NOT NULL DEFAULT 'active',
 next_run_at TIMESTAMPTZ, revision INTEGER NOT NULL DEFAULT 1,
 created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
 updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
 PRIMARY KEY(account_id,schedule_id), UNIQUE(account_id,user_id,schedule_id),
 FOREIGN KEY(account_id,user_id,session_id) REFERENCES bot_session(account_id,user_id,session_id),
 CHECK(state IN ('active','paused','cancelled','completed')),
 CHECK(revision>=1), CHECK((state='active')=(next_run_at IS NOT NULL))
);
CREATE TABLE IF NOT EXISTS bot_task (
 account_id VARCHAR(128) NOT NULL, user_id VARCHAR(128) NOT NULL,
 task_id VARCHAR(64) NOT NULL, session_id VARCHAR(64), parent_task_id VARCHAR(64),
 kind VARCHAR(16) NOT NULL, seq BIGINT, request_id VARCHAR(64) NOT NULL,
 input JSONB NOT NULL, accepted_authz JSONB NOT NULL,
 state VARCHAR(24) NOT NULL DEFAULT 'queued', result JSONB,
 memory_status VARCHAR(20) NOT NULL DEFAULT 'not_requested', memory_operation_id VARCHAR(128),
 schedule_id VARCHAR(64), scheduled_at TIMESTAMPTZ, schedule_revision INTEGER, schedule_snapshot JSONB,
 owner_id VARCHAR(64), attempt INTEGER NOT NULL DEFAULT 0, lease_until TIMESTAMPTZ,
 retry_count INTEGER NOT NULL DEFAULT 0, next_attempt_at TIMESTAMPTZ,
 cancel_requested BOOLEAN NOT NULL DEFAULT FALSE, error JSONB,
 next_event_seq BIGINT NOT NULL DEFAULT 1, finished_at TIMESTAMPTZ,
 created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
 updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
 PRIMARY KEY(account_id,task_id), UNIQUE(account_id,user_id,task_id),
 UNIQUE(account_id,user_id,task_id,parent_task_id),
 FOREIGN KEY(account_id,user_id,session_id) REFERENCES bot_session(account_id,user_id,session_id),
 FOREIGN KEY(account_id,user_id,parent_task_id) REFERENCES bot_task(account_id,user_id,task_id),
 FOREIGN KEY(account_id,user_id,schedule_id) REFERENCES bot_schedule(account_id,user_id,schedule_id),
 CHECK(kind IN ('chat','ingest','memory_write','delivery')),
 CHECK((kind='chat' AND session_id IS NOT NULL AND seq IS NOT NULL AND seq>=1) OR (kind<>'chat' AND seq IS NULL)),
 CHECK(state IN ('queued','running','retry_wait','waiting_confirmation','succeeded','cancelled','failed')),
 CHECK(memory_status IN ('not_requested','pending','stored','searchable','unchanged','failed')),
 CHECK(attempt>=0 AND retry_count>=0 AND next_event_seq>=1),
 CHECK(state<>'running' OR (owner_id IS NOT NULL AND lease_until IS NOT NULL)),
 CHECK(state<>'retry_wait' OR next_attempt_at IS NOT NULL),
 CHECK(state NOT IN ('succeeded','cancelled','failed') OR finished_at IS NOT NULL),
 CHECK(state<>'succeeded' OR result IS NOT NULL),
 CHECK((schedule_id IS NULL AND scheduled_at IS NULL AND schedule_revision IS NULL AND schedule_snapshot IS NULL) OR
       (kind='chat' AND schedule_id IS NOT NULL AND scheduled_at IS NOT NULL AND schedule_revision IS NOT NULL AND schedule_revision>=1 AND schedule_snapshot IS NOT NULL))
);
CREATE TABLE IF NOT EXISTS bot_step (
 account_id VARCHAR(128) NOT NULL, user_id VARCHAR(128) NOT NULL,
 task_id VARCHAR(64) NOT NULL, step_no INTEGER NOT NULL, kind VARCHAR(32) NOT NULL,
 operation_id VARCHAR(64) NOT NULL, state VARCHAR(24) NOT NULL DEFAULT 'pending',
 input JSONB NOT NULL, output JSONB, external_receipt JSONB,
 public_progress JSONB NOT NULL DEFAULT '{}',
 created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
 updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
 PRIMARY KEY(account_id,task_id,step_no), UNIQUE(account_id,operation_id),
 FOREIGN KEY(account_id,user_id,task_id) REFERENCES bot_task(account_id,user_id,task_id),
 CHECK(step_no>=1), CHECK(state IN ('pending','running','waiting_confirmation','succeeded','failed','cancelled')),
 CHECK(kind IN ('retrieve','model','tool','file_publish','memory_submit','memory_poll','memory_apply')),
 CHECK(state<>'succeeded' OR output IS NOT NULL)
);
ALTER TABLE bot_step ADD COLUMN IF NOT EXISTS public_progress JSONB NOT NULL DEFAULT '{}';
ALTER TABLE bot_step DROP CONSTRAINT IF EXISTS bot_step_state_check;
ALTER TABLE bot_step ADD CONSTRAINT bot_step_state_check
 CHECK(state IN ('pending','running','waiting_confirmation','succeeded','failed','cancelled'));
CREATE TABLE IF NOT EXISTS bot_event (
 account_id VARCHAR(128) NOT NULL, user_id VARCHAR(128) NOT NULL,
 task_id VARCHAR(64) NOT NULL, event_seq BIGINT NOT NULL, type VARCHAR(24) NOT NULL,
 payload JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
 PRIMARY KEY(account_id,task_id,event_seq),
 FOREIGN KEY(account_id,user_id,task_id) REFERENCES bot_task(account_id,user_id,task_id),
 CHECK(event_seq>=1),
 CHECK(type IN ('progress','answer.delta','answer.reset','task.completed','task.updated','memory.updated','delivery.updated','step.started','step.completed','step.failed','step.cancelled','step.updated'))
);
ALTER TABLE bot_event DROP CONSTRAINT IF EXISTS bot_event_type_check;
ALTER TABLE bot_event ADD CONSTRAINT bot_event_type_check
 CHECK(type IN ('progress','answer.delta','answer.reset','task.completed','task.updated','memory.updated','delivery.updated','step.started','step.completed','step.failed','step.cancelled','step.updated'));
CREATE TABLE IF NOT EXISTS bot_file (
 account_id VARCHAR(128) NOT NULL, user_id VARCHAR(128) NOT NULL,
 file_id VARCHAR(64) NOT NULL, purpose VARCHAR(16) NOT NULL, source_kind VARCHAR(16) NOT NULL,
 task_id VARCHAR(64) NOT NULL, previous_file_id VARCHAR(64), name VARCHAR(255) NOT NULL,
 mime_type VARCHAR(128), size_bytes BIGINT, storage_key VARCHAR(1024), sha256 CHAR(64), source_url VARCHAR(2048),
 scope JSONB NOT NULL DEFAULT '{"memory_scope":"user","resource_ids":[]}',
 state VARCHAR(16) NOT NULL DEFAULT 'processing',
 metadata JSONB NOT NULL DEFAULT '{"summary":null,"source_time":null,"parsed_ranges":[],"memory_sources":[]}',
 created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
 updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
 PRIMARY KEY(account_id,file_id), UNIQUE(account_id,user_id,file_id), UNIQUE(account_id,storage_key),
 FOREIGN KEY(account_id,user_id,task_id) REFERENCES bot_task(account_id,user_id,task_id),
 FOREIGN KEY(account_id,user_id,previous_file_id) REFERENCES bot_file(account_id,user_id,file_id),
 CHECK(purpose IN ('source','artifact')), CHECK(source_kind IN ('upload','url','generated')),
 CHECK(purpose<>'artifact' OR source_kind='generated'), CHECK(source_kind<>'url' OR source_url IS NOT NULL),
 CHECK(state IN ('processing','ready','failed')), CHECK(size_bytes>=0 AND size_bytes<=52428800),
 CHECK(state<>'ready' OR (mime_type IS NOT NULL AND size_bytes IS NOT NULL AND storage_key IS NOT NULL AND sha256 IS NOT NULL))
);
CREATE TABLE IF NOT EXISTS bot_delivery (
 account_id VARCHAR(128) NOT NULL, user_id VARCHAR(128) NOT NULL,
 delivery_id VARCHAR(64) NOT NULL, task_id VARCHAR(64) NOT NULL, source_task_id VARCHAR(64) NOT NULL,
 channel VARCHAR(16) NOT NULL DEFAULT 'web', target_id VARCHAR(256) NOT NULL,
 operation_id VARCHAR(64) NOT NULL, content JSONB NOT NULL,
 state VARCHAR(24) NOT NULL DEFAULT 'pending', provider_receipt JSONB,
 retry_count INTEGER NOT NULL DEFAULT 0, next_attempt_at TIMESTAMPTZ,
 created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
 updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
 PRIMARY KEY(account_id,delivery_id), UNIQUE(account_id,task_id), UNIQUE(account_id,operation_id),
 UNIQUE(account_id,source_task_id,channel,target_id),
 FOREIGN KEY(account_id,user_id,task_id) REFERENCES bot_task(account_id,user_id,task_id),
 FOREIGN KEY(account_id,user_id,task_id,source_task_id) REFERENCES bot_task(account_id,user_id,task_id,parent_task_id),
 FOREIGN KEY(account_id,user_id,source_task_id) REFERENCES bot_task(account_id,user_id,task_id),
 CHECK(channel='web'),
 CHECK(state IN ('pending','sending','sent','retry_wait','waiting_confirmation','cancelled','failed')),
 CHECK(state<>'sent' OR provider_receipt IS NOT NULL), CHECK(retry_count>=0)
);
CREATE TABLE IF NOT EXISTS bot_forget (
 account_id VARCHAR(128) NOT NULL, user_id VARCHAR(128) NOT NULL,
 forget_id VARCHAR(64) NOT NULL, task_id VARCHAR(64) NOT NULL, memory_uri VARCHAR(2048) NOT NULL,
 source_refs JSONB NOT NULL DEFAULT '[]', fact_scope JSONB NOT NULL DEFAULT '{"mode":"entry"}',
 state VARCHAR(16) NOT NULL DEFAULT 'pending', operation_id VARCHAR(64) NOT NULL,
 relearn_request_id VARCHAR(64), created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
 updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
 PRIMARY KEY(account_id,forget_id), UNIQUE(account_id,operation_id),
 FOREIGN KEY(account_id,user_id,task_id) REFERENCES bot_task(account_id,user_id,task_id),
 CHECK(state IN ('pending','effective','superseded','failed')),
 CHECK(state<>'superseded' OR relearn_request_id IS NOT NULL)
);
CREATE TABLE IF NOT EXISTS bot_request (
 account_id VARCHAR(128) NOT NULL, user_id VARCHAR(128) NOT NULL,
 operation VARCHAR(256) NOT NULL, request_id VARCHAR(64) NOT NULL, request_hash CHAR(64) NOT NULL,
 response_status SMALLINT NOT NULL, response_body JSONB NOT NULL,
 expires_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP + INTERVAL '90 days',
 created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
 updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
 PRIMARY KEY(account_id,user_id,operation,request_id), CHECK(response_status BETWEEN 100 AND 599)
);
CREATE INDEX IF NOT EXISTS ix_bot_session_owner ON bot_session(account_id,user_id,updated_at,session_id);
CREATE UNIQUE INDEX IF NOT EXISTS uq_bot_task_seq ON bot_task(account_id,session_id,seq);
CREATE UNIQUE INDEX IF NOT EXISTS uq_bot_task_chat_request ON bot_task(account_id,session_id,request_id) WHERE kind='chat';
CREATE UNIQUE INDEX IF NOT EXISTS uq_bot_task_child ON bot_task(account_id,parent_task_id,kind,request_id);
CREATE UNIQUE INDEX IF NOT EXISTS uq_bot_task_schedule ON bot_task(account_id,schedule_id,scheduled_at);
CREATE INDEX IF NOT EXISTS ix_bot_task_claim ON bot_task(account_id,state,next_attempt_at,created_at,task_id);
CREATE INDEX IF NOT EXISTS ix_bot_task_lease ON bot_task(account_id,state,lease_until,task_id);
CREATE INDEX IF NOT EXISTS ix_bot_task_parent ON bot_task(account_id,parent_task_id,kind);
CREATE INDEX IF NOT EXISTS ix_bot_task_memory_order ON bot_task(account_id,user_id,created_at,task_id)
 WHERE kind='memory_write' AND state NOT IN ('succeeded','cancelled','failed');
CREATE INDEX IF NOT EXISTS ix_bot_task_cluster_lease ON bot_task(state,lease_until,account_id);
CREATE INDEX IF NOT EXISTS ix_bot_task_retention ON bot_task(state,finished_at,task_id);
CREATE INDEX IF NOT EXISTS ix_bot_event_retention ON bot_event(created_at,account_id,task_id,event_seq);
CREATE INDEX IF NOT EXISTS ix_bot_file_owner ON bot_file(account_id,user_id,purpose,created_at,file_id);
CREATE INDEX IF NOT EXISTS ix_bot_schedule_due ON bot_schedule(account_id,state,next_run_at,schedule_id);
CREATE INDEX IF NOT EXISTS ix_bot_schedule_global_due ON bot_schedule(state,next_run_at,schedule_id);
CREATE INDEX IF NOT EXISTS ix_bot_forget_effective ON bot_forget(account_id,user_id,state,memory_uri);
CREATE INDEX IF NOT EXISTS ix_bot_request_expiry ON bot_request(expires_at,account_id,user_id);

ALTER TABLE bot_session ENABLE ROW LEVEL SECURITY;
ALTER TABLE bot_session FORCE ROW LEVEL SECURITY;
DO $$ BEGIN
 CREATE POLICY bot_tenant_isolation ON bot_session
 USING (account_id = current_setting('app.account_id', true) OR current_setting('app.bot_worker') = 'on')
 WITH CHECK (account_id = current_setting('app.account_id', true));
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

ALTER TABLE bot_task ENABLE ROW LEVEL SECURITY;
ALTER TABLE bot_task FORCE ROW LEVEL SECURITY;
DO $$ BEGIN
 CREATE POLICY bot_tenant_isolation ON bot_task
 USING (account_id = current_setting('app.account_id', true) OR current_setting('app.bot_worker') = 'on')
 WITH CHECK (account_id = current_setting('app.account_id', true));
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

ALTER TABLE bot_step ENABLE ROW LEVEL SECURITY;
ALTER TABLE bot_step FORCE ROW LEVEL SECURITY;
DO $$ BEGIN
 CREATE POLICY bot_tenant_isolation ON bot_step
 USING (account_id = current_setting('app.account_id', true) OR current_setting('app.bot_worker') = 'on')
 WITH CHECK (account_id = current_setting('app.account_id', true));
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

ALTER TABLE bot_event ENABLE ROW LEVEL SECURITY;
ALTER TABLE bot_event FORCE ROW LEVEL SECURITY;
DO $$ BEGIN
 CREATE POLICY bot_tenant_isolation ON bot_event
 USING (account_id = current_setting('app.account_id', true) OR current_setting('app.bot_worker') = 'on')
 WITH CHECK (account_id = current_setting('app.account_id', true));
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

ALTER TABLE bot_file ENABLE ROW LEVEL SECURITY;
ALTER TABLE bot_file FORCE ROW LEVEL SECURITY;
DO $$ BEGIN
 CREATE POLICY bot_tenant_isolation ON bot_file
 USING (account_id = current_setting('app.account_id', true) OR current_setting('app.bot_worker') = 'on')
 WITH CHECK (account_id = current_setting('app.account_id', true));
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

ALTER TABLE bot_schedule ENABLE ROW LEVEL SECURITY;
ALTER TABLE bot_schedule FORCE ROW LEVEL SECURITY;
DO $$ BEGIN
 CREATE POLICY bot_tenant_isolation ON bot_schedule
 USING (account_id = current_setting('app.account_id', true) OR current_setting('app.bot_worker') = 'on')
 WITH CHECK (account_id = current_setting('app.account_id', true));
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

ALTER TABLE bot_delivery ENABLE ROW LEVEL SECURITY;
ALTER TABLE bot_delivery FORCE ROW LEVEL SECURITY;
DO $$ BEGIN
 CREATE POLICY bot_tenant_isolation ON bot_delivery
 USING (account_id = current_setting('app.account_id', true) OR current_setting('app.bot_worker') = 'on')
 WITH CHECK (account_id = current_setting('app.account_id', true));
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

ALTER TABLE bot_forget ENABLE ROW LEVEL SECURITY;
ALTER TABLE bot_forget FORCE ROW LEVEL SECURITY;
DO $$ BEGIN
 CREATE POLICY bot_tenant_isolation ON bot_forget
 USING (account_id = current_setting('app.account_id', true) OR current_setting('app.bot_worker') = 'on')
 WITH CHECK (account_id = current_setting('app.account_id', true));
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

ALTER TABLE bot_request ENABLE ROW LEVEL SECURITY;
ALTER TABLE bot_request FORCE ROW LEVEL SECURITY;
DO $$ BEGIN
 CREATE POLICY bot_tenant_isolation ON bot_request
 USING (account_id = current_setting('app.account_id', true) OR current_setting('app.bot_worker') = 'on')
 WITH CHECK (account_id = current_setting('app.account_id', true));
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;
