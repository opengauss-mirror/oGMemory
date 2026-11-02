"""Fail-closed snapshot and evidence helpers for isolated BEAM evaluation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
from datetime import datetime, timedelta, timezone

PROTOCOL = "beam-full-snapshot-v1"
REQUIRED_TABLES = {"context_nodes", "vector_index", "relation_edges",
                   "edge_projection_jobs", "session_archives", "outbox_events"}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def required_corpus_tables(config):
    """Require only enabled features, without filtering the snapshot inventory.

    A graph-off runtime need not initialize the graph projection queue.
    An explicitly enabled SQL graph still requires
    its projection queue, even if retrieval expansion is disabled.
    """
    backend = config.get("graph_backend")
    require(backend in (None, "none", "sql"), "Unsupported graph backend in evaluation runtime")
    tables = REQUIRED_TABLES.copy()
    if backend in (None, "none"):
        tables.discard("edge_projection_jobs")
    return tables


def ingestion_failure_log(log):
    """Retain log evidence; ignore ONLY the exact recoverable wakeup retry block.

    Callers must still require zero unfinished/failed outbox and projection jobs
    before checkpointing. No provider/extraction exception is filtered here.
    """
    pattern = (
        r"^\d{4}-[^\n]*\[index\.outbox_worker\] ERROR Failed to process event [^:\n]+: "
        r"Graph projection is pending; retain durable wakeup for retry\n"
        r"Traceback \(most recent call last\):\n"
        r'  File "/opt/ogmem/index/outbox_worker.py", line \d+, in process_event\n'
        r'    raise RuntimeError\("Graph projection is pending; retain durable wakeup for retry"\)\n'
        r"RuntimeError: Graph projection is pending; retain durable wakeup for retry\n"
    )
    return re.sub(pattern, "", log, flags=re.MULTILINE)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def schema_fingerprint_text(schema):
    """Ignore only pg_dump's existing volatile headers, not DDL or whitespace."""
    return "\n".join(line for line in schema.splitlines()
                     if not line.startswith(("\\restrict ", "\\unrestrict ", "-- Dumped")))


def write_json(path, value):
    """Exclusive evidence creation: never overwrite a prior run."""
    with Path(path).open("x") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def inspect_container(name):
    require(bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]+", name)), "Invalid container name")
    return json.loads(subprocess.check_output(["docker", "inspect", name], text=True))[0]


def require_eval_container(name, run_id, role):
    info = inspect_container(name)
    labels = info["Config"].get("Labels") or {}
    require(labels.get("anttrail.eval.run") == run_id and labels.get("anttrail.eval.role") == role,
            f"{name} is not labelled for this isolated evaluation ({run_id}/{role})")
    return info


def docker_text(name, *command):
    return subprocess.check_output(["docker", "exec", name, *command], text=True)


def validate_agent_mounts(info, protected_paths):
    protected = [Path(path).resolve() for path in protected_paths if path is not None]
    for mount in info.get("Mounts", []):
        if mount.get("Type") != "bind":
            continue
        source = Path(mount["Source"]).resolve()
        require(not any(path == source or path.is_relative_to(source) or source.is_relative_to(path)
                        for path in protected), "Answering container can access evaluator/code/snapshot artifacts")


def identifier(value):
    return '"' + value.replace('"', '""') + '"'


def literal(value):
    return "'" + value.replace("'", "''") + "'"


class SnapshotDB:
    """Only dedicated beam_eval_* databases may be replaced. No cutoff deletes."""

    def __init__(self, container, database):
        require(bool(re.fullmatch(r"beam_eval_[a-z0-9_]{8,48}", database)),
                "Refusing non-disposable DB: use a unique beam_eval_* database")
        self.container, self.database = container, database
        info = inspect_container(container)
        env = dict(item.split("=", 1) for item in info["Config"]["Env"] if "=" in item)
        self.user = env.get("POSTGRES_USER", "postgres")
        self.container_id = info["Id"]

    def sql(self, query, *, admin=False):
        return subprocess.check_output(
            ["docker", "exec", "-i", self.container, "psql", "-X", "-At",
             "-v", "ON_ERROR_STOP=1", "-U", self.user,
             "-d", "postgres" if admin else self.database, "-c", query], text=True).strip()

    def assert_idle(self):
        # Docker stop can return before PostgreSQL has observed disconnects.
        # Wait briefly for natural teardown; never terminate clients or relax
        # the zero-client precondition for a destructive snapshot restoration.
        deadline = time.monotonic() + 30
        while True:
            count = self.sql("SELECT count(*) FROM pg_stat_activity WHERE datname="
                             + literal(self.database) + " AND pid<>pg_backend_pid()", admin=True)
            if count == "0":
                return
            remaining = deadline - time.monotonic()
            require(remaining > 0, "Database still has clients; stop all isolated writers before snapshot/restore")
            time.sleep(min(.1, remaining))

    def fingerprint(self):
        if getattr(self, "legacy_fingerprint", False):
            return self.fingerprint_legacy()
        return self.fingerprint_batched()

    def fingerprint_batched(self):
        """Same complete inventory with one psql process and one schema dump."""
        program = r"""
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY;
SELECT format('SELECT json_build_object(''kind'',''table'',''name'',%L,''value'',json_build_object(''count'',count(*),''rows_md5'',md5(COALESCE(string_agg(h,'''' ORDER BY h),'''')))) FROM (SELECT md5(row_to_json(t)::text) h FROM %I.%I t) rows;',
              n.nspname||'.'||c.relname,n.nspname,c.relname)
FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
WHERE c.relkind IN ('r','m') AND n.nspname NOT LIKE 'pg_%'
  AND n.nspname <> 'information_schema' ORDER BY n.nspname,c.relname
\gexec
SELECT format('SELECT json_build_object(''kind'',''sequence'',''name'',%L,''value'',last_value::text||''|''||CASE WHEN is_called THEN ''t'' ELSE ''f'' END) FROM %I.%I;',
              '"'||replace(schemaname,'"','""')||'"."'||replace(sequencename,'"','""')||'"',schemaname,sequencename)
FROM pg_sequences WHERE schemaname NOT LIKE 'pg_%' ORDER BY schemaname,sequencename
\gexec
COMMIT;
"""
        output = subprocess.check_output([
            "docker", "exec", "-i", self.container, "psql", "-X", "-qAt",
            "-v", "ON_ERROR_STOP=1", "-U", self.user, "-d", self.database,
        ], input=program, text=True)
        result = {"tables": {}, "sequences": {}}
        for line in output.splitlines():
            row = json.loads(line)
            result["tables" if row["kind"] == "table" else "sequences"][row["name"]] = row["value"]
        schema = subprocess.check_output(["docker", "exec", self.container, "pg_dump",
            "-U", self.user, "-d", self.database, "--schema-only"], text=True)
        schema = schema_fingerprint_text(schema)
        result["schema_sha256"] = hashlib.sha256(schema.encode()).hexdigest()
        return result

    def fingerprint_legacy(self):
        """Hash ALL application tables, sequences and schema, including old updated rows."""
        tables = json.loads(self.sql("""SELECT COALESCE(json_agg(x ORDER BY schema,name),'[]') FROM (
            SELECT n.nspname AS schema,c.relname AS name FROM pg_class c
            JOIN pg_namespace n ON n.oid=c.relnamespace
            WHERE c.relkind IN ('r','m') AND n.nspname NOT LIKE 'pg_%'
              AND n.nspname<>'information_schema') x"""))
        result = {"tables": {}, "sequences": {}}
        for table in tables:
            name = identifier(table["schema"]) + "." + identifier(table["name"])
            raw = self.sql(f"SELECT json_build_object('count',count(*),'rows_md5',"
                f"md5(COALESCE(string_agg(h,'' ORDER BY h),''))) FROM "
                f"(SELECT md5(row_to_json(t)::text) AS h FROM {name} t) rows")
            result["tables"][table["schema"] + "." + table["name"]] = json.loads(raw)
        sequences = json.loads(self.sql("""SELECT COALESCE(json_agg(x ORDER BY schemaname,sequencename),'[]')
            FROM (SELECT schemaname,sequencename FROM pg_sequences
            WHERE schemaname NOT LIKE 'pg_%') x"""))
        for seq in sequences:
            name = identifier(seq["schemaname"]) + "." + identifier(seq["sequencename"])
            result["sequences"][name] = self.sql(f"SELECT last_value,is_called FROM {name}")
        schema = subprocess.check_output(["docker", "exec", self.container, "pg_dump",
            "-U", self.user, "-d", self.database, "--schema-only"], text=True)
        # Recent pg_dump adds random psql restrict tokens, unrelated to DB schema.
        schema = schema_fingerprint_text(schema)
        result["schema_sha256"] = hashlib.sha256(schema.encode()).hexdigest()
        return result

    def freeze(self, directory, metadata):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        self.assert_idle()
        before = self.fingerprint()
        required_tables = required_corpus_tables(metadata.get("runtime", {}).get("config", {}))
        require({"public." + name for name in required_tables} <= set(before["tables"]),
                "Required corpus/graph tables are missing")
        require(before["tables"]["public.outbox_events"]["count"] == 0,
                "Outbox must be drained before freezing (including directory jobs)")
        if "public.edge_projection_jobs" in before["tables"]:
            require(self.sql("SELECT count(*) FROM edge_projection_jobs WHERE upper(status)<>'COMPLETE'") == "0",
                    "Projection jobs have not settled")
        with (directory / "snapshot.dump").open("xb") as handle:
            subprocess.run(["docker", "exec", self.container, "pg_dump", "-U", self.user,
                            "-d", self.database, "-Fc"], stdout=handle, check=True)
        require(before == self.fingerprint(), "Database changed while freezing; snapshot invalid")
        locale = json.loads(self.sql("SELECT json_build_object('encoding',pg_encoding_to_char(encoding),"
            "'collate',datcollate,'ctype',datctype,'provider',datlocprovider) FROM pg_database WHERE datname="
            + literal(self.database), admin=True))
        require(locale["provider"] == "c", "This restore implementation requires libc database locale")
        manifest = {**metadata, "protocol": PROTOCOL, "database": self.database,
                    "postgres_container_id": self.container_id, "locale": locale,
                    "dump_sha256": digest(directory / "snapshot.dump"), "inventory": before}
        write_json(directory / "snapshot.json", manifest)
        return manifest

    def restore(self, directory):
        directory = Path(directory)
        manifest = json.loads((directory / "snapshot.json").read_text())
        require(manifest["protocol"] == PROTOCOL, "Unsupported snapshot protocol")
        require(manifest["database"] == self.database, "Snapshot/database mismatch")
        require(manifest["postgres_container_id"] == self.container_id, "PostgreSQL instance changed")
        require(digest(directory / "snapshot.dump") == manifest["dump_sha256"], "Snapshot dump changed")
        self.assert_idle()
        # Validate every identifier/value BEFORE the destructive operation.
        locale = manifest["locale"]
        require(locale.get("provider") == "c", "Unsupported database locale provider")
        create = (f"CREATE DATABASE {identifier(self.database)} TEMPLATE template0 "
                  f"ENCODING {literal(locale['encoding'])} LC_COLLATE {literal(locale['collate'])} "
                  f"LC_CTYPE {literal(locale['ctype'])}")
        with (directory / "snapshot.dump").open("rb") as handle:
            subprocess.run(["docker", "exec", "-i", self.container, "pg_restore", "--list"],
                           stdin=handle, stdout=subprocess.DEVNULL, check=True)
        self.sql(f"DROP DATABASE {identifier(self.database)}", admin=True)
        self.sql(create, admin=True)
        with (directory / "snapshot.dump").open("rb") as handle:
            subprocess.run(["docker", "exec", "-i", self.container, "pg_restore", "-U", self.user,
                "-d", self.database, "--exit-on-error", "--single-transaction"], stdin=handle, check=True)
        actual = self.fingerprint()
        require(actual == manifest["inventory"], "Restored DB content/schema/sequence hash mismatch")
        return {"inventory_sha256": json_digest(actual), "dump_sha256": manifest["dump_sha256"]}


def audit_session(text, expected_question):
    records = [json.loads(line) for line in text.splitlines() if line.strip()]
    messages = [record["message"] for record in records if record.get("type") == "message"]
    # Match decoded content, not JSON syntax (which escapes quotes/newlines).
    # Metadata is deliberately excluded: it cannot prove the user saw a question.
    def contains_question(message):
        content = message.get("content", [])
        if isinstance(content, str):
            return expected_question in content
        return isinstance(content, list) and any(
            isinstance(part, dict) and part.get("type") in {"text", "input_text"}
            and isinstance(part.get("text"), str) and expected_question in part["text"]
            for part in content)

    require(bool(expected_question) and any(
        message.get("role") == "user" and contains_question(message)
        for message in messages), "Session log lacks the expected question")
    require(any(message.get("role") == "assistant" for message in messages), "Session log lacks an answer")
    structured = 0
    literals = 0
    for message in messages:
        if message.get("role") in {"tool", "toolResult"}:
            structured += 1
        if message.get("tool_calls") or message.get("function_call"):
            structured += 1
        for part in message.get("content", []) if isinstance(message.get("content"), list) else []:
            if part.get("type") in {"toolCall", "tool_use", "function_call", "tool_result"}:
                structured += 1
        if message.get("role") == "assistant":
            literals += len(re.findall(r"DSML|<tool_call>|<function_call>", json.dumps(message, ensure_ascii=False)))
    require(structured == 0, f"INVALID: answering session contains {structured} structured tool calls/results")
    return {"structured_tool_calls": structured, "literal_tool_syntax_occurrences": literals}


# Production Python/module directories copied into both supported Docker images.
# The resolved configuration is bound separately; Docker build files, local
# configs and optional non-deployed research trees are not runtime source files.
SOURCE_DIRS = ("core", "commit", "extraction", "fs", "index", "providers", "retrieval",
               "server", "service", "session", "cli", "skills", "features", "scene",
               "perf", "scripts")


def source_hashes(root):
    root = Path(root)
    return {str(path.relative_to(root)): digest(path) for directory in SOURCE_DIRS
            for path in sorted((root / directory).rglob("*"))
            if path.is_file() and path.suffix in {".py", ".yaml", ".yml", ".sql", ".sh"}}


def native_policy_manifest(config):
    """Inspect native defaults/dispatch without constructing a service or a provider."""
    import inspect
    import core  # noqa: F401 -- Preserve cold-process native import order.
    from extraction.tools import Extractor
    from server.memory_service import MemoryService
    from retrieval import query_planner
    probe = Extractor.__new__(Extractor)
    # Mirror only the native constructor's configured dispatch flag for this
    # read-only receipt. No constructor/provider/service is invoked.
    probe._dual_pass_enabled = config.dual_pass_enabled and not config.node_processing_enabled
    temperatures = list(probe._native_temperatures())
    return dict(
        after_turn_threshold=config.after_turn_threshold,
        dual_pass_enabled=config.dual_pass_enabled,
        node_processing_enabled=config.node_processing_enabled,
        extraction_temperature_arguments=temperatures,
        extraction_effective_temperatures=[config.llm_temperature if t is None else t for t in temperatures],
        recall_score_gap_ratio=inspect.signature(MemoryService._truncate_by_score_gap).parameters["gap_ratio"].default,
        routing_keyword_counts={k: len(getattr(query_planner, k)) for k in (
            "_MEMORY_PATTERNS", "_SKILL_PATTERNS", "_RESOURCE_PATTERNS")})


def runtime_preflight(memory, source_root, database, account, user):
    """Read post-entrypoint config through the same loader as API/index services."""
    program = r'''
import hashlib,json,pathlib,psycopg2,sys,dataclasses,os
sys.path.insert(0,"/opt/ogmem")
import core
from providers.unified_config import get_config
from scripts.beam_eval_safety import native_policy_manifest
# docker exec does not inherit environment changes made by an entrypoint's
# supervisor. Prefer its actual resolved file, not the original YAML template.
resolved=pathlib.Path('/tmp/ogmem.resolved.yaml')
config_path=os.environ.get('BEAM_RUNTIME_CONFIG') or (str(resolved) if resolved.is_file() else os.environ.get('OGMEM_CONFIG'))
c=get_config(config_path) if config_path else get_config()
names=("sql_connection_string","opengauss_connection_string")
targets=[]
for name in names:
    conn=psycopg2.connect(getattr(c,name))
    with conn.cursor() as cur:
        cur.execute("SELECT current_database(),system_identifier::text FROM pg_control_system()")
        targets.append(cur.fetchone())
    conn.close()
fields=("account_id","user_id","openai_llm_model","openai_embedding_model",
        "opengauss_dimension","opengauss_table_name","graph_backend","enable_graph_expansion",
        "graph_edge_embedding_profile","graph_expand_top_k","graph_expand_limit",
        "graph_supplement_enabled","graph_min_weight","graph_edge_similarity_threshold",
        "enable_cache","http_port","storage_backend",
        "provider","embedding_provider","llm_temperature","llm_max_tokens",
        "llm_request_timeout_seconds","llm_reasoning_effort","llm_is_reasoning_model",
        "retrieval_expand_hierarchy","retrieval_hybrid_enabled","retrieval_result_ranker_strategy",
        "retrieval_candidate_pool_enabled","retrieval_owner_scope_filter_enabled",
        "retrieval_global_result_rerank_enabled","topic_detection_enabled",
        "scene_enabled","scene_generation_enabled","scene_retrieval_enabled",
        "after_turn_threshold","skill_sedimentation_enabled","source_occurrence_reconciliation_enabled",
        "dual_pass_enabled","node_processing_enabled","card_format_enabled","extraction_mode")
root=pathlib.Path("/opt/ogmem")
dirs=json.loads(sys.argv[1])
hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
        for d in dirs for p in sorted((root/d).rglob("*"))
        if p.is_file() and p.suffix in {".py",".yaml",".yml",".sql",".sh"}}
config_hash=hashlib.sha256(json.dumps(dataclasses.asdict(c),sort_keys=True,default=str).encode()).hexdigest()
print(json.dumps({"targets":targets,"config":{k:getattr(c,k,None) for k in fields},
                  "resolved_config_sha256":config_hash,"source_hashes":hashes,
                  "native_policy":native_policy_manifest(c),
                  "config_path":config_path or "loader default"}))
'''
    actual = json.loads(docker_text(memory, "python3", "-c", program, json.dumps(SOURCE_DIRS)))
    require(actual["targets"][0] == actual["targets"][1], "Storage and vector SQL targets differ")
    require(actual["targets"][0][0] == database, "Runtime points at a different database")
    config = actual["config"]
    require(config["account_id"] == account and config["user_id"] == user,
            "Resolved memory identity differs from expected account/user")
    require(config["opengauss_table_name"] == "vector_index", "This runner requires vector_index")
    require(config["storage_backend"] == "sql", "Full-snapshot BEAM evaluation requires SQL storage")
    require(config["enable_cache"] is False, "Disable runtime cache before evaluation")
    expected = source_hashes(source_root)
    require(expected and actual["source_hashes"] == expected, "Runtime/source hashes differ")
    started = inspect_container(memory)["State"]["StartedAt"]
    # Docker's state/log clocks can straddle a clock correction at startup.
    # Keep the read bounded, allowing a small clock-skew margin. Both actual
    # SQL connections above still independently prove the database identity.
    cutoff = (datetime.fromisoformat(started.replace("Z", "+00:00"))
              - timedelta(seconds=30)).astimezone(timezone.utc).isoformat()
    logs = subprocess.run(["docker", "logs", "--since", cutoff, memory],
                          capture_output=True, text=True, check=True)
    targets = {}
    for line in (logs.stdout + "\n" + logs.stderr).splitlines():
        match = re.search(r"\[entrypoint\] Patched ([\w.]+): ([^:\s]+):(\d+)/([\w]+)", line)
        if match:
            targets[match[1]] = {"host": match[2], "port": int(match[3]), "database": match[4]}
    # Current standalone_runtime no longer emits the legacy Patched lines.
    # Live SQL connections, PostgreSQL cluster identity and source hashes above
    # remain authoritative; reject contradictory logs when present.
    require(all(target["database"] == database for target in targets.values()),
            "Startup log reports a different database")
    actual["startup_db_targets"] = targets
    return actual


def validate_agent_config(config, account, user, memory_url):
    """Shared static checks; never repair model, prompt, routing or tool policy."""
    require(config.get("tools", {}).get("deny") == ["*"], "Answering tools must all be denied")
    agents = config.get("agents", {})
    require(agents.get("list", []) in ([], [{"id": "main"}]),
            "Only the isolated main agent is permitted; per-agent overrides are forbidden")
    require(agents.get("defaults", {}).get("skipBootstrap") is True,
            "Set agents.defaults.skipBootstrap=true for an artifact-free workspace")
    workspace = agents["defaults"].get("workspace")
    require(workspace == "/home/node/.openclaw/workspace", "Use the dedicated standard workspace")
    plugin = config["plugins"]["entries"]["og-memory-context-engine"]
    require(config["plugins"]["slots"]["contextEngine"] == "og-memory-context-engine"
            and plugin.get("enabled") is True, "Memory context plugin is not active")
    values = plugin["config"]
    require(values.get("accountId") == account and values.get("userId") == user,
            "Plugin account/user differs from corpus identity")
    require(values.get("memoryApiBaseUrl", "").rstrip("/") == memory_url.rstrip("/"),
            "Agent uses a different memory endpoint")
    require(values.get("autoCapture") is False, "Disable QA autoCapture before evaluation")
    return workspace


def agent_preflight(container, account, user, memory_url, *, fresh, batched=False):
    if batched:
        program = r'''
const fs=require('fs');const base='/home/node/.openclaw';
const config=JSON.parse(fs.readFileSync(base+'/openclaw.json','utf8'));
const workspace=config.agents.defaults.workspace;
const dir=base+'/agents/main/sessions';
const sessions=fs.existsSync(dir+'/sessions.json')?JSON.parse(fs.readFileSync(dir+'/sessions.json','utf8')):{};
console.log(JSON.stringify({config,contents:fs.readdirSync(workspace),sessions,
logs:fs.existsSync(dir)?fs.readdirSync(dir).filter(n=>n.endsWith('.jsonl')):[]}));
'''
        data = json.loads(docker_text(container, "node", "-e", program))
        config = data["config"]
    else:
        config = json.loads(docker_text(container, "cat", "/home/node/.openclaw/openclaw.json"))
    workspace = validate_agent_config(config, account, user, memory_url)
    agents = config["agents"]
    contents = data["contents"] if batched else docker_text(container, "find", workspace, "-mindepth", "1", "-print").strip()
    require(not contents, "Answering workspace is not empty")
    sessions = data["sessions"] if batched else json.loads(docker_text(container, "sh", "-c",
        "if test -f /home/node/.openclaw/agents/main/sessions/sessions.json; then "
        "cat /home/node/.openclaw/agents/main/sessions/sessions.json; else printf '{}'; fi"))
    require(not fresh or not sessions, "Answering container has previous sessions")
    if fresh:
        logs = "\n".join(data["logs"]) if batched else docker_text(container, "sh", "-c",
            "find /home/node/.openclaw/agents/main/sessions -name '*.jsonl' -print 2>/dev/null || true")
        require(not logs.strip(), "Answering container contains previous session logs")
    info = inspect_container(container)
    return {"config_sha256": json_digest(config), "model": agents["defaults"].get("model"),
            "container_id": info["Id"], "image_id": info["Image"],
            "workspace_empty": True, "tools_denied": True}, sessions


def collect_session(container, session_key, question):
    sessions = json.loads(docker_text(container, "cat", "/home/node/.openclaw/agents/main/sessions/sessions.json"))
    require(session_key in sessions, "Cannot locate answering session")
    session_id = sessions[session_key]["sessionId"]
    require(bool(re.fullmatch(r"[a-fA-F0-9-]{36}", session_id)), "Unsafe session-log identifier")
    text = docker_text(container, "cat", f"/home/node/.openclaw/agents/main/sessions/{session_id}.jsonl")
    return text
