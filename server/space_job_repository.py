"""Transactional scheduling and publication for system-owned shared memory."""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import asdict
from datetime import UTC, datetime

from psycopg2.extras import Json, execute_values

from core.models import ContextNode, IndexRecord
from index.index_record_builder import build_index_records
from server.space_locks import lock_space_account
from server.space_privacy import redact_item
from server.space_rules import digest as fingerprint


def maintenance_state(item, now):
    """Evaluate the persisted policy using the original lifetime anchors."""
    if item["state"] in {"ARCHIVED", "SUPERSEDED", "EXPIRED"}:
        return "PURGED" if item["purge_after"] and item["purge_after"] <= now else None
    if item["state"] not in {"ACTIVE", "RECOMPUTING"}:
        return None
    if item["expires_at"] <= now:
        return "EXPIRED"
    policy = item["policy_snapshot"]
    idle = max(0, (now - (item["last_used_at"] or item["created_at"])).total_seconds() / 86400)
    age = (now - item["created_at"]).total_seconds() / 86400
    score = item["quality_score"] * math.exp2(-idle / policy["half_life_days"])
    if age >= policy["min_age_days"] and score < policy["archive_threshold"]:
        return "ARCHIVED"
    return None


class SpaceJobRepository:
    """Mixin using SpaceRepository's tenant-scoped transaction cursor."""

    def worker_accounts(self):
        conn = self.pool.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT account_id FROM cp_accounts WHERE status='active' ORDER BY account_id")
                return [row[0] for row in cur.fetchall()]
        finally:
            conn.rollback()
            self.pool.return_connection(conn)

    @staticmethod
    def _source_principal(uri):
        parts = uri.removeprefix("ctx://").split("/")
        if len(parts) < 4 or parts[1] not in {"users", "agents"}:
            return ""
        return ("user:" if parts[1] == "users" else "agent:") + parts[2]

    def dispatch_source_events(self, account_id, limit=100):
        with self.transaction(account_id) as cur:
            cur.execute("""SELECT event_id,uri,payload,seq FROM outbox_events
                WHERE account_id=%s AND event_type='SOURCE_CHANGED' AND status='SOURCE_PENDING'
                  AND NOT (payload ? 'space_dispatch_receipt')
                ORDER BY seq LIMIT %s FOR UPDATE SKIP LOCKED""", (account_id, limit))
            events = cur.fetchall()
            for event in events:
                cur.execute("""SELECT DISTINCT r.space_id,r.rule_id,v.spec FROM space_rules r
                    JOIN space_rule_revisions v USING(account_id,space_id,rule_id,rule_rev)
                    JOIN spaces s USING(account_id,space_id)
                    JOIN space_source_bindings b USING(account_id,space_id)
                    WHERE r.account_id=%s AND b.principal=%s AND r.state='ACTIVE' AND s.state='ACTIVE'
                      AND (NOT(v.spec ? 'source_subjects') OR v.spec->'source_subjects' ? b.principal)""",
                    (account_id, self._source_principal(event["uri"])))
                targets = cur.fetchall()
                for target in targets:
                    debounce = target["spec"]["trigger_policy"]["debounce_seconds"]
                    cur.execute("""UPDATE space_rule_runtime SET dirty_generation=dirty_generation+1,
                        next_run_at=LEAST(next_run_at,NOW()+(%s * INTERVAL '1 second')),updated_at=NOW()
                        WHERE account_id=%s AND space_id=%s AND rule_id=%s""",
                        (debounce, account_id, target["space_id"], target["rule_id"]))
                receipt = {"change_id": event["event_id"], "targets": [
                    {"space_id": t["space_id"], "rule_id": t["rule_id"]} for t in targets],
                    "confirmed_at": datetime.now(UTC).isoformat()}
                cur.execute("""UPDATE outbox_events SET payload=jsonb_set(payload, '{space_dispatch_receipt}', %s::jsonb, true),status='DONE'
                    WHERE event_id=%s""", (Json(receipt), event["event_id"]))
            cur.execute("""DELETE FROM outbox_events WHERE event_id IN (
                SELECT event_id FROM outbox_events WHERE account_id=%s AND event_type='SOURCE_CHANGED'
                AND status='DONE' AND created_at<NOW()-INTERVAL '7 days' ORDER BY created_at LIMIT %s)""", (account_id, limit))
            cur.execute("""DELETE FROM space_requests WHERE account_id=%s AND (actor_id,operation,request_key) IN (
                SELECT actor_id,operation,request_key FROM space_requests WHERE account_id=%s AND expires_at<=NOW()
                ORDER BY expires_at LIMIT %s)""", (account_id, account_id, limit))
            return len(events)

    def reconcile_due_rules(self, account_id, limit=100):
        with self.transaction(account_id) as cur:
            lock_space_account(cur, account_id)
            cur.execute("""UPDATE space_jobs SET state='FAILED',claim_token=NULL,lease_until=NULL,
                    last_error='lease attempts exhausted',snapshot=snapshot - 'sources',updated_at=NOW()
                WHERE account_id=%s AND state='RUNNING' AND lease_until<=NOW() AND attempts>=5""", (account_id,))
            cur.execute("""SELECT r.*,v.spec,v.spec_hash,t.dirty_generation,t.applied_generation
                FROM space_rules r JOIN space_rule_revisions v USING(account_id,space_id,rule_id,rule_rev)
                JOIN space_rule_runtime t USING(account_id,space_id,rule_id)
                JOIN spaces s USING(account_id,space_id)
                WHERE r.account_id=%s AND r.state='ACTIVE' AND s.state='ACTIVE' AND t.next_run_at<=NOW()
                AND COALESCE((SELECT j.updated_at
                      + ((v.spec->'trigger_policy'->>'min_interval_seconds')::double precision * INTERVAL '1 second')
                    FROM space_jobs j
                    WHERE j.account_id=r.account_id AND j.space_id=r.space_id AND j.rule_id=r.rule_id
                      AND j.rule_rev=r.rule_rev AND j.operation IN ('GENERATE','RECOMPUTE') AND j.state='SUCCEEDED'
                    ORDER BY j.generation DESC LIMIT 1), NOW())<=NOW()
                AND NOT EXISTS (SELECT 1 FROM space_jobs j WHERE j.account_id=r.account_id
                    AND j.space_id=r.space_id AND j.rule_id=r.rule_id AND j.state IN ('QUEUED','RUNNING'))
                ORDER BY t.next_run_at,r.space_id,r.rule_id LIMIT %s""", (account_id, limit))
            rules = cur.fetchall()
            for rule in rules:
                generation = max(rule["dirty_generation"], rule["applied_generation"] + 1)
                cur.execute("""SELECT COALESCE(MAX(generation),0) AS generation FROM space_jobs
                    WHERE account_id=%s AND space_id=%s AND rule_id=%s""",
                    (account_id, rule["space_id"], rule["rule_id"]))
                generation = max(generation, cur.fetchone()["generation"] + 1)
                snapshot = {"rule": {key: rule[key] for key in ("rule_id", "rule_rev", "name", "state", "spec", "spec_hash")}}
                cur.execute("""INSERT INTO space_jobs(account_id,space_id,job_id,rule_id,rule_rev,
                    operation,generation,snapshot) VALUES (%s,%s,%s,%s,%s,'RECOMPUTE',%s,%s)""",
                    (account_id, rule["space_id"], str(uuid.uuid4()), rule["rule_id"], rule["rule_rev"], generation, Json(snapshot)))
                cur.execute("""UPDATE space_rule_runtime SET dirty_generation=%s,updated_at=NOW()
                    WHERE account_id=%s AND space_id=%s AND rule_id=%s""",
                    (generation, account_id, rule["space_id"], rule["rule_id"]))
            return len(rules)

    def claim_job(self, account_id, lease_seconds=60):
        with self.transaction(account_id) as cur:
            lock_space_account(cur, account_id)
            cur.execute("""SELECT COUNT(*) AS count FROM space_jobs WHERE account_id=%s
                AND state='RUNNING' AND lease_until>NOW()""", (account_id,))
            if cur.fetchone()["count"] >= 4:
                return None
            cur.execute("""SELECT * FROM space_jobs WHERE account_id=%s AND attempts<5 AND
                ((state='QUEUED' AND next_run_at<=NOW()) OR (state='RUNNING' AND lease_until<=NOW()))
                ORDER BY next_run_at,job_id LIMIT 1 FOR UPDATE SKIP LOCKED""", (account_id,))
            job = cur.fetchone()
            if job is None:
                return None
            cur.execute("""UPDATE space_jobs SET state='RUNNING',attempts=attempts+1,
                claim_token=%s,lease_until=NOW()+(%s * INTERVAL '1 second'),updated_at=NOW()
                WHERE account_id=%s AND space_id=%s AND job_id=%s RETURNING *""",
                (str(uuid.uuid4()), lease_seconds, account_id, job["space_id"], job["job_id"]))
            return dict(cur.fetchone())

    def heartbeat(self, job, lease_seconds=60):
        with self.transaction(job["account_id"], job["space_id"]) as cur:
            cur.execute("""UPDATE space_jobs SET lease_until=NOW()+(%s * INTERVAL '1 second')
                WHERE account_id=%s AND space_id=%s AND job_id=%s AND state='RUNNING'
                AND claim_token=%s AND lease_until>NOW()""",
                (lease_seconds, job["account_id"], job["space_id"], job["job_id"], job["claim_token"]))
            return cur.rowcount == 1

    def _job_sources(self, cur, job, max_sources=100, lock=False, uris=None, after="", unprocessed=False):
        spec = job["snapshot"]["rule"]["spec"]
        subjects = spec.get("source_subjects")
        cur.execute("""SELECT b.binding_id,b.grant_rev,n.uri,n.version AS content_rev,n.content,
                n.abstract,n.metadata,n.owner_space,n.category
            FROM space_source_bindings b JOIN space_members m USING(account_id,space_id,principal)
            JOIN context_nodes n ON n.account_id=b.account_id AND n.owner_space=b.principal
            LEFT JOIN cp_users u ON u.account_id=m.account_id AND u.user_id=m.user_id
            LEFT JOIN cp_agents a ON a.account_id=m.account_id AND a.agent_id=m.agent_id
            LEFT JOIN cp_users au ON au.account_id=a.account_id AND au.user_id=a.owner_user_id
            JOIN cp_users grantor ON grantor.account_id=b.account_id AND grantor.user_id=b.authorized_by
            WHERE b.account_id=%s AND b.space_id=%s AND b.state='ACTIVE'
              AND (%s IS NULL OR b.principal=ANY(%s)) AND n.category=ANY(%s) AND b.memory_types ? n.category
              AND n.status='ACTIVE' AND ((u.status='active') OR (a.status='active' AND au.status='active'))
              AND grantor.status='active' AND (grantor.role='admin' OR b.principal='user:'||grantor.user_id
                   OR a.owner_user_id=grantor.user_id)
              AND (%s IS NULL OR n.uri=ANY(%s)) AND n.uri>%s
              AND (%s=FALSE OR NOT EXISTS (
                  SELECT 1 FROM space_item_sources x JOIN space_items i USING(account_id,space_id,item_id)
                  WHERE x.account_id=b.account_id AND x.space_id=b.space_id AND x.binding_id=b.binding_id
                    AND x.source_uri=n.uri AND x.grant_rev=b.grant_rev AND x.source_content_rev=n.version
                    AND x.source_epoch=COALESCE(n.metadata->>'shared_source_epoch','')
                    AND i.rule_id=%s AND COALESCE(i.checked_rule_rev,i.rule_rev)=%s))
            ORDER BY n.uri LIMIT %s""" + (" FOR SHARE OF n" if lock else ""),
            (job["account_id"], job["space_id"], subjects, subjects, spec["memory_types"], uris, uris, after,
             unprocessed, job["rule_id"], job["rule_rev"], max_sources))
        sources = [dict(row) for row in cur.fetchall()]
        if getattr(self, "content_store", None) is not None:
            sources = self.content_store.load_content(sources)
        for source in sources:
            source["sha256"] = hashlib.sha256(source["content"].encode()).hexdigest()
            source["epoch"] = source["metadata"].get("shared_source_epoch", "")
        return sources

    def job_snapshot(self, job, max_sources=100):
        account, space = job["account_id"], job["space_id"]
        with self.transaction(account, space) as cur:
            cur.execute("SELECT source_after FROM space_rule_runtime WHERE account_id=%s AND space_id=%s AND rule_id=%s",
                        (account, space, job["rule_id"]))
            after = cur.fetchone()["source_after"]
            # Stale provenance is repaired in bounded target batches before advancing the source scan.
            cur.execute("""SELECT DISTINCT i.item_id FROM space_items i
                JOIN space_item_sources x USING(account_id,space_id,item_id)
                LEFT JOIN space_source_bindings b ON b.account_id=x.account_id AND b.space_id=x.space_id AND b.binding_id=x.binding_id
                LEFT JOIN space_members m ON m.account_id=b.account_id AND m.space_id=b.space_id AND m.principal=b.principal
                LEFT JOIN context_nodes n ON n.account_id=x.account_id AND n.uri=x.source_uri
                LEFT JOIN cp_users u ON u.account_id=m.account_id AND u.user_id=m.user_id
                LEFT JOIN cp_agents a ON a.account_id=m.account_id AND a.agent_id=m.agent_id
                LEFT JOIN cp_users au ON au.account_id=a.account_id AND au.user_id=a.owner_user_id
                LEFT JOIN cp_users grantor ON grantor.account_id=b.account_id AND grantor.user_id=b.authorized_by
                WHERE i.account_id=%s AND i.space_id=%s AND i.rule_id=%s AND i.state IN ('ACTIVE','RECOMPUTING')
                  AND (COALESCE(i.checked_rule_rev,i.rule_rev)<>%s OR b.state<>'ACTIVE' OR b.grant_rev<>x.grant_rev OR m.principal IS NULL
                       OR (u.status IS DISTINCT FROM 'active' AND (a.status IS DISTINCT FROM 'active' OR au.status IS DISTINCT FROM 'active'))
                       OR grantor.status IS DISTINCT FROM 'active'
                       OR (grantor.role='admin' OR b.principal='user:'||grantor.user_id OR a.owner_user_id=grantor.user_id) IS NOT TRUE
                       OR n.uri IS NULL OR n.status<>'ACTIVE' OR n.version<>x.source_content_rev
                       OR COALESCE(n.metadata->>'shared_source_epoch','')<>x.source_epoch)
                ORDER BY i.item_id LIMIT 100""", (account, space, job["rule_id"], job["rule_rev"]))
            stale_ids = [row["item_id"] for row in cur.fetchall()]
            repair = bool(stale_ids)
            if repair:
                cur.execute("""SELECT item_id,source_uri FROM space_item_sources WHERE account_id=%s
                    AND space_id=%s AND item_id=ANY(%s) ORDER BY item_id,source_uri""", (account, space, stale_ids))
                rows = cur.fetchall()
                selected_ids, selected_uris = [], set()
                for item_id in stale_ids:
                    refs = {row["source_uri"] for row in rows if row["item_id"] == item_id}
                    if len(refs) > max_sources:
                        raise ValueError("shared item provenance exceeds the task source budget")
                    if len(selected_uris | refs) > max_sources:
                        break
                    selected_ids.append(item_id)
                    selected_uris |= refs
                sources = self._job_sources(cur, job, max_sources, uris=sorted(selected_uris))
            else:
                page = self._job_sources(cur, job, max_sources, after=after, unprocessed=True)
                if not page and after:
                    after = ""
                    page = self._job_sources(cur, job, max_sources, unprocessed=True)
                page_uris = [source["uri"] for source in page]
                sources = page
                cur.execute("""SELECT i.item_id FROM space_items i WHERE i.account_id=%s AND i.space_id=%s
                    AND i.rule_id=%s AND i.state IN ('ACTIVE','RECOMPUTING')
                    AND EXISTS(SELECT 1 FROM space_item_sources x WHERE x.account_id=i.account_id AND x.space_id=i.space_id
                        AND x.item_id=i.item_id AND x.source_uri=ANY(%s))
                    AND NOT EXISTS(SELECT 1 FROM space_item_sources x WHERE x.account_id=i.account_id AND x.space_id=i.space_id
                        AND x.item_id=i.item_id AND NOT(x.source_uri=ANY(%s)))
                    ORDER BY i.item_id LIMIT 100""", (account, space, job["rule_id"], page_uris, page_uris))
                selected_ids = [row["item_id"] for row in cur.fetchall()]
            cur.execute("""SELECT i.*,n.content,n.abstract,n.version AS content_rev,n.metadata
                FROM space_items i JOIN context_nodes n USING(account_id,uri)
                WHERE i.account_id=%s AND i.space_id=%s AND i.item_id=ANY(%s) ORDER BY i.item_id""",
                (account, space, selected_ids))
            items = [dict(row) for row in cur.fetchall()]
            cur.execute("""SELECT item_id,source_uri,grant_rev,source_content_rev,source_hash,rtrim(source_epoch) AS source_epoch
                FROM space_item_sources WHERE account_id=%s AND space_id=%s AND item_id=ANY(%s)""",
                (account, space, selected_ids))
            contributions = cur.fetchall()
            for item in items:
                item["sources"] = [dict(source) for source in contributions if source["item_id"] == item["item_id"]]
            signature = fingerprint([(s["binding_id"], s["grant_rev"], s["uri"], s["content_rev"], s["sha256"], s["epoch"]) for s in sources])
            cur.execute("""SELECT 1 FROM space_jobs WHERE account_id=%s AND space_id=%s AND rule_id=%s
                AND rule_rev=%s AND state='SUCCEEDED' AND result->>'source_fingerprint'=%s LIMIT 1""",
                (account, space, job["rule_id"], job["rule_rev"], signature))
            unchanged = bool(cur.fetchone()) and not repair
            next_after = after if repair else (page[-1]["uri"] if len(page) == max_sources else "")
            snapshot = {**job["snapshot"], "source_after": after,
                        "next_source_after": next_after, "continue_scan": repair or bool(next_after),
                        "source_selection": sorted(selected_uris) if repair else page_uris,
                        "window": fingerprint([s["uri"] for s in sources]),
                        "target_revisions": [{"item_id": i["item_id"], "content_rev": i["content_rev"]} for i in items],
                        "source_fingerprint": signature, "unchanged": unchanged}
            snapshot.pop("sources", None)
            model_input = {**snapshot, "sources": sources}
            if len(json.dumps(model_input, ensure_ascii=False).encode()) > 20 * 1024 * 1024:
                raise ValueError("shared memory source snapshot exceeds 20 MiB")
            cur.execute("""UPDATE space_jobs SET snapshot=%s WHERE account_id=%s AND space_id=%s
                AND job_id=%s AND claim_token=%s AND state='RUNNING' AND lease_until>NOW()""",
                (Json(snapshot), account, space, job["job_id"], job["claim_token"]))
            if cur.rowcount != 1:
                raise RuntimeError("shared memory task lease changed")
            job["snapshot"] = snapshot
            return {**model_input, "items": items}

    def fail_job(self, job, error):
        with self.transaction(job["account_id"], job["space_id"]) as cur:
            cur.execute("""UPDATE space_jobs SET state=CASE WHEN attempts>=5 THEN 'FAILED' ELSE 'QUEUED' END,
                claim_token=NULL,lease_until=NULL,last_error=%s,snapshot=snapshot - 'sources',
                next_run_at=NOW()+(LEAST(30,POWER(2,attempts)) * INTERVAL '1 second'),updated_at=NOW()
                WHERE account_id=%s AND space_id=%s AND job_id=%s AND claim_token=%s
                AND state='RUNNING' AND lease_until>NOW()""",
                (str(error)[:2000], job["account_id"], job["space_id"], job["job_id"], job["claim_token"]))
            if not cur.rowcount:
                return
            cur.execute("""UPDATE space_rule_runtime SET next_run_at=NOW()+INTERVAL '300 seconds'
                WHERE account_id=%s AND space_id=%s AND rule_id=%s""",
                (job["account_id"], job["space_id"], job["rule_id"]))

    def _finish_job(self, cur, job, result, state="SUCCEEDED"):
        cur.execute("""UPDATE space_jobs SET state=%s,result=%s,claim_token=NULL,lease_until=NULL,updated_at=NOW()
            WHERE account_id=%s AND space_id=%s AND job_id=%s""",
            (state, Json(result), job["account_id"], job["space_id"], job["job_id"]))
        delay = job["snapshot"]["rule"]["spec"]["trigger_policy"]["reconcile_seconds"]
        if job["snapshot"].get("continue_scan"):
            delay = job["snapshot"]["rule"]["spec"]["trigger_policy"]["min_interval_seconds"]
        minimum = job["snapshot"]["rule"]["spec"]["trigger_policy"]["min_interval_seconds"]
        cur.execute("""UPDATE space_rule_runtime SET applied_generation=GREATEST(applied_generation,%s),
            source_after=CASE WHEN %s='SUCCEEDED' THEN %s ELSE source_after END,
            next_run_at=NOW()+(CASE WHEN dirty_generation>%s THEN %s ELSE %s END * INTERVAL '1 second'),
            updated_at=NOW() WHERE account_id=%s AND space_id=%s AND rule_id=%s""",
            (job["generation"], state, job["snapshot"].get("next_source_after", ""), job["generation"], minimum, delay,
             job["account_id"], job["space_id"], job["rule_id"]))

    def complete_job(self, job, plans):
        privacy = job['snapshot']['rule']['spec'].get('privacy_policy')
        plans = [redact_item(plan, privacy) for plan in plans]
        account, space = job["account_id"], job["space_id"]
        with self.transaction(account, space) as cur:
            lock_space_account(cur, account)
            cur.execute("""SELECT j.*,r.rule_rev AS current_rev,r.state AS rule_state,s.state AS space_state
                FROM space_jobs j JOIN space_rules r USING(account_id,space_id,rule_id)
                JOIN spaces s USING(account_id,space_id) WHERE j.account_id=%s AND j.space_id=%s AND j.job_id=%s
                AND j.state='RUNNING' AND j.claim_token=%s AND j.lease_until>NOW() FOR UPDATE OF j""",
                (account, space, job["job_id"], job["claim_token"]))
            current = cur.fetchone()
            if current is None:
                raise RuntimeError("shared memory task lease changed")
            sources = self._job_sources(cur, job, lock=True, uris=job["snapshot"]["source_selection"])
            signature = fingerprint([(s["binding_id"], s["grant_rev"], s["uri"], s["content_rev"], s["sha256"], s["epoch"]) for s in sources])
            if (current["current_rev"] != job["rule_rev"] or current["rule_state"] != "ACTIVE"
                    or current["space_state"] != "ACTIVE" or signature != job["snapshot"]["source_fingerprint"]):
                result = {"reason": "input_changed"}
                self._finish_job(cur, job, result, "SUPERSEDED")
                cur.execute("""UPDATE space_rule_runtime SET dirty_generation=dirty_generation+1,next_run_at=NOW()
                    WHERE account_id=%s AND space_id=%s AND rule_id=%s""", (account, space, job["rule_id"]))
                return result
            target_ids = [item["item_id"] for item in job["snapshot"]["target_revisions"]]
            topics = sorted({plan["topic"].strip().casefold() for plan in plans if "topic" in plan})
            cur.execute("""SELECT i.*,n.version AS content_rev,n.metadata FROM space_items i JOIN context_nodes n USING(account_id,uri)
                WHERE i.account_id=%s AND i.space_id=%s AND i.rule_id=%s AND (i.item_id=ANY(%s) OR i.item_id IN (
                    SELECT DISTINCT ON (n2.metadata->>'topic',i2.acl_origin) i2.item_id FROM space_items i2
                    JOIN context_nodes n2 USING(account_id,uri) WHERE i2.account_id=%s AND i2.space_id=%s AND i2.rule_id=%s
                    AND n2.metadata->>'topic'=ANY(%s) AND (n2.metadata->>'source_window'=%s OR i2.acl_origin='manual')
                    ORDER BY n2.metadata->>'topic',i2.acl_origin,i2.updated_at DESC,i2.item_id)) ORDER BY i.item_id FOR UPDATE OF i,n""",
                (account, space, job["rule_id"], target_ids, account, space, job["rule_id"], topics, job["snapshot"]["window"]))
            items = {row["item_id"]: dict(row) for row in cur.fetchall()}
            expected = {i["item_id"]: i["content_rev"] for i in job["snapshot"]["target_revisions"]}
            if any(items.get(key, {}).get("content_rev") != rev for key, rev in expected.items()):
                self._finish_job(cur, job, {"reason": "target_changed"}, "SUPERSEDED")
                return {"reason": "target_changed"}
            cur.execute("""WITH manual AS (
                    SELECT i.item_id,i.acl_mode,n.metadata->>'topic' AS topic FROM space_items i
                    JOIN context_nodes n USING(account_id,uri)
                    WHERE i.account_id=%s AND i.space_id=%s AND i.rule_id=%s AND i.acl_origin='manual'
                      AND i.superseded_by IS NULL AND n.metadata->>'topic'=ANY(%s)
                ), restricted_counts AS (
                    SELECT topic,COUNT(*) AS count FROM manual WHERE acl_mode='restricted' GROUP BY topic
                ), shared_readers AS (
                    SELECT m.topic,a.principal FROM manual m JOIN space_read_acl a ON a.item_id=m.item_id
                      AND a.account_id=%s AND a.space_id=%s WHERE m.acl_mode='restricted'
                    GROUP BY m.topic,a.principal HAVING COUNT(*)=(SELECT count FROM restricted_counts c WHERE c.topic=m.topic)
                ) SELECT m.topic,BOOL_OR(m.acl_mode='restricted') AS restricted,
                    COALESCE((SELECT ARRAY_AGG(r.principal ORDER BY r.principal) FROM shared_readers r WHERE r.topic=m.topic),
                             ARRAY[]::varchar[]) AS readers FROM manual m GROUP BY m.topic""",
                (account, space, job["rule_id"], topics, account, space))
            manual_domains = {row["topic"]: {"mode": "restricted" if row["restricted"] else "inherit",
                                             "readers": list(row["readers"])} for row in cur.fetchall()}
            result = {"source_fingerprint": signature, "items": []}
            kept = [plan["item_id"] for plan in plans if plan["action"] == "KEEP"]
            if kept:
                rule_managed = [key for key in kept if items[key]["acl_origin"] == "rule"]
                if rule_managed:
                    acl = job["snapshot"]["rule"]["spec"]["output_acl"]
                    self._set_output_acl(cur, account, space, rule_managed, acl["mode"], acl["readers"], "rule")
                cur.execute("""UPDATE space_items SET checked_rule_rev=%s
                    WHERE account_id=%s AND space_id=%s AND item_id=ANY(%s)
                    AND COALESCE(checked_rule_rev,rule_rev)<>%s""",
                    (job["rule_rev"], account, space, kept, job["rule_rev"]))
            for plan in plans:
                if plan["action"] == "KEEP":
                    continue
                if plan["action"] == "ARCHIVE":
                    item = items[plan["item_id"]]
                    self._transition_item(cur, item, "ARCHIVED", job["job_id"])
                    result["items"].append({"item_id": item["item_id"], "state": "ARCHIVED"})
                else:
                    result["items"].append(self._publish_plan(cur, job, plan, sources, items, manual_domains))
            self._finish_job(cur, job, result)
            return result

    def _publish_plan(self, cur, job, plan, sources, items, manual_domains):
        account, space = job["account_id"], job["space_id"]
        spec = job["snapshot"]["rule"]["spec"]
        evidence = [source for source in sources if source["uri"] in plan["source_uris"]]
        evidence_hash = fingerprint(sorted((s["uri"], s["sha256"]) for s in evidence))
        acl = spec["output_acl"]
        previous = items.get(plan.get("item_id"))
        if previous is None and spec["merge_policy"] == "by_topic":
            matching = [item for item in items.values() if item["metadata"].get("topic") == plan["topic"].strip().casefold()
                        and item["metadata"].get("source_window") == job["snapshot"]["window"]]
            if matching:
                previous = max(matching, key=lambda item: item["created_at"])
        policy = manual_domains.get(plan["topic"].strip().casefold())
        if previous and previous["acl_origin"] == "manual":
            cur.execute("SELECT principal FROM space_read_acl WHERE account_id=%s AND space_id=%s AND item_id=%s ORDER BY principal",
                        (account, space, previous["item_id"]))
            policy = {"mode": previous["acl_mode"], "readers": [row["principal"] for row in cur.fetchall()]}
        acl_mode = policy["mode"] if policy else acl["mode"]
        readers = policy["readers"] if policy else acl["readers"]
        acl_origin = "manual" if policy else "rule"
        domain = {"mode": acl_mode, "readers": sorted(readers) if acl_mode == "restricted" else []}
        topic = fingerprint([plan["topic"].strip().casefold(), domain, job["snapshot"]["window"], 'experience'])
        cur.execute("""SELECT item_id,state FROM space_items WHERE account_id=%s AND space_id=%s
            AND rule_id=%s AND rule_rev=%s AND topic_key=%s AND evidence_fingerprint=%s AND kind='experience'
            AND state IN ('ACTIVE','RECOMPUTING')""",
            (account, space, job["rule_id"], job["rule_rev"], topic, evidence_hash))
        same = cur.fetchone()
        if (same is None and previous and previous['state'] in {'ACTIVE', 'RECOMPUTING'}
                and previous['kind'] == 'experience' and previous["rule_rev"] == job["rule_rev"]
                and previous["evidence_fingerprint"] == evidence_hash):
            same = {"item_id": previous["item_id"], "state": previous["state"]}
        if same:
            self._replace_contributions(cur, account, space, same["item_id"], evidence)
            if policy:
                self._set_output_acl(cur, account, space, [same["item_id"]], acl_mode, readers, acl_origin)
            self._supersede_item(cur, previous, same["item_id"], job["job_id"])
            self._system_audit(cur, account, space, job["job_id"], same["item_id"], "REFRESH_SOURCES")
            return dict(same)
        item_id = str(uuid.uuid4())
        uri = f"ctx://{account}/spaces/{space}/memories/experience/{item_id}"
        projection_event_id = str(uuid.uuid4())
        metadata = {"shared_projection_event_id": projection_event_id, "rule_id": job["rule_id"], "rule_rev": job["rule_rev"], "job_id": job["job_id"],
                    "content_hash": hashlib.sha256(plan["content"].encode()).hexdigest(),
                    "topic": plan["topic"].strip().casefold(), "source_window": job["snapshot"]["window"],
                    **plan.get("metadata", {})}
        if getattr(self, "content_store", None) is not None:
            metadata.update(self.content_store.store_shared(
                uri, plan["content"], plan["abstract"], plan["content"][:2000], metadata, cur=cur,
            ))
        cur.execute("""INSERT INTO context_nodes(uri,account_id,owner_space,category,context_type,
            content,abstract,overview,metadata) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (uri, account, f"space:{space}", 'experience', 'MEMORY',
             plan["content"], plan["abstract"], plan["content"][:2000], Json(metadata)))
        retention_policy = spec["maintenance_policy"]
        cur.execute("""INSERT INTO space_items(account_id,space_id,item_id,rule_id,rule_rev,uri,kind,
            topic_key,evidence_fingerprint,acl_mode,acl_origin,quality_score,expires_at,policy_snapshot)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW()+(%s * INTERVAL '1 day'),%s)""",
            (account, space, item_id, job["rule_id"], job["rule_rev"], uri, 'experience', topic, evidence_hash,
             acl_mode, acl_origin, plan["quality_score"], retention_policy["retention_days"], Json(retention_policy)))
        self._replace_contributions(cur, account, space, item_id, evidence)
        self._set_output_acl(cur, account, space, [item_id], acl_mode, readers, acl_origin)
        self._supersede_item(cur, previous, item_id, job["job_id"])
        filters = {"account_id": account, "owner_space": f"space:{space}", "space_key": space,
                   "authorization_resource_id": uri, "content_rev": 1, "state": "ACTIVE",
                   "context_type": 'MEMORY', "category": 'experience'}
        node = ContextNode(uri, filters["context_type"], 'experience', 2,
                           f"space:{space}", plan["abstract"], plan["content"][:2000],
                           plan["content"], metadata=metadata)
        records = []
        for record in build_index_records(node):
            record.filters.update(filters)
            record.metadata["content_rev"] = 1
            records.append(asdict(record))
        self._space_event(cur, account, uri, "UPSERT_CONTEXT", {"records": records, "content_rev": 1}, projection_event_id)
        self._system_audit(cur, account, space, job["job_id"], item_id, "GENERATE")
        return {"item_id": item_id, "uri": uri, "state": "ACTIVE"}

    def _supersede_item(self, cur, previous, item_id, job_id):
        if previous is None or previous["item_id"] == item_id:
            return
        if previous["state"] in {"ACTIVE", "RECOMPUTING"}:
            self._transition_item(cur, previous, "SUPERSEDED", job_id)
        cur.execute("""UPDATE space_items SET superseded_by=%s WHERE account_id=%s AND space_id=%s AND item_id=%s""",
                    (item_id, previous["account_id"], previous["space_id"], previous["item_id"]))

    def _set_output_acl(self, cur, account, space, item_ids, mode, readers, origin):
        cur.execute("UPDATE space_items SET acl_mode=%s,acl_origin=%s WHERE account_id=%s AND space_id=%s AND item_id=ANY(%s)",
                    (mode, origin, account, space, item_ids))
        cur.execute("DELETE FROM space_read_acl WHERE account_id=%s AND space_id=%s AND item_id=ANY(%s)",
                    (account, space, item_ids))
        if mode == "restricted":
            cur.execute("""INSERT INTO space_read_acl(account_id,space_id,item_id,principal)
                SELECT account_id,space_id,item_id,principal FROM space_members
                CROSS JOIN UNNEST(%s::varchar[]) AS items(item_id)
                WHERE account_id=%s AND space_id=%s AND principal=ANY(%s)""", (item_ids, account, space, readers))

    def _replace_contributions(self, cur, account, space, item_id, evidence):
        cur.execute("DELETE FROM space_item_sources WHERE account_id=%s AND space_id=%s AND item_id=%s",
                    (account, space, item_id))
        execute_values(cur, """INSERT INTO space_item_sources(account_id,space_id,item_id,binding_id,source_uri,
            grant_rev,source_content_rev,source_hash,source_epoch) VALUES %s""",
            [(account, space, item_id, source["binding_id"], source["uri"], source["grant_rev"],
              source["content_rev"], source["sha256"], source["epoch"]) for source in evidence], page_size=100)

    def _space_event(self, cur, account, uri, event_type, payload, event_id=None):
        cur.execute("""INSERT INTO outbox_events(event_id,event_type,uri,account_id,payload)
            VALUES (%s,%s,%s,%s,%s)""", (event_id or str(uuid.uuid4()), event_type, uri, account, Json(payload)))

    def _system_audit(self, cur, account, space, job_id, item_id, action):
        cur.execute("""INSERT INTO cp_audit_logs(log_id,account_id,actor,target,action,result,details)
            VALUES (%s,%s,'system:space-worker',%s,%s,'success',%s)""",
            (str(uuid.uuid4()), account, f"space:{space}", f"space.{action.lower()}",
             Json({"job_id": job_id, "space_id": space, "item_id": item_id})))

    def _transition_item(self, cur, item, state, job_id):
        purge = state == "PURGED"
        cur.execute("""UPDATE space_items SET state=%s,purge_after=COALESCE(purge_after,
            NOW()+(%s * INTERVAL '1 day')),updated_at=NOW()
            WHERE account_id=%s AND space_id=%s AND item_id=%s""",
            (state, item["policy_snapshot"]["archive_keep_days"], item["account_id"], item["space_id"], item["item_id"]))
        cur.execute("""UPDATE context_nodes SET status='ARCHIVED',version=version+1,updated_at=NOW(),
            content=CASE WHEN %s THEN '' ELSE content END,abstract=CASE WHEN %s THEN '' ELSE abstract END,
            overview=CASE WHEN %s THEN '' ELSE overview END,
            metadata=CASE WHEN %s THEN json_build_object('topic',metadata->>'topic',
                'source_window',metadata->>'source_window','job_id',%s::text,'state',%s::text)::jsonb ELSE metadata END
            WHERE account_id=%s AND uri=%s""",
            (purge, purge, purge, purge, job_id, state, item["account_id"], item["uri"]))
        self._space_event(cur, item["account_id"], item["uri"], "DELETE_CONTEXT",
                          {"ids_to_delete": [IndexRecord.generate_id(item['uri'], 0),
                                             IndexRecord.generate_id(item['uri'] + '/content.md', 2)]})
        self._system_audit(cur, item["account_id"], item["space_id"], job_id, item["item_id"], state)

    def maintain_due_items(self, account_id, limit=100):
        with self.transaction(account_id) as cur:
            lock_space_account(cur, account_id)
            cur.execute("SELECT NOW() AS now")
            now = cur.fetchone()["now"]
            cur.execute("""SELECT i.*,r.state AS rule_state,s.state AS space_state FROM space_items i
                JOIN space_rules r USING(account_id,space_id,rule_id) JOIN spaces s USING(account_id,space_id)
                WHERE i.account_id=%s
                AND NOT EXISTS (SELECT 1 FROM space_jobs j WHERE j.account_id=i.account_id
                    AND j.space_id=i.space_id AND j.rule_id=i.rule_id AND j.state IN ('QUEUED','RUNNING'))
                AND
                ((i.state IN ('ARCHIVED','SUPERSEDED','EXPIRED') AND i.purge_after<=NOW()) OR
                 (i.state IN ('ACTIVE','RECOMPUTING') AND (i.expires_at<=NOW() OR r.state='ARCHIVED'
                   OR s.state='ARCHIVED' OR i.created_at<=NOW()-((i.policy_snapshot->>'min_age_days')::double precision * INTERVAL '1 day'))))
                ORDER BY i.updated_at,i.item_id LIMIT %s FOR UPDATE OF i SKIP LOCKED""", (account_id, limit))
            items = cur.fetchall()
            changed = 0
            for item in items:
                state = maintenance_state(item, now)
                if item["state"] in {"ACTIVE", "RECOMPUTING"} and "ARCHIVED" in (item["rule_state"], item["space_state"]):
                    state = "ARCHIVED"
                if state is None:
                    cur.execute("""UPDATE space_items SET updated_at=NOW()
                        WHERE account_id=%s AND space_id=%s AND item_id=%s""", (account_id, item["space_id"], item["item_id"]))
                    continue
                job_id = str(uuid.uuid4())
                cur.execute("""SELECT COALESCE(MAX(generation),0)+1 AS generation FROM space_jobs
                    WHERE account_id=%s AND space_id=%s AND rule_id=%s""", (account_id, item["space_id"], item["rule_id"]))
                generation = cur.fetchone()["generation"]
                cur.execute("""INSERT INTO space_jobs(account_id,space_id,job_id,rule_id,rule_rev,operation,
                    generation,state,snapshot,result) VALUES (%s,%s,%s,%s,%s,%s,%s,'SUCCEEDED',%s,%s)""",
                    (account_id, item["space_id"], job_id, item["rule_id"], item["rule_rev"],
                     "PURGE" if state == "PURGED" else "MAINTAIN", generation,
                     Json({"item_id": item["item_id"], "policy_snapshot": item["policy_snapshot"]}), Json({"state": state})))
                self._transition_item(cur, item, state, job_id)
                changed += 1
        if getattr(self, "content_store", None) is not None:
            with self.transaction(account_id) as cur:
                self.content_store.reclaim_deleted_content(cur, account_id, min(limit, 10))
        return changed
