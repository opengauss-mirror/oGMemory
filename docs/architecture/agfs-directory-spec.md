# AGFS Directory Specification v1

## Overview

ContextEngine stores ContextNodes as directories in AGFS (Adaptive Globally-addressable File System). Each node corresponds to a URI and contains multiple files representing different content layers and metadata.

## URI → Physical Path Mapping

```
ctx://{account}/users/{user}/memories/{category}/{slug}
→ /accounts/{account}/users/{user}/memories/{category}/{slug}/

ctx://{account}/agents/{agent}/memories/{category}/{slug}
→ /accounts/{account}/agents/{agent}/memories/{category}/{slug}/

ctx://{account}/agents/{agent}/skills/{skill_name}
→ /accounts/{account}/agents/{agent}/skills/{skill_name}/
```

### URI Components

| Component | Description | Examples |
|-----------|-------------|----------|
| account | Top-level tenant isolation | `acme-corp` |
| user | User ID (user-space nodes) | `user-123` |
| agent | Agent ID (agent-space nodes) | `agent-gpt-4` |
| category | Memory category | `profile`, `preferences`, `entities`, `events`, `cases`, `patterns`, `skills` |
| slug | URL-safe identifier | `coffee_preferences`, `event_20250317_abc123` |

## Node Directory Structure

Every ContextNode is stored as a directory with the following files:

```
/accounts/{account}/users/{user}/memories/profile/
├── content.md          # Full original content
├── .relations.json     # Outgoing relation edges
├── .abstract.md        # L0: ≤100 char summary
├── .overview.md        # L1: Structured overview
├── .meta.json          # L2: Metadata + status (commit point)
└── .outbox/            # Pending outbox events
    ├── {event_id}.json
    └── dlq/            # Dead letter queue for failed events
```

### File Descriptions

| File | Purpose | Format |
|------|---------|--------|
| `content.md` | Full original content | Markdown |
| `.relations.json` | Outgoing RelationEdge list | JSON array |
| `.abstract.md` | L0 summary for vector recall | Markdown, ≤100 chars |
| `.overview.md` | L1 structured overview | Markdown |
| `.meta.json` | Metadata + NodeStatus | JSON (see schema below) |
| `.outbox/{event_id}.json` | Pending outbox events | JSON (OutboxEvent) |

## .meta.json Schema

```json
{
  "uri": "ctx://acme/users/alice/memories/profile",
  "context_type": "MEMORY",
  "category": "profile",
  "level": 3,
  "owner_space": "user:alice",
  "status": "ACTIVE",
  "created_at": "2025-03-17T10:00:00Z",
  "updated_at": "2025-03-17T11:30:00Z",
  "version": 3,
  "tags": ["user-profile", "verified"]
}
```

### NodeStatus Values

| Status | Description | Visible to Retrieval |
|--------|-------------|---------------------|
| `PENDING` | Write in progress (steps ①-③ complete, ④ incomplete) | No |
| `ACTIVE` | Fully written, ready for retrieval | Yes |
| `BROKEN` | Repair Job detected corruption, needs manual intervention | No |

## Atomic Write Order (Critical)

The 4-step write order MUST be followed strictly. Repair Job branch logic depends on this order.

```
Step ①: Write content.md (largest file, do first)
   ↓
Step ②: Write .relations.json
   ↓
Step ③: Write .abstract.md + .overview.md (parallel safe)
   ↓
Step ④: Write .meta.json with status=ACTIVE ★ COMMIT POINT
   ↓
Step ⑤: Register OutboxEvent to .outbox/{event_id}.json
```

### Why This Order Matters

| Failure Point | Repair Job Detection | Recovery Action |
|---------------|---------------------|-----------------|
| Before step ① | No .meta.json, no content.md | Skip (nothing to repair) |
| After step ①, before ② | content.md exists, no .meta.json | Detect orphan, recreate from content |
| After step ②, before ③ | .relations.json exists, no .meta.json | Detect PENDING, attempt completion |
| After step ③, before ④ | .abstract.md/.overview.md exist, status=PENDING | Detect PENDING, update status=ACTIVE |
| After step ④ | .meta.json status=ACTIVE | Normal, skip |

## ContextFS.exists() Implementation

```python
def exists(self, uri: str, ctx: RequestContext) -> bool:
    """Check if node exists and is ACTIVE."""
    meta_path = uri_to_path(uri) / ".meta.json"
    if not meta_path.exists():
        return False
    meta = json.loads(meta_path.read_text())
    return meta.get("status") == "ACTIVE"
```

**Key rule:** PENDING nodes are NOT visible to upper layers.

## .relations.json Schema

```json
[
  {
    "from_uri": "ctx://acme/users/alice/memories/entities/coffee_shop",
    "to_uri": "ctx://acme/users/alice/memories/events/visit_20250315",
    "relation_type": "related_to",
    "weight": 0.85,
    "reason": "Alice visited this coffee shop on March 15"
  }
]
```

## OutboxEvent Persistence

OutboxEvents are stored in `.outbox/` alongside the node directory:

```
.outbox/
├── {event_id}.json     # PENDING or PROCESSING events
└── dlq/                # FAILED events after MAX_RETRY retries
    └── {event_id}.json
```

### OutboxEvent Schema

```json
{
  "event_id": "550e8400-e29b-41d4-a716-446655440000",
  "event_type": "UPSERT_CONTEXT",
  "uri": "ctx://acme/users/alice/memories/profile",
  "payload": {
    "records": [
      {
        "id": "a1b2c3d4e5f6g7h8",
        "uri": "ctx://acme/users/alice/memories/profile",
        "level": 0,
        "text": "Alice is a software engineer who prefers Python",
        "filters": {"account_id": "acme", "owner_space": "user:alice"},
        "metadata": {"category": "profile", "context_type": "MEMORY"}
      }
    ]
  },
  "status": "PENDING",
  "retry_count": 0,
  "created_at": "2025-03-17T10:00:00Z"
}
```

**Payload formats by event_type:**
- `UPSERT_CONTEXT`: `payload.records` = serialized L0/L1/L2 IndexRecord list
- `DELETE_CONTEXT`: `payload` = `{}`
- `MOVE_CONTEXT`: `payload` = `{"new_uri": "ctx://..."}`
- `UPSERT_RELATION`: `payload` = `{"edges": [RelationEdge list]}`

## Repair Job Branch Logic

Repair Job scans `/accounts/**/` and checks each `.meta.json`:

```python
for node_dir in scan_accounts():
    meta_path = node_dir / ".meta.json"

    if not meta_path.exists():
        content_exists = (node_dir / "content.md").exists()
        if content_exists:
            # Recovery: recreate .meta.json, register OutboxEvent
            repair_missing_metadata(node_dir)
        # else: skip (nothing to repair)

    meta = json.loads(meta_path.read_text())
    status = meta.get("status")

    if status == "PENDING":
        if all_files_present(node_dir):
            # Recovery: update status to ACTIVE, register OutboxEvent
            activate_pending_node(node_dir)
        else:
            # Recovery: update status to BROKEN, alert
            mark_broken(node_dir)

    if status == "ACTIVE":
        # Normal, skip
        pass

    if status == "BROKEN":
        # Already marked, skip (requires manual intervention)
        pass
```

## Multi-Tenant Isolation

### Physical Namespace Separation

```
/accounts/{account}/           ← Account-level isolation
├── users/                     ← User-owned memories
│   └── {user_id}/
│       └── memories/
└── agents/                    ← Agent-owned memories
    └── {agent_id}/
        └── memories/
```

### Access Control Rules

| Request | Rule |
|---------|------|
| Read `ctx://{A}/...` | Allow only if RequestContext.account_id == A |
| Write `ctx://{A}/...` | Allow only if RequestContext.account_id == A |
| List `/accounts/{A}/...` | Allow only if RequestContext.account_id == A |
| Cross-account read | **DENIED** (v1) |

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v1 | 2025-03-17 | Initial spec, Phase 0 frozen |
