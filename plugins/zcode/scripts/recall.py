import time
from urllib.parse import quote

from capture import get, text_value


def recall(client, event, identity):
    deadline = time.monotonic() + 5
    name = get(event, "hookEventName", "hook_event_name")
    query = event.get("prompt") if name == "UserPromptSubmit" else (
        f"ZCode session {event.get('source', 'startup')} in {event.get('cwd', '')}. "
        "Recall project conventions, decisions and user preferences."
    )
    query = text_value(query or "", 12000).strip()
    if not query:
        return ""
    code, body, _ = client.request("compose", {**identity, "prompt": query}, timeout=3)
    parts = []
    if code == 200:
        for key in ("identityContext", "episodicContext", "sessionContext", "taskContext", "retrievedEvidence"):
            if isinstance(body.get(key), str) and body[key].strip():
                parts.append(body[key].strip())
    memory = "\n\n".join(parts)
    skills = []
    remaining = deadline - time.monotonic()
    if name == "UserPromptSubmit" and remaining > 0.15:
        agent = quote(identity["agentId"], safe="")
        encoded = quote(query.encode()[:500].decode(errors="ignore"), safe="")
        try:
            code, catalog, _ = client.request(f"skills?q={encoded}&limit=3&agent_id={agent}",
                                               timeout=min(0.8, remaining))
            rows = catalog.get("skills", []) if code == 200 else []
            if not isinstance(rows, list):
                rows = []
            rows = [row for row in rows[:3] if isinstance(row, dict) and row.get("name")]
            skills = [f"- {str(row['name'])[:40]}: {str(row.get('description', ''))[:60]}" for row in rows]
            remaining = deadline - time.monotonic()
            if rows and remaining > 0.15 and float(rows[0].get("score") or 0) >= 0.20:
                skill = quote(str(rows[0]["name"]), safe="")
                source = quote(str(rows[0].get("source") or "generated"), safe="")
                code, detail, _ = client.request(
                    f"skills/{skill}?source={source}&agent_id={agent}&include_files=false",
                    timeout=min(0.8, remaining),
                )
                if code == 200 and isinstance(detail.get("content"), str):
                    skills.append(detail["content"][:800])
        except (OSError, ValueError, TypeError):
            pass
    if not memory and not skills:
        return ""
    context = memory[:1500] + "\n\nPublished Skills:\n" + "\n".join(skills)[:750] if skills else memory
    return ("[oGMemory reference data]\n" + context)[:2400].encode()[:9600].decode(errors="ignore")
