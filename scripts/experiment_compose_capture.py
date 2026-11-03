"""Copy the first actual OpenClaw compose response; never re-run retrieval."""
import json
import re
import subprocess


def capture_compose(agent_container, memory_container, session):
    if not isinstance(session, str) or not re.fullmatch(r"[a-z0-9_-]+", session):
        raise ValueError("Unsafe answering session key")
    result = subprocess.run(
        ["docker", "exec", agent_container, "cat", "/tmp/ogmem_ws/" + session + ".compose.json"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        value = json.loads(result.stdout)
    else:
        program = r"""const fs=require('fs');
const sessions=JSON.parse(fs.readFileSync('/home/node/.openclaw/agents/main/sessions/sessions.json','utf8'));
const id=sessions[process.argv[1]]?.sessionId;
if(!/^[a-fA-F0-9-]{36}$/.test(id||''))throw Error('Missing answering session UUID');
console.log(id);"""
        identity = subprocess.check_output(
            ["docker", "exec", agent_container, "node", "-e", program, session], text=True,
        ).strip()
        if not re.fullmatch(r"[a-fA-F0-9-]{36}", identity):
            raise ValueError("Unsafe answering session UUID")
        value = json.loads(subprocess.check_output(
            ["docker", "exec", memory_container, "cat", "/tmp/ogmem-compose/" + identity + ".json"],
            text=True,
        ))
    if not isinstance(value, dict) or "stats" not in value or "retrievedEvidence" not in value:
        raise ValueError("Incomplete actual compose capture; mount the passive BEAM capture hook")
    return json.dumps(value, ensure_ascii=False)
