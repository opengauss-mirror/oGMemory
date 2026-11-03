---
name: og-add-history
description: Add current project's Claude Code conversation history to oGMemory
---

# oGMemory Add History

Add current project's Claude Code historical conversations to oGMemory by calling `/api/v1/after_turn` for each transcript.

## When to use

- User wants to import past conversations into memory
- User says "add history" or "import conversations"
- User types `/og-add-history`

## Instructions

### Step 1: Estimate cost and confirm

Show the user a cost estimate for the **current project**:

```bash
python3 -c "
import os
cwd = os.environ.get('PWD', os.getcwd())
proj = cwd.replace('/', '-').replace('_', '-')
base = os.path.expanduser(f'~/.claude/projects/{proj}')
if not os.path.isdir(base):
    base = os.path.expanduser(f'~/.claude/projects/-{proj}')
total_size = 0; total_files = 0; skipped = 0
if os.path.isdir(base):
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d != 'subagents']
        for f in files:
            if f.endswith('.jsonl') and not f.endswith('.ingest-offset'):
                path = os.path.join(root, f)
                off = path + '.ingest-offset'
                if os.path.isfile(off):
                    try:
                        with open(off) as of:
                            if int(of.read().strip()) >= os.path.getsize(path):
                                skipped += 1; continue
                    except: pass
                total_size += os.path.getsize(path); total_files += 1
print(f'Project: {proj}')
print(f'Transcripts: {total_files} (skipped {skipped} already ingested)')
print(f'Total size: {total_size / 1024 / 1024:.1f} MB')
print(f'Estimated tokens (embedding): ~{total_size // 4:,}')
print(f'LLM extraction calls: ~{total_files}')
print(f'Estimated time: ~{total_files * 3 // 60 + 1} minutes')
"
```

**Ask the user to confirm before proceeding.**

### Step 2: Run batch add

For each transcript: read JSONL, extract messages (same parsing logic as `call_after_turn.py`), POST to `/api/v1/after_turn`.

```bash
python3 -c "
import os, json, urllib.request
cwd = os.environ.get('PWD', os.getcwd())
proj = cwd.replace('/', '-').replace('_', '-')
base = os.path.expanduser(f'~/.claude/projects/{proj}')
if not os.path.isdir(base): base = os.path.expanduser(f'~/.claude/projects/-{proj}')
OG_URL = os.environ.get('OG_MEMORY_URL', 'http://localhost:8090')
ok = skip = fail = dedup = 0
for root, dirs, files in os.walk(base):
    dirs[:] = [d for d in dirs if d != 'subagents']
    for f in sorted(files):
        if not f.endswith('.jsonl') or f.endswith('.ingest-offset'): continue
        path = os.path.join(root, f)
        sid = f.replace('.jsonl', '')
        off_path = path + '.ingest-offset'
        if os.path.isfile(off_path):
            try:
                with open(off_path) as of:
                    if int(of.read().strip()) >= os.path.getsize(path):
                        print(f'DEDUP {sid}'); dedup += 1; continue
            except: pass
        msgs = []
        for line in open(path, errors='replace'):
            line = line.strip()
            if not line: continue
            try: e = json.loads(line)
            except: continue
            if e.get('isSidechain') or e.get('isApiErrorMessage'): continue
            if e.get('type') not in ('user','assistant'): continue
            m = e.get('message', {})
            if m.get('role') not in ('user','assistant'): continue
            c = m.get('content','')
            t = c if isinstance(c,str) else '\n'.join(b.get('text','') for b in c if isinstance(b,dict) and b.get('type')=='text')
            t = t.strip()
            if t: msgs.append({'role':m['role'],'content':t})
        if not msgs: print(f'SKIP {sid}'); skip += 1; continue
        payload = json.dumps({'sessionId':sid,'messages':msgs,'accountId':os.environ.get('OG_MEMORY_ACCOUNT_ID','acct-demo'),'userId':os.environ.get('OG_MEMORY_USER_ID','u-claude'),'agentId':'claude-code','hook_event_name':'Stop'}).encode()
        req = urllib.request.Request(f'{OG_URL}/api/v1/after_turn', data=payload, headers={'Content-Type':'application/json'}, method='POST')
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                if resp.status < 300:
                    print(f'OK {sid}: {len(msgs)} msgs'); ok += 1
                    with open(off_path,'w') as of: of.write(str(os.path.getsize(path)))
                else: print(f'FAIL {sid}: {resp.status}'); fail += 1
        except Exception as ex: print(f'FAIL {sid}: {ex}'); fail += 1
print(f'\nDone: ok={ok} fail={fail} skip={skip} dedup={dedup}')
"
```

### Step 3: Report results

Report the counts and verify service health:

```bash
curl -s http://localhost:8090/api/v1/health
```

## Important notes

- Background extraction/embedding continues after the POST returns
- Already-added sessions are skipped via `.ingest-offset` files
- Subagent transcripts are automatically excluded
- By default only the current project's history is ingested
