---
name: og-compose
description: Compose full memory context from oGMemory (profile, archives, session state, working set)
---

# oGMemory Compose

Retrieve full composed memory context from oGMemory, including profile, archive history, session state, and relevant working set items.

## When to use

- User asks to search or look up past conversations
- User asks "what did we discuss about X"
- User wants to recall previous decisions or context
- User types `/og-compose`

## Instructions

1. Take the user's query from the arguments or infer it from the conversation context
2. Call the compose API and extract all context layers:

```bash
curl -s -X POST "${OG_MEMORY_URL:-http://localhost:8090}/api/v1/compose" \
  -H "Content-Type: application/json" \
  -d "{\"prompt\":\"<query>\",\"accountId\":\"${OG_MEMORY_ACCOUNT_ID:-acct-demo}\",\"userId\":\"${OG_MEMORY_USER_ID:-u-claude}\",\"agentId\":\"claude-code\"}" \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
parts = []
for key, label in [('identityContext','Profile'),('episodicContext','Archives'),('sessionContext','Session'),('retrievedEvidence','Working Set')]:
    val = (data.get(key) or '').strip()
    if val: parts.append(f'## {label}\n{val}')
if parts: print('\n\n'.join(parts))
else: print('No relevant context found.')
"
```

3. Present the results to the user in a clear format
4. If no results, suggest alternative queries

## Examples

```
/og-compose marketplace plugin setup
/og-compose how did we fix the AGFS mount issue
/og-compose openclaw context engine
```
