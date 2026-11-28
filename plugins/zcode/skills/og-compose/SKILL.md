---
name: og-compose
description: Query previously stored oGMemory facts for the user's current task.
---

For an explicit memory lookup, run `python "${ZCODE_SKILL_DIR}/../../scripts/ogmemory_zcode_hook.py" --query "<query>"`.
Use the configured Python interpreter and pass the query as a single argument. Summarize the returned context for the user.
