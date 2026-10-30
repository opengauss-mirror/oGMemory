# oGMemory QwenPaw plugin

Native QwenPaw plugin for automatic recall, conversation writes and canonical Agent Trace capture. Uses the `PluginApi` and AgentScope middleware interfaces introduced in QwenPaw 2.1, including later versions that retain these interfaces.

Installation, configuration and cross-session examples: [QwenPaw integration](../../docs/integrations/qwenpaw.md).

## Runtime

- `on_model_call` retrieves memory through `compose` and inserts it into the current model input.
- `on_reply` writes completed user/assistant text through `after_turn`. Pending batches and their stable request IDs remain in `AgentState.middle_context` until acknowledged.
- Explicit requests containing `remember`, `记住`, `记下` or `保存到记忆` trigger extraction even below the server's accumulation threshold.
- Native stream events produce canonical message, tool, permission and turn events through `/api/v1/trace/events`.
- A parked reply awaiting tool confirmation keeps its logical turn open until the native reply ends.
- When compression changes the context, the plugin writes captured conversation text and calls `compact` to extract accumulated memory regardless of the ordinary turn threshold. During a reply this runs at reply completion; standalone compression flushes immediately. Pending writes and compaction are retried on the next reply. QwenPaw keeps its own compressed context.
- The native `/compact` command uses a separate runtime path. A `PRE_DISPATCH` hook persists an extraction marker in QwenPaw's session, submits pending conversation writes, and requests extraction before QwenPaw handles the command. Memory service failures retain pending work while QwenPaw continues compression. The next reply or `/compact` retries it, including after a restart.
- Compression records its boundaries. Shutdown and uninstall drain the bounded Trace queue and dispose sessions.
- Explicit `ogmemory_search` uses QwenPaw's current request identity, including when its stream advances across heartbeat tasks.

QwenPaw retains ownership of its local context, model, tools and session persistence. oGMemory retains ownership of memory extraction, indexing, retrieval and Trace storage.

## Development

Run in the QwenPaw Python environment:

```bash
python -m pytest --confcutdir=tests/unit tests/unit/qwenpaw_plugin tests/unit/e2e -q
```

The shared [memory evaluation guide](../../docs/testing/memory-evaluation-guide-zh.md) covers dataset selection, host setup, write verification, QA and judging. Select `--agent qwenpaw --transport plugin`, or use `tests/e2e/run_qwenpaw_plugin_locomo.py` as the entry point.
