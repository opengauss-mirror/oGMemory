# Entity and preference cards

The extraction YAML defines the Markdown layout. Enable prewrite layout validation:

```yaml
memory:
  extraction:
    card_format_enabled: true
```

`OGMEM_CARD_FORMAT_ENABLED=true` is the environment equivalent. The setting applies to the HTTP memory service. It defaults to false for existing deployments.

Both types use a single-line abstract and an overview with one core fact per `- ` bullet. Entity content has one H1 display name, an optional introductory sentence, and named H2 sections with fact bullets. Preference content has one H1 topic and preference bullets. Preserve exact facts and source provenance; the layout check does not establish factual correctness.

With `node_processing_enabled: true`, eager uses one native extraction pass and lazy uses one ReAct run, followed by entity/preference node processing. Otherwise the configured extraction passes and candidate reconciliation apply. Validation runs after candidate selection and before checkpoint acceptance, and again on the built node before storage. A malformed card enters the existing durable source recovery task. The synchronous turn receives `ok=true, status=queued, reason=memory_card_format`; queued acknowledges the durable task, and the source completes when recovery finishes. An asynchronous turn retains its normal `processing` acknowledgement and registers recovery in its worker after a format failure.

Turns without a client request ID receive a fresh identity, persisted with their source batch. Identical text in a later turn has its own identity. Background recovery reuses the saved identity and source, with at most two generation retries. Transient provider failures have a separate three-retry budget with backoff. The validation error is included in the retry's extraction prompt. Accepted candidates and completed writes retain their existing checkpoint/receipt protection.

Raw conversation chunks are persisted before structured extraction. When model retries are exhausted, recovery completes from these source memories and creates a deterministic archive marked `memory_extraction_degraded=true`. Confirmed writes are preserved through their receipts. Inspect completion through `wait_until_idle` and its recovery task status.

HTTP and MCP startup discover pending ordinary source batches from persisted session state, including tasks interrupted by a process restart. Custom service hosts call `MemoryService.start_extraction_recovery()` when starting and `shutdown()` when stopping. Gunicorn hosts use `-c python:server.gunicorn_config`; the standalone entrypoints configure this hook automatically.

AGFS session requests and recovery share a renewable storage lease and reload saved state after acquiring ownership. SQL recovery enumerates the content-free `extraction_work_accounts` directory, then binds tenant RLS for each scan. Schema initialization backfills existing session accounts once; subsequent session writes maintain this directory in the same transaction.

Scope: ordinary `after_turn` source extraction. Compaction and dispose retain their lifecycle recovery contract.
