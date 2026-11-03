# Two-stage topic processing

Enable `memory.extraction.node_processing_enabled: true` with eager or lazy mode.
Eager uses a single native extraction pass; lazy uses one ReAct run with its
existing read tools and bounded iterations. Both use the configured extraction
temperature and retain separate
entity/preference drafts, including complementary drafts across spans. The service
persists these drafts before processing topic nodes at temperature 0.

The processor partitions drafts by category, owner, scope and preference holder.
It reads the paginated identity catalog, selects lexical/exact candidates, supplements
larger catalogs with vector prefetch, and reads complete candidate snapshots.
Exact identities and bound targets receive priority within the processing budget;
optional snapshots use the remaining space. Oversized draft batches run sequentially
against shared working snapshots. Each target produces one final versioned update,
preserving the combined source references and confirmed aliases.
Drafts retain the complete focused extraction span, including neighboring turns and
the source time anchor. Split batches carry only their contributors' spans. Durable
checkpoints preserve these contributors for recovery and version-conflict replay.
Legacy drafts retain their saved message context. An individual span or required
node snapshot that exceeds the budget still requires a smaller source write.
New nodes use batch-local read identities while later batches edit or reuse them.
The final combined candidate becomes one create with a storage-allocated identity.
One bounded tool response partitions drafts into create/update/noop/defer decisions.
Existing updates use exact local edits and additions, with whole-card summaries.
Code renders new sections and validates final entity/preference card structure.
Confirmed names are saved in the existing routing and object alias metadata.

Owner/category identity fences cover extraction, processing and commit. This initial
implementation serializes topic writes for one owner while independent owners can
proceed concurrently. These fences can increase same-owner queue latency.
Updates materialize full read-bound candidates and reuse the existing versioned writer.
Committed operations retain their receipts. A proven pre-write version conflict creates
a new checkpoint revision, refreshes the node and reprocesses the update.

Processing failures use the existing bounded asynchronous recovery. Saved drafts
allow stage B recovery without repeating stage A. Exhausted recovery stays visible.
Occurrence qualification/reconciliation and session summaries retain their own costs.

Candidate recall and semantic decisions are probabilistic. Card validation checks
structure, while source grounding and meaning preservation still require evaluation.
