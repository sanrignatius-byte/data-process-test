# Scaling UnifiedGraph to Million-Document Corpora

This note captures practical steps to evolve the current in-memory prototype into a
production pipeline at 10^6 documents.

## Current bottlenecks

- **In-memory full graph objects** (`nodes`, `edges`, adjacency dicts) do not fit 10^6 docs.
- **Exhaustive DFS from all element nodes** scales poorly with corpus size.
- **Single-machine JSON read/write** is not suitable for distributed indexing and retrieval.

## Recommended architecture

1. **Offline graph build (distributed):**
   - Build citation and intra-doc edges in partitioned jobs.
   - Store edges in Parquet (partitioned by `src_doc_bucket`).

2. **Serving graph store:**
   - Use a KV/graph backend (e.g., RocksDB/TiKV/Neo4j/JanusGraph depending on ops constraints).
   - Materialize compact adjacency lists keyed by node id.

3. **Two-stage path retrieval:**
   - Stage A: candidate docs (citation neighborhood + embedding ANN prefilter).
   - Stage B: constrained path search only on candidate-induced subgraph.

4. **Search budget controls:**
   - Limit seed nodes per query.
   - Limit neighbor expansion per hop.
   - Keep only top-K partial paths by optimistic score (beam search).

## What is already added in code

- `max_start_nodes`: cap seed element nodes used for path discovery.
- `neighbor_limit`: cap per-step DFS expansion after confidence ranking.
- Alignment edges are zero-cost identity links for logical scoring (`ALIGNMENT` does not increase logical hops).

## Next milestones

- Add **beam search mode** (priority queue) for deterministic bounded search.
- Add **subgraph extraction API**: query-time build of local induced graph around candidate docs.
- Add telemetry:
  - expanded nodes/edges per query
  - prune ratio
  - alignment-hit ratio
  - median path latency

## Suggested rollout

1. Keep current implementation for ≤10k docs experiments.
2. Move storage format to Parquet + adjacency KV for 100k docs.
3. Introduce distributed build + bounded retrieval for 1M docs.
