# Experiment Plan — Graph-Augmented Retrieval

**Problem**: On M4query_v1 (53 docs, 473 queries, 1798 passages), the best precision (R@1=0.2505) and best recall (R@10=0.6406) come from two disjoint configurations that cannot be simultaneously achieved.
**Method Thesis**: Combining typed cross-document element edges (figure/formula/table) with a richer intra-document chunk-v2 graph under a precision-oriented reranking mode can close the R@1 vs R@10 gap without a method switch.
**Date**: 2026-04-19

---

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|----------------|-----------------------------|---------------|
| C1 | Explicit bridge-edge rerank with static prior improves precision over dense baseline | R@1 and MRR consistently above dense baseline across 0.6B and 4B | B1 (baseline) |
| C5 | Typed cross-doc element edges (figure/formula/table) provide R@10 uplift | R@10=0.6406 via `explicit+typed(w=0.2)` static_plus_neighbor — already proven | B2 |
| C_NEW | Typed cross-doc + chunk-v2 graph simultaneously achieves R@1 ≥ 0.25 AND R@10 ≥ 0.62 under static_prior | A single config beating both thresholds | B3 |
| C_PARA | Fixed-length paragraph merge (n=400/500) improves corpus quality and graph connectivity | New corpus v3 baseline R@1 ≥ v1_enriched 0.2389 | B4 |

---

## Paper Storyline

**Main paper must prove:**
- The document graph provides measurable retrieval uplift on multimodal scientific documents (C1 done, C5 done)
- The uplift extends to both precision (R@1) and recall (R@10) with the right combination (C_NEW)

**Appendix can support:**
- bbl citation boost strengthens typed_crossdoc once coverage is expanded
- Paragraph-merged corpus provides a cleaner dense baseline
- The three-axes validation (retrieval / QA / data synthesis) from C4

**Experiments intentionally cut:**
- Summary virtual nodes as retrieval signal (C6: closed)
- Intra-doc virtual edges in the candidate pool (C2: harmful, proven)
- LLM-generated virtual summaries as cross-doc bridge (too expensive, superseded by typed edges)

---

## Experiment Blocks

### Block 1: Baseline Confirmation (already done — document here)
- **Claim tested**: C1
- **Why**: Anchors all comparison tables
- **Config**: 0.6B dense `v1_enriched` → explicit-only graph rerank, static_prior, chunk_v2 graph
- **Result**: R@1=0.2505, R@5=0.4852, R@10=0.5391, MRR=0.6162
- **Reference**: `exp:20260419_deliverable_420`, dir `graph_06b_v1_explicit_v2chunk`
- **Priority**: DONE

### Block 2: Typed Cross-Doc R@10 Uplift (already done — document here)
- **Claim tested**: C5
- **Why**: Anchors cross-doc claim
- **Config**: 0.6B ranking + explicit+typed(w=0.2), static_plus_neighbor, chunk_v1 graph
- **Result**: R@1=0.1818, R@5=0.5423, R@10=0.6406, MRR=0.5413
- **Reference**: `exp:20260419_typed_crossdoc`, dir `typed_crossdoc/`
- **Priority**: DONE

### Block 3: Combined — chunk-v2 graph + typed_crossdoc, static_prior [MUST-RUN]
- **Claim tested**: C_NEW — can one config achieve R@1 ≥ 0.25 AND R@10 ≥ 0.62?
- **Why this block exists**: The two best configs optimize different axes (static_prior → R@1, static_plus_neighbor → R@10). Typed edges under static_prior have not been tested with chunk_v2 graph.
- **Dataset**: `v1_enriched` corpus, 473 queries, `augmented_v2/`
- **Compared systems**:
  - [ref A] explicit_only + chunk_v2, static_prior (R@1=0.2505, R@10=0.5391)
  - [ref B] explicit+typed(w=0.2), static_plus_neighbor, chunk_v1 (R@1=0.1818, R@10=0.6406)
  - **new** explicit+typed(w=0.1), chunk_v2, static_prior
  - **new** explicit+typed(w=0.2), chunk_v2, static_prior
  - **new** explicit+typed(w=0.3), chunk_v2, static_prior
  - **new** explicit+typed(w=0.2), chunk_v2, static_plus_neighbor (cross-check)
- **Metrics**: R@1 (primary), R@10 (primary), MRR, R@5
- **Success criterion**: R@1 ≥ 0.25 AND R@10 ≥ 0.62 in any single config
- **Failure interpretation**: If typed_crossdoc under static_prior still hurts R@1 (< 0.24), the two axes are fundamentally incompatible at current edge quality; next step is bbl expansion to improve signal precision.
- **Table target**: Main comparison table (Table 2 in future paper)
- **Priority**: **MUST-RUN**
- **Cost**: CPU-only, ~10 min (no new embeddings; reuses existing ranking + graph files)

**SLURM sketch** (`slurm_scripts/32_combo_typed_chunkv2.sh`):
```bash
#!/bin/bash
#SBATCH --job-name=32_combo_typed_chunkv2
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00

source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh
conda activate /projects/myyyx1/envs/minerU
cd /projects/myyyx1/data-process-test

RANKING="data/05_eval/dense_retrieval/augmented_v2/ranking_v1_enriched_qwen06b.jsonl"
QRELS="data/05_eval/dense_retrieval/augmented_v2/qrels_v1.jsonl"
CORPUS="data/05_eval/dense_retrieval/augmented_v2/corpus_v1_enriched.jsonl"
CHUNK_V2="data/01_graphs/chunk_virtual_nodes_v2.json"
HUB_CANDS="data/02_enriched/hub_candidates_enriched_v3.json"
TYPED="data/01_graphs/typed_crossdoc_edges.json"
OUTBASE="data/05_eval/dense_retrieval/combo_typed_chunkv2"

for W in 0.1 0.2 0.3; do
  python scripts/eval_graph_topk_rerank.py \
    --ranking "$RANKING" --qrels "$QRELS" --corpus "$CORPUS" \
    --chunk-graph "$CHUNK_V2" --hub-candidates "$HUB_CANDS" \
    --typed-crossdoc-edges "$TYPED" \
    --graph-sources explicit typed_crossdoc \
    --typed-crossdoc-weight "$W" \
    --prior-mode weighted \
    --output-dir "${OUTBASE}/explicit_typed_w${W}_v2chunk_static_prior"
done

# Also test static_plus_neighbor with chunk_v2
python scripts/eval_graph_topk_rerank.py \
  --ranking "$RANKING" --qrels "$QRELS" --corpus "$CORPUS" \
  --chunk-graph "$CHUNK_V2" --hub-candidates "$HUB_CANDS" \
  --typed-crossdoc-edges "$TYPED" \
  --graph-sources explicit typed_crossdoc \
  --typed-crossdoc-weight 0.2 \
  --prior-mode weighted \
  --output-dir "${OUTBASE}/explicit_typed_w02_v2chunk_neighbor"
```

### Block 4: Paragraph Merge — corpus v3 [MUST-RUN for 4.20 TODO]
- **Claim tested**: C_PARA
- **Why**: Mentor 4.16 plan requires paragraph merge with fixed-length chunks n=400/500, new edges chunk-para and section-chunk. Tests whether richer chunk boundaries improve the dense retrieval baseline.
- **Dataset**: Same 53 docs, rebuild corpus from scratch
- **Steps**:
  1. Write `scripts/build_paragraph_chunks.py` — reads existing paragraph nodes, merges same-section paragraphs to target length (400 or 500 tokens by nltk word count), outputs new chunk nodes + edges chunk-para + section-chunk
  2. Rebuild augmented corpus (v3) using merged chunks as the text unit
  3. Run `eval_dense_retrieval.py` baseline (0.6B)
  4. If v3 baseline ≥ v1_enriched, run full graph rerank suite on v3
- **Metrics**: R@1, R@10, MRR on 0.6B dense baseline
- **Success criterion**: v3 dense baseline R@1 > 0.2389 (v1_enriched dense baseline)
- **Failure interpretation**: Paragraph merge does not help the base embedding quality; chunk granularity is already well-matched in v1_enriched.
- **Table target**: Ablation table — corpus construction choices
- **Priority**: **MUST-RUN** (4.20 mentor deliverable)
- **Cost**: ~1h compute (needs new script), no GPU needed for chunk construction; 0.6B embedding eval needs A5000

**SLURM sketch** (`slurm_scripts/33_paragraph_merge.sh`):
```bash
#!/bin/bash
#SBATCH --job-name=33_paragraph_merge
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Step 1: Build paragraph-merged chunks (n=400 and n=500)
python scripts/build_paragraph_chunks.py \
  --graph data/01_graphs/latex_reference_graph_v2.json \
  --elements data/02_enriched/multimodal_elements_enriched.json \
  --chunk-size 400 \
  --output data/01_graphs/paragraph_chunks_n400.json

python scripts/build_paragraph_chunks.py \
  --graph data/01_graphs/latex_reference_graph_v2.json \
  --elements data/02_enriched/multimodal_elements_enriched.json \
  --chunk-size 500 \
  --output data/01_graphs/paragraph_chunks_n500.json

# Step 2: Build corpus v3 (merge chunk text into passages)
python scripts/build_graph_augmented_corpus.py \
  --paragraph-chunks data/01_graphs/paragraph_chunks_n400.json \
  --output-dir data/05_eval/dense_retrieval/augmented_v3_n400

# Step 3: Dense retrieval baseline on v3
python scripts/eval_dense_retrieval.py \
  --data-dir data/05_eval/dense_retrieval/augmented_v3_n400 \
  --model-name Qwen/Qwen3-Embedding-0.6B \
  --output data/05_eval/dense_retrieval/augmented_v3_n400/eval_v3_n400_qwen06b.json \
  --batch-size 8
```

### Block 5: bbl Coverage Expansion [NICE-TO-HAVE]
- **Claim tested**: C5 (strengthens citation boost in typed_crossdoc)
- **Why**: Current `citation_graph.json` covers only 59 docs / 123 edges. The +0.05 boost applies to only 10% of typed_crossdoc edges. Expanding bbl extraction to all 53 eval docs may make cite_boost meaningful.
- **Steps**:
  1. Re-run bbl extractor on 53 eval docs
  2. Rebuild `citation_graph.json` (expect 53 docs / 300-600 edges)
  3. Rebuild `typed_crossdoc_edges.json` with new boost values
  4. Re-run Block 3 best config with new edges
- **Priority**: NICE-TO-HAVE (no GPU needed, ~30 min)

### Block 6: QA-Side Validation — graph guided evidence recall [NICE-TO-HAVE]
- **Claim tested**: C4 (three-axes: retrieval ✓, QA ??, data synthesis ??)
- **Why**: gap:G4 — retrieval gains are proven but QA uplift is still missing
- **Steps**: With C-Pool 78 queries + human-annotated qrels (pending mentor decision), run graph rerank and measure evidence recall improvement
- **Blocked by**: C-Pool qrels (mentor decision 4.20)
- **Priority**: NICE-TO-HAVE / BLOCKED

---

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|-----------|------|------|---------------|------|------|
| **M1** | Combine typed_crossdoc + chunk-v2 | B3 (R001-R004) | If R@1≥0.25 AND R@10≥0.62 → new paper-worthy config | CPU ~10 min | typed edges hurt R@1 under static_prior |
| **M2** | Paragraph merge baseline | B4 step 1-3 | If v3 dense > v1_enriched → continue graph rerank on v3 | 1h, A5000 | Needs new script (build_paragraph_chunks.py) |
| **M3** | bbl expansion | B5 | If cite_boost coverage > 30% → retrain typed edges | 30 min CPU | bbl files might be missing for some docs |
| **M4** | C-Pool QA validation | B6 | Blocked on qrels decision | 0 now | Mentor decides manual vs auto |

---

## Compute and Data Budget

- **Block 3** (combo experiment): CPU-only, ~10 min. All files already exist.
- **Block 4** (paragraph merge): new Python script needed (~2h dev), then 1-2h GPU for embedding
- **Block 5** (bbl): pure CPU preprocessing, ~30 min
- **Block 6** (QA): blocked on qrels, 0 cost now
- **Biggest bottleneck**: dev time for `build_paragraph_chunks.py`

---

## Risks and Mitigations

- **Risk**: typed_crossdoc under static_prior still dilutes R@1 (the cross-doc signal has noise).
  **Mitigation**: Lower weight to 0.1 or use `--merge-combine max` to limit double-counting. If still bad, accept that R@1 and R@10 require different methods and present them as complementary configs.

- **Risk**: Paragraph merge doesn't beat v1_enriched dense baseline.
  **Mitigation**: Treat it as a negative result — report it as evidence that enrichment quality matters more than chunk boundary choice, and deprioritize entity/keywords nodes.

- **Risk**: bbl files are missing or incomplete for many of the 53 eval docs.
  **Mitigation**: Use partial coverage (≥ 30 docs) — even partial expansion should 2-3x the current cite edge count.

- **Risk**: C_NEW fails (no config achieves R@1 ≥ 0.25 AND R@10 ≥ 0.62 simultaneously).
  **Mitigation**: Present current best dual configs as complementary — precision mode (R@1=0.2505) and recall mode (R@10=0.6406) — and reframe as "task-specific graph configuration" rather than a single global optimum.

---

## Final Checklist

- [x] C1 supported (done)
- [x] C5 supported (done)
- [ ] C_NEW: combo config — **Block 3, run today**
- [ ] C_PARA: paragraph merge — **Block 4, needs new script**
- [ ] C4 third axis (QA) — blocked on C-Pool qrels
- [x] Summary line explicitly closed (C6)
- [x] Intra-doc virtual edges explicitly rejected from corpus (C2)
- [x] Nice-to-have separated from must-run
