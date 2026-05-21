# M4 Long-Chain Construction from Pure MinerU: Strategic Analysis

**Date**: 2026-05-21
**Question**: 面对纯 MinerU 产物，如何构造稳定的、能生成 M4 的长链？当前对齐思路有没有搞头？

---

## 1. Problem Anchor

- **Bottom-line problem**: Given pure MinerU (PDF-parsed) outputs with no LaTeX source, construct multi-hop chains of 2-3 cross-document hops where each hop is reliable enough that the full chain doesn't collapse from error compounding.
- **Must-solve bottleneck**: Cross-doc hops are the reliability bottleneck. A 2-hop chain with per-hop precision 0.9 has chain-level precision 0.81; at per-hop 0.05 (current cross-doc visual edges), chain-level is 0.0025.
- **Success condition**: Produce chains where ≥50% of cross-doc hops are verifiably correct (not just recall-level), enabling chain-level precision to stay above noise for 2-3 hop chains.

---

## 2. What Edges We Actually Have from Pure MinerU

### Tier 1: Strong edges (alignment WORKS)

| Edge type | Method | Evidence | Reliability |
|-----------|--------|----------|-------------|
| **Intra-doc element** | MinerU regex → LaTeX \ref recovery | C15: 84% ref recall, 90.8% extraction | HIGH — directly replaces LaTeX reference graph |
| **Cross-doc citation** | LaTeX \cite → MinerU XGBoost | C18: AUC 0.852, F1 0.746, 75% edges prob≥0.95 | HIGH after noise filter — title_match is 88% of signal |

### Tier 2: Weak edges (alignment FAILS)

| Edge type | Method | Evidence | Reliability |
|-----------|--------|----------|-------------|
| **Cross-doc visual** | CLIP + TF-IDF rerank | C16: 87% caption_sim=0, 5.1% text support | LOW — recall only, not promotable |
| **Cross-doc VLM-judged** | GPT-5.4 on CLIP top-K | idea:008 full-160: 6.3% strong-edge rate | LOW yield — all 10 strong edges cluster in fairness/bias, degraded-caption 0/37 |

### Tier 3: Untested but plausible

| Edge type | Potential method | Why plausible |
|-----------|------------------|---------------|
| **Co-citation** | Reverse C18 graph: papers sharing cited targets | Dense: 27K direct pairs → potentially 200K+ co-citation pairs |
| **Bibliographic coupling** | Papers cited together by same source | Same mechanism, different graph direction |
| **Shared-venue/topic** | ArXiv category, title embedding similarity | Cheap, high-recall, useful as pre-filter |

---

## 3. The Critical Insight: We've Been Framing the Cross-Doc Hop Wrong

Look at the actual L3 pass chain structure:

```
element_A (figure, paper A) → paragraph_bridge (in paper A) → element_B (table, paper B)
```

The cross-doc "hop" is NOT a direct element→element visual similarity edge. It is:

1. **Paragraph bridge**: A text paragraph in paper A that *discusses* paper B's work
2. **Citation relationship**: The paragraph mentions paper B — this is detectable via C18
3. **Target element**: element_B in paper B is the specific figure/table/formula being discussed

Decomposing the cross-doc hop into two operations:
- **Op 1 (cross-doc)**: Identify that paragraph_in_A discusses paper B → **C18 citation edges (AUC 0.852)**
- **Op 2 (intra-doc)**: Find the relevant element in paper B → **C15 intra-doc edges (84% recall)**

**Both operations use edge types where the alignment approach WORKS.**

The multimodal richness of M4 comes from the **endpoints** being figures/tables/formulas, NOT from the cross-doc edge being visual. The chain "compare Figure 3 in paper A with Table 2 in paper B" doesn't need a visual-similarity edge between Figure 3 and Table 2 — it needs to know that paper A discusses paper B's Table 2 in a specific paragraph.

---

## 4. Why the Alignment Approach IS Viable (But Needs Re-Scoping)

### What alignment does well

LaTeX source provides ground truth for two structurally distinct edge types:

1. **Reference edges** (\ref, \cite): intra-doc element references + cross-doc paper citations
2. These are **text-structural** relationships — they exist in the document's explicit reference graph

Both transfer to MinerU features because the signal is in the document's text structure (regex patterns, title mentions, section labels), not in visual semantics.

### What alignment CANNOT do

Cross-doc visual-semantic edges (Figure 3 in paper A "looks similar to" Table 2 in paper B) have NO LaTeX ground truth. LaTeX has no `\similar-figure{paperB-table2}` markup. CLIP-based edges are inherently zero-shot and will stay at recall quality regardless of how much LaTeX training data we have.

### The correct scope for alignment

```
Align:    LaTeX \cite{paperB}  →  MinerU "paper B's title appears in this paragraph"  (C18 ✓)
Align:    LaTeX \ref{fig:3}    →  MinerU "Figure 3" regex near this paragraph         (C15 ✓)
Don't:    LaTeX ???            →  MinerU "this figure looks like that table"           (no GT exists)
```

---

## 5. Proposed Chain Construction Pipeline

### Phase A: Build the reliable cross-doc skeleton (citation graph)

1. Take C18 predicted edges (53,435 edges, 27,349 doc-pairs)
2. Apply G11 noise filters:
   - Drop Acknowledgement/Funding/author-list sections
   - Require cite_pattern>0 OR title_match>0.2 for edges with text_sim<0.75
   - Per-section probability calibration
3. Result: ~35,000-40,000 filtered high-confidence cross-doc citation edges
4. These edges form the **cross-doc skeleton**: for each (paper_A, paper_B) pair, we know which paragraphs in A discuss B

### Phase B: Attach elements to paragraphs (intra-doc navigation)

1. For each paragraph that cites paper B, use C15 intra-doc edges to find nearby elements:
   - Elements whose `position_idx` is close to the paragraph
   - Elements referenced by regex patterns in the paragraph text
2. This gives us: `element_in_A → paragraph_citing_B → ???`

### Phase C: Resolve target elements (cross-doc element alignment)

This is the missing piece. Given "paper A's paragraph discusses paper B's Table 2", how do we find Table 2 in paper B?

Three approaches, in order of reliability:

1. **Explicit mention**: The paragraph says "Table 2 of [paper B] shows..." → regex extract element reference. Low coverage but perfect precision.
2. **Element type + topic matching**: The paragraph discusses "training curves" and paper B has a figure captioned "Training loss vs epochs" → embedding match between paragraph text and paper B's element captions. This is intra-doc retrieval within paper B.
3. **L3 pass template**: Use existing 146 L3 passes as templates — for each pass that is cross-doc (87/146), the paragraph bridge already connects to a specific element in paper B. Mine these as training data for a cross-doc element resolver.

### Phase D: Compose chains

Chain = element_A → paragraph_bridge (cites B) → element_B

For 3-hop chains:
```
element_A (paper A) → paragraph_citing_B → element_B (paper B) → paragraph_citing_C → element_C (paper C)
```

The middle element_B serves as both the target of hop 1 and the anchor of hop 2.

### Phase E: Validate chain quality

For each constructed chain:
1. **Hop validity**: Does paragraph_bridge actually cite the target paper? (C18 score)
2. **Element relevance**: Is element_B actually the element discussed in the paragraph? (cross-doc element resolver score)
3. **Chain coherence**: Can the full chain be verbalized as a coherent multi-hop question? (LLM judge)

---

## 6. Concrete Next Steps (Priority-Ordered)

### P0: Noise-filter the C18 citation graph (G11)

- Implement section-based filter + cite_pattern threshold
- Output: `data/04_xdoc_citation/predicted_xdoc_edges_filtered.jsonl`
- Cost: ~1 hour engineering, 0 GPU

### P0: Build the cross-doc element resolver

- For each (paragraph_citing_B, paper_B) pair, retrieve the most likely target element in B
- Start with explicit mention extraction (regex "Figure/Table N of [B]")
- Fall back to embedding match between paragraph text and B's element captions
- Validate on the 87 existing cross-doc L3 passes
- Cost: ~2 hours engineering, 0 GPU

### P1: Compose first M4 chains

- Wire citation skeleton + intra-doc edges + element resolver into `cross_doc_pairs.py`
- Generate first batch of cross-doc chains
- Validate against existing L3 pass cross-doc subset
- Cost: ~3 hours engineering

### P1: Chain-to-session projection v1

- Implement Phase 0 verbalize: locked-schema LLM pass on `reasoning_chain` free-text
- Produce 2-turn sessions from existing L3 passes
- Apply turn-dependency + coref_resolution QC
- Cost: ~2 hours engineering + ~$5 LLM tokens (146 passes × 2-turn verbalization)

### P2: Expand beyond direct citation

- Co-citation: papers A and B both cite paper C → potential cross-doc relationship
- Bibliographic coupling: paper C cites both A and B → they're in the same reference context
- These expand the cross-doc skeleton from 27K to potentially 100K+ doc-pairs

---

## 7. Verdict on the Alignment Approach

**The alignment approach (LaTeX GT → MinerU features) is viable and is the right strategy — but only for text-structural edges (citations, references).** It was never going to work for visual-semantic cross-doc edges because LaTeX has no ground truth for those.

The mistake wasn't the alignment strategy — it was assuming we NEED visual cross-doc edges for M4 chains. We don't. The paragraph bridge + citation relationship provides the cross-doc connectivity, and the endpoints provide the multimodal richness. This is exactly how the existing 146 L3 passes work: 59.6% are cross-doc, all use paragraph bridges, and all endpoint pairs are multimodal (figure/table/formula).

**What to stop doing:**
- Trying to make CLIP cross-doc visual edges "strong" via VLM/recaption/LLM-rerank (idea:008). The ~6% strong-edge ceiling is real and the yield is too low to build chains on.
- Treating cross-doc visual edges as the primary M4 cross-doc mechanism.

**What to start doing:**
- Treating citation edges (C18) as the cross-doc backbone for M4 chains.
- Building the cross-doc element resolver (Phase C above) — the one genuinely missing piece.
- Composing citation + intra-doc + element resolution into end-to-end chain construction.

---

## 8. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Citation graph too sparse for M4 coverage | Medium | High | Co-citation expansion (Phase D), shared-topic edges |
| Cross-doc element resolver low precision | Medium | High | Start with explicit mention only (high precision, low recall), expand gradually |
| Section info missing from C18 edges | High (currently "unknown") | Medium | Re-run inference with section labels; already have section info in MinerU structure.json |
| Chain verbalization produces robotic dialog | Medium | Low | Stylistic post-pass with persona variation; accept for v1 |
| Some L3 pass paragraph bridges don't actually cite the target paper | Low | Medium | Validate C18 score on each existing cross-doc L3 pass before using as template |

---

## 9. Summary

**Alignment has 搞头 — 但方向要收窄。** 对齐在 citation/reference 这两种文本结构边上成立（C15/C18 都证明了），在 visual-semantic 边上不成立也不需要成立。M4 长链的跨文档跳不靠"这张图和那张图像"，而是靠"这段文字引了那篇论文"。现有的 146 条 L3 pass 已经是这个结构——跨文档跳全部走 paragraph bridge + citation 关系。把 citation graph（C18）和 intra-doc edges（C15）拼起来，缺的只是 cross-doc element resolver 这一块，补上就能出链。
