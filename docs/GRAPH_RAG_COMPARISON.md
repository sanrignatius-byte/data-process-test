# Graph RAG vs Document Graph: Comparison and Design Rationale

*Created: 2026-03-15 | Based on mentor discussion 2026-03-04*

---

## 1. Background

Mentor suggested researching **Graph RAG** approaches (specifically Microsoft GraphRAG and query-sentence graphs) to understand whether they offer components worth borrowing and to articulate why our Document Graph approach is better suited for academic paper understanding.

---

## 2. What is Graph RAG?

### 2.1 Microsoft GraphRAG (entity-community approach)

**Pipeline:**
1. Text chunking → LLM-based entity extraction (named entities, relations)
2. Entity deduplication + community detection (Leiden algorithm)
3. Community summary generation (LLM summarizes each community)
4. Hierarchical retrieval: query → community match → entity subgraph → answer

**Key properties:**
- Operates at entity level (people, organizations, concepts)
- Communities capture co-occurrence clusters, not document structure
- Designed for long-form documents / knowledge bases (e.g., novels, reports)
- Cost: LLM call per chunk for entity extraction → $$$$ at scale

**Published baseline (GraphRAG paper, Edge et al. 2024):**
- 1M-token corpus: ~$4-8 per graph build
- Estimated for our 82-paper corpus: ~$15-30 build cost (entity extraction)

### 2.2 Query-sentence graph

**Concept:**
- For each passage, generate hypothetical queries it could answer (HyDE-style)
- Build a graph where passages sharing similar hypothetical queries are connected
- Retrieval: map user query to hypothetical query space → find connected passages

**Properties:**
- Captures semantic proximity not captured by lexical overlap
- Expensive: one LLM call per passage to generate hypothetical queries
- For our corpus (~50k paragraphs): significant cost
- Passages-only graph — does not explicitly represent figures/tables/formulas

---

## 3. Our Document Graph Approach

**Pipeline:**
1. MinerU parsing → figures, tables, formulas, paragraphs extracted automatically
2. LaTeX source → `\ref{}` graph → exact cross-modal links at zero marginal cost
3. Natural reading order (backbone edges) → structural connectivity for free
4. Cross-document citation edges from `.bbl` files
5. Hub scoring → identify structurally important nodes (bridges, core sections)
6. Query generation from hub-anchored paths → multi-hop, multi-modal queries

**Node types:** paragraph, section, subsection, figure, table, formula
**Edge types:** backbone, paragraph_ref, element_ref, section_contains_*, cross_doc_cite

---

## 4. Side-by-Side Comparison

| Dimension | GraphRAG (entity) | Query-sentence graph | Our Document Graph |
|---|---|---|---|
| **Primary node type** | Named entities | Text passages | Structural elements (fig/table/formula/para/sec) |
| **Edge derivation** | LLM co-occurrence | Hypothetical query similarity | LaTeX `\ref{}` + backbone + citation |
| **Build cost (82 papers)** | ~$15-30 (entity extraction) | ~$5-15 (hypothetical queries) | **<$1** (MinerU + LaTeX parsing, no LLM) |
| **Scalability to 10k docs** | Prohibitive without sampling | Expensive | Low-cost (MinerU runs offline) |
| **Multimodal nodes** | ✗ (text-only entities) | ✗ (text passages only) | **✓** (figures, tables, formulas as 1st-class nodes) |
| **Structural fidelity** | Low (entity ≠ document structure) | Medium (paragraph-level) | **High** (respects document hierarchy: section → para → element) |
| **Interpretability** | Medium (entity names readable) | Low (hypothetical queries) | **High** (LaTeX labels, section titles, captions) |
| **Cross-document links** | Via entity co-occurrence | Via shared queries | Via `.bbl` citation graph (explicit paper references) |
| **Hub identification** | Community centrality | Most-queried passages | Bridge score × PageRank × core_module (section keyword) |
| **Query generation** | Retrieval only, no generation | Retrieval only | **Multi-hop multi-modal query synthesis** |
| **Evidence localization** | Entity-level | Passage-level | **Element-level** (figure/table ID returned) |

---

## 5. What We Can Borrow

### 5.1 Community detection for document-level summarization ✓

GraphRAG's community hierarchy concept is useful for **document summarization tasks**:
- Within-document: section nodes can form communities based on element co-reference
- Cross-document: connected-component analysis of citation graph already in place
- **Lightweight adaptation**: use section containment edges to define "communities" without LLM extraction

### 5.2 Hypothetical query generation for C-Pool expansion ✓

The query-sentence graph idea maps well to our C-Pool strategy:
- For high-importance hub nodes (architecture figures, main result tables), generate 3-5 hypothetical queries they could answer
- Use these as additional C-Pool entries with known evidence nodes
- Cost: targeted — only for top-60 bridge hubs, not all 50k paragraphs

**Estimated cost:** ~60 hub elements × 5 queries × $0.002/query = **$0.60 total**

### 5.3 Hierarchical retrieval for long-document queries ✓

GraphRAG's hierarchical summarization (community → local context) maps to:
- Section-level retrieval → paragraph-level → element-level
- Our section containment edges already encode this hierarchy
- Can be implemented as a 2-stage retrieval: section BM25 → element rerank

---

## 6. Why Our Approach is Better Suited for Academic Papers

### 6.1 Multimodality is first-class

Academic papers communicate through figures, tables, and formulas — not just text. GraphRAG treats these as opaque referenced objects (if at all). Our graph treats them as typed nodes with captions and content, enabling:
- Figure + formula multi-hop queries
- Evidence localization to specific figure IDs
- MoDora-style enrichment of element semantics

### 6.2 Structural signals are free and accurate

LaTeX `\ref{}` provides exact cross-modal links — no LLM needed, zero hallucination risk. GraphRAG's entity extraction can hallucinate relations; our edges are deterministic (either a `\ref{}` exists or it doesn't).

### 6.3 Section hierarchy enables importance scoring without entity extraction

Core module scoring using `introduction / main results / method` keywords on section titles is:
- Free (regex matching)
- Generalizable (same keywords apply across ML papers)
- Already implemented

GraphRAG achieves similar importance weighting via entity frequency, but requires the expensive extraction step.

### 6.4 Patent-relevant differentiation

Our system's novel contributions not present in GraphRAG:
1. **Multi-modal bridge hub scoring** — combining PageRank, modality diversity, and section importance
2. **MoDora-style element enrichment** as a pre-retrieval step
3. **M4 query synthesis** from graph paths (multi-hop, multi-modal, multi-document, multi-turn)
4. **Dual-track QC** (objective vs subjective) with persona-aware generation
5. **C-Pool universal query library** with evidence localization QC

---

## 7. Roadmap for Incorporating Graph RAG Ideas

| Action | Priority | Effort | Expected Value |
|---|---|---|---|
| Community detection on section containment graph | P1 | Low (NetworkX) | Document summary capability |
| Hypothetical query generation for top-60 hubs | P1 | Low ($0.60) | C-Pool expansion + richer evidence graph |
| 2-stage retrieval: section → element rerank | P2 | Medium | Better long-document recall |
| Entity extraction for key domain terms only | P3 | High (LLM cost) | Minimal marginal gain given existing structure |

---

## 8. Conclusion

GraphRAG and query-sentence graphs are designed for **unstructured text corpora** where document structure is unavailable. For academic papers with LaTeX sources, our Document Graph approach extracts richer, more accurate structural information at a fraction of the cost.

The key borrowable concept from Graph RAG is **community-based summarization**, which we can implement cheaply via section containment edges without entity extraction. The query-sentence graph concept is valuable for C-Pool expansion on targeted hub nodes.

These additions further differentiate our patent claim: a **structure-aware, cost-efficient, multimodal document graph** that subsumes the useful properties of Graph RAG approaches without their computational overhead.

---

*See also: `docs/GRAPH_ARCHITECTURE.md` for full node/edge specification*
*See also: `docs/PATENT_TECHNICAL_SUMMARY.md` for patent claim structure*
