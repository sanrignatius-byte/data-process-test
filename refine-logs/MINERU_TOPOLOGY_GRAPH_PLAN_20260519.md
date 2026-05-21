# MinerU-Only Topology Graph — Implementation Plan

**Date**: 2026-05-19 (updated with VL embedding strategy)
**Status**: v1 graph done; topology + hub + candidate layers pending
**Context**: DPT project — migrating from LaTeX-centric graph to pure MinerU pipeline for old_53 experimental group
**Assets**: 706 figures + 229 tables with images = 935 image assets on disk, all accessible

---

## 0. Background

### Why this migration

The LaTeX graph pipeline (`build_latex_reference_graph.py` → `analyze_latex_graph_topology.py` → `generate_multihop_l1_queries.py`) depends on two data sources merged together:

1. **LaTeX `.tex`/`.bbl`** — provides `\label{}`/`\ref{}` structural references, section hierarchy, paragraph boundaries
2. **MinerU `content_list.json`/`structure.json`** — provides PDF-grounded visual elements (figures, tables, formulas) with `page_idx`, `bbox`, `image_path`

The merger (`map_label_to_element`) maps LaTeX labels to MinerU element IDs through number matching + caption Jaccard similarity. This merger is **lossy** and the LaTeX source is **not guaranteed** for all documents.

### What changed with v1

`experiments/build_mineru_only_graph_v1.py` proved that **content_list.json alone** can produce a rich graph without any LaTeX dependency. The key insight: content_list.json items are already in reading order with proper `page_idx`, `bbox`, and type annotations — no label mapping needed.

| Metric | LaTeX-dependent v0 | Pure MinerU v1 |
|---|---|---|
| Text elements (per doc) | ~1 (monolithic blob) | ~113 (paragraph-level) |
| Section nodes | 0 | ~20 |
| Edge types | 4 | 6 |
| Figure caption coverage | ~0% | 65% |
| Table caption coverage | ~0% | 77% |

### What's still missing

v1 produces elements and edges as dicts/lists. The LaTeX pipeline's downstream consumers (`analyze_latex_graph_topology.py`, query generation, candidate scoring) expect a **unified topology graph** with typed `Node` and `Edge` objects, plus **hub scoring**, **multi-hop path discovery**, and **cross-doc edges**. These layers don't exist yet for the pure MinerU path.

---

## 1. Target Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   v1 (already done)                      │
│  build_mineru_only_graph_v1.py                          │
│  Input: content_list.json + structure.json              │
│  Output: elements dict + edges list                     │
│  • 8,877 elements (section/text/formula/figure/table)  │
│  • 58,887 edges (6 types)                               │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              Phase 1: Topology Graph Builder             │
│  experiments/build_mineru_topology_graph.py             │
│                                                         │
│  Converts v1 dicts → unified Node + Edge data model     │
│  • Node: node_id, doc_id, node_type, text_snippet,     │
│          page_idx, position_idx, element_id, metadata   │
│  • Edge: source_id, target_id, doc_id, edge_type,       │
│          weight, metadata                                │
│  • Builds adjacencies (out_adj, in_adj)                 │
│  • No LaTeX dependency whatsoever                        │
│                                                         │
│  Output: data/05_eval/mineru_topology_graph_v1.json     │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│           Phase 2: Hub Scoring + Multi-Hop Discovery     │
│  experiments/build_mineru_hub_candidates.py              │
│                                                         │
│  • PageRank on topology graph → hub scores              │
│  • Modality diversity scoring                           │
│  • Keyword boost (introduction/overview/architecture)   │
│  • DFS multi-hop path discovery (max 3 hops)            │
│  • Path scoring: modality diversity, bridge richness,   │
│    edge type mix, position compactness                  │
│  • Top-N candidate paths per modality pair              │
│                                                         │
│  Output: data/05_eval/mineru_hub_scores_v1.json         │
│  Output: data/05_eval/mineru_multihop_candidates_v1.json│
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│          Phase 3: Cross-Doc + Bridge Text                │
│  experiments/build_mineru_crossdoc_bridges.py            │
│                                                         │
│  • Sentence-level bridge text extraction                │
│    (not just whole referring_paragraphs)                │
│  • Cross-doc edges via:                                  │
│    a) Section-title embedding similarity (fast)          │
│    b) Citation graph if available                        │
│  • Bridge quality scoring (verb density, specificity)   │
│                                                         │
│  Output: data/05_eval/mineru_crossdoc_edges_v1.json     │
│  Output: data/05_eval/mineru_bridge_texts_v1.json       │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│          Phase 4: API Query Generation Smoke Test        │
│  experiments/crossdoc_company_api_smoke.py ← existing   │
│                                                         │
│  • Feed mineru_multihop_candidates → prompt builder     │
│  • 9-call smoke test using company API                   │
│  • Gate: parsed_ok ≥ 8/9, each query has 3 evidence     │
│    spans + explicit chain roles                         │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Phase 1 — Topology Graph Builder

**File**: `experiments/build_mineru_topology_graph.py`

### 2.1 Node schema

Lifted from `src/models/__init__.py` — `Node` dataclass:

```python
@dataclass
class Node:
    node_id: str          # "1104.3913::text::00042"
    doc_id: str           # "1104.3913"
    node_type: str        # "text" | "section" | "figure" | "table" | "formula"
    label: str = ""       # Human-readable: "Figure 2" | "Introduction" | first 80 chars of text
    text_snippet: Optional[str] = None   # content[:500] for text nodes
    page_idx: Optional[int] = None       # from content_list
    position_idx: Optional[int] = None   # reading-order position
    element_id: Optional[str] = None     # reference back to v1 element_id
    mapped_element_id: Optional[str] = None  # for compat with downstream consumers
    metadata: Dict[str, Any] = {}        # extra fields (bbox, image_path, caption, etc.)
```

### 2.2 Edge schema

```python
@dataclass
class Edge:
    source_id: str
    target_id: str
    doc_id: str           # for intra-doc edges; "src->tgt" for cross-doc
    edge_type: str        # see table below
    weight: float = 1.0
    metadata: Dict[str, Any] = {}

    def key(self) -> Tuple[str, str, str]:
        return (self.source_id, self.target_id, self.edge_type)
```

### 2.3 Edge type mapping — v1 → topology

| v1 edge_type | Topology edge_type | Direction | Weight |
|---|---|---|---|
| `next_element` | `backbone` | text[i] → text[i+1] | 0.4 |
| `prev_element` | `backbone` (reverse) | text[i+1] → text[i] | 0.4 |
| `regex_reference` | `element_ref` | text → figure/table/formula | 0.8 |
| `co_reference` | `element_ref` (bidirectional) | element ↔ element | 0.6 |
| `section_contains` | `section_contains_element` or `section_contains_paragraph` | section → child | 0.5 |
| `same_page_cross_type` | `same_page_cross_type` | element ↔ element | 0.1–0.6 |
| *(new)* embedding similarity | `embedding_similarity` | element ↔ element | cosine |
| *(new)* cross-doc cite | `cross_doc_cite` | paragraph (doc A) → element (doc B) | 0.7 |

### 2.4 Node type mapping — v1 → topology

| v1 element_type | Topology node_type | Notes |
|---|---|---|
| `text` | `paragraph` | Core bridge nodes |
| `section` | `section` | From text_level ≤ 2 |
| `figure` | `figure` | With image_path in metadata |
| `table` | `table` | With table_body in metadata |
| `formula` | `formula` | With LaTeX in metadata |

### 2.5 Implementation steps

1. **Load v1 output**: Read `mineru_elements_v1.json` and `mineru_edges_v1.jsonl`
2. **Create Node objects**: One per v1 element, with proper `node_id = f"{doc_id}::text::{counter:05d}"` format
3. **Create Edge objects**: Convert v1 edges to topology edges, mapping types per table above
4. **Build adjacencies**: `out_adj: Dict[str, Set[str]]`, `in_adj: Dict[str, Set[str]]`
5. **Backbone edges**: Ensure consecutive text nodes within same doc+page are connected (already done via next_element in v1)
6. **Deduplication**: Use `Edge.key()` to avoid duplicate edges
7. **Output**: Single JSON with `nodes`, `edges`, `adjacency`, and `metadata`

### 2.6 Acceptance criteria

- All 53 docs processed, 0 skipped
- Node count matches v1 element count (~8,877)
- Edge count matches v1 (~58,887)
- Every node has valid `doc_id`, `node_type`, `page_idx`
- Graph is traversable: all text nodes reachable via backbone edges
- py_compile passes, VS Code diagnostics clean

---

## 3. Phase 2 — Hub Scoring + Multi-Hop Discovery

**File**: `experiments/build_mineru_hub_candidates.py`

### 3.1 Algorithm

Directly adapted from `analyze_latex_graph_topology.py`:

**PageRank**:
- Standard PageRank (damping=0.85, max_iter=40) on the full topology graph
- Treat undirected (sum both in/out degrees) since edges are mostly bidirectional in our case

**Hub scoring** (for ranking nodes):
```
hub_score = PageRank_zscore
          + modality_diversity_bonus  (0.3 × unique_modalities_in_neighborhood / 4)
          + backbone_depth_bonus      (0.2 × normalized_position_in_doc)
          + keyword_boost             (0.05–0.15 for introduction/overview/architecture)
          + connectivity_bonus        (0.1 × log(1 + degree))
```

**Multi-hop path discovery**:
1. Select top-K hub nodes (K=50) as seeds
2. From each seed, DFS up to max_hops=3, tracking visited nodes
3. Filter: path endpoints must be different modality types (cross-modal)
4. Filter: path must contain at least one `element_ref` edge (not just backbone)
5. Score each path:
```
path_score = mean_hub_score_of_endpoints
           + modality_diversity (count of unique types in path / 4)
           + bridge_richness   (total text length of paragraph nodes in path / 1000)
           - long_gap_penalty  (page distance / 10)
           + edge_type_bonus   (0.1 for regex_reference edges in path)
```

### 3.2 Output format

`mineru_multihop_candidates_v1.json`:
```json
{
  "metadata": {
    "builder": "mineru_hub_candidates_v1",
    "topology_source": "mineru_topology_graph_v1.json",
    "num_seeds": 50,
    "max_hops": 3,
    "created_at": "..."
  },
  "hubs": [
    {
      "node_id": "1104.3913::text::00042",
      "hub_score": 0.85,
      "pagerank": 0.012,
      "modality_diversity": 3,
      "node_type": "paragraph",
      "text_preview": "..."
    }
  ],
  "candidates": [
    {
      "candidate_id": "cand_001",
      "path": ["1104.3913::text::00005", "1104.3913::figure::2", "1104.3913::text::00042", "1104.3913::table::1"],
      "endpoint_types": ["paragraph", "table"],
      "hop_count": 3,
      "edge_types": ["element_ref", "backbone", "element_ref"],
      "score": 0.72,
      "bridge_texts": ["As shown in Figure 2, the framework...", "Table 1 summarizes..."],
      "modality_diversity": 3,
      "page_span": 2
    }
  ]
}
```

### 3.3 Acceptance criteria

- ≥ 20 hub nodes with hub_score > 0.5
- ≥ 100 multi-hop candidate paths
- ≥ 80% of paths are cross-modal (different endpoint types)
- Path lengths: 2-hop ≥ 50%, 3-hop ≥ 30%
- All paths contain at least one non-backbone edge

---

## 4. File Manifest

All new files go under `experiments/` (experimental lane, no production `src/` changes):

```
experiments/
├── build_mineru_only_graph_v1.py      ← DONE
├── build_mineru_topology_graph.py     ← Phase 1
├── build_mineru_hub_candidates.py     ← Phase 2
├── build_mineru_vl_edges.py           ← Phase VL (new — CLIP embeddings)
└── build_mineru_crossdoc_bridges.py   ← Phase 3 (revised — consumes VL edges)
```

All outputs go under `data/05_eval/`:

```
data/05_eval/
├── mineru_only_graph_v1_latest/       ← symlink to latest v1
├── mineru_topology_graph_v1_latest/   ← symlink (Phase 1)
├── mineru_hub_candidates_v1_latest/   ← symlink (Phase 2)
├── mineru_vl_edges_v1_latest/         ← symlink (Phase VL)
│   └── mineru_vl_embeddings_v1/       (numpy files: figure + text embeddings)
└── mineru_crossdoc_bridges_v1_latest/ ← symlink (Phase 3)
```

Each output directory contains:
- `mineru_*_v1.json` — main artifact
- `summary.json` — statistics
- `report.md` — human-readable summary

---

## 5. Key Design Decisions

1. **No LaTeX dependency at any layer.** content_list.json provides reading order, page_idx, type annotations, and text — everything LaTeX provided plus PDF grounding.

2. **Node ID format**: `{doc_id}::{node_type}::{counter:05d}` — consistent with existing `analyze_latex_graph_topology.py` convention so downstream consumers work unchanged.

3. **Backbone edges are the spine.** Text paragraph → text paragraph sequential edges form the reading-order backbone. Multi-hop paths traverse backbone to bridge between modalities.

4. **element_ref edges are the signals.** Regex references (Figure N, Table N, Eq. N) from text to elements form the cross-modal reference edges — the MinerU equivalent of LaTeX `\ref{}`.

5. **Same data model as LaTeX path.** Using the same `Node`/`Edge` dataclasses from `src/models/__init__.py` ensures the hub scoring, PageRank, and DFS algorithms from `analyze_latex_graph_topology.py` can be reused with minimal changes.

6. **Experimental lane only.** No modifications to `src/` or production scripts. All new code in `experiments/`, all outputs in `data/05_eval/`.

---

## 6. Downstream Integration Path

Once Phases 1–3 are complete, the topology graph can feed directly into the existing query generation pipeline:

```
mineru_topology_graph_v1.json
        │
        ▼
mineru_hub_candidates_v1.json
        │
        ▼
generate_multihop_l1_queries.py  ← reuse existing script with
    --candidates mineru_hub_candidates_v1.json
    --topology mineru_topology_graph_v1.json
    --bridge-texts mineru_bridge_texts_v1.json
        │
        ▼
M4query_v3 (pure MinerU queries)
```

The only change needed in `generate_multihop_l1_queries.py` is the bridge text resolution path: instead of `_find_para_for_line()` (which depends on LaTeX line numbers), use `mineru_bridge_texts_v1.json` which maps paragraph_id → sentence directly.

---

## 7. Phase VL — CLIP Visual Embeddings (Cross-Cutting)

### 8.1 Why VL embeddings

Current v1 edges are purely structural (reading order, positional window, regex patterns). They miss **semantic** relationships that VL models can capture:

| Current edge | Problem | VL replacement |
|---|---|---|
| `same_page_cross_type` (29K edges) | Purely positional — two elements on same page within ±5 window get an edge regardless of semantic relevance | `visual_similarity` — figures with similar visual content; `text_describes_figure` — paragraph that actually explains the figure |
| No cross-doc figure edges | Can't connect figures across documents | `cross_doc_figure_sim` — top-k similar figures by visual embedding |
| `context_before/after` is paragraph-level | Context window is position-based, not semantics-based | `paragraph_describes_figure` — CLIP score between figure image and paragraph text |
| `regex_reference` is string-match only | "Figure 2" matches regardless of whether the paragraph actually discusses it | CLIP alignment score as a re-ranker for regex edges |

### 8.2 Available VL approaches

Given the environment (CPU-only, no GPU, no torch/transformers), three tiers:

#### Tier 1 — Pure text embedding (zero new deps, CPU, fast)

Use figure captions + context text as proxy for visual content. Embed with `numpy`-based TF-IDF or a lightweight sentence embedding model.

- **Pro**: No new dependencies, instant
- **Con**: Doesn't use actual image pixels; figures with similar captions get high similarity even if visually different
- **Setup**: `pip install sentence-transformers` → `all-MiniLM-L6-v2` (80MB, CPU-friendly, ~100 sentences/sec)
- **Use case**: Baseline for text-based figure similarity; cross-doc section similarity

#### Tier 2 — CLIP ViT-B/32 on CPU (one new dep, CPU, ~8 min for 935 images)

Install `open_clip_torch` (or `clip`) with ViT-B/32. The model is ~350MB. On CPU: ~0.5s per image → 935 images ≈ 8 minutes.

- **Pro**: Actual visual embeddings; well-established for figure-text matching
- **Con**: Requires `torch` install (~800MB); CPU inference is slow but acceptable for 935 images
- **Setup**: `pip install torch open_clip_torch` → ViT-B/32
- **Use case**: Figure visual similarity, figure-paragraph alignment, cross-doc figure matching

#### Tier 3 — API-based VL embedding (no local deps, uses company API)

Send figure images (base64-encoded) to the company API for embedding. The same API already used for query generation may support multimodal inputs.

- **Pro**: No local model install; best embedding quality; no CPU/GPU concerns
- **Con**: API cost; rate limits; 935 API calls; requires API to support image inputs
- **Setup**: Same company API already used in `experiments/crossdoc_company_api_smoke.py`
- **Use case**: Best quality figure embeddings if API supports multimodal

### 8.3 Recommended approach: Tier 2 (CLIP on CPU)

CLIP ViT-B/32 is the pragmatic sweet spot:
1. One-time `pip install torch open_clip_torch` (~800MB)
2. 8 minutes CPU inference for all 935 images
3. Produces 512-dim embeddings for both images AND text in the same latent space
4. This means we can compute `cosine(figure_image, paragraph_text)` directly — no separate text model needed
5. Established in academic literature for figure retrieval (Semantic Scholar, Arxiv papers use CLIP for figure indexing)

### 8.4 New edge types from VL embeddings

| Edge type | Source → Target | How it's computed | Weight |
|---|---|---|---|
| `visual_similarity` | figure → figure (same doc or cross-doc) | cosine(CLIP(image_a), CLIP(image_b)) > 0.7 | cosine score |
| `cross_doc_visual_sim` | figure (doc A) → figure (doc B) | same as above, filtered to cross-doc only | cosine score |
| `text_describes_figure` | paragraph → figure | cosine(CLIP(text), CLIP(figure_image)) > 0.25 | cosine score × richness_bonus |
| `figure_described_by_text` | figure → paragraph | reverse direction of above | same weight |
| `table_similarity` | table → table | CLIP on table image (if available) or table_body text embedding | cosine score |

### 8.5 How VL edges replace/augment existing edges

**`same_page_cross_type` → `text_describes_figure`**:
- Current: 29K edges, any text within ±5 positions on same page connects to any figure/table
- VL replacement: only paragraphs with CLIP score > 0.25 get connected → much sparser, more meaningful
- Expected reduction: 29K → ~2-5K high-quality alignment edges

**Cross-doc figure similarity (currently nonexistent)**:
- VL: for each of the 706 figures, find top-3 most similar figures in other docs
- Expected: ~2,100 cross-doc figure edges

**Formula similarity (formula → formula)**:
- Since formulas have LaTeX source in `structure.json`, text embedding of LaTeX is better than visual
- Use sentence-transformer on LaTeX content → `formula_similarity` edges

### 8.6 Implementation plan for VL

Add a new script: `experiments/build_mineru_vl_edges.py`

```
Input:
  - mineru_elements_v1.json (figure image_paths, text content, formula LaTeX)
  - 935 figure/table images on disk

Pipeline:
  1. Load CLIP ViT-B/32
  2. Embed all figure images → figure_embeddings.npy (935 × 512)
  3. Embed all paragraph texts (first 200 chars) → text_embeddings.npy (~6000 × 512)
  4. Embed all formula LaTeX → formula_embeddings.npy (~876 × 512)
  5. Compute similarity matrices:
     a. figure × figure → top-5 per figure (same-doc + cross-doc)
     b. paragraph × figure → top-3 per figure (filtered to same doc)
     c. formula × formula → top-5 per formula
  6. Output edges as JSONL with edge_type, source_id, target_id, weight, metadata

Output:
  data/05_eval/mineru_vl_edges_v1.jsonl
  data/05_eval/mineru_vl_embeddings_v1/  (numpy files for reuse)
```

---

## 8. Phase 3 — Cross-Doc + VL Bridge + Sentence Bridges (Revised)

With VL edges available, Phase 3 becomes significantly richer:

**File**: `experiments/build_mineru_crossdoc_bridges.py`

**Inputs**:
- Topology graph (Phase 1)
- VL edges (Section 8.6)
- v1 elements (for raw text access)

**Three sub-modules**:

### 9a. Sentence-level bridge text (unchanged from original plan)
- Split referring paragraphs into sentences
- Find exact sentence containing regex match
- Score by verb density, specificity, length

### 9b. Cross-doc edges — now with three sources

| Source | Edge type | How |
|---|---|---|
| VL figure similarity | `cross_doc_visual_sim` | Already computed in Phase VL |
| Section-title text embedding | `cross_doc_section_sim` | TF-IDF or sentence-transformer on section titles |
| Paragraph embedding similarity | `cross_doc_semantic` | Top-1 paragraph per doc, cross-doc cosine > 0.6 |

### 9c. Figure-paragraph alignment (replacing same_page_cross_type)

Use CLIP `text_describes_figure` scores to:
1. For each figure, identify the 1-3 paragraphs that actually describe it
2. These replace the ~29K same_page_cross_type edges as the primary text-figure connection
3. Also identify figures that NO paragraph describes well (CLIP score < 0.2) — flag as "orphan figures" for human review

---

## 9. Implementation Order & Dependencies

```
Phase 1 (topology graph)
  ├── Depends on: v1 output (mineru_elements_v1.json + mineru_edges_v1.jsonl)
  ├── No API calls
  ├── Estimated: ~1 script, ~300 lines
  └── Unblocks: Phase 2, Phase VL, Phase 3

Phase VL (CLIP embeddings)
  ├── Depends on: v1 figure image_paths + text content
  ├── Requires: pip install torch open_clip_torch (~800MB, one-time)
  ├── CPU inference: ~8 min for 935 images + 6000 texts
  ├── Estimated: ~1 script, ~350 lines
  └── Unblocks: Phase 3 (cross-doc visual edges, text_describes_figure)

Phase 2 (hub + candidates)
  ├── Depends on: Phase 1 output
  ├── No API calls (pure graph algorithm)
  ├── Estimated: ~1 script, ~400 lines
  └── Unblocks: Phase 4 (query generation)

Phase 3 (cross-doc + VL bridge + sentence bridges)
  ├── Depends on: Phase 1 output + Phase VL output
  ├── May use sentence-transformers for text embedding (CPU, 80MB)
  ├── Estimated: ~1 script, ~400 lines
  └── Unblocks: cross-doc query generation, bridge text quality

Phase 4 (API smoke test)
  ├── Depends on: Phase 2 output
  ├── Uses company API (9 calls)
  └── Gate: parsed_ok ≥ 8/9
```
