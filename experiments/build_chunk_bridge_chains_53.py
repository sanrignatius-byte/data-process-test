#!/usr/bin/env python3
"""
Build cross-document long chains for the 53-paper subset using chunk-level
(paragraph text) bridge matching instead of sparse keyword overlap.

Method:
  1. Load paragraph texts from MinerU elements (6003 paragraphs across 53 docs)
  2. Load enriched element descriptions (figures/formulas/tables)
  3. TF-IDF vectorize all chunks
  4. For each doc pair: compute doc-level centroid similarity, keep top-K docs
  5. Within top doc pairs: compute chunk-chunk cross-doc similarities
  6. For each matched chunk pair: use topology graph to find linked elements
  7. Build paper-level bridge graph, enumerate multi-hop chains

Cost: $0 (no LLM calls — TF-IDF + topology graph only)
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, vstack

ROOT = Path(__file__).resolve().parent.parent

MINERU_ELEMENTS = ROOT / "data/05_eval/mineru_only_graph_v1_latest/mineru_elements_v1.json"
TOPOLOGY_GRAPH = ROOT / "data/05_eval/mineru_topology_graph_v1_latest/mineru_topology_graph_v1.json"
TOPOLOGY_SUMMARY = ROOT / "data/05_eval/mineru_topology_graph_v1_latest/summary.json"
ENRICHED = ROOT / "data/02_enriched/multimodal_elements_enriched.json"


def get_53_docs() -> list[str]:
    with open(TOPOLOGY_SUMMARY) as f:
        s = json.load(f)
    return sorted(s["backbone_reachability"]["component_counts"].keys())


def _is_boilerplate(text: str) -> bool:
    """Detect author blocks, acknowledgments, arXiv headers, extraction failures,
    and other non-scientific boilerplate that produces spurious TF-IDF matches.

    Based on VLM judge audit of chunk-bridge chains (2026-05-22): early version
    had 91/300 topic_only verdicts, mostly from boilerplate matching at high
    TF-IDF scores (0.7-0.9). This filter targets every category the VLM flagged.
    """
    t = text.lower()

    # --- Category A: Author / affiliation blocks (any length) ---
    author_markers = ["§", "†", "‡", "¶", "∗"]
    if any(c in text for c in author_markers):
        return True

    # Email addresses
    if "@" in text and (".edu" in t or ".com" in t or ".org" in t or ".cn" in t or
                        ".jp" in t or ".de" in t or ".uk" in t or ".fr" in t):
        return True

    # Author-affiliation combo: name patterns + institution
    author_affil_patterns = [
        "university of", "department of", "institute of",
        "college of", "school of", "laboratory",
        "@cs.", "@edu", "@gmail", ".edu",
        "allen institute", "google research", "microsoft research",
        "deepmind", "openai", "meta ai", "facebook ai",
        "ibm research", "amazon", "apple inc",
    ]
    if sum(1 for p in author_affil_patterns if p in t) >= 2:
        return True

    # --- Category B: Acknowledgments / funding ---
    ack_patterns = [
        "acknowledgment", "acknowledgements", "acknowledge",
        "we thank", "the authors thank", "would like to thank",
        "this work was supported", "this research was supported",
        "this work is supported", "funded by", "supported by",
        "grant number", "grant no.", "nsf grant", "nsf award",
        "nih grant", "erc grant", "erc advanced", "erc starting",
        "national science foundation", "national natural science",
        "european research council", "darpa", "iaria",
        "work was partially supported", "research was partially funded",
        "was sponsored by", "is sponsored by",
    ]
    if any(p in t for p in ack_patterns):
        return True

    # --- Category C: Conference / journal metadata headers ---
    conf_patterns = [
        "proceedings of the", "international conference on",
        "published as a conference paper", "published at",
        "conference paper at", "published in",
        "©", "copyright", "all rights reserved",
        "permission to make digital", "personal use",
        "acm sig", "ieee", "association for computational linguistics",
    ]
    if sum(1 for p in conf_patterns if p in t) >= 2:
        return True
    # "Published as a conference paper at ICLR" — single strong signal
    if "published as a conference paper" in t:
        return True

    # --- Category D: arXiv / DOI / preprint headers ---
    arxiv_patterns = [
        "arxiv:", "arxiv preprint", "arxiv.", "https://arxiv",
        "doi:", "doi.org/", "preprint", "under review",
        "submitted to", "manuscript submitted",
    ]
    if any(p in t for p in arxiv_patterns):
        return True

    # --- Category E: Extraction failure boilerplate ---
    extraction_patterns = [
        "content is not provided", "no table body",
        "placeholder text", "not provided beyond",
        "for further explanation",
        "column/row content is not", "table content is",
        "figure content is", "image content is",
        "missing table content", "cannot extract columns",
        "cannot extract", "unable to extract",
        "no html", "no table html", "raw table",
        "provide the raw table",
    ]
    if any(p in t for p in extraction_patterns):
        return True

    # --- Category F: Pure URL / reference dump ---
    url_count = t.count("http://") + t.count("https://") + t.count("www.")
    if url_count >= 2:
        return True

    # --- Category G: Pure reference / bibliography lines ---
    if len(text) < 100 and t.strip().startswith("[") and "]" in t[:10]:
        return True

    # --- Category H: QED / proof markers / page ornaments ---
    # Descriptions of visual markers (hollow squares, end-of-proof symbols) that
    # happen to include inline math are still boilerplate — the math is incidental.
    qed_markers = [
        "hollow square", "empty square", "square marker", "square glyph",
        "black square", "white square", "square symbol",
        "qed", "∎", "□", "■",
        "proof end marker", "end-of-proof",
    ]
    if any(p in t for p in qed_markers):
        # Even with math, these are visual-ornament descriptions
        return True

    # --- Category I: Et al. citation lists (pure author name strings) ---
    # "Jieyu Zhao, Tianlu Wang, Mark Yatskar, Vicente Ordonez, Kai-Wei Chang"
    # Count commas — pure name lists with many commas and short sentences
    comma_count = text.count(",")
    if comma_count >= 4 and len(text) < 300 and "et al" not in t:
        # Likely a pure author list
        if not any(w in t for w in ["model", "method", "data", "result", "use", "show",
                                     "propose", "learn", "train", "test", "evaluate"]):
            return True

    return False


def _is_enriched_junk(text: str) -> bool:
    """Lightweight junk filter for enriched element (figure/table/formula) chunks.

    Only catches extraction failures and visual-ornament descriptions.
    Unlike _is_boilerplate, does NOT flag author/affiliation/QED patterns
    that could appear legitimately in scientific figure descriptions.
    """
    t = text.lower()

    # Extraction failures — the enriched caption is a placeholder error message
    extraction_patterns = [
        "content is not provided", "no table body",
        "placeholder text", "not provided beyond",
        "column/row content is not",
        "missing table content", "cannot extract columns",
        "cannot extract", "unable to extract",
        "no html", "no table html", "raw table",
        "provide the raw table",
    ]
    if any(p in t for p in extraction_patterns):
        return True

    # Visual ornament descriptions masquerading as scientific elements.
    # These describe page-decoration symbols (hollow squares, QED markers)
    # that MinerU classified as figure/formula elements.
    ornament_patterns = [
        "hollow square", "empty square", "square glyph",
        "square marker", "square symbol",
        "proof end marker", "end-of-proof",
    ]
    if any(p in t for p in ornament_patterns):
        return True

    return False


def _has_scientific_content(text: str) -> bool:
    """Check if paragraph contains scientific/mathematical content worth bridging."""
    t = text.lower()

    # Math notation — strong positive signal
    math_indicators = [
        "$", "\\frac", "\\sum", "\\int", "\\mathbb", "\\mathcal",
        "\\mathbf", "\\text", "\\begin", "\\end",
    ]
    if any(m in text for m in math_indicators):
        return True

    # Subscripts/superscripts used in formulas (but not in author markers)
    if "_" in text and any(c in text for c in "{}()") :
        return True

    # Scientific keywords
    sci_keywords = [
        "model", "method", "result", "experiment", "dataset", "training",
        "accuracy", "performance", "algorithm", "loss", "function",
        "distribution", "parameter", "estimate", "prediction", "classification",
        "regression", "fairness", "bias", "causal", "graph", "equation",
        "metric", "sample", "feature", "layer", "network", "embedding",
        "cluster", "probability", "variance", "error", "optimization",
        "gradient", "inference", "hypothesis", "constraint", "objective",
        "evaluation", "benchmark", "baseline", "measure", "correlation",
    ]
    return any(kw in t for kw in sci_keywords)


def load_paragraphs(
    mineru_path: Path, doc_filter: set[str],
    require_scientific: bool = True,
    max_per_doc: int = 60,
) -> dict[str, list[dict]]:
    """Load text paragraphs from MinerU elements. Filters boilerplate, keeps scientific content."""
    with open(mineru_path) as f:
        data = json.load(f)

    doc_paras: dict[str, list[dict]] = defaultdict(list)
    for doc_id, doc_data in data.get("documents", {}).items():
        if doc_id not in doc_filter:
            continue
        for elem_id, elem in doc_data.get("elements", {}).items():
            if elem.get("element_type") != "text":
                continue
            content = (elem.get("content") or "").strip()
            context_after = (elem.get("context_after") or "").strip()
            text = content
            if context_after and len(text) < 200:
                text = content + " " + context_after
            if len(text) < 80:
                continue
            if _is_boilerplate(text):
                continue
            if require_scientific and not _has_scientific_content(text):
                continue
            doc_paras[doc_id].append({
                "element_id": elem_id,
                "text": text,
                "page_idx": elem.get("page_idx", 0),
                "position_idx": elem.get("position_idx", 0),
            })
        # Cap per doc to avoid over-representation
        if len(doc_paras[doc_id]) > max_per_doc:
            # Keep paragraphs distributed across the paper
            step = len(doc_paras[doc_id]) / max_per_doc
            doc_paras[doc_id] = [doc_paras[doc_id][int(i * step)] for i in range(max_per_doc)]

    return dict(doc_paras)


def load_enriched_chunks(enriched_path: Path, doc_filter: set[str]) -> dict[str, list[dict]]:
    """Load enriched element descriptions as additional chunks."""
    with open(enriched_path) as f:
        data = json.load(f)

    doc_chunks: dict[str, list[dict]] = defaultdict(list)
    for doc_id, doc_data in data.get("documents", {}).items():
        if doc_id not in doc_filter:
            continue
        for elem_id, elem in doc_data.get("elements", {}).items():
            enriched = elem.get("enriched_content", "") or ""
            caption = elem.get("caption", "") or ""
            title = elem.get("enriched_title", "") or ""
            text = f"{title}. {caption} {enriched}".strip()
            if len(text) < 40:
                continue
            # Lightweight filter for enriched chunks: only catch extraction
            # failures and visual-ornament descriptions. Full _is_boilerplate
            # is too aggressive (e.g. "square marker" in scatter plots).
            if _is_enriched_junk(text):
                continue
            doc_chunks[doc_id].append({
                "element_id": elem_id,
                "text": text,
                "element_type": elem.get("element_type", ""),
                "is_enriched": True,
            })
    return dict(doc_chunks)


def load_topology_element_links(topo_path: Path) -> dict[str, list[str]]:
    """Build mapping: element_id -> [connected element_ids] via topology graph edges."""
    with open(topo_path) as f:
        g = json.load(f)

    e2n = g.get("element_to_node", {})
    edges = g.get("edges", [])

    # Build node -> element reverse map
    n2e = {}
    for eid, nid in e2n.items():
        n2e.setdefault(nid, []).append(eid)

    # For each element, find all other elements connected via element_ref edges
    linked: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        if edge.get("edge_type") not in ("element_ref", "same_page_cross_type"):
            continue
        src_eids = n2e.get(edge["source_id"], [])
        tgt_eids = n2e.get(edge["target_id"], [])
        for se in src_eids:
            for te in tgt_eids:
                if se != te:
                    linked[se].append(te)
                    linked[te].append(se)

    # Deduplicate
    for eid in linked:
        linked[eid] = list(set(linked[eid]))

    return dict(linked)


def build_chunk_corpus(
    paragraphs: dict[str, list[dict]],
    enriched_chunks: dict[str, list[dict]],
    topo_links: dict[str, list[str]],
) -> tuple[list[dict], np.ndarray, TfidfVectorizer]:
    """Combine paragraphs and enriched chunks into a single corpus, TF-IDF vectorize."""
    all_chunks: list[dict] = []
    all_texts: list[str] = []

    for doc_id in sorted(set(list(paragraphs.keys()) + list(enriched_chunks.keys()))):
        for p in paragraphs.get(doc_id, []):
            all_chunks.append({
                "doc_id": doc_id,
                "chunk_id": p["element_id"],
                "text": p["text"],
                "chunk_type": "paragraph",
                "linked_elements": topo_links.get(p["element_id"], []),
            })
            all_texts.append(p["text"])
        for e in enriched_chunks.get(doc_id, []):
            all_chunks.append({
                "doc_id": doc_id,
                "chunk_id": e["element_id"],
                "text": e["text"],
                "chunk_type": "enriched",
                "element_type": e.get("element_type", ""),
                "linked_elements": topo_links.get(e["element_id"], []),
            })
            all_texts.append(e["text"])

    print(f"Total chunks: {len(all_chunks)} ({sum(1 for c in all_chunks if c['chunk_type']=='paragraph')} paragraphs, "
          f"{sum(1 for c in all_chunks if c['chunk_type']=='enriched')} enriched)")

    # TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=8000,
        stop_words="english",
        sublinear_tf=True,
        ngram_range=(1, 2),
        max_df=0.8,
        min_df=2,
    )
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    print(f"TF-IDF matrix: {tfidf_matrix.shape}, vocab={len(vectorizer.vocabulary_)}")

    return all_chunks, tfidf_matrix, vectorizer


def find_cross_doc_pairs(
    all_chunks: list[dict],
    tfidf_matrix: csr_matrix,
    top_k_docs: int = 8,
    top_k_pairs_per_pair: int = 5,
    min_similarity: float = 0.15,
) -> list[dict]:
    """Find cross-doc chunk pairs with highest TF-IDF cosine similarity."""

    # Build doc -> chunk indices mapping
    doc_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(all_chunks):
        doc_to_indices[c["doc_id"]].append(i)

    doc_ids = sorted(doc_to_indices.keys())
    n_docs = len(doc_ids)

    # Compute per-doc centroid vectors
    doc_centroids = np.zeros((n_docs, tfidf_matrix.shape[1]), dtype=np.float64)
    doc_indices_map = {}
    for di, doc_id in enumerate(doc_ids):
        indices = doc_to_indices[doc_id]
        row_sum = np.array(tfidf_matrix[indices].sum(axis=0)).flatten()
        doc_centroids[di] = row_sum / len(indices)
        doc_indices_map[doc_id] = indices

    # Doc-doc similarity
    doc_sim = cosine_similarity(doc_centroids)
    # Mask out self-similarity and lower triangle
    for i in range(n_docs):
        doc_sim[i, i] = -1

    pairs: list[dict] = []
    total_candidate_pairs = 0

    for di in range(n_docs):
        doc_a = doc_ids[di]
        # Find top-k most similar docs
        sims_to_others = [(doc_sim[di, dj], dj) for dj in range(n_docs) if dj != di]
        sims_to_others.sort(key=lambda x: -x[0])
        top_docs = sims_to_others[:top_k_docs]

        for doc_sim_score, dj in top_docs:
            doc_b = doc_ids[dj]
            if doc_b <= doc_a:
                continue  # avoid double counting

            idx_a = doc_indices_map[doc_a]
            idx_b = doc_indices_map[doc_b]

            # Sub-matrices
            mat_a = tfidf_matrix[idx_a]
            mat_b = tfidf_matrix[idx_b]

            # Pairwise chunk-chunk similarities (n_a × n_b)
            chunk_sim = cosine_similarity(mat_a, mat_b)

            # Find top pairs
            n_a, n_b = chunk_sim.shape
            for ci in range(n_a):
                for cj in range(n_b):
                    sim = chunk_sim[ci, cj]
                    if sim >= min_similarity:
                        total_candidate_pairs += 1
                        pairs.append({
                            "source_doc": doc_a,
                            "target_doc": doc_b,
                            "source_chunk_idx": idx_a[ci],
                            "target_chunk_idx": idx_b[cj],
                            "similarity": float(sim),
                        })

    print(f"Raw candidate pairs (sim >= {min_similarity}): {total_candidate_pairs}")

    # Deduplicate: keep top-k pairs per doc pair
    import collections
    pair_by_docs: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for p in pairs:
        key = (p["source_doc"], p["target_doc"])
        pair_by_docs[key].append(p)

    filtered_pairs = []
    for key, plist in pair_by_docs.items():
        plist.sort(key=lambda x: -x["similarity"])
        filtered_pairs.extend(plist[:top_k_pairs_per_pair])

    print(f"Filtered pairs (top-{top_k_pairs_per_pair} per doc pair): {len(filtered_pairs)} "
          f"across {len(pair_by_docs)} doc pairs")
    return filtered_pairs


def build_chain_from_pairs(
    chunk_pairs: list[dict],
    all_chunks: list[dict],
    max_hops: int = 2,
    max_chains: int = 300,
) -> tuple[list[dict], dict[str, dict[str, list[dict]]]]:
    """Build paper-level graph from chunk pairs, enumerate chains."""

    # Aggregate pairs into paper-level edges with bridge info
    paper_graph: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for p in chunk_pairs:
        a, b = p["source_doc"], p["target_doc"]
        paper_graph[a][b].append(p)
        paper_graph[b][a].append(p)

    chains: list[dict] = []

    # 1-hop chains
    for doc_a in sorted(paper_graph.keys()):
        for doc_b in sorted(paper_graph[doc_a].keys()):
            if doc_b <= doc_a:
                continue
            bridge_pairs = sorted(paper_graph[doc_a][doc_b],
                                  key=lambda x: -x["similarity"])[:3]
            for bp in bridge_pairs:
                src_chunk = all_chunks[bp["source_chunk_idx"]]
                tgt_chunk = all_chunks[bp["target_chunk_idx"]]
                chains.append({
                    "chain_id": f"chunk_eb1_{doc_a}_{doc_b}_{len(chains)}",
                    "paper_path": [doc_a, doc_b],
                    "cross_doc_hops": 1,
                    "hops": [{
                        "from_doc": doc_a, "to_doc": doc_b,
                        "from_chunk_id": src_chunk["chunk_id"],
                        "to_chunk_id": tgt_chunk["chunk_id"],
                        "bridge_type": "chunk_similarity",
                        "similarity": round(bp["similarity"], 4),
                        "bridge_text_source": src_chunk["text"][:500],
                        "bridge_text_target": tgt_chunk["text"][:500],
                        "from_linked_elements": src_chunk.get("linked_elements", []),
                        "to_linked_elements": tgt_chunk.get("linked_elements", []),
                    }],
                    "total_score": round(bp["similarity"], 4),
                })

    # 2-hop chains
    if max_hops >= 2:
        for doc_a in sorted(paper_graph.keys()):
            for doc_b in sorted(paper_graph[doc_a].keys()):
                if doc_b <= doc_a:
                    continue
                for doc_c in sorted(paper_graph[doc_b].keys()):
                    if doc_c == doc_a:
                        continue
                    ab_pairs = sorted(paper_graph[doc_a][doc_b],
                                     key=lambda x: -x["similarity"])[:2]
                    bc_pairs = sorted(paper_graph[doc_b][doc_c],
                                     key=lambda x: -x["similarity"])[:2]

                    for bp_ab in ab_pairs:
                        for bp_bc in bc_pairs:
                            score = bp_ab["similarity"] + bp_bc["similarity"]
                            src_c = all_chunks[bp_ab["source_chunk_idx"]]
                            mid_c1 = all_chunks[bp_ab["target_chunk_idx"]]
                            mid_c2 = all_chunks[bp_bc["source_chunk_idx"]]
                            tgt_c = all_chunks[bp_bc["target_chunk_idx"]]

                            # Joint elements at B
                            b_elems_ab = set(src_c.get("linked_elements", [])) | set(mid_c1.get("linked_elements", []))
                            b_elems_bc = set(mid_c2.get("linked_elements", [])) | set(tgt_c.get("linked_elements", []))
                            joint_elems = b_elems_ab & b_elems_bc if b_elems_ab and b_elems_bc else set()

                            chains.append({
                                "chain_id": f"chunk_eb2_{doc_a}_{doc_b}_{doc_c}_{len(chains)}",
                                "paper_path": [doc_a, doc_b, doc_c],
                                "cross_doc_hops": 2,
                                "hops": [
                                    {
                                        "from_doc": doc_a, "to_doc": doc_b,
                                        "from_chunk_id": src_c["chunk_id"],
                                        "to_chunk_id": mid_c1["chunk_id"],
                                        "bridge_type": "chunk_similarity",
                                        "similarity": round(bp_ab["similarity"], 4),
                                        "bridge_text_source": src_c["text"][:500],
                                        "bridge_text_target": mid_c1["text"][:500],
                                    },
                                    {
                                        "from_doc": doc_b, "to_doc": doc_c,
                                        "from_chunk_id": mid_c2["chunk_id"],
                                        "to_chunk_id": tgt_c["chunk_id"],
                                        "bridge_type": "chunk_similarity",
                                        "similarity": round(bp_bc["similarity"], 4),
                                        "bridge_text_source": mid_c2["text"][:500],
                                        "bridge_text_target": tgt_c["text"][:500],
                                    },
                                ],
                                "joint_elements_at_b": sorted(joint_elems),
                                "total_score": round(score, 4),
                            })

                    if len(chains) >= max_chains:
                        break
                if len(chains) >= max_chains:
                    break
            if len(chains) >= max_chains:
                break

    chains.sort(key=lambda c: -c["total_score"])
    return chains[:max_chains], dict(paper_graph)


def analyze_diversity(
    chains: list[dict],
    all_chunks: list[dict],
    paper_graph: dict[str, dict[str, list[dict]]],
    docs_53: set[str],
) -> dict:
    """Analyze chain diversity, coverage, and bridge topics."""
    from collections import Counter

    # Paper coverage
    papers_in_chains = set()
    for c in chains:
        for p in c["paper_path"]:
            papers_in_chains.add(p)

    # Paper frequency
    paper_freq = Counter()
    for c in chains:
        for p in c["paper_path"]:
            paper_freq[p] += 1

    # Bridge text key phrases (simple: extract noun phrases from bridge texts)
    bridge_texts = []
    for c in chains:
        for h in c["hops"]:
            bridge_texts.append(h.get("bridge_text_source", ""))
            bridge_texts.append(h.get("bridge_text_target", ""))

    # Simple keyword extraction from bridge texts
    topic_keywords = [
        "fairness", "bias", "discrimination", "parity", "calibration",
        "causal", "structural equation", "counterfactual", "mediation",
        "coreference", "resolution", "ontonotes", "winobias", "gender",
        "embedding", "representation", "clustering", "classification",
        "neural network", "training", "optimization", "gradient", "loss",
        "recidivism", "compas", "probability", "acceptance", "stack exchange",
        "deep learning", "adversarial", "regularization", "generalization",
        "algorithm", "metric", "distance", "distribution", "model",
        "prediction", "accuracy", "dataset", "benchmark", "evaluation",
    ]
    topic_counts = Counter()
    all_bridge_text = " ".join(bridge_texts).lower()
    for kw in topic_keywords:
        cnt = all_bridge_text.count(kw.lower())
        if cnt > 0:
            topic_counts[kw] = cnt

    # Score distribution
    scores_1hop = [c["total_score"] for c in chains if c["cross_doc_hops"] == 1]
    scores_2hop = [c["total_score"] for c in chains if c["cross_doc_hops"] == 2]

    # Chunk type pair distribution
    chunk_type_pairs = Counter()
    for c in chains:
        for h in c["hops"]:
            src_idx = None
            tgt_idx = None
            for i, chunk in enumerate(all_chunks):
                if chunk["chunk_id"] == h["from_chunk_id"]:
                    src_idx = i
                if chunk["chunk_id"] == h["to_chunk_id"]:
                    tgt_idx = i
            if src_idx is not None and tgt_idx is not None:
                pair_type = f"{all_chunks[src_idx]['chunk_type']}↔{all_chunks[tgt_idx]['chunk_type']}"
                chunk_type_pairs[pair_type] += 1

    return {
        "papers_in_chains": len(papers_in_chains),
        "papers_total": len(docs_53),
        "paper_coverage_pct": round(len(papers_in_chains) / len(docs_53) * 100, 1),
        "paper_freq_distribution": {
            "once": sum(1 for v in paper_freq.values() if v == 1),
            "2_5": sum(1 for v in paper_freq.values() if 2 <= v <= 5),
            "6_10": sum(1 for v in paper_freq.values() if 6 <= v <= 10),
            "gt_10": sum(1 for v in paper_freq.values() if v > 10),
        },
        "top_hub_papers": paper_freq.most_common(8),
        "topic_keywords": topic_counts.most_common(20),
        "score_1hop": {
            "min": round(min(scores_1hop), 3) if scores_1hop else 0,
            "max": round(max(scores_1hop), 3) if scores_1hop else 0,
            "median": round(sorted(scores_1hop)[len(scores_1hop)//2], 3) if scores_1hop else 0,
        },
        "score_2hop": {
            "min": round(min(scores_2hop), 3) if scores_2hop else 0,
            "max": round(max(scores_2hop), 3) if scores_2hop else 0,
            "median": round(sorted(scores_2hop)[len(scores_2hop)//2], 3) if scores_2hop else 0,
        },
        "chunk_type_pairs": dict(chunk_type_pairs.most_common()),
        "doc_pairs_with_bridges": sum(1 for a in paper_graph for b in paper_graph[a] if a < b),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build chunk-level bridge chains for the 53-paper subset"
    )
    parser.add_argument("--min-similarity", type=float, default=0.15)
    parser.add_argument("--top-k-docs", type=int, default=8,
                       help="Per doc, how many other docs to compare chunks with")
    parser.add_argument("--top-k-pairs", type=int, default=5,
                       help="Max chunk pairs per doc pair")
    parser.add_argument("--max-hops", type=int, default=2)
    parser.add_argument("--max-chains", type=int, default=300)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--no-enriched", action="store_true",
                       help="Skip enriched element chunks, use paragraphs only")
    args = parser.parse_args()

    docs_53 = set(get_53_docs())
    print(f"Target docs: {len(docs_53)}")

    # 1. Load paragraphs
    paragraphs = load_paragraphs(MINERU_ELEMENTS, docs_53)
    total_paras = sum(len(v) for v in paragraphs.values())
    print(f"Paragraphs loaded: {total_paras} from {len(paragraphs)} docs")

    # 2. Load enriched chunks
    enriched = {}
    if not args.no_enriched:
        enriched = load_enriched_chunks(ENRICHED, docs_53)
        total_enriched = sum(len(v) for v in enriched.values())
        print(f"Enriched chunks loaded: {total_enriched} from {len(enriched)} docs")

    # 3. Load topology links
    topo_links = load_topology_element_links(TOPOLOGY_GRAPH)
    print(f"Topology element links: {len(topo_links)} elements with connections")

    # 4. Build chunk corpus + TF-IDF
    all_chunks, tfidf_matrix, vectorizer = build_chunk_corpus(
        paragraphs, enriched, topo_links
    )

    # 5. Find cross-doc chunk pairs
    pairs = find_cross_doc_pairs(
        all_chunks, tfidf_matrix,
        top_k_docs=args.top_k_docs,
        top_k_pairs_per_pair=args.top_k_pairs,
        min_similarity=args.min_similarity,
    )

    # 6. Build chains
    chains, paper_graph = build_chain_from_pairs(
        pairs, all_chunks,
        max_hops=args.max_hops,
        max_chains=args.max_chains,
    )
    hop1 = sum(1 for c in chains if c["cross_doc_hops"] == 1)
    hop2 = sum(1 for c in chains if c["cross_doc_hops"] == 2)
    print(f"Chains: {len(chains)} total ({hop1} 1-hop, {hop2} 2-hop)")

    # 7. Diversity analysis
    analysis = analyze_diversity(chains, all_chunks, paper_graph, docs_53)

    # 8. Output
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.output_dir:
        out_dir = ROOT / args.output_dir
    else:
        out_dir = ROOT / f"data/05_eval/chunk_bridge_chains_53_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write chunk pairs
    pairs_path = out_dir / "chunk_pairs.jsonl"
    with open(pairs_path, "w", encoding="utf-8") as f:
        for p in pairs:
            src = all_chunks[p["source_chunk_idx"]]
            tgt = all_chunks[p["target_chunk_idx"]]
            out = {
                "source_doc": p["source_doc"],
                "target_doc": p["target_doc"],
                "source_chunk_id": src["chunk_id"],
                "target_chunk_id": tgt["chunk_id"],
                "source_chunk_type": src["chunk_type"],
                "target_chunk_type": tgt["chunk_type"],
                "source_text": src["text"][:500],
                "target_text": tgt["text"][:500],
                "similarity": p["similarity"],
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    # Write chains
    chains_path = out_dir / "chains.jsonl"
    with open(chains_path, "w", encoding="utf-8") as f:
        for c in chains:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # Summary
    summary = {
        "method": "chunk_bridge",
        "description": "Cross-doc chains built from TF-IDF paragraph/enriched-chunk matching",
        "corpus": "old_53",
        "created_at": ts,
        "config": {
            "min_similarity": args.min_similarity,
            "top_k_docs": args.top_k_docs,
            "top_k_pairs_per_pair": args.top_k_pairs,
            "max_hops": args.max_hops,
            "use_enriched_chunks": not args.no_enriched,
        },
        "corpus_stats": {
            "total_chunks": len(all_chunks),
            "paragraphs": sum(1 for c in all_chunks if c["chunk_type"] == "paragraph"),
            "enriched": sum(1 for c in all_chunks if c["chunk_type"] == "enriched"),
            "vocab_size": len(vectorizer.vocabulary_),
        },
        "pairs": {
            "raw_candidates_filtered": len(pairs),
        },
        "chains": {
            "total": len(chains),
            "1_hop": hop1,
            "2_hop": hop2,
        },
        "diversity": analysis,
        "output_dir": str(out_dir.relative_to(ROOT)),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Print results
    print(f"\n=== Chunk-Bridge Chains (old_53) ===")
    print(f"Total chunks: {len(all_chunks)} ({summary['corpus_stats']['paragraphs']} para + "
          f"{summary['corpus_stats']['enriched']} enriched)")
    print(f"Doc pairs with bridges: {analysis['doc_pairs_with_bridges']}")
    print(f"Chains: {len(chains)} ({hop1} 1-hop, {hop2} 2-hop)")
    print(f"Paper coverage: {analysis['papers_in_chains']}/{analysis['papers_total']} "
          f"({analysis['paper_coverage_pct']}%)")
    print(f"\nPaper frequency distribution:")
    pf = analysis["paper_freq_distribution"]
    print(f"  Once: {pf['once']}, 2-5: {pf['2_5']}, 6-10: {pf['6_10']}, >10: {pf['gt_10']}")
    print(f"\nTop hub papers:")
    for p, f in analysis["top_hub_papers"]:
        print(f"  {p}: {f} chains")
    print(f"\nChunk pair types:")
    for pt, cnt in analysis["chunk_type_pairs"].items():
        print(f"  {pt}: {cnt}")
    print(f"\nScore distribution:")
    print(f"  1-hop: min={analysis['score_1hop']['min']:.3f}, "
          f"median={analysis['score_1hop']['median']:.3f}, "
          f"max={analysis['score_1hop']['max']:.3f}")
    if scores_2hop := analysis["score_2hop"]:
        print(f"  2-hop: min={scores_2hop['min']:.3f}, "
              f"median={scores_2hop['median']:.3f}, "
              f"max={scores_2hop['max']:.3f}")
    print(f"\nOutput: {out_dir}")

    # Print example chains
    print(f"\n=== Example chains ===")
    for c in chains[:5]:
        path = " → ".join(c["paper_path"])
        print(f"\n{c['chain_id']}")
        print(f"  Path: {path}")
        print(f"  Hops: {c['cross_doc_hops']}, Score: {c['total_score']:.4f}")
        for i, h in enumerate(c["hops"]):
            print(f"  Bridge {i+1}: sim={h['similarity']:.4f}")
            print(f"    Source text: {h['bridge_text_source'][:200]}...")
            print(f"    Target text: {h['bridge_text_target'][:200]}...")
        if c.get("joint_elements_at_b"):
            print(f"  Joint elements at B: {c['joint_elements_at_b'][:5]}")


if __name__ == "__main__":
    main()
