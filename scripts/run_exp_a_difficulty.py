#!/usr/bin/env python3
"""Experiment A: Difficulty Gradient Validation.

Runs BM25 retrieval on Level 1/2/3 queries and shows that
Recall@10 decreases as difficulty level increases, validating
the M2 difficulty taxonomy.

Uses the same BM25Lite and chunk-building logic as run_phase0_eval_ab.py.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
M2_DIR = DATA_DIR / "m2"

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{1,}")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "")]


# ─── BM25 (copied from run_phase0_eval_ab.py for self-containment) ───

class BM25Lite:
    def __init__(self, docs: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.N = len(docs)
        self.doc_lens = [len(d) for d in docs]
        self.avgdl = sum(self.doc_lens) / max(1, self.N)
        self.df: Dict[str, int] = defaultdict(int)
        self.tf_docs: List[Counter] = []
        for d in docs:
            tf = Counter(d)
            self.tf_docs.append(tf)
            for t in tf.keys():
                self.df[t] += 1

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))

    def score(self, query_tokens: List[str], doc_idx: int) -> float:
        tf = self.tf_docs[doc_idx]
        dl = self.doc_lens[doc_idx]
        s = 0.0
        for t in set(query_tokens):
            f = tf.get(t, 0)
            if f <= 0:
                continue
            idf_val = self.idf(t)
            denom = f + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-6))
            s += idf_val * (f * (self.k1 + 1)) / max(denom, 1e-6)
        return s


# ─── Chunk building ───

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str


def _to_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, (int, float, bool)):
        return str(v)
    return ""


def build_chunks(elements_json: Path, max_chars: int = 1800) -> List[Chunk]:
    data = json.loads(elements_json.read_text(encoding="utf-8"))
    docs = data.get("documents", {}) or {}
    chunks: List[Chunk] = []
    for doc_id, d in docs.items():
        elements = d.get("elements", {}) or {}
        for element_id, e in elements.items():
            fields = [
                _to_text(e.get("caption")),
                _to_text(e.get("content")),
                _to_text(e.get("context_before")),
                _to_text(e.get("context_after")),
                _to_text(e.get("enriched_title")),
                _to_text(e.get("enriched_content")),
            ]
            text = "\n".join([x for x in fields if x]).strip()
            if not text:
                continue
            if len(text) > max_chars:
                text = text[:max_chars]
            chunks.append(Chunk(chunk_id=element_id, doc_id=str(doc_id), text=text))
    return chunks


# ─── Hit evaluation ───

def span_overlap(span: str, text: str) -> float:
    span = (span or "").strip()
    text = (text or "").strip()
    if not span or not text:
        return 0.0
    if span in text:
        return 1.0
    m = SequenceMatcher(None, span, text).find_longest_match(0, len(span), 0, len(text))
    return m.size / max(1, len(span))


def query_ground_truth_ids(q: Dict[str, Any]) -> Set[str]:
    """Extract ground truth element IDs from a query."""
    gt_ids: Set[str] = set()
    # From required_evidence_spans
    for s in (q.get("required_evidence_spans") or []):
        eid = s.get("element_id", "") if isinstance(s, dict) else ""
        if eid:
            gt_ids.add(eid)
    # From element_ids (Level 2/3)
    for eid in (q.get("element_ids") or []):
        if eid:
            gt_ids.add(eid)
    # From figure_id (Level 1)
    if q.get("figure_id"):
        gt_ids.add(q["figure_id"])
    return gt_ids


def is_hit(q: Dict[str, Any], top_k_chunk_ids: List[str], chunks: List[Chunk],
           overlap_threshold: float = 0.3) -> bool:
    """Check if any ground truth element is hit in top-k."""
    gt_ids = query_ground_truth_ids(q)
    if not gt_ids:
        return False

    # Direct ID match
    for cid in top_k_chunk_ids:
        if cid in gt_ids:
            return True

    # Span overlap fallback
    spans = []
    for s in (q.get("required_evidence_spans") or []):
        st = (s.get("span") if isinstance(s, dict) else None) or ""
        if st.strip():
            spans.append(st.strip())
    if not spans:
        # Fall back to text_evidence
        te = (q.get("text_evidence") or "").strip()
        if te:
            spans.append(te)

    if spans:
        chunk_map = {c.chunk_id: c for c in chunks}
        for cid in top_k_chunk_ids:
            c = chunk_map.get(cid)
            if not c:
                continue
            for sp in spans:
                if span_overlap(sp, c.text) >= overlap_threshold:
                    return True
    return False


def reciprocal_rank(q: Dict[str, Any], ranked_chunk_ids: List[str],
                    chunks: List[Chunk], k: int = 10,
                    overlap_threshold: float = 0.3) -> float:
    """Compute reciprocal rank for a query within top-k."""
    gt_ids = query_ground_truth_ids(q)
    spans = []
    for s in (q.get("required_evidence_spans") or []):
        st = (s.get("span") if isinstance(s, dict) else None) or ""
        if st.strip():
            spans.append(st.strip())
    if not spans:
        te = (q.get("text_evidence") or "").strip()
        if te:
            spans.append(te)

    chunk_map = {c.chunk_id: c for c in chunks}
    for rank, cid in enumerate(ranked_chunk_ids[:k], 1):
        if cid in gt_ids:
            return 1.0 / rank
        if spans:
            c = chunk_map.get(cid)
            if c:
                for sp in spans:
                    if span_overlap(sp, c.text) >= overlap_threshold:
                        return 1.0 / rank
    return 0.0


def evidence_coverage(q: Dict[str, Any], top_k_chunk_ids: List[str],
                      chunks: List[Chunk], overlap_threshold: float = 0.3) -> float:
    """Fraction of ground-truth evidence elements found in top-k."""
    gt_ids = query_ground_truth_ids(q)
    if not gt_ids:
        return 0.0

    chunk_map = {c.chunk_id: c for c in chunks}
    found = set()
    for cid in top_k_chunk_ids:
        if cid in gt_ids:
            found.add(cid)
    # Span overlap fallback
    spans_by_eid: Dict[str, List[str]] = {}
    for s in (q.get("required_evidence_spans") or []):
        eid = s.get("element_id", "") if isinstance(s, dict) else ""
        sp = (s.get("span", "") if isinstance(s, dict) else "").strip()
        if eid and sp:
            spans_by_eid.setdefault(eid, []).append(sp)
    for cid in top_k_chunk_ids:
        c = chunk_map.get(cid)
        if not c:
            continue
        for eid, sps in spans_by_eid.items():
            if eid in found:
                continue
            for sp in sps:
                if span_overlap(sp, c.text) >= overlap_threshold:
                    found.add(eid)
                    break

    return len(found) / len(gt_ids)


# ─── Main ───

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment A: Difficulty Gradient")
    ap.add_argument(
        "--elements",
        default=str(DATA_DIR / "multimodal_elements.json"),
        help="Path to multimodal_elements.json (chunk corpus)",
    )
    ap.add_argument(
        "--enriched-elements",
        default="",
        help="Path to enriched elements (data111/multimodal_elements_enriched.json)",
    )
    ap.add_argument("--top-k", type=int, default=10, help="Top-K for retrieval")
    ap.add_argument(
        "--output",
        default=str(M2_DIR / "exp_a_difficulty_gradient.json"),
        help="Output report path",
    )
    args = ap.parse_args()

    # Load chunks
    elem_path = Path(args.enriched_elements) if args.enriched_elements else Path(args.elements)
    if not elem_path.exists():
        elem_path = Path(args.elements)
    print(f"Loading chunks from {elem_path}...")
    chunks = build_chunks(elem_path)
    print(f"  {len(chunks)} chunks loaded")

    # Tokenize chunks
    chunk_tokens = [tokenize(c.text) for c in chunks]
    chunk_ids = [c.chunk_id for c in chunks]

    # Build BM25 index
    print("Building BM25 index...")
    bm25 = BM25Lite(chunk_tokens)

    # Load queries per level
    levels = {}
    l1 = load_jsonl(M2_DIR / "level1_single_element.jsonl")
    l2 = load_jsonl(M2_DIR / "level2_dual_evidence.jsonl")
    l3 = load_jsonl(M2_DIR / "level3_reasoning_chain.jsonl")

    if l1:
        levels[1] = l1
    if l2:
        levels[2] = l2
    if l3:
        levels[3] = l3

    if not levels:
        print("ERROR: No level data found. Run package_m2_levels.py first.")
        return

    print(f"\nLoaded queries: " + ", ".join(f"L{k}={len(v)}" for k, v in sorted(levels.items())))

    # Run BM25 retrieval per level
    results: Dict[int, Dict[str, Any]] = {}
    k = args.top_k

    for level, queries in sorted(levels.items()):
        hits = 0
        rr_sum = 0.0
        cov_sum = 0.0
        n_valid = 0

        for q in queries:
            gt_ids = query_ground_truth_ids(q)
            if not gt_ids:
                continue  # skip queries without ground truth

            n_valid += 1
            query_text = q.get("query", "")
            q_tokens = tokenize(query_text)

            # Score all chunks
            scores = [(bm25.score(q_tokens, i), i) for i in range(len(chunks))]
            scores.sort(reverse=True)
            top_k_indices = [idx for _, idx in scores[:k]]
            top_k_ids = [chunk_ids[idx] for idx in top_k_indices]

            if is_hit(q, top_k_ids, chunks):
                hits += 1
            rr_sum += reciprocal_rank(q, top_k_ids, chunks, k=k)
            cov_sum += evidence_coverage(q, top_k_ids, chunks)

        recall = hits / max(1, n_valid)
        mrr = rr_sum / max(1, n_valid)
        avg_cov = cov_sum / max(1, n_valid)

        results[level] = {
            "level": level,
            "label": ["", "single_element", "dual_evidence", "reasoning_chain"][level],
            "n_queries": len(queries),
            "n_valid": n_valid,
            "recall_at_k": round(recall, 4),
            "mrr": round(mrr, 4),
            "avg_evidence_coverage": round(avg_cov, 4),
        }

        print(f"\n  Level {level} ({results[level]['label']}):")
        print(f"    Queries: {n_valid} (with ground truth)")
        print(f"    Recall@{k}: {recall:.4f}")
        print(f"    MRR:        {mrr:.4f}")
        print(f"    Avg Evidence Coverage: {avg_cov:.4f}")

    # Difficulty gradient analysis
    print(f"\n{'='*60}")
    print(f"Experiment A: Difficulty Gradient Summary")
    print(f"{'='*60}")
    level_nums = sorted(results.keys())
    gradient_valid = True
    for i in range(len(level_nums) - 1):
        l_curr = level_nums[i]
        l_next = level_nums[i + 1]
        r_curr = results[l_curr]["recall_at_k"]
        r_next = results[l_next]["recall_at_k"]
        delta = r_next - r_curr
        direction = "DROP" if delta < 0 else ("FLAT" if delta == 0 else "RISE")
        print(f"  L{l_curr}→L{l_next}: Recall {r_curr:.4f} → {r_next:.4f}  ({direction} {abs(delta):.4f})")
        if delta >= 0:
            gradient_valid = False

    print()
    if gradient_valid:
        print("  VERDICT: Difficulty gradient CONFIRMED (Recall strictly decreases)")
    else:
        print("  VERDICT: Difficulty gradient NOT confirmed (Recall does not strictly decrease)")
    print(f"{'='*60}")

    # Save report
    report = {
        "experiment": "A_difficulty_gradient",
        "top_k": k,
        "chunk_count": len(chunks),
        "gradient_confirmed": gradient_valid,
        "levels": {str(k): v for k, v in results.items()},
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    main()
