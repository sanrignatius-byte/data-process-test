#!/usr/bin/env python3
"""Experiment C: QA Triangle — Evidence Coverage under BM25 vs Graph retrieval.

For Level 2 and Level 3 queries:
  Path A: BM25 top-5 → concatenate as context → LLM generates answer
  Path B: BM25+Graph top-5 → concatenate as context → LLM generates answer

Evaluate: what fraction of ground-truth evidence elements are covered by each answer.

Requires LLM API calls (company provider by default, model gpt-5.4).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.token_logger import log_run

DATA_DIR = PROJECT_ROOT / "data"
M2_DIR = DATA_DIR / "m2"

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{1,}")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "")]


# ─── BM25 (same as run_exp_a_difficulty.py) ───

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


# ─── Graph signal (neighbor propagation, simplified) ───

def load_element_adjacency(hub_candidates_json: Path) -> Dict[str, Set[str]]:
    if not hub_candidates_json.exists():
        return {}
    obj = json.loads(hub_candidates_json.read_text(encoding="utf-8"))
    pairs = obj.get("pairs", []) or []
    adj: Dict[str, Set[str]] = defaultdict(set)
    for p in pairs:
        a = str(p.get("element_a_id", "") or "").strip()
        b = str(p.get("element_b_id", "") or "").strip()
        if a and b:
            adj[a].add(b)
            adj[b].add(a)
    for ab in (obj.get("adjacent_bridge_adjacency") or []):
        eids_i = [str(e).strip() for e in (ab.get("elements_i") or []) if str(e).strip()]
        eids_j = [str(e).strip() for e in (ab.get("elements_j") or []) if str(e).strip()]
        for ei in eids_i:
            for ej in eids_j:
                adj[ei].add(ej)
                adj[ej].add(ei)
    return dict(adj)


def graph_rerank(
    bm25_scores: List[Tuple[float, int]],
    chunk_ids: List[str],
    adjacency: Dict[str, Set[str]],
    top_n: int = 50,
    decay: float = 0.2,
) -> List[Tuple[float, int]]:
    """BM25 + 1-hop neighbor propagation rerank."""
    if not adjacency:
        return bm25_scores

    # Normalize BM25 scores
    max_s = max((s for s, _ in bm25_scores[:top_n]), default=1.0)
    if max_s <= 0:
        max_s = 1.0
    norm_scores = {chunk_ids[idx]: s / max_s for s, idx in bm25_scores[:top_n]}

    # Propagate
    boost: Dict[str, float] = defaultdict(float)
    for cid, score in norm_scores.items():
        for nbr in adjacency.get(cid, set()):
            boost[nbr] = max(boost[nbr], score * decay)

    # Re-score
    cid_to_idx = {cid: i for i, cid in enumerate(chunk_ids)}
    reranked = []
    for s, idx in bm25_scores:
        cid = chunk_ids[idx]
        new_s = s + boost.get(cid, 0.0) * max_s
        reranked.append((new_s, idx))
    reranked.sort(reverse=True)
    return reranked


# ─── Evidence evaluation ───

def span_overlap(span: str, text: str) -> float:
    span = (span or "").strip()
    text = (text or "").strip()
    if not span or not text:
        return 0.0
    if span in text:
        return 1.0
    m = SequenceMatcher(None, span, text).find_longest_match(0, len(span), 0, len(text))
    return m.size / max(1, len(span))


def _normalize_element_id(eid: str) -> str:
    """Normalize L1-style short IDs (fig_/tab_/eq_) to chunk-style IDs (figure_/table_/formula_)."""
    eid = re.sub(r'^([0-9.]+)_fig_', r'\1_figure_', eid)
    eid = re.sub(r'^([0-9.]+)_tab_', r'\1_table_', eid)
    eid = re.sub(r'^([0-9.]+)_eq_', r'\1_formula_', eid)
    return eid


def query_ground_truth_ids(q: Dict[str, Any]) -> Set[str]:
    gt_ids: Set[str] = set()
    for s in (q.get("required_evidence_spans") or []):
        eid = s.get("element_id", "") if isinstance(s, dict) else ""
        if eid:
            gt_ids.add(_normalize_element_id(eid))
    for eid in (q.get("element_ids") or []):
        if eid:
            gt_ids.add(_normalize_element_id(eid))
    return gt_ids


def evidence_coverage(q: Dict[str, Any], top_k_chunk_ids: List[str],
                      chunks: List[Chunk], overlap_threshold: float = 0.3) -> float:
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


# ─── LLM QA ───

_COMPANY_API_URL: str = ""
_COMPANY_API_KEY: str = ""


def call_qa_llm(query: str, context: str, model: str, provider: str) -> Tuple[str, int, int]:
    """Ask LLM to answer a query given retrieved context. Returns (answer, in_tok, out_tok)."""
    system = (
        "You are a research assistant. Answer the question using ONLY the provided context. "
        "Cite specific evidence from the context. If the context is insufficient, say so."
    )
    prompt = f"## Context\n{context}\n\n## Question\n{query}\n\nAnswer concisely (max 3 sentences):"

    if provider == "company":
        from local_api_logger import wrap_requests_call
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {_COMPANY_API_KEY}",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 512,
            "temperature": 0.2,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        stream = wrap_requests_call(
            model=model,
            url=_COMPANY_API_URL,
            headers=headers,
            payload=payload,
            user="exp_c_qa",
            verify=False,
        )
        # Collect SSE stream
        content_parts: List[str] = []
        in_tok, out_tok = 0, 0
        for line in stream:
            if isinstance(line, bytes):
                line = line.decode("utf-8", errors="replace")
            line = line.strip() if isinstance(line, str) else str(line).strip()
            if not line or not line.startswith("data: "):
                continue
            data = line[6:].strip()
            if data == "[DONE]":
                continue
            try:
                parsed = json.loads(data)
                if "usage" in parsed and parsed["usage"]:
                    in_tok = parsed["usage"].get("prompt_tokens", 0)
                    out_tok = parsed["usage"].get("completion_tokens", 0)
                choices = parsed.get("choices") or []
                if choices:
                    delta = choices[0].get("delta", {})
                    if c := delta.get("content"):
                        content_parts.append(c)
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
        return "".join(content_parts), in_tok, out_tok

    elif provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=512,
            temperature=0.2,
        )
        text = r.choices[0].message.content if r.choices else ""
        in_tok = int(getattr(getattr(r, "usage", None), "prompt_tokens", 0) or 0)
        out_tok = int(getattr(getattr(r, "usage", None), "completion_tokens", 0) or 0)
        return str(text or ""), in_tok, out_tok

    else:
        raise ValueError(f"Unsupported provider: {provider}")


def answer_mentions_evidence(answer: str, q: Dict[str, Any]) -> float:
    """Check how many ground-truth evidence spans are mentioned in the LLM answer."""
    gt_spans: List[str] = []
    for s in (q.get("required_evidence_spans") or []):
        sp = (s.get("span", "") if isinstance(s, dict) else "").strip()
        if sp:
            gt_spans.append(sp)
    if not gt_spans:
        return 1.0  # no spans to check

    answer_lower = answer.lower()
    found = 0
    for sp in gt_spans:
        # Check if key terms from the span appear in the answer
        sp_tokens = set(tokenize(sp))
        if not sp_tokens:
            found += 1
            continue
        answer_tokens = set(tokenize(answer))
        overlap = len(sp_tokens & answer_tokens) / len(sp_tokens)
        if overlap >= 0.4:
            found += 1
    return found / len(gt_spans)


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
    ap = argparse.ArgumentParser(description="Experiment C: QA Triangle")
    ap.add_argument("--elements", default=str(DATA_DIR / "multimodal_elements.json"))
    ap.add_argument("--enriched-elements", default="")
    ap.add_argument(
        "--hub-candidates",
        default=str(DATA_DIR / "hub_candidates_enriched.json"),
        help="Hub candidates for graph adjacency",
    )
    ap.add_argument("--top-k", type=int, default=5, help="Top-K chunks as context")
    ap.add_argument("--limit", type=int, default=0, help="Limit queries per level (0=all)")
    ap.add_argument("--provider", choices=["company", "openai"], default="company")
    ap.add_argument("--model", default=None)
    ap.add_argument("--delay", type=float, default=0.5)
    ap.add_argument("--dry-run", action="store_true", help="Skip LLM calls, just measure retrieval coverage")
    ap.add_argument("--output", default=str(M2_DIR / "exp_c_qa_triangle.json"))
    args = ap.parse_args()

    if args.model is None:
        args.model = "gpt-5.4" if args.provider == "company" else "gpt-4o"

    # Load chunks
    elem_path = Path(args.enriched_elements) if args.enriched_elements else Path(args.elements)
    if not elem_path.exists():
        elem_path = Path(args.elements)
    print(f"Loading chunks from {elem_path}...")
    chunks = build_chunks(elem_path)
    print(f"  {len(chunks)} chunks")

    chunk_tokens = [tokenize(c.text) for c in chunks]
    chunk_ids = [c.chunk_id for c in chunks]

    print("Building BM25 index...")
    bm25 = BM25Lite(chunk_tokens)

    # Load graph adjacency
    hub_path = Path(args.hub_candidates)
    adjacency = load_element_adjacency(hub_path)
    print(f"  Graph adjacency: {len(adjacency)} elements with neighbors")

    # Load Level 2 and 3 queries
    l2 = load_jsonl(M2_DIR / "level2_dual_evidence.jsonl")
    l3 = load_jsonl(M2_DIR / "level3_reasoning_chain.jsonl")
    queries_by_level: Dict[int, List[Dict[str, Any]]] = {}
    if l2:
        queries_by_level[2] = l2
    if l3:
        queries_by_level[3] = l3

    if not queries_by_level:
        print("ERROR: No Level 2/3 data. Run package_m2_levels.py first.")
        return

    # Setup LLM
    if not args.dry_run:
        if args.provider == "company":
            global _COMPANY_API_URL, _COMPANY_API_KEY
            _COMPANY_API_KEY = os.environ.get("COMPANY_API_KEY", "")
            _COMPANY_API_URL = os.environ.get("COMPANY_API_URL", "")
            if not _COMPANY_API_URL or not _COMPANY_API_KEY:
                print("ERROR: Set COMPANY_API_URL and COMPANY_API_KEY in .env")
                sys.exit(1)

    total_in_tok = 0
    total_out_tok = 0
    results: Dict[str, Any] = {}

    for level, queries in sorted(queries_by_level.items()):
        if args.limit > 0:
            queries = queries[:args.limit]

        level_label = {2: "dual_evidence", 3: "reasoning_chain"}.get(level, f"level_{level}")
        print(f"\n--- Level {level} ({level_label}): {len(queries)} queries ---")

        bm25_cov_sum = 0.0
        graph_cov_sum = 0.0
        bm25_qa_score_sum = 0.0
        graph_qa_score_sum = 0.0
        n_valid = 0
        n_qa = 0

        for qi, q in enumerate(queries):
            gt_ids = query_ground_truth_ids(q)
            if not gt_ids:
                continue
            n_valid += 1

            query_text = q.get("query", "")
            q_tokens = tokenize(query_text)

            # BM25 ranking
            bm25_scores = [(bm25.score(q_tokens, i), i) for i in range(len(chunks))]
            bm25_scores.sort(reverse=True)
            bm25_top_ids = [chunk_ids[idx] for _, idx in bm25_scores[:args.top_k]]

            # Graph-enhanced ranking
            graph_scores = graph_rerank(bm25_scores, chunk_ids, adjacency)
            graph_top_ids = [chunk_ids[idx] for _, idx in graph_scores[:args.top_k]]

            # Retrieval coverage (no LLM needed)
            bm25_cov = evidence_coverage(q, bm25_top_ids, chunks)
            graph_cov = evidence_coverage(q, graph_top_ids, chunks)
            bm25_cov_sum += bm25_cov
            graph_cov_sum += graph_cov

            if not args.dry_run:
                # Build context strings
                chunk_map = {c.chunk_id: c for c in chunks}
                bm25_context = "\n\n---\n\n".join(
                    f"[{cid}]\n{chunk_map[cid].text}" for cid in bm25_top_ids if cid in chunk_map
                )
                graph_context = "\n\n---\n\n".join(
                    f"[{cid}]\n{chunk_map[cid].text}" for cid in graph_top_ids if cid in chunk_map
                )

                # LLM QA
                try:
                    ans_bm25, it1, ot1 = call_qa_llm(query_text, bm25_context, args.model, args.provider)
                    total_in_tok += it1
                    total_out_tok += ot1

                    ans_graph, it2, ot2 = call_qa_llm(query_text, graph_context, args.model, args.provider)
                    total_in_tok += it2
                    total_out_tok += ot2

                    bm25_qa = answer_mentions_evidence(ans_bm25, q)
                    graph_qa = answer_mentions_evidence(ans_graph, q)
                    bm25_qa_score_sum += bm25_qa
                    graph_qa_score_sum += graph_qa
                    n_qa += 1

                    print(f"  [{qi+1}/{len(queries)}] bm25_cov={bm25_cov:.2f} graph_cov={graph_cov:.2f} "
                          f"bm25_qa={bm25_qa:.2f} graph_qa={graph_qa:.2f}")

                    if args.delay > 0:
                        time.sleep(args.delay)

                except Exception as e:
                    print(f"  [{qi+1}/{len(queries)}] LLM ERROR: {e}")
                    if "rate" in str(e).lower() or "429" in str(e):
                        time.sleep(30)
                    continue
            else:
                if (qi + 1) % 50 == 0 or qi == 0:
                    print(f"  [{qi+1}/{len(queries)}] bm25_cov={bm25_cov:.2f} graph_cov={graph_cov:.2f}")

        avg_bm25_cov = bm25_cov_sum / max(1, n_valid)
        avg_graph_cov = graph_cov_sum / max(1, n_valid)
        avg_bm25_qa = bm25_qa_score_sum / max(1, n_qa) if n_qa > 0 else None
        avg_graph_qa = graph_qa_score_sum / max(1, n_qa) if n_qa > 0 else None

        level_result = {
            "level": level,
            "label": level_label,
            "n_queries": len(queries),
            "n_valid": n_valid,
            "retrieval": {
                "bm25_avg_coverage": round(avg_bm25_cov, 4),
                "graph_avg_coverage": round(avg_graph_cov, 4),
                "delta": round(avg_graph_cov - avg_bm25_cov, 4),
            },
        }
        if avg_bm25_qa is not None:
            level_result["qa"] = {
                "n_qa_pairs": n_qa,
                "bm25_avg_evidence_mention": round(avg_bm25_qa, 4),
                "graph_avg_evidence_mention": round(avg_graph_qa, 4),
                "delta": round(avg_graph_qa - avg_bm25_qa, 4),
            }

        results[str(level)] = level_result
        print(f"\n  Level {level} Summary:")
        print(f"    Retrieval coverage: BM25={avg_bm25_cov:.4f}  Graph={avg_graph_cov:.4f}  "
              f"delta={avg_graph_cov - avg_bm25_cov:+.4f}")
        if avg_bm25_qa is not None:
            print(f"    QA evidence mention: BM25={avg_bm25_qa:.4f}  Graph={avg_graph_qa:.4f}  "
                  f"delta={avg_graph_qa - avg_bm25_qa:+.4f}")

    # Save report
    report = {
        "experiment": "C_qa_triangle",
        "model": args.model,
        "provider": args.provider,
        "top_k": args.top_k,
        "dry_run": args.dry_run,
        "total_input_tokens": total_in_tok,
        "total_output_tokens": total_out_tok,
        "levels": results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport saved to {out_path}")

    # Token logging (Iron Rule 1)
    if total_in_tok > 0 or total_out_tok > 0:
        log_run(
            script="run_exp_c_qa_triangle",
            model=f"{args.provider}:{args.model}",
            purpose=f"Exp C QA triangle — BM25 vs Graph evidence coverage across difficulty levels",
            input_tokens=total_in_tok,
            output_tokens=total_out_tok,
            extra={
                "levels_evaluated": list(results.keys()),
                "dry_run": args.dry_run,
                "output": str(out_path),
            },
        )


if __name__ == "__main__":
    main()
