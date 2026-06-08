#!/usr/bin/env python3
"""
Run QC (regex + LLM) on generated Huawei L1 queries.

QC Pipeline:
  1. Regex QC (qc_real_user_query) — 12+ rule-based checks
  2. LLM QC (run_llm_qc) — ablation test + grounding check

Usage:
  python scripts/qc_huawei_queries.py \
    --queries data/03_queries/huawei_l1_queries_v2.jsonl \
    --candidates data/03_queries/huawei_rich_enriched_candidates.json \
    --output data/06_audit/huawei_qc_report.json \
    --run-llm-qc
"""

import argparse, json, os, sys, time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.qc.pipelines import qc_real_user_query
from src.qc.llm_judge import run_llm_qc

# ── QC thresholds ──
# Hard-fail issues: if any of these appear, the query FAILS
HARD_FAIL_ISSUES = {
    "empty_query",
    "meta_language",
    "single_element_answer",
    "text_evidence_over_reliance",
}

# Soft-fail issues: if only these appear, the query WARNS
SOFT_FAIL_ISSUES = {
    "yes_no_question",
    "query_too_short",
    "query_too_long",
    "lazy_query_too_long",
}


def run_regex_qc(
    queries: list[dict],
    pairs_by_id: dict[str, dict],
) -> List[dict]:
    """Run regex-based QC on all queries, return per-query results."""
    results = []
    for i, q in enumerate(queries):
        pair_id = q.get("pair_id", "")
        pair = pairs_by_id.get(pair_id, {})

        issues, metrics = qc_real_user_query(q, pair)

        # Determine verdict
        hard_hits = HARD_FAIL_ISSUES & set(issues)
        soft_hits = SOFT_FAIL_ISSUES & set(issues)
        if hard_hits:
            verdict = "FAIL"
        elif soft_hits:
            verdict = "WARN"
        else:
            verdict = "PASS"

        results.append({
            "query_idx": i,
            "pair_id": pair_id,
            "query": q.get("query", "")[:120],
            "verdict": verdict,
            "issues": issues,
            "metrics": metrics,
        })

    return results


def run_llm_qc(
    queries: list[dict],
    pairs_by_id: dict[str, dict],
    model: str = "gpt-5.4",
    limit: int = 0,
    delay: float = 0.5,
    dry_run: bool = False,
) -> List[dict]:
    """Run LLM-based ablation + grounding QC."""
    from src.api import set_company_credentials
    api_url = os.environ.get("COMPANY_API_URL", "")
    api_key = os.environ.get("COMPANY_API_KEY", "")
    set_company_credentials(url=api_url, key=api_key)

    # Prepare query objects with pair metadata
    targets = []
    for i, q in enumerate(queries):
        pair_id = q.get("pair_id", "")
        pair = pairs_by_id.get(pair_id, {})
        targets.append({
            **q,
            "pair": pair,
            "idx": i,
        })

    if limit > 0:
        targets = targets[:limit]

    if dry_run:
        print(f"  [DRY RUN] Would run LLM QC on {len(targets)} queries")
        return []

    llm_results = []
    for i, t in enumerate(targets):
        try:
            result = run_llm_qc(
                query_obj=t,
                pair=t.get("pair", {}),
                provider="company",
                model=model,
                delay=delay,
            )
            llm_results.append({
                "query_idx": t["idx"],
                "pair_id": t.get("pair_id", ""),
                "ablation_pass": result.get("ablation_pass", False),
                "grounding_pass": result.get("grounding_pass", False),
                "llm_verdict": "PASS" if result.get("ablation_pass") and result.get("grounding_pass") else "FAIL",
                "llm_details": result,
            })
            print(f"  [{i+1}/{len(targets)}] LLM QC: "
                  f"ablation={'PASS' if result.get('ablation_pass') else 'FAIL'}, "
                  f"grounding={'PASS' if result.get('grounding_pass') else 'FAIL'}")
        except Exception as e:
            llm_results.append({
                "query_idx": t["idx"],
                "pair_id": t.get("pair_id", ""),
                "llm_error": str(e)[:100],
            })
            print(f"  [{i+1}/{len(targets)}] LLM QC ERROR: {e}")
        time.sleep(delay)

    return llm_results


def main():
    ap = argparse.ArgumentParser(description="QC for Huawei L1 queries")
    ap.add_argument("--queries", default="data/03_queries/huawei_l1_queries_v2.jsonl")
    ap.add_argument("--candidates", default="data/03_queries/huawei_rich_enriched_candidates.json")
    ap.add_argument("--output", default="data/06_audit/huawei_qc_report.json")
    ap.add_argument("--run-llm-qc", action="store_true", help="Run LLM-based QC (slower, costs tokens)")
    ap.add_argument("--llm-limit", type=int, default=0, help="Limit LLM QC to N queries")
    ap.add_argument("--delay", type=float, default=0.5)
    args = ap.parse_args()

    # Load data
    queries = []
    with open(args.queries, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))

    candidates = json.loads(Path(args.candidates).read_text(encoding="utf-8"))
    pairs_by_id = {p.get("pair_id", ""): p for p in candidates.get("pairs", [])}

    print(f"QC Pipeline")
    print(f"  Queries: {len(queries)}")
    print(f"  Candidates: {len(pairs_by_id)} pairs")
    print(f"  LLM QC: {'enabled' if args.run_llm_qc else 'disabled'}")

    # ── Phase 1: Regex QC ──
    print(f"\n{'='*50}")
    print(f"  PHASE 1: Regex QC (qc_real_user_query)")
    print(f"{'='*50}")
    regex_results = run_regex_qc(queries, pairs_by_id)

    verdicts = defaultdict(int)
    issue_counts = defaultdict(int)
    for r in regex_results:
        verdicts[r["verdict"]] += 1
        for issue in r["issues"]:
            issue_counts[issue] += 1

    print(f"\n  Verdicts: {dict(verdicts)}")
    print(f"  Issues:   {dict(issue_counts)}")

    for r in regex_results:
        if r["verdict"] != "PASS":
            print(f"  [{r['verdict']}] {r['query'][:80]}...  issues={r['issues']}")

    # ── Phase 2: LLM QC ──
    llm_results = []
    if args.run_llm_qc:
        print(f"\n{'='*50}")
        print(f"  PHASE 2: LLM QC (ablation + grounding)")
        print(f"{'='*50}")

        # Import here to avoid loading unless needed
        import openai
        llm_results = run_llm_qc(
            queries, pairs_by_id,
            model="gpt-5.4",
            limit=args.llm_limit,
            delay=args.delay,
        )

        llm_verdicts = defaultdict(int)
        for r in llm_results:
            if "llm_error" in r:
                llm_verdicts["ERROR"] += 1
            else:
                llm_verdicts[r.get("llm_verdict", "UNKNOWN")] += 1
        print(f"\n  LLM Verdicts: {dict(llm_verdicts)}")

    # ── Save report ──
    report = {
        "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "total_queries": len(queries),
        "regex_qc": {
            "verdicts": dict(verdicts),
            "issue_counts": dict(issue_counts),
            "per_query": regex_results,
        },
        "llm_qc": {
            "results": llm_results,
        } if llm_results else None,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"\n  QC Report: {out_path}")
    print(f"  Pass: {verdicts.get('PASS', 0)}, Warn: {verdicts.get('WARN', 0)}, "
          f"Fail: {verdicts.get('FAIL', 0)}")


if __name__ == "__main__":
    main()
