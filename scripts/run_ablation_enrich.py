#!/usr/bin/env python3
"""Enrichment ablation study: 2×2 design + matched-pair subset.

两个维度：
  Axis 1 (查询侧): raw queries (keyword_boost) vs enriched queries (section_enriched)
  Axis 2 (语料侧): raw elements vs enriched elements (MoDora, data111/)

六个条件:
  1A  raw    query  + raw     corpus   ← 全局基线
  2A  enrich query  + raw     corpus   ← 已跑 (section enrichment 仅影响生成侧)
  1B  raw    query  + enriched corpus
  2B  enrich query  + enriched corpus  ← 全链路 enrichment
  1A_matched  raw    query (127 L2 + 28 L3 matched pairs)
  2A_matched  enrich query (same matched pairs)

路径约定:
  Raw elements       : data/multimodal_elements.json
  Enriched elements  : data111/multimodal_elements_enriched.json
  Raw L2/L3 queries  : data/m2/l2_production_2026-03-24_keyword_boost_pass.jsonl
                       data/m2/l3_production_2026-03-24_keyword_boost_pass.jsonl
  Enriched L2/L3     : data/m2/l2_production_2026-03-26_section_enriched_pass.jsonl
                       data/m2/l3_production_2026-03-26_section_enriched_pass.jsonl
  L1 (fixed)         : data/m2/level1_single_element.jsonl
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
EVAL_SCRIPT = ROOT / "scripts" / "run_m2_classic_eval.py"

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
RAW_ELEMENTS   = ROOT / "data" / "multimodal_elements.json"
ENRICH_ELEMENTS = ROOT / "data111" / "multimodal_elements_enriched.json"

RAW_L2   = ROOT / "data/m2/l2_production_2026-03-24_keyword_boost_pass.jsonl"
RAW_L3   = ROOT / "data/m2/l3_production_2026-03-24_keyword_boost_pass.jsonl"
ENRICH_L2 = ROOT / "data/m2/l2_production_2026-03-26_section_enriched_pass.jsonl"
ENRICH_L3 = ROOT / "data/m2/l3_production_2026-03-26_section_enriched_pass.jsonl"
L1        = ROOT / "data/m2/level1_single_element.jsonl"

HUB_CANDIDATES = ROOT / "data/m2/hub_candidates_enriched_keyword_boost_full_2026-03-24.json"
OUT_DIR = ROOT / "data/m2"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_matched_subsets(rebuild: bool = False) -> tuple[Path, Path, Path, Path]:
    """生成 raw/enriched 的 matched-pair 子集（只保留两者都有的 pair_id）。

    如果输出文件已存在且 rebuild=False，直接复用（避免 Windows 只读文件写入错误）。
    用 --rebuild-matched 强制重新生成。
    """
    m_l2r = OUT_DIR / "l2_matched_raw.jsonl"
    m_l2e = OUT_DIR / "l2_matched_enriched.jsonl"
    m_l3r = OUT_DIR / "l3_matched_raw.jsonl"
    m_l3e = OUT_DIR / "l3_matched_enriched.jsonl"

    all_exist = all(p.exists() for p in [m_l2r, m_l2e, m_l3r, m_l3e])
    if all_exist and not rebuild:
        print(f"  Matched 子集已存在，直接复用（用 --rebuild-matched 强制重建）")
        print(f"  L2 raw={_count(m_l2r)} enrich={_count(m_l2e)}  "
              f"L3 raw={_count(m_l3r)} enrich={_count(m_l3e)}")
        return m_l2r, m_l2e, m_l3r, m_l3e

    l2r = load_jsonl(RAW_L2)
    l2e = load_jsonl(ENRICH_L2)
    l3r = load_jsonl(RAW_L3)
    l3e = load_jsonl(ENRICH_L3)

    shared_l2 = {r["pair_id"] for r in l2r} & {r["pair_id"] for r in l2e}
    shared_l3 = {r["pair_id"] for r in l3r} & {r["pair_id"] for r in l3e}

    print(f"  Matched L2 pairs: {len(shared_l2)} / raw={len(l2r)} enrich={len(l2e)}")
    print(f"  Matched L3 pairs: {len(shared_l3)} / raw={len(l3r)} enrich={len(l3e)}")

    save_jsonl([r for r in l2r if r["pair_id"] in shared_l2], m_l2r)
    save_jsonl([r for r in l2e if r["pair_id"] in shared_l2], m_l2e)
    save_jsonl([r for r in l3r if r["pair_id"] in shared_l3], m_l3r)
    save_jsonl([r for r in l3e if r["pair_id"] in shared_l3], m_l3e)

    return m_l2r, m_l2e, m_l3r, m_l3e


def run_condition(
    label: str,
    elements: Path,
    q1: Path,
    q2: Path,
    output: Path,
    dry_run: bool = False,
    methods: List[str] = None,
) -> Optional[dict]:
    """运行一个 eval 条件，返回结果 dict。"""
    if methods is None:
        methods = ["bm25", "proximity", "hits", "rrf", "lm_dirichlet"]

    cmd = [
        sys.executable, str(EVAL_SCRIPT),
        "--q1", str(q1),
        "--q2", str(q2),
        "--q3", str(L1),
        "--elements", str(elements),
        "--hub-candidates", str(HUB_CANDIDATES),
        "--methods", *methods,
        "--top-k", "20",
        "--split-levels",
        "--output", str(output),
    ]

    print(f"\n{'='*60}")
    print(f"  条件: {label}")
    print(f"  elements : {elements.relative_to(ROOT)}")
    print(f"  L2 queries: {q1.relative_to(ROOT)}  ({_count(q1)} rows)")
    print(f"  L3 queries: {q2.relative_to(ROOT)}  ({_count(q2)} rows)")
    print(f"  output   : {output.relative_to(ROOT)}")
    print(f"{'='*60}")

    if dry_run:
        print("  [DRY RUN] 跳过执行")
        return None

    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"  ⚠️  条件 {label} 运行失败 (exit {result.returncode})")
        return None

    if output.exists():
        return json.loads(output.read_text(encoding="utf-8"))
    return None


def _count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for l in path.open(encoding="utf-8") if l.strip())


# ---------------------------------------------------------------------------
# Result comparison table
# ---------------------------------------------------------------------------

def extract_metric(result: dict, level: str, method: str, metric: str) -> Optional[float]:
    if result is None:
        return None
    if level == "overall":
        return result.get("metrics", {}).get(method, {}).get(metric)
    return result.get("level_metrics", {}).get(level, {}).get(method, {}).get(metric)


def print_comparison_table(results: Dict[str, dict], method: str = "bm25") -> None:
    levels    = ["l1", "l2", "l3", "overall"]
    metrics   = ["recall@10", "mrr"]
    conditions = list(results.keys())

    print(f"\n{'='*80}")
    print(f"  Enrichment Ablation — method={method.upper()}")
    print(f"{'='*80}")
    header = f"{'条件':<18}" + "".join(f"{'L1 R@10':>10}{'L1 MRR':>9}{'L2 R@10':>10}{'L2 MRR':>9}{'L3 R@10':>10}{'L3 MRR':>9}")
    print(header)
    print("-" * 80)

    for cond, res in results.items():
        if res is None:
            print(f"{cond:<18}  (无结果)")
            continue
        row = f"{cond:<18}"
        for lv in ["l1", "l2", "l3"]:
            r10 = extract_metric(res, lv, method, "recall@10")
            mrr = extract_metric(res, lv, method, "mrr")
            row += f"{r10:>10.4f}" if r10 is not None else f"{'—':>10}"
            row += f"{mrr:>9.4f}"  if mrr is not None else f"{'—':>9}"
        print(row)

    print(f"\n  注：L1 在所有条件下使用相同的 974 条查询（固定锚点）")
    print(f"      1A_matched / 2A_matched 只含 overlapping pair_ids，样本数更小但对比更纯净\n")


def save_summary(results: Dict[str, dict], output: Path, methods: List[str]) -> None:
    """把所有条件的关键指标汇总保存为一个 JSON。"""
    summary = {}
    for cond, res in results.items():
        if res is None:
            summary[cond] = None
            continue
        summary[cond] = {}
        for method in methods:
            summary[cond][method] = {}
            for level in ["l1", "l2", "l3", "overall"]:
                r10 = extract_metric(res, level, method, "recall@10")
                mrr = extract_metric(res, level, method, "mrr")
                ndcg = extract_metric(res, level, method, "ndcg@10")
                summary[cond][method][level] = {
                    "recall@10": r10, "mrr": mrr, "ndcg@10": ndcg
                }
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n汇总结果已保存: {output.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Enrichment ablation study (2x2 + matched)")
    parser.add_argument("--conditions", nargs="+",
                        default=["1A", "2A", "1B", "2B", "1A_matched", "2A_matched"],
                        help="要运行的条件列表 (默认全部)")
    parser.add_argument("--methods", nargs="+",
                        default=["bm25", "proximity", "hits", "rrf", "lm_dirichlet"],
                        help="检索方法")
    parser.add_argument("--dry-run", action="store_true", help="只打印命令不执行")
    parser.add_argument("--skip-existing", action="store_true",
                        help="如果输出文件已存在则跳过该条件")
    parser.add_argument("--rebuild-matched", action="store_true",
                        help="强制重新生成 matched-pair 子集（默认复用已有文件）")
    args = parser.parse_args()

    # --- 前置检查 ---
    missing = []
    for label, path in [
        ("Raw elements",    RAW_ELEMENTS),
        ("Enriched elements", ENRICH_ELEMENTS),
        ("Raw L2",   RAW_L2),   ("Raw L3",   RAW_L3),
        ("Enrich L2", ENRICH_L2), ("Enrich L3", ENRICH_L3),
        ("L1",        L1),
        ("Hub candidates", HUB_CANDIDATES),
    ]:
        if not path.exists():
            missing.append(f"  ❌ {label}: {path.relative_to(ROOT)}")
    if missing:
        print("缺失文件：")
        for m in missing:
            print(m)
        print("\n请确认路径后重试。")
        sys.exit(1)

    print("✅ 所有输入文件就绪")

    # --- 生成 matched 子集 ---
    need_matched = any(c in args.conditions for c in ["1A_matched", "2A_matched"])
    if need_matched:
        print("\n生成 matched-pair 子集...")
        m_l2r, m_l2e, m_l3r, m_l3e = build_matched_subsets(rebuild=args.rebuild_matched)
    else:
        m_l2r = m_l2e = m_l3r = m_l3e = None

    # --- 条件定义 ---
    condition_defs = {
        "1A": dict(elements=RAW_ELEMENTS,    q1=RAW_L2,   q2=RAW_L3,
                   output=OUT_DIR / "ablation_1A_raw_query_raw_corpus.json"),
        "2A": dict(elements=RAW_ELEMENTS,    q1=ENRICH_L2, q2=ENRICH_L3,
                   output=OUT_DIR / "ablation_2A_enrich_query_raw_corpus.json"),
        "1B": dict(elements=ENRICH_ELEMENTS, q1=RAW_L2,   q2=RAW_L3,
                   output=OUT_DIR / "ablation_1B_raw_query_enrich_corpus.json"),
        "2B": dict(elements=ENRICH_ELEMENTS, q1=ENRICH_L2, q2=ENRICH_L3,
                   output=OUT_DIR / "ablation_2B_enrich_query_enrich_corpus.json"),
        "1A_matched": dict(elements=RAW_ELEMENTS, q1=m_l2r, q2=m_l3r,
                           output=OUT_DIR / "ablation_1A_matched.json"),
        "2A_matched": dict(elements=RAW_ELEMENTS, q1=m_l2e, q2=m_l3e,
                           output=OUT_DIR / "ablation_2A_matched.json"),
    }

    # --- 逐条件运行 ---
    results: Dict[str, dict] = {}
    for cond in args.conditions:
        if cond not in condition_defs:
            print(f"未知条件: {cond}，跳过")
            continue
        defn = condition_defs[cond]

        # matched 条件需要子集文件已生成
        if defn["q1"] is None or defn["q2"] is None:
            print(f"条件 {cond} 的 matched 子集未生成，跳过")
            results[cond] = None
            continue

        out = defn["output"]
        if args.skip_existing and out.exists():
            print(f"\n条件 {cond}: 输出已存在，加载缓存结果")
            results[cond] = json.loads(out.read_text(encoding="utf-8"))
            continue

        results[cond] = run_condition(
            label=cond,
            elements=defn["elements"],
            q1=defn["q1"],
            q2=defn["q2"],
            output=out,
            dry_run=args.dry_run,
            methods=args.methods,
        )

    if args.dry_run:
        print("\n[DRY RUN 结束] 以上为实际执行时的命令预览。")
        return

    # --- 打印对比表格 ---
    finished = {k: v for k, v in results.items() if v is not None}
    if finished:
        for method in ["bm25", "hits"]:
            if method in args.methods:
                print_comparison_table(finished, method=method)

        # 保存汇总
        summary_path = OUT_DIR / "ablation_enrich_summary.json"
        save_summary(results, summary_path, args.methods)
    else:
        print("\n没有可用的结果（全部跳过或失败）。")


if __name__ == "__main__":
    main()
