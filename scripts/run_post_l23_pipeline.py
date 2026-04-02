#!/usr/bin/env python3
"""Detached end-to-end L2/L3 production runner.

This task is intended to survive terminal / chat disconnects and finish the
full workflow autonomously:

1. Wait for any existing L2/L3 generation jobs to exit.
2. Regenerate baseline L2/L3 with the section-aware topology materials.
3. Run baseline retrieval / packaging / Exp-A checks.
4. Run section/subsection API enrichment.
5. Regenerate L2/L3 with section-level semantic context.
6. Re-run retrieval / packaging / Exp-A checks and compare against baseline.
7. Re-run C-Pool proxy retrieval evaluation on the current graph materials.
8. Update the external weekly report + code walkthrough files.
9. Commit selected repo artifacts and push to origin/main.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA111 = ROOT / "data111"
M2 = DATA / "m2"
DATE_TAG = os.environ.get("DATE_TAG", "2026-03-24")

_EXTERNAL_BASE = Path(os.environ.get("EXTERNAL_REPORT_DIR", "/projects/_hdd/myyyx1"))
WEEKLY_REPORT = _EXTERNAL_BASE / f"weekly_report_{DATE_TAG}.md"
CODE_WALKTHROUGH = _EXTERNAL_BASE / f"code_walkthrough_{DATE_TAG}.md"


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print(f"[{ts} UTC] {msg}", flush=True)


def load_dotenv(dotenv_path: Path, override_empty: bool = True) -> Dict[str, str]:
    """Load .env into os.environ and return the parsed key/value map."""
    loaded: Dict[str, str] = {}
    if not dotenv_path.exists():
        return loaded
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        loaded[key] = value
        if key not in os.environ or (override_empty and not os.environ.get(key)):
            os.environ[key] = value
    return loaded


def build_child_env() -> Dict[str, str]:
    env = os.environ.copy()
    required = ("COMPANY_API_URL", "COMPANY_API_KEY")
    missing = [k for k in required if not env.get(k)]
    if missing:
        raise SystemExit(f"Missing required env vars after .env load: {missing}")
    return env


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def wait_for_pids(pids: Iterable[int], poll_sec: int = 20) -> None:
    alive = [pid for pid in pids if pid_alive(pid)]
    if not alive:
        log("No live baseline generation PIDs found; continuing immediately.")
        return
    log(f"Waiting for baseline generation PIDs to finish: {alive}")
    while True:
        alive = [pid for pid in pids if pid_alive(pid)]
        if not alive:
            break
        log(f"Still running: {alive}")
        time.sleep(poll_sec)
    log("Baseline generation PIDs exited.")


def run_cmd(cmd: List[str], *, cwd: Path = ROOT, env: Optional[Dict[str, str]] = None) -> None:
    log("RUN: " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def file_has_content(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"Wrote JSON: {path}")


def fmt_num(v: Any, digits: int = 4) -> str:
    if isinstance(v, (int, float)):
        return f"{v:.{digits}f}"
    return str(v)


def run_generation(
    *,
    candidates: Path,
    output: Path,
    query_style: str,
    reference_graph: Path,
    topology_candidates: Path,
    env: Dict[str, str],
    section_enrich: Optional[Path] = None,
) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "generate_multihop_l1_queries.py"),
        "--candidates", str(candidates),
        "--output", str(output),
        "--pass-only",
        "--provider", "company",
        "--model", "gpt-5.4",
        "--topology-candidates", str(topology_candidates),
        "--reference-graph", str(reference_graph),
        "--delay", "0.5",
        "--query-style", query_style,
    ]
    if section_enrich is not None:
        cmd.extend(["--section-enrich", str(section_enrich)])
    run_cmd(cmd, cwd=ROOT, env=env)


def run_package(env: Dict[str, str]) -> None:
    run_cmd([sys.executable, str(ROOT / "scripts" / "package_m2_levels.py")], cwd=ROOT, env=env)


def run_exp_a(tag: str, env: Dict[str, str]) -> Path:
    out = M2 / f"exp_a_difficulty_{tag}.json"
    run_cmd(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_exp_a_difficulty.py"),
            "--enriched-elements", str(DATA111 / "multimodal_elements_enriched.json"),
            "--output", str(out),
        ],
        cwd=ROOT,
        env=env,
    )
    return out


def run_phase0_eval(
    *,
    q1: Path,
    q2: Path,
    output: Path,
    hub_candidates: Path,
    hubs: Path,
    env: Dict[str, str],
    q3: Optional[Path] = None,
) -> Path:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_phase0_eval_ab.py"),
        "--q1", str(q1),
        "--q2", str(q2),
        "--elements", str(DATA111 / "multimodal_elements_enriched.json"),
        "--hubs", str(hubs),
        "--hub-candidates", str(hub_candidates),
        "--citation-graph", str(DATA / "citation_graph.json"),
        "--output", str(output),
        "--hub-weight", "0.15",
        "--nprop-weight", "0.20",
        "--cite-weight", "0",
        "--neighbor-hops", "1",
    ]
    if q3 is not None:
        cmd.extend(["--q3", str(q3)])
    run_cmd(cmd, cwd=ROOT, env=env)
    return output


def run_section_enrich(reference_graph: Path, output: Path, env: Dict[str, str]) -> None:
    run_cmd(
        [
            sys.executable,
            str(ROOT / "scripts" / "enrich_section_nodes.py"),
            "--reference-graph", str(reference_graph),
            "--output", str(output),
            "--provider", "company",
            "--model", "gpt-5.4",
            "--delay", "0.5",
        ],
        cwd=ROOT,
        env=env,
    )


def build_cpool_proxy(output: Path, summary: Path, env: Dict[str, str]) -> None:
    run_cmd(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_cpool_proxy_queries.py"),
            "--input", str(DATA / "c_pool_universal_queries.json"),
            "--elements", str(DATA111 / "multimodal_elements_enriched.json"),
            "--output", str(output),
            "--summary", str(summary),
        ],
        cwd=ROOT,
        env=env,
    )


def extract_phase0_summary(path: Path) -> Dict[str, Any]:
    obj = read_json(path)
    cfg = obj.get("config", {}) or {}
    metrics = obj.get("metrics", {}) or {}
    out: Dict[str, Any] = {
        "report_path": str(path),
        "query_count": cfg.get("n_queries"),
        "chunk_count": cfg.get("n_chunks"),
        "decision": obj.get("decision", {}),
        "metrics": {},
    }
    for name in (
        "bm25",
        "graph_hub_rerank",
        "graph_neighbor_prop",
        "graph_ppr",
        "graph_citation_walk",
        "graph_full",
    ):
        row = metrics.get(name, {}) or {}
        out["metrics"][name] = {
            "n": row.get("n"),
            "recall_at_10": row.get("recall_at_10"),
            "mrr": row.get("mrr"),
        }
    return out


def extract_exp_a_summary(path: Path) -> Dict[str, Any]:
    obj = read_json(path)
    levels = obj.get("levels", {}) or {}
    return {
        "report_path": str(path),
        "chunk_count": obj.get("chunk_count"),
        "gradient_confirmed": obj.get("gradient_confirmed"),
        "gradient_metric": obj.get("gradient_metric"),
        "levels": {
            key: {
                "n_queries": value.get("n_queries"),
                "recall_at_k": value.get("recall_at_k"),
                "mrr": value.get("mrr"),
                "avg_evidence_coverage": value.get("avg_evidence_coverage"),
            }
            for key, value in levels.items()
        },
    }


def collect_dataset_counts(
    *,
    l2_full: Path,
    l2_pass: Path,
    l3_full: Path,
    l3_pass: Path,
    section_enrich: Optional[Path] = None,
) -> Dict[str, Any]:
    payload = {
        "l2_full_count": count_jsonl(l2_full),
        "l2_pass_count": count_jsonl(l2_pass),
        "l3_full_count": count_jsonl(l3_full),
        "l3_pass_count": count_jsonl(l3_pass),
        "level2_count": count_jsonl(M2 / "level2_dual_evidence.jsonl"),
        "level3_count": count_jsonl(M2 / "level3_reasoning_chain.jsonl"),
    }
    if section_enrich is not None and section_enrich.exists():
        meta = read_json(section_enrich).get("metadata", {}) or {}
        payload["section_enrich"] = {
            "path": str(section_enrich),
            "processed": meta.get("processed"),
            "failed": meta.get("failed"),
            "total_requested": meta.get("total_requested"),
            "input_tokens": meta.get("input_tokens"),
            "output_tokens": meta.get("output_tokens"),
        }
    return payload


def build_delta(section_summary: Dict[str, Any], baseline_summary: Dict[str, Any]) -> Dict[str, Any]:
    delta: Dict[str, Any] = {}
    for key in ("l2_pass_count", "l3_pass_count", "level2_count", "level3_count"):
        delta[key] = section_summary["counts"][key] - baseline_summary["counts"][key]

    for method in ("bm25", "graph_neighbor_prop", "graph_full"):
        s_metrics = section_summary["phase0_eval"]["metrics"].get(method, {})
        b_metrics = baseline_summary["phase0_eval"]["metrics"].get(method, {})
        delta[f"{method}_mrr"] = round(
            float(s_metrics.get("mrr") or 0.0) - float(b_metrics.get("mrr") or 0.0), 4
        )
        delta[f"{method}_recall_at_10"] = round(
            float(s_metrics.get("recall_at_10") or 0.0) - float(b_metrics.get("recall_at_10") or 0.0), 4
        )
    return delta


def render_weekly_report(compare: Dict[str, Any]) -> str:
    baseline = compare["baseline"]
    section = compare["section_enriched"]
    cpool = compare["cpool"]
    delta = compare["delta"]

    b_full = baseline["phase0_eval"]["metrics"]["graph_full"]
    s_full = section["phase0_eval"]["metrics"]["graph_full"]
    c_full = cpool["phase0_eval"]["metrics"]["graph_full"]
    c_bm25 = cpool["phase0_eval"]["metrics"]["bm25"]

    lines = [
        "# Weekly Report — Document Graph for Document Understanding",
        "",
        "**Week**: 2026-03-17 → 2026-03-24 | **Author**: Yao",
        "",
        "---",
        "",
        "## 本周完成",
        "",
        f"- **Section-aware 图重建完成**：82/82 文档恢复 section/paragraph metadata，共 1417 个 section、8650 个 paragraph；新的 graph / hubs / candidates 已单独落到 `data/latex_sections_rebuild_{DATE_TAG}`",
        "- **全局关键词节点强化完成**：`introduction / summary / architecture / structure` 相关 section 与 paragraph 会进入 keyword boost 排序，已用于新的 hub 与 candidate 生成",
        f"- **新候选池重建完成**：section-aware + keyword boost 重新 enrich 后得到 295 对可用 pairs，其中 L2 217 对、L3 78 对",
        f"- **L2/L3 baseline 全量重跑完成**：L2 ` {baseline['counts']['l2_pass_count']} / {baseline['counts']['l2_full_count']} ` pass，L3 ` {baseline['counts']['l3_pass_count']} / {baseline['counts']['l3_full_count']} ` pass；Graph Full MRR ` {fmt_num(b_full.get('mrr'))} `",
        f"- **section/subsection API enrich + rerun 完成**：section enrich ` {section['counts'].get('section_enrich', {}).get('processed', 0)} ` 个节点，L2/L3 rerun 后 Graph Full MRR ` {fmt_num(s_full.get('mrr'))} `，相对 baseline ` {delta['graph_full_mrr']:+.4f} `",
        f"- **Query 能力测试补上 C-Pool**：proxy 支持 ` {cpool['proxy_summary']['supported_queries']} / {cpool['proxy_summary']['total_queries']} ` 条通用 query；graph_full MRR ` {fmt_num(c_full.get('mrr'))} `，相对 BM25 ` {(float(c_full.get('mrr') or 0.0) - float(c_bm25.get('mrr') or 0.0)):+.4f} `",
        "",
        "## 需要帮助",
        "",
        "- **下一轮重点应放在 section-level retrieval 接入**：当前 section enrich 已进入 query prompt，但检索 corpus 仍是 element 级，通用 query 的上限还被 section 缺失卡住",
        "- **如果继续放大 C-Pool，需要补 section chunk / section node 入检索**：单纯调 graph 权重收益有限，重点要转到 section-aware retrieval",
        "",
        "## 下周计划",
        "",
        "| 交付物 | 标准 |",
        "|--------|------|",
        "| L2/L3 最终口径冻结 | baseline vs section-enriched 对比表 + 推荐版本 |",
        "| section-aware retrieval | section 节点进入检索分数链路，而不只停在 query 生成 |",
        "| 通用 query 扩容 | C-Pool 从 43 条补足到 50-100 条种子库 |",
        "| 图检索机制汇报 | 用数学/伪代码讲清 rerank / propagation / scoring |",
    ]
    return "\n".join(lines) + "\n"


def render_code_walkthrough(compare: Dict[str, Any]) -> str:
    baseline = compare["baseline"]
    section = compare["section_enriched"]
    cpool = compare["cpool"]
    delta = compare["delta"]

    b_full = baseline["phase0_eval"]["metrics"]["graph_full"]
    s_full = section["phase0_eval"]["metrics"]["graph_full"]
    b_np = baseline["phase0_eval"]["metrics"]["graph_neighbor_prop"]
    s_np = section["phase0_eval"]["metrics"]["graph_neighbor_prop"]

    sec_meta = section["counts"].get("section_enrich", {})

    lines = [
        "# 活跃代码逐块讲解 — 阅后即焚",
        "",
        "> 周会备战材料 | 2026-03-24",
        "",
        "---",
        "",
        "## 这轮真正新增了什么",
        "",
        "| 模块 | 文件 | 作用 |",
        "|------|------|------|",
        "| Section-aware 图重建 | `scripts/analyze_latex_graph_topology.py` | 真正把 `section/subsection/subsubsection` 建进 topology，并加上 keyword boost |",
        "| 新候选重映射 | `scripts/enrich_hub_candidates.py` + `data/m2/hub_candidates_enriched_keyword_boost_full_2026-03-24.json` | 把新 graph candidates 映射回 MinerU element，得到新的 L2/L3 池 |",
        "| L3 prompt 增强 | `scripts/generate_multihop_l1_queries.py` | 注入真实 bridge text、graph path、bridge quality，并支持 `--section-enrich` |",
        "| Section API enrich | `scripts/enrich_section_nodes.py` | 用 company provider 对 section/subsection 做语义总结，并保留 token log |",
        "| 断线可续跑任务 | `scripts/run_post_l23_pipeline.py` | `L2/L3 -> eval -> section enrich -> rerun -> compare -> 更新汇报 -> git push` 一条龙后台任务 |",
        "",
        "## 数据流口径",
        "",
        "```",
        "LaTeX source rebuild",
        "  -> latex_reference_graph.json",
        "  -> latex_graph_hubs_keyword_boost.json",
        "  -> latex_hub_multihop_candidates_keyword_boost.json",
        "  -> hub_candidates_enriched_keyword_boost_full_2026-03-24.json",
        "  -> l2/l3_candidates_keyword_boost_2026-03-24.json",
        "  -> generate_multihop_l1_queries.py (baseline)",
        "  -> run_phase0_eval_ab.py + package_m2_levels.py + run_exp_a_difficulty.py",
        "  -> enrich_section_nodes.py",
        "  -> generate_multihop_l1_queries.py --section-enrich ... (rerun)",
        "  -> compare baseline vs section-enriched",
        "```",
        "",
        "## 明天汇报时最值得讲的点",
        "",
        f"- **baseline 产量**：L2 ` {baseline['counts']['l2_pass_count']} / {baseline['counts']['l2_full_count']} ` pass，L3 ` {baseline['counts']['l3_pass_count']} / {baseline['counts']['l3_full_count']} ` pass。",
        f"- **section enrich 规模**：` {sec_meta.get('processed', 0)} ` processed，` {sec_meta.get('failed', 0)} ` failed，token ` {sec_meta.get('input_tokens', 0)} in / {sec_meta.get('output_tokens', 0)} out `。",
        f"- **核心对比**：Graph Full MRR ` {fmt_num(b_full.get('mrr'))} -> {fmt_num(s_full.get('mrr'))} `，delta ` {delta['graph_full_mrr']:+.4f} `。",
        f"- **neighbor propagation 对比**：` {fmt_num(b_np.get('mrr'))} -> {fmt_num(s_np.get('mrr'))} `，delta ` {delta['graph_neighbor_prop_mrr']:+.4f} `。",
        f"- **C-Pool 结论**：proxy 支持 ` {cpool['proxy_summary']['supported_queries']} / {cpool['proxy_summary']['total_queries']} `，graph_full MRR ` {fmt_num(cpool['phase0_eval']['metrics']['graph_full'].get('mrr'))} `。",
        "",
        "## 如果导师追问“图怎么影响分数”",
        "",
        "1. 先用 BM25 给每个 element chunk 一个初始分。",
        "2. hub prior 给桥接元素一个静态加分。",
        "3. neighbor propagation 把高分 chunk 的分数沿 element adjacency 传播到邻居。",
        "4. graph_full 把 `bm25 + hub_boost + neighbor_prop (+ citation_walk)` 线性组合后再排序。",
        "5. 现在 section enrich 影响的是 query prompt；下一步要做的是让 section 节点直接进入 chunk scoring / rerank。",
        "",
        "## 一句话结论",
        "",
        "Section-aware graph 已经真正进到候选与生成链，section enrich 也接上了 API + 日志，但通用 query 的下一阶段重点不再是多造 prompt，而是把 section 节点直接接进 retrieval score。",  # noqa: E501
    ]
    return "\n".join(lines) + "\n"


def update_external_reports(compare: Dict[str, Any]) -> None:
    WEEKLY_REPORT.write_text(render_weekly_report(compare), encoding="utf-8")
    CODE_WALKTHROUGH.write_text(render_code_walkthrough(compare), encoding="utf-8")
    log(f"Updated weekly report: {WEEKLY_REPORT}")
    log(f"Updated walkthrough: {CODE_WALKTHROUGH}")


def git_stage_commit_push(env: Dict[str, str]) -> None:
    stage_paths = [
        ROOT / "CLAUDE.md",
        ROOT / "scripts" / "analyze_latex_graph_topology.py",
        ROOT / "scripts" / "generate_multihop_l1_queries.py",
        ROOT / "scripts" / "build_cpool_proxy_queries.py",
        ROOT / "scripts" / "enrich_section_nodes.py",
        ROOT / "scripts" / "run_post_l23_pipeline.py",
        ROOT / "data" / f"latex_sections_rebuild_{DATE_TAG}",
        M2 / f"hub_candidates_enriched_keyword_boost_full_{DATE_TAG}.json",
        M2 / f"l2_candidates_keyword_boost_{DATE_TAG}.json",
        M2 / f"l3_candidates_keyword_boost_{DATE_TAG}.json",
        M2 / f"l2_production_{DATE_TAG}_keyword_boost.jsonl",
        M2 / f"l2_production_{DATE_TAG}_keyword_boost_pass.jsonl",
        M2 / f"l3_production_{DATE_TAG}_keyword_boost.jsonl",
        M2 / f"l3_production_{DATE_TAG}_keyword_boost_pass.jsonl",
        M2 / f"section_nodes_enriched_{DATE_TAG}.json",
        M2 / f"l2_production_{DATE_TAG}_section_enriched.jsonl",
        M2 / f"l2_production_{DATE_TAG}_section_enriched_pass.jsonl",
        M2 / f"l3_production_{DATE_TAG}_section_enriched.jsonl",
        M2 / f"l3_production_{DATE_TAG}_section_enriched_pass.jsonl",
        M2 / f"_eval_l23_baseline_keyword_boost_{DATE_TAG}.json",
        M2 / f"_eval_l23_section_enriched_{DATE_TAG}.json",
        M2 / f"exp_a_difficulty_{DATE_TAG}_keyword_boost_baseline.json",
        M2 / f"exp_a_difficulty_{DATE_TAG}_section_enriched.json",
        M2 / f"c_pool_universal_queries_proxy_keyword_boost_{DATE_TAG}.jsonl",
        M2 / f"c_pool_universal_queries_proxy_keyword_boost_summary_{DATE_TAG}.json",
        M2 / f"_eval_cpool_proxy_keyword_boost_{DATE_TAG}.json",
        M2 / f"post_l23_section_compare_{DATE_TAG}.json",
    ]
    existing = [str(p.relative_to(ROOT)) for p in stage_paths if p.exists()]
    if not existing:
        log("No staged paths exist; skipping git commit/push.")
        return

    run_cmd(["git", "add", "--"] + existing, cwd=ROOT, env=env)

    status = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=str(ROOT),
        env=env,
        check=False,
    )
    if status.returncode == 0:
        log("No staged changes after git add; skipping commit/push.")
        return

    commit_msg = "Add section-aware L2/L3 rerun, section enrich comparison, and C-pool eval"
    run_cmd(["git", "commit", "-m", commit_msg], cwd=ROOT, env=env)
    run_cmd(["git", "push", "origin", "main"], cwd=ROOT, env=env)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run detached end-to-end L2/L3 pipeline")
    ap.add_argument("--l2-pid", type=int, default=0)
    ap.add_argument("--l3-pid", type=int, default=0)
    args = ap.parse_args()

    load_dotenv(ROOT / ".env", override_empty=True)
    env = build_child_env()

    reference_graph = DATA / f"latex_sections_rebuild_{DATE_TAG}" / "latex_reference_graph.json"
    topology_candidates = DATA / f"latex_sections_rebuild_{DATE_TAG}" / "latex_hub_multihop_candidates_keyword_boost.json"
    hubs = DATA / f"latex_sections_rebuild_{DATE_TAG}" / "latex_graph_hubs_keyword_boost.json"
    hub_candidates = M2 / f"hub_candidates_enriched_keyword_boost_full_{DATE_TAG}.json"
    l2_candidates = M2 / f"l2_candidates_keyword_boost_{DATE_TAG}.json"
    l3_candidates = M2 / f"l3_candidates_keyword_boost_{DATE_TAG}.json"

    baseline_l2 = M2 / f"l2_production_{DATE_TAG}_keyword_boost.jsonl"
    baseline_l2_pass = M2 / f"l2_production_{DATE_TAG}_keyword_boost_pass.jsonl"
    baseline_l3 = M2 / f"l3_production_{DATE_TAG}_keyword_boost.jsonl"
    baseline_l3_pass = M2 / f"l3_production_{DATE_TAG}_keyword_boost_pass.jsonl"

    section_enrich_out = M2 / f"section_nodes_enriched_{DATE_TAG}.json"
    section_l2 = M2 / f"l2_production_{DATE_TAG}_section_enriched.jsonl"
    section_l2_pass = M2 / f"l2_production_{DATE_TAG}_section_enriched_pass.jsonl"
    section_l3 = M2 / f"l3_production_{DATE_TAG}_section_enriched.jsonl"
    section_l3_pass = M2 / f"l3_production_{DATE_TAG}_section_enriched_pass.jsonl"

    baseline_eval_out = M2 / f"_eval_l23_baseline_keyword_boost_{DATE_TAG}.json"
    section_eval_out = M2 / f"_eval_l23_section_enriched_{DATE_TAG}.json"
    baseline_exp_a_out = M2 / f"exp_a_difficulty_{DATE_TAG}_keyword_boost_baseline.json"
    section_exp_a_out = M2 / f"exp_a_difficulty_{DATE_TAG}_section_enriched.json"

    cpool_proxy = M2 / f"c_pool_universal_queries_proxy_keyword_boost_{DATE_TAG}.jsonl"
    cpool_proxy_summary = M2 / f"c_pool_universal_queries_proxy_keyword_boost_summary_{DATE_TAG}.json"
    cpool_eval_out = M2 / f"_eval_cpool_proxy_keyword_boost_{DATE_TAG}.json"

    compare_out = M2 / f"post_l23_section_compare_{DATE_TAG}.json"

    wait_for_pids([args.l2_pid, args.l3_pid])

    log("Starting baseline L2 generation.")
    run_generation(
        candidates=l2_candidates,
        output=baseline_l2,
        query_style="mixed",
        reference_graph=reference_graph,
        topology_candidates=topology_candidates,
        env=env,
    )
    log("Starting baseline L3 generation.")
    run_generation(
        candidates=l3_candidates,
        output=baseline_l3,
        query_style="academic",
        reference_graph=reference_graph,
        topology_candidates=topology_candidates,
        env=env,
    )

    run_phase0_eval(
        q1=baseline_l2_pass,
        q2=baseline_l3_pass,
        output=baseline_eval_out,
        hub_candidates=hub_candidates,
        hubs=hubs,
        env=env,
    )
    run_package(env)
    baseline_exp_a_out = run_exp_a(f"{DATE_TAG}_keyword_boost_baseline", env)
    baseline_counts = collect_dataset_counts(
        l2_full=baseline_l2,
        l2_pass=baseline_l2_pass,
        l3_full=baseline_l3,
        l3_pass=baseline_l3_pass,
    )

    log("Starting section/subsection API enrich.")
    run_section_enrich(reference_graph=reference_graph, output=section_enrich_out, env=env)

    log("Starting section-enriched L2 generation.")
    run_generation(
        candidates=l2_candidates,
        output=section_l2,
        query_style="mixed",
        reference_graph=reference_graph,
        topology_candidates=topology_candidates,
        section_enrich=section_enrich_out,
        env=env,
    )
    log("Starting section-enriched L3 generation.")
    run_generation(
        candidates=l3_candidates,
        output=section_l3,
        query_style="academic",
        reference_graph=reference_graph,
        topology_candidates=topology_candidates,
        section_enrich=section_enrich_out,
        env=env,
    )

    run_phase0_eval(
        q1=section_l2_pass,
        q2=section_l3_pass,
        output=section_eval_out,
        hub_candidates=hub_candidates,
        hubs=hubs,
        env=env,
    )
    run_package(env)
    section_exp_a_out = run_exp_a(f"{DATE_TAG}_section_enriched", env)
    section_counts = collect_dataset_counts(
        l2_full=section_l2,
        l2_pass=section_l2_pass,
        l3_full=section_l3,
        l3_pass=section_l3_pass,
        section_enrich=section_enrich_out,
    )

    log("Building / evaluating C-Pool proxy.")
    build_cpool_proxy(output=cpool_proxy, summary=cpool_proxy_summary, env=env)
    run_phase0_eval(
        q1=cpool_proxy,
        q2=cpool_proxy,
        output=cpool_eval_out,
        hub_candidates=hub_candidates,
        hubs=hubs,
        env=env,
    )

    baseline_summary = {
        "counts": baseline_counts,
        "phase0_eval": extract_phase0_summary(baseline_eval_out),
        "exp_a": extract_exp_a_summary(baseline_exp_a_out),
    }
    section_summary = {
        "counts": section_counts,
        "phase0_eval": extract_phase0_summary(section_eval_out),
        "exp_a": extract_exp_a_summary(section_exp_a_out),
    }
    cpool_summary = {
        "proxy_summary": read_json(cpool_proxy_summary),
        "phase0_eval": extract_phase0_summary(cpool_eval_out),
    }

    compare = {
        "metadata": {
            "date_tag": DATE_TAG,
            "reference_graph": str(reference_graph),
            "topology_candidates": str(topology_candidates),
            "hub_candidates": str(hub_candidates),
        },
        "baseline": baseline_summary,
        "section_enriched": section_summary,
        "cpool": cpool_summary,
    }
    compare["delta"] = build_delta(section_summary, baseline_summary)
    write_json(compare_out, compare)

    update_external_reports(compare)
    git_stage_commit_push(env)
    log("Detached post-L2/L3 pipeline finished successfully.")


if __name__ == "__main__":
    main()
