#!/usr/bin/env python3
"""Experimental enriched-only M4 material construction.

This is an experiment-lane wrapper around the existing Method C production
prototype in ``scripts/pilot_method_c.py``.  It does not introduce a new prompt
policy.  It only adds:

1. enriched-only candidate filtering;
2. material/prompt pack export for inspection;
3. optional small company-API smoke generation through the shared src.api layer.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import pilot_method_c as method_c  # noqa: E402
from src.api import parse_json, set_company_credentials  # noqa: E402
from src.qc.pipelines import qc_multihop_query  # noqa: E402
from src.utils.token_logger import log_run  # noqa: E402
from src.qc.checks import is_noisy_enrichment  # noqa: E402


DEFAULT_CANDIDATES = ROOT / "data/03_queries/method_c_true2_candidates_2026-04-12T050859Z.json"
DEFAULT_OUT_ROOT = ROOT / "data/05_eval"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def compact(text: Any, limit: int = 700) -> str:
    return " ".join(str(text or "").split()).strip()[:limit]


def non_noisy_enrichment(elem: dict[str, Any]) -> tuple[str, str]:
    title = compact(elem.get("enriched_title"), 240)
    content = compact(elem.get("enriched_content"), 1200)
    if title and is_noisy_enrichment(title):
        title = ""
    if content and is_noisy_enrichment(content):
        content = ""
    return title, content


def is_enriched_element(elem: dict[str, Any]) -> bool:
    title, content = non_noisy_enrichment(elem)
    return bool(title or content)


def strip_to_enriched_element(elem: dict[str, Any]) -> dict[str, Any]:
    """Keep only stable identity fields plus enriched text for material review."""
    title, content = non_noisy_enrichment(elem)
    return {
        "element_id": elem.get("element_id", ""),
        "doc_id": elem.get("doc_id", ""),
        "element_type": elem.get("element_type", ""),
        "caption": compact(elem.get("caption"), 400),
        "image_path": elem.get("image_path") or "",
        "enriched_title": title,
        "enriched_content": content,
        "enrichment_issues": elem.get("enrichment_issues", []) or [],
    }


def real_path_elements(pair: dict[str, Any]) -> list[dict[str, Any]]:
    elems = []
    for key in ("element_a", "element_b"):
        elem = pair.get(key)
        if isinstance(elem, dict):
            elems.append(elem)
    for elem in pair.get("node_group") or []:
        if isinstance(elem, dict) and not elem.get("is_synthetic_bridge"):
            elems.append(elem)
    # Deduplicate by element_id while preserving order.
    out = []
    seen = set()
    for elem in elems:
        eid = elem.get("element_id")
        if eid and eid not in seen:
            seen.add(eid)
            out.append(elem)
    return out


def pair_passes_enriched_only(pair: dict[str, Any], min_hop: int, max_hop: int) -> tuple[bool, str]:
    hop = int(pair.get("hop_distance") or 0)
    if hop < min_hop or hop > max_hop:
        return False, "hop_out_of_range"
    elems = real_path_elements(pair)
    if len(elems) < 2:
        return False, "missing_real_elements"
    if not all(is_enriched_element(elem) for elem in elems):
        return False, "unenriched_real_element"
    view = method_c.build_method_c_view(pair, max_bridge_nodes=2)
    if int(view.get("compressed_bridge_count") or 0) < 2:
        return False, "compressed_bridge_lt_2"
    return True, "ok"


def sample_diverse(pairs: list[dict[str, Any]], n: int, per_doc: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    if per_doc <= 0:
        shuffled = list(pairs)
        rng.shuffle(shuffled)
        return shuffled[:n] if n > 0 else shuffled
    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        by_doc[str(pair.get("doc_id", ""))].append(pair)
    selected = []
    for bucket in by_doc.values():
        rng.shuffle(bucket)
        selected.extend(bucket[:per_doc])
    rng.shuffle(selected)
    return selected[:n] if n > 0 else selected


def build_material(pair: dict[str, Any], max_bridge_nodes: int) -> dict[str, Any]:
    view = method_c.build_method_c_view(pair, max_bridge_nodes=max_bridge_nodes)
    prompt = method_c.build_prompt(pair, view)
    return {
        "material_id": f"m4_material_{pair['pair_id']}",
        "pair_id": pair["pair_id"],
        "doc_id": pair["doc_id"],
        "hop_distance": pair.get("hop_distance"),
        "pair_type": pair.get("pair_type"),
        "path": pair.get("path", []),
        "element_a": strip_to_enriched_element(pair["element_a"]),
        "element_b": strip_to_enriched_element(pair["element_b"]),
        "method_c": method_c.build_result_method_c_summary(view),
        "prompt": prompt,
        "checks": {
            "real_path_elements": len(real_path_elements(pair)),
            "all_real_elements_enriched": all(is_enriched_element(e) for e in real_path_elements(pair)),
            "compressed_bridge_count": int(view.get("compressed_bridge_count") or 0),
        },
    }


def run_company_generation(
    materials: list[dict[str, Any]],
    pairs_by_id: dict[str, dict[str, Any]],
    *,
    model: str,
    delay: float,
) -> tuple[list[dict[str, Any]], int, int]:
    generated: list[dict[str, Any]] = []
    total_in = 0
    total_out = 0
    for idx, material in enumerate(materials, start=1):
        pair = pairs_by_id[material["pair_id"]]
        view = method_c.build_method_c_view(pair, max_bridge_nodes=2)
        print(f"  API [{idx}/{len(materials)}] {pair['pair_id']} ...", end=" ", flush=True)
        try:
            raw, in_tok, out_tok = method_c.generate_qa(material["prompt"], model)
        except Exception as exc:  # pragma: no cover - operational path
            print(f"ERROR {exc}")
            generated.append({
                "pair_id": pair["pair_id"],
                "error": str(exc),
                "raw_response": "",
            })
            continue
        total_in += in_tok
        total_out += out_tok
        parsed = parse_json(raw)
        if not parsed:
            print("PARSE_FAIL")
            generated.append({
                "pair_id": pair["pair_id"],
                "parse_error": True,
                "raw_response": raw,
                "tokens": {"in": in_tok, "out": out_tok},
            })
            continue
        qc_obj = method_c.build_qc_obj(parsed, pair, view)
        qc_pair = method_c.build_qc_pair(pair, view)
        rule_issues, rule_metrics = qc_multihop_query(qc_obj, qc_pair)
        passed = len(rule_issues) == 0
        print("PASS" if passed else f"RULE_FAIL {rule_issues}")
        generated.append({
            "pair_id": pair["pair_id"],
            "material_id": material["material_id"],
            "generated": parsed,
            "qc": {
                "passed_rule_qc": passed,
                "rule_issues": rule_issues,
                "rule_metrics": rule_metrics,
            },
            "tokens": {"in": in_tok, "out": out_tok},
        })
        if delay > 0:
            time.sleep(delay)
    return generated, total_in, total_out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--api-limit", type=int, default=0)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--company-api-url", default=os.environ.get("COMPANY_API_URL", ""))
    parser.add_argument("--company-api-key", default=os.environ.get("COMPANY_API_KEY", ""))
    parser.add_argument("--min-hop", type=int, default=4)
    parser.add_argument("--max-hop", type=int, default=5)
    parser.add_argument("--max-bridge-nodes", type=int, default=2)
    parser.add_argument("--per-doc-cap", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else (
        DEFAULT_OUT_ROOT / f"m4_enriched_materials_{utc_stamp()}"
    )
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates_path = Path(args.candidates)
    if not candidates_path.is_absolute():
        candidates_path = ROOT / candidates_path
    data = read_json(candidates_path)
    raw_pairs = list(data.get("pairs") or [])

    reject = Counter()
    eligible = []
    for pair in raw_pairs:
        ok, reason = pair_passes_enriched_only(pair, args.min_hop, args.max_hop)
        if ok:
            eligible.append(pair)
        else:
            reject[reason] += 1

    selected = sample_diverse(eligible, args.limit, args.per_doc_cap, args.seed)
    materials = [build_material(pair, args.max_bridge_nodes) for pair in selected]

    pairs_by_id = {pair["pair_id"]: pair for pair in selected}
    generated: list[dict[str, Any]] = []
    total_in = 0
    total_out = 0
    if args.api_limit > 0:
        if not args.company_api_url or not args.company_api_key:
            raise SystemExit("company API credentials missing; set COMPANY_API_URL and COMPANY_API_KEY")
        set_company_credentials(args.company_api_url, args.company_api_key)
        generated, total_in, total_out = run_company_generation(
            materials[:args.api_limit],
            pairs_by_id,
            model=args.model,
            delay=args.delay,
        )

    material_path = out_dir / "m4_material_pack.jsonl"
    candidates_out = out_dir / "m4_enriched_candidates.json"
    prompt_path = out_dir / "prompt_batch.jsonl"
    generated_path = out_dir / "generated_m4_smoke.jsonl"
    summary_path = out_dir / "summary.json"
    report_path = out_dir / "report.md"

    write_jsonl(material_path, materials)
    write_jsonl(prompt_path, [
        {
            "material_id": material["material_id"],
            "pair_id": material["pair_id"],
            "prompt": material["prompt"],
        }
        for material in materials
    ])
    candidates_out.write_text(
        json.dumps({
            "metadata": {
                "source": "construct_m4_enriched_materials.py",
                "source_candidates": rel(candidates_path),
                "selection": "enriched_only_method_c_true_two_bridge",
                "min_hop": args.min_hop,
                "max_hop": args.max_hop,
                "max_bridge_nodes": args.max_bridge_nodes,
                "seed": args.seed,
            },
            "summary": {
                "raw_pairs": len(raw_pairs),
                "eligible_pairs": len(eligible),
                "selected_pairs": len(selected),
                "reject_reasons": dict(reject),
            },
            "pairs": selected,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if generated:
        write_jsonl(generated_path, generated)

    pair_types = Counter(material["pair_type"] for material in materials)
    hops = Counter(str(material["hop_distance"]) for material in materials)
    bridge_counts = Counter(str(material["checks"]["compressed_bridge_count"]) for material in materials)
    rule_pass = sum(1 for row in generated if row.get("qc", {}).get("passed_rule_qc"))
    parse_failures = sum(1 for row in generated if row.get("parse_error"))
    api_errors = sum(1 for row in generated if row.get("error"))

    summary = {
        "status": "ok",
        "source_candidates": rel(candidates_path),
        "output_dir": rel(out_dir),
        "raw_pairs": len(raw_pairs),
        "eligible_pairs": len(eligible),
        "selected_materials": len(materials),
        "reject_reasons": dict(reject),
        "pair_type_counts": dict(pair_types),
        "hop_counts": dict(hops),
        "compressed_bridge_counts": dict(bridge_counts),
        "api_smoke": {
            "requested": args.api_limit,
            "generated_rows": len(generated),
            "rule_qc_pass": rule_pass,
            "parse_failures": parse_failures,
            "api_errors": api_errors,
            "model": args.model,
            "tokens": {"in": total_in, "out": total_out},
        },
        "files": {
            "materials": rel(material_path),
            "candidates": rel(candidates_out),
            "prompts": rel(prompt_path),
            "generated": rel(generated_path) if generated else "",
            "summary": rel(summary_path),
            "report": rel(report_path),
            "local_api_logger": "api_logs_cannt_delete",
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report = [
        "# M4 Enriched-Only Material Construction",
        "",
        f"- source candidates: `{summary['source_candidates']}`",
        f"- raw / eligible / selected: **{len(raw_pairs)} / {len(eligible)} / {len(materials)}**",
        f"- reject reasons: `{dict(reject)}`",
        f"- pair types: `{dict(pair_types)}`",
        f"- hop counts: `{dict(hops)}`",
        f"- compressed bridge counts: `{dict(bridge_counts)}`",
        f"- API smoke: `{summary['api_smoke']}`",
        "",
        "## Files",
        "",
    ]
    for key, value in summary["files"].items():
        if value:
            report.append(f"- {key}: `{value}`")
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    if total_in or total_out:
        log_run(
            script="experiments/construct_m4_enriched_materials.py",
            model=f"company:{args.model}",
            input_tokens=total_in,
            output_tokens=total_out,
            purpose="M4 enriched-only material API smoke",
            extra={
                "pairs_processed": len(generated),
                "queries_written": len(generated) - parse_failures - api_errors,
                "qc_pass": rule_pass,
                "qc_fail": len(generated) - rule_pass - parse_failures - api_errors,
                "parse_failures": parse_failures + api_errors,
                "output": rel(generated_path) if generated else "",
            },
        )

    latest = DEFAULT_OUT_ROOT / "m4_enriched_materials_latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(out_dir, target_is_directory=True)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Latest: {latest}")


if __name__ == "__main__":
    main()
