#!/usr/bin/env python3
"""Audit the reasoning-depth heuristic used in generate_multihop_l1_queries.py.

Two subcommands:

1) export-sample
   Export a stratified CSV sample for manual annotation.

2) evaluate
   Compare manual labels against heuristic predictions and compute metrics.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
GEN_SCRIPT = ROOT / "scripts" / "generate_multihop_l1_queries.py"


def load_reasoning_classifier():
    spec = importlib.util.spec_from_file_location("generate_multihop_l1_queries", GEN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {GEN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.classify_reasoning_structure


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def classify_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    classify_reasoning_structure = load_reasoning_classifier()
    out: List[Dict[str, Any]] = []
    for row in rows:
        analysis = classify_reasoning_structure(
            row.get("reasoning_chain", ""),
            row.get("answer", ""),
            row.get("required_evidence_spans", []) or [],
        )
        out.append({
            "query_id": row.get("query_id", ""),
            "query": row.get("query", ""),
            "answer": row.get("answer", ""),
            "query_style": row.get("query_style", ""),
            "pair_type": row.get("pair_type", ""),
            "reasoning_structure_pred": analysis.get("structure", "parallel"),
            "reasoning_depth_pred": analysis.get("reasoning_depth_estimate", 1),
            "is_true_multihop_pred": analysis.get("is_true_multihop", False),
            "serial_markers": " | ".join(analysis.get("serial_markers_found", [])),
            "parallel_markers": " | ".join(analysis.get("parallel_markers_found", [])),
            "distinct_evidence_elements": analysis.get("distinct_evidence_elements", 0),
        })
    return out


def stratified_sample(rows: List[Dict[str, Any]], sample_size: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("reasoning_structure_pred", "parallel")),
            str(row.get("query_style", "unknown")),
        )
        buckets[key].append(row)

    keys = sorted(buckets)
    if not keys:
        return []

    target_per_bucket = max(1, sample_size // len(keys))
    sampled: List[Dict[str, Any]] = []
    leftovers: List[Dict[str, Any]] = []

    for key in keys:
        bucket = list(buckets[key])
        rng.shuffle(bucket)
        take = min(len(bucket), target_per_bucket)
        sampled.extend(bucket[:take])
        leftovers.extend(bucket[take:])

    if len(sampled) < sample_size:
        rng.shuffle(leftovers)
        sampled.extend(leftovers[: sample_size - len(sampled)])

    rng.shuffle(sampled)
    return sampled[:sample_size]


def export_sample(args: argparse.Namespace) -> None:
    rows = load_jsonl(Path(args.input))
    classified = classify_rows(rows)
    sampled = stratified_sample(classified, args.sample_size, args.seed)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query_id",
        "query_style",
        "pair_type",
        "reasoning_structure_pred",
        "reasoning_depth_pred",
        "is_true_multihop_pred",
        "distinct_evidence_elements",
        "serial_markers",
        "parallel_markers",
        "query",
        "answer",
        "manual_structure",
        "manual_depth",
        "manual_is_true_multihop",
        "notes",
    ]
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sampled:
            writer.writerow({
                **row,
                "manual_structure": "",
                "manual_depth": "",
                "manual_is_true_multihop": "",
                "notes": "",
            })

    summary = Counter((r["reasoning_structure_pred"], r["query_style"]) for r in sampled)
    print(f"Exported {len(sampled)} rows to {out_path}")
    print("Sample distribution (pred_structure, query_style):")
    for key, count in sorted(summary.items()):
        print(f"  {key}: {count}")


def safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return num / den


def compute_prf(rows: List[Dict[str, str]], label: str) -> Dict[str, float]:
    tp = sum(
        1 for r in rows
        if r["manual_structure"] == label and r["reasoning_structure_pred"] == label
    )
    fp = sum(
        1 for r in rows
        if r["manual_structure"] != label and r["reasoning_structure_pred"] == label
    )
    fn = sum(
        1 for r in rows
        if r["manual_structure"] == label and r["reasoning_structure_pred"] != label
    )
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "support": tp + fn,
    }


def evaluate(args: argparse.Namespace) -> None:
    path = Path(args.input)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    labeled = [
        r for r in rows
        if (r.get("manual_structure", "") or "").strip()
    ]
    if not labeled:
        raise SystemExit("No manually labeled rows found. Fill manual_structure first.")

    valid_labels = {"parallel", "serial", "mixed"}
    bad_labels = sorted({
        (r.get("manual_structure", "") or "").strip()
        for r in labeled
        if (r.get("manual_structure", "") or "").strip() not in valid_labels
    })
    if bad_labels:
        raise SystemExit(f"Invalid manual_structure labels: {bad_labels}")

    confusion: Dict[str, Counter] = defaultdict(Counter)
    for row in labeled:
        gold = row["manual_structure"].strip()
        pred = (row.get("reasoning_structure_pred", "") or "").strip()
        confusion[gold][pred] += 1

    structures = ["parallel", "serial", "mixed"]
    accuracy = safe_div(
        sum(confusion[s][s] for s in structures),
        len(labeled),
    )

    depth_errors: List[float] = []
    multihop_agreement = 0
    multihop_total = 0
    for row in labeled:
        pred_depth = row.get("reasoning_depth_pred", "").strip()
        gold_depth = row.get("manual_depth", "").strip()
        if pred_depth and gold_depth:
            try:
                depth_errors.append(abs(int(pred_depth) - int(gold_depth)))
            except ValueError:
                pass
        pred_mh = (row.get("is_true_multihop_pred", "") or "").strip().lower()
        gold_mh = (row.get("manual_is_true_multihop", "") or "").strip().lower()
        if pred_mh and gold_mh:
            multihop_total += 1
            if pred_mh == gold_mh:
                multihop_agreement += 1

    report = {
        "input_csv": str(path),
        "num_labeled": len(labeled),
        "accuracy_structure": round(accuracy, 4),
        "macro_prf": {
            label: compute_prf(labeled, label)
            for label in structures
        },
        "confusion_matrix": {
            gold: {pred: confusion[gold][pred] for pred in structures}
            for gold in structures
        },
        "mean_abs_depth_error": round(mean(depth_errors), 4) if depth_errors else None,
        "true_multihop_agreement": round(safe_div(multihop_agreement, multihop_total), 4)
        if multihop_total else None,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Audit reasoning-depth heuristic quality")
    sub = ap.add_subparsers(dest="command", required=True)

    ap_export = sub.add_parser("export-sample", help="Export a stratified CSV for manual annotation")
    ap_export.add_argument("--input", required=True, help="Input generated query JSONL")
    ap_export.add_argument("--output", required=True, help="Output CSV path")
    ap_export.add_argument("--sample-size", type=int, default=40, help="Number of rows to export")
    ap_export.add_argument("--seed", type=int, default=42, help="Random seed")
    ap_export.set_defaults(func=export_sample)

    ap_eval = sub.add_parser("evaluate", help="Evaluate manual labels against heuristic predictions")
    ap_eval.add_argument("--input", required=True, help="Annotated CSV from export-sample")
    ap_eval.add_argument("--output", required=True, help="Output JSON report path")
    ap_eval.set_defaults(func=evaluate)
    return ap


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
