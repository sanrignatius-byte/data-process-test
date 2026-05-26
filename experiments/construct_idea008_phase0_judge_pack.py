#!/usr/bin/env python3
"""Construct a judge-ready Phase 0 pack for idea:008.

The pack samples existing MinerU cross-document visual candidates and renders a
caption-independent edge-judging prompt for each pair.  It does not call any API.
The goal is to make the next VLM/LLM/manual audit consume one stable artifact:
image paths, text context, rerank scores, degradation strata, and prompts.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RERANK = (
    ROOT
    / "data/05_eval/mineru_crossdoc_text_rerank_v1_latest"
    / "mineru_crossdoc_text_rerank_edges_v1.jsonl"
)
DEFAULT_TOPOLOGY = (
    ROOT
    / "data/05_eval/mineru_topology_graph_v1_latest"
    / "mineru_topology_graph_v1.json"
)

TIER_ORDER = [
    "strong_text_supported",
    "strong_enriched_supported",
    "text_supported_candidate",
    "weak_text_support",
    "visual_only_risky",
]

DEFAULT_TIER_WEIGHTS = {
    "strong_text_supported": 36,
    "strong_enriched_supported": 24,
    "text_supported_candidate": 36,
    "weak_text_support": 40,
    "visual_only_risky": 24,
}


JUDGE_TEMPLATE = """You are judging one candidate cross-document multimodal graph edge.

Task:
Decide whether Element A and Element B should be promoted from a CLIP-recall candidate
to a STRONG cross-document semantic edge for a scientific-document graph.

Use the images if they are attached by the caller. Use captions/context as supporting
evidence, but do not require caption token overlap. The important question is whether
the two elements express the same method, dataset, metric, experimental comparison,
visual pattern, or reusable scientific role strongly enough that a multi-hop query can
rely on this edge.

Return only valid JSON with exactly these fields:
- verdict: one of ["strong_edge", "weak_related", "visual_layout_only", "unrelated", "insufficient"]
- confidence: number from 0 to 1
- shared_semantics: short phrase naming the shared concept, or "" if none
- evidence_a: one concrete visual/textual cue from Element A
- evidence_b: one concrete visual/textual cue from Element B
- rationale: 2-4 sentences
- failure_mode: one of ["none", "caption_degraded", "layout_false_positive", "generic_caption", "missing_image", "insufficient_context", "other"]

Element A:
- node_id: {source_id}
- doc_id: {source_doc}
- type: {source_type}
- label: {source_label}
- image_path: {source_image_path}
- caption: {source_caption}
- local_context: {source_context}
- enriched_preview: {source_enriched}

Element B:
- node_id: {target_id}
- doc_id: {target_doc}
- type: {target_type}
- label: {target_label}
- image_path: {target_image_path}
- caption: {target_caption}
- local_context: {target_context}
- enriched_preview: {target_enriched}

Audit metadata:
- candidate_id: {candidate_id}
- caption_bucket: {caption_bucket}
- source_caption_quality: {source_caption_quality}
- target_caption_quality: {target_caption_quality}
- existing_support_tier: {support_tier}
"""


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compact(value: Any, limit: int = 1000) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def meaningful_tokens(text: str) -> list[str]:
    generic = {
        "figure",
        "fig",
        "table",
        "image",
        "images",
        "page",
        "panel",
        "results",
        "result",
    }
    out = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", text.lower()):
        if token in generic:
            continue
        if token.replace("-", "").isdigit():
            continue
        out.append(token)
    return out


def caption_quality(caption: str) -> str:
    text = compact(caption, 1000)
    low = text.lower()
    if not text:
        return "empty"
    if re.fullmatch(r"(?:figure|fig\.?|table)\s*\d+[a-z]?[.:]?", text, flags=re.I):
        return "generic_number"
    if re.search(r"</?(?:td|tr|table|tbody|thead|html|span|div)\b", low):
        return "html_fragment"
    tokens = meaningful_tokens(text)
    if len(tokens) <= 2:
        return "too_short"
    if len(tokens) <= 5:
        return "thin"
    return "clean"


def pair_caption_bucket(src_quality: str, tgt_quality: str, row: dict[str, Any]) -> str:
    src_clean = src_quality == "clean"
    tgt_clean = tgt_quality == "clean"
    if not src_clean and not tgt_clean:
        return "degraded_both"
    if not src_clean or not tgt_clean:
        return "degraded_one"
    if float(row.get("caption_sim") or 0.0) == 0.0:
        return "clean_caption_zero_overlap"
    return "clean_text_overlap"


def load_nodes(path: Path) -> dict[str, dict[str, Any]]:
    data = read_json(path)
    return {str(node["node_id"]): node for node in data.get("nodes", []) if node.get("node_id")}


def node_context(node: dict[str, Any], limit: int = 900) -> str:
    meta = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
    parts = [
        node.get("text_snippet") or "",
        meta.get("content_preview") or "",
        meta.get("context_before") or "",
        meta.get("context_after") or "",
    ]
    text = " ".join(str(part).strip() for part in parts if part and str(part).strip())
    return compact(text, limit) if text else "(no context)"


def node_summary(node: dict[str, Any], enriched_preview: str) -> dict[str, Any]:
    meta = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
    image_path = str(meta.get("image_path") or "")
    return {
        "node_id": node.get("node_id"),
        "doc_id": node.get("doc_id"),
        "node_type": node.get("node_type"),
        "element_id": node.get("element_id") or node.get("mapped_element_id"),
        "label": node.get("label") or "",
        "page_idx": node.get("page_idx"),
        "position_idx": node.get("position_idx"),
        "caption": compact(meta.get("caption") or node.get("text_snippet") or node.get("label") or "", 1200),
        "context": node_context(node, 1000),
        "image_path": image_path,
        "image_exists": bool(image_path and Path(image_path).exists()),
        "bbox": meta.get("bbox"),
        "source": meta.get("source"),
        "enriched_preview": compact(enriched_preview, 900),
    }


def canonical_pair_key(row: dict[str, Any]) -> tuple[str, str]:
    a = str(row.get("source_id"))
    b = str(row.get("target_id"))
    return tuple(sorted([a, b]))  # type: ignore[return-value]


def dedupe_directional(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = canonical_pair_key(row)
        old = best.get(key)
        if old is None:
            best[key] = row
            continue
        new_score = (float(row.get("combined_score") or 0.0), float(row.get("visual_score") or 0.0))
        old_score = (float(old.get("combined_score") or 0.0), float(old.get("visual_score") or 0.0))
        if new_score > old_score:
            best[key] = row
    return list(best.values())


def scaled_tier_quotas(target_size: int, available: Counter[str]) -> dict[str, int]:
    base_total = sum(DEFAULT_TIER_WEIGHTS.values())
    quotas: dict[str, int] = {}
    for tier in TIER_ORDER:
        quota = round(DEFAULT_TIER_WEIGHTS[tier] * target_size / base_total)
        quotas[tier] = min(quota, available.get(tier, 0))
    while sum(quotas.values()) < min(target_size, sum(available.values())):
        added = False
        for tier in TIER_ORDER:
            if quotas[tier] < available.get(tier, 0):
                quotas[tier] += 1
                added = True
                if sum(quotas.values()) >= target_size:
                    break
        if not added:
            break
    while sum(quotas.values()) > target_size:
        for tier in reversed(TIER_ORDER):
            if quotas[tier] > 0:
                quotas[tier] -= 1
                break
    return quotas


def choose_from_tier(rows: list[dict[str, Any]], quota: int, rng: random.Random) -> list[dict[str, Any]]:
    if quota <= 0:
        return []
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row["caption_bucket"])].append(row)
    for bucket_rows in buckets.values():
        bucket_rows.sort(key=lambda r: int(r.get("rank_after_rerank") or 10**9))

    selected: list[dict[str, Any]] = []
    bucket_names = sorted(buckets)
    while len(selected) < quota and any(buckets.values()):
        for bucket in bucket_names:
            if buckets[bucket] and len(selected) < quota:
                selected.append(buckets[bucket].pop(0))

    if len(selected) < quota:
        leftovers = [row for bucket_rows in buckets.values() for row in bucket_rows]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: quota - len(selected)])
    return selected


def render_prompt(candidate: dict[str, Any]) -> str:
    source = candidate["source"]
    target = candidate["target"]
    return JUDGE_TEMPLATE.format(
        candidate_id=candidate["candidate_id"],
        source_id=source["node_id"],
        source_doc=source["doc_id"],
        source_type=source["node_type"],
        source_label=source["label"],
        source_image_path=source["image_path"] or "(missing)",
        source_caption=source["caption"] or "(no caption)",
        source_context=source["context"],
        source_enriched=source["enriched_preview"] or "(no enriched preview)",
        target_id=target["node_id"],
        target_doc=target["doc_id"],
        target_type=target["node_type"],
        target_label=target["label"],
        target_image_path=target["image_path"] or "(missing)",
        target_caption=target["caption"] or "(no caption)",
        target_context=target["context"],
        target_enriched=target["enriched_preview"] or "(no enriched preview)",
        caption_bucket=candidate["caption_bucket"],
        source_caption_quality=candidate["source_caption_quality"],
        target_caption_quality=candidate["target_caption_quality"],
        support_tier=candidate["support_tier"],
    )


def enrich_rows(rows: list[dict[str, Any]], nodes: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    enriched = []
    for row in rows:
        source = nodes.get(str(row.get("source_id")))
        target = nodes.get(str(row.get("target_id")))
        if not source or not target:
            continue
        meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        src_caption = str(meta.get("source_caption") or "")
        tgt_caption = str(meta.get("target_caption") or "")
        src_quality = caption_quality(src_caption)
        tgt_quality = caption_quality(tgt_caption)
        new = dict(row)
        new["source_caption_quality"] = src_quality
        new["target_caption_quality"] = tgt_quality
        new["caption_bucket"] = pair_caption_bucket(src_quality, tgt_quality, row)
        new["source"] = node_summary(source, str(meta.get("source_enriched_preview") or ""))
        new["target"] = node_summary(target, str(meta.get("target_enriched_preview") or ""))
        enriched.append(new)
    return enriched


def build_candidates(rows: list[dict[str, Any]], target_size: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    by_tier: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_tier[str(row.get("support_tier"))].append(row)
    for tier_rows in by_tier.values():
        tier_rows.sort(key=lambda r: int(r.get("rank_after_rerank") or 10**9))

    available = Counter({tier: len(by_tier.get(tier, [])) for tier in TIER_ORDER})
    quotas = scaled_tier_quotas(target_size, available)
    sampled: list[dict[str, Any]] = []
    for tier in TIER_ORDER:
        sampled.extend(choose_from_tier(by_tier.get(tier, []), quotas[tier], rng))

    sampled.sort(key=lambda r: (TIER_ORDER.index(str(r.get("support_tier"))) if str(r.get("support_tier")) in TIER_ORDER else 999, int(r.get("rank_after_rerank") or 10**9)))
    final: list[dict[str, Any]] = []
    for idx, row in enumerate(sampled, 1):
        scores = {
            "combined_score": row.get("combined_score"),
            "visual_score": row.get("visual_score"),
            "caption_sim": row.get("caption_sim"),
            "context_sim": row.get("context_sim"),
            "enriched_sim": row.get("enriched_sim"),
            "all_text_sim": row.get("all_text_sim"),
            "text_support": row.get("text_support"),
            "rank_after_rerank": row.get("rank_after_rerank"),
        }
        candidate = {
            "candidate_id": f"idea008_phase0_{idx:04d}",
            "source_id": row.get("source_id"),
            "target_id": row.get("target_id"),
            "doc_pair": row.get("doc_id"),
            "support_tier": row.get("support_tier"),
            "caption_bucket": row.get("caption_bucket"),
            "source_caption_quality": row.get("source_caption_quality"),
            "target_caption_quality": row.get("target_caption_quality"),
            "scores": scores,
            "source": row["source"],
            "target": row["target"],
            "heuristic_label_hint": heuristic_label_hint(row),
            "prompt": "",
        }
        candidate["prompt"] = render_prompt(candidate)
        final.append(candidate)

    summary = {
        "target_size": target_size,
        "sample_size": len(final),
        "seed": seed,
        "quotas": quotas,
        "sample_tier_counts": dict(Counter(c["support_tier"] for c in final)),
        "sample_caption_bucket_counts": dict(Counter(c["caption_bucket"] for c in final)),
        "image_missing_count": sum(
            1
            for c in final
            if not c["source"]["image_exists"] or not c["target"]["image_exists"]
        ),
    }
    return final, summary


def heuristic_label_hint(row: dict[str, Any]) -> str:
    text_support = float(row.get("text_support") or 0.0)
    enriched = float(row.get("enriched_sim") or 0.0)
    visual = float(row.get("visual_score") or 0.0)
    tier = str(row.get("support_tier"))
    bucket = str(row.get("caption_bucket"))
    if tier in {"strong_text_supported", "strong_enriched_supported"} and max(text_support, enriched) >= 0.25:
        return "likely_positive_control"
    if tier == "visual_only_risky" or (visual >= 0.90 and text_support < 0.05):
        return "likely_layout_negative_control"
    if "degraded" in bucket and text_support < 0.10:
        return "degraded_caption_hard_case"
    return "ambiguous_candidate"


def write_outputs(out_dir: Path, candidates: list[dict[str, Any]], summary: dict[str, Any], metadata: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = dict(summary)
    summary.update(metadata)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    with (out_dir / "phase0_candidates.jsonl").open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            handle.write(json.dumps(candidate, ensure_ascii=False) + "\n")

    with (out_dir / "prompt_batch.jsonl").open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            handle.write(
                json.dumps(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "source_image_path": candidate["source"]["image_path"],
                        "target_image_path": candidate["target"]["image_path"],
                        "prompt": candidate["prompt"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    fieldnames = [
        "candidate_id",
        "support_tier",
        "caption_bucket",
        "heuristic_label_hint",
        "source_id",
        "target_id",
        "source_doc",
        "target_doc",
        "source_type",
        "target_type",
        "combined_score",
        "visual_score",
        "caption_sim",
        "context_sim",
        "enriched_sim",
        "all_text_sim",
        "source_image_exists",
        "target_image_exists",
    ]
    with (out_dir / "phase0_candidates.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for c in candidates:
            s = c["scores"]
            writer.writerow(
                {
                    "candidate_id": c["candidate_id"],
                    "support_tier": c["support_tier"],
                    "caption_bucket": c["caption_bucket"],
                    "heuristic_label_hint": c["heuristic_label_hint"],
                    "source_id": c["source_id"],
                    "target_id": c["target_id"],
                    "source_doc": c["source"]["doc_id"],
                    "target_doc": c["target"]["doc_id"],
                    "source_type": c["source"]["node_type"],
                    "target_type": c["target"]["node_type"],
                    "combined_score": s["combined_score"],
                    "visual_score": s["visual_score"],
                    "caption_sim": s["caption_sim"],
                    "context_sim": s["context_sim"],
                    "enriched_sim": s["enriched_sim"],
                    "all_text_sim": s["all_text_sim"],
                    "source_image_exists": c["source"]["image_exists"],
                    "target_image_exists": c["target"]["image_exists"],
                }
            )

    write_report(out_dir, candidates, summary)


def write_report(out_dir: Path, candidates: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = [
        "# idea:008 Phase 0 Judge Pack",
        "",
        "Purpose: test whether caption-independent judgment can promote MinerU+CLIP cross-doc recall edges into strong semantic graph edges.",
        "",
        "## Summary",
        "",
        f"- sample size: **{summary['sample_size']}**",
        f"- seed: `{summary['seed']}`",
        f"- missing image pairs: **{summary['image_missing_count']}**",
        f"- source rerank: `{summary['source_rerank']}`",
        f"- source topology: `{summary['source_topology']}`",
        "",
        "## Tier Counts",
        "",
    ]
    for key, value in sorted(summary["sample_tier_counts"].items(), key=lambda kv: (TIER_ORDER.index(kv[0]) if kv[0] in TIER_ORDER else 999, kv[0])):
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Caption Buckets", ""])
    for key, value in sorted(summary["sample_caption_bucket_counts"].items()):
        lines.append(f"- `{key}`: {value}")
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `phase0_candidates.jsonl`: full candidate records with prompts and image paths.",
            "- `prompt_batch.jsonl`: minimal API-ready prompt records.",
            "- `phase0_candidates.csv`: spreadsheet-friendly audit index.",
            "- `summary.json`: construction metadata and counts.",
            "",
            "## Prompt Samples",
            "",
        ]
    )
    for c in candidates[:8]:
        lines.extend(
            [
                f"### {c['candidate_id']} — {c['support_tier']} / {c['caption_bucket']}",
                "",
                f"- hint: `{c['heuristic_label_hint']}`",
                f"- pair: `{c['source_id']}` -> `{c['target_id']}`",
                f"- scores: combined={c['scores']['combined_score']} visual={c['scores']['visual_score']} caption={c['scores']['caption_sim']} enriched={c['scores']['enriched_sim']}",
                f"- source image: `{c['source']['image_path']}`",
                f"- target image: `{c['target']['image_path']}`",
                "",
                "```text",
                compact(c["prompt"], 2200),
                "```",
                "",
            ]
        )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Construct idea:008 Phase 0 judge pack")
    parser.add_argument("--rerank", default=str(DEFAULT_RERANK))
    parser.add_argument("--topology", default=str(DEFAULT_TOPOLOGY))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--target-size", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20260520)
    args = parser.parse_args()

    rerank_path = Path(args.rerank)
    topology_path = Path(args.topology)
    if not rerank_path.is_absolute():
        rerank_path = ROOT / rerank_path
    if not topology_path.is_absolute():
        topology_path = ROOT / topology_path

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) if args.output_dir else ROOT / f"data/05_eval/idea008_phase0_judge_pack_{timestamp}"
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    raw_rows = iter_jsonl(rerank_path)
    deduped_rows = dedupe_directional(raw_rows)
    nodes = load_nodes(topology_path)
    enriched_rows = enrich_rows(deduped_rows, nodes)
    candidates, sample_summary = build_candidates(enriched_rows, args.target_size, args.seed)
    metadata = {
        "builder": "construct_idea008_phase0_judge_pack.py",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_rerank": str(rerank_path),
        "source_topology": str(topology_path),
        "raw_edge_count": len(raw_rows),
        "deduped_pair_count": len(deduped_rows),
        "usable_pair_count": len(enriched_rows),
        "raw_tier_counts": dict(Counter(str(r.get("support_tier")) for r in raw_rows)),
        "deduped_tier_counts": dict(Counter(str(r.get("support_tier")) for r in enriched_rows)),
        "deduped_caption_bucket_counts": dict(Counter(str(r.get("caption_bucket")) for r in enriched_rows)),
    }
    write_outputs(out_dir, candidates, sample_summary, metadata)
    latest = ROOT / "data/05_eval/idea008_phase0_judge_pack_latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(out_dir)
    print(f"[ok] raw edges: {len(raw_rows)}")
    print(f"[ok] deduped pairs: {len(deduped_rows)}")
    print(f"[ok] sample size: {len(candidates)}")
    print(f"[ok] wrote {out_dir}")
    print(f"[ok] latest -> {latest}")


if __name__ == "__main__":
    main()
