#!/usr/bin/env python3
"""Element-level linguistic cross-document validation.

Key difference from build_linguistic_xdoc_bridges.py:
  - Input = element-element pairs (caption + enriched_content), NOT section text
  - Bypasses the section→element cartesian projection that killed chain quality
  - Genette+RST+asymmetry runs directly on element text

Pipeline:
  1. Select element pairs from cross_doc_sim_edges doc pairs
     (ranked by caption+enriched_content text similarity)
  2. Genette+RST classification on element text (no decontext needed)
  3. Asymmetric verification
  4. Inject gold+strong+weak pairs as hard cross-doc edges into Document Graph
  5. BFS chain finding (cap=5000, min-hops=2, max-hops=4)
  6. Chain-level head-to-head judge (VALIDATED vs REJECTED baseline)

T1 gate:
  - Revive: VALIDATED usable >= 15% AND VALIDATED - REJECTED >= 10pp
  - Ambiguous: usable 5-15%
  - Close: usable < 5% OR delta <= 5pp
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api.llm import call_llm, parse_json, set_company_credentials
from src.utils.token_logger import log_run

DEFAULT_MODEL = "gpt-5.4"

# ── Genette+RST prompt (adapted for element-level, no decontext) ──────────

ELEMENT_RST_SYSTEM = """You are a rhetorical structure analyst. Given two scientific
elements (figures, tables, or formulas from potentially different papers), classify the
rhetorical relationship between them using Genette's transtextuality framework.

Genette's 5 types, adapted for scientific elements:
1. **direct_quotation** (Intertextuality): Element B explicitly copies, cites, or
   reproduces content from Element A. Evidence: same numbers, same formulas, same figure layout.
2. **paratextual** (Paratextuality): Elements share framing — same problem domain,
   same dataset, same evaluation metric, but don't directly interact.
3. **commentary** (Metatextuality): Element B discusses, critiques, or benchmarks
   against what Element A shows. B "speaks about" A's content.
4. **architextual** (Architextuality): Elements belong to the same genre/methodology
   family — both are ablation studies, both use Transformer, both are fairness papers.
5. **transformation** (Hypertextuality): Element B's content is a direct derivation,
   extension, or modification of Element A's method/theory/result. B "transforms" A.

Additionally, classify the RST relation (Mann & Thompson):
- Cause-Effect: A's finding causes/explains B's result
- Elaboration: B provides more detail about A's claim
- Contrast: A and B show opposing/different results
- Background: A provides context for understanding B
- Evidence: B provides evidence for A's claim
- Summary: B summarizes A
- Joint: A and B are parallel/independent contributions to the same topic

Return JSON:
{
  "genette_type": "<one of the 5 types>",
  "rst_relation": "<one of the 7 RST relations>",
  "is_causal_chain": true/false,
  "bidirectional": true/false,
  "evidence": "<one sentence with specific facts from both elements that support this classification>",
  "confidence": 0.85
}

Causal chain = Genette type is 'transformation' AND RST relation is one of
{Cause-Effect, Evidence, Elaboration} AND the reasoning goes from A to B.
Bidirectional = the relationship holds in both directions (A↔B rather than A→B)."""


def build_element_rst_prompt(
    caption_a: str,
    enriched_a: str,
    elem_a_type: str,
    caption_b: str,
    enriched_b: str,
    elem_b_type: str,
) -> str:
    text_a = f"{caption_a}\n{enriched_a}"[:800]
    text_b = f"{caption_b}\n{enriched_b}"[:800]
    return f"""Classify the cross-document relationship between these two elements:

Element A ({elem_a_type}):
{text_a}

Element B ({elem_b_type}):
{text_b}

What is the relationship from A to B?"""


# ── Asymmetric verification (same as original, adapted for elements) ─────

ELEMENT_ASYMMETRIC_SYSTEM = """You are testing whether a cross-document relationship is
symmetric or asymmetric. Many scientific cross-document relationships are ASYMMETRIC:
Paper B builds on Paper A, but Paper A does not need Paper B to be understood.

Given two elements and their classified relationship (A→B), test the REVERSE direction (B→A).
Can you construct a meaningful rhetorical relationship from B to A?

Return JSON:
{
  "reverse_valid": true/false,
  "reverse_genette_type": "<type or 'none'>",
  "asymmetry_explanation": "<one sentence>",
  "asymmetry_score": 0.0-1.0
}

asymmetry_score: 0.0 = perfectly symmetric, 1.0 = completely one-sided (A→B only)."""


# ── Element pair selection ────────────────────────────────────────────────

def load_elements_index(elements_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load all elements keyed by element_id."""
    with open(elements_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    elements: Dict[str, Dict[str, Any]] = {}
    docs = data.get("documents", {})
    for doc_id, doc in docs.items():
        for eid, el in doc.get("elements", {}).items():
            elements[eid] = el
    return elements


def load_section_edges(edges_path: Path) -> List[Dict[str, Any]]:
    with open(edges_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("edges", data if isinstance(data, list) else [])


def tokenize_simple(text: str) -> Set[str]:
    """Simple tokenization for Jaccard similarity."""
    tokens = re.findall(r'[a-zA-Z_]{3,}', text.lower())
    return set(tokens)


def select_element_pairs(
    section_edges: List[Dict],
    elements: Dict[str, Dict[str, Any]],
    top_k: int = 1500,
    min_text_len: int = 20,
) -> List[Dict[str, Any]]:
    """Select element-element pairs from section-edge doc pairs.

    Quota-enforced: ~1/3 cross-modal (figure+table, figure+formula, table+formula),
    ~2/3 same-modal (figure+figure, table+table, formula+formula).
    Within each modality bucket, pairs are ranked by enriched_content token Jaccard
    (since captions rarely overlap across modalities, enriched_content is more reliable).
    """
    # Build doc→elements mapping
    doc_elements: Dict[str, List[Tuple[str, Dict]]] = defaultdict(list)
    for eid, el in elements.items():
        doc_id = el.get("doc_id", "")
        if not doc_id:
            for sep in ["_figure_", "_table_", "_formula_"]:
                if sep in eid:
                    doc_id = eid.split(sep)[0]
                    break
        if doc_id:
            doc_elements[doc_id].append((eid, el))

    # Per-modality-bucket accumulators
    buckets: Dict[str, List[Tuple[float, Dict]]] = defaultdict(list)
    seen_pairs: Set[Tuple[str, str]] = set()

    for edge in section_edges:
        src_doc = edge.get("source_doc", "").strip()
        tgt_doc = edge.get("target_doc", "").strip()
        if not src_doc or not tgt_doc or src_doc == tgt_doc:
            continue

        src_elems = doc_elements.get(src_doc, [])
        tgt_elems = doc_elements.get(tgt_doc, [])

        if not src_elems or not tgt_elems:
            continue

        for src_eid, src_el in src_elems:
            src_type = src_el.get("element_type", "unknown")
            src_cap = (src_el.get("caption", "") or "")[:500]
            src_enr = (src_el.get("enriched_content", "") or "")[:500]
            src_text = f"{src_cap} {src_enr}".strip()
            if len(src_text) < min_text_len:
                continue
            src_tokens = tokenize_simple(src_text)
            src_enr_tokens = tokenize_simple(src_enr)

            for tgt_eid, tgt_el in tgt_elems:
                pair_key = tuple(sorted([src_eid, tgt_eid]))
                if pair_key in seen_pairs:
                    continue

                tgt_type = tgt_el.get("element_type", "unknown")
                tgt_cap = (tgt_el.get("caption", "") or "")[:500]
                tgt_enr = (tgt_el.get("enriched_content", "") or "")[:500]
                tgt_text = f"{tgt_cap} {tgt_enr}".strip()
                if len(tgt_text) < min_text_len:
                    continue

                tgt_tokens = tokenize_simple(tgt_text)
                tgt_enr_tokens = tokenize_simple(tgt_enr)

                if not src_tokens or not tgt_tokens:
                    continue

                # Jaccard on full text (caption+enriched)
                jaccard_full = len(src_tokens & tgt_tokens) / max(len(src_tokens | tgt_tokens), 1)
                # Jaccard on enriched only (better for cross-modal)
                jaccard_enr = 0.0
                if src_enr_tokens and tgt_enr_tokens:
                    jaccard_enr = len(src_enr_tokens & tgt_enr_tokens) / max(len(src_enr_tokens | tgt_enr_tokens), 1)

                # Modality bucket key
                types_key = "+".join(sorted([src_type, tgt_type]))
                bucket_key = types_key if types_key in _MODALITY_BUCKETS else "other"

                # Use enriched-only Jaccard for cross-modal, full Jaccard for same-modal
                if src_type != tgt_type:
                    score = jaccard_enr if jaccard_enr > 0 else jaccard_full * 0.5
                else:
                    score = jaccard_full

                seen_pairs.add(pair_key)
                buckets[bucket_key].append((score, {
                    "element_a_id": src_eid,
                    "element_b_id": tgt_eid,
                    "element_a_type": src_type,
                    "element_b_type": tgt_type,
                    "caption_a": src_cap,
                    "caption_b": tgt_cap,
                    "enriched_a": src_enr,
                    "enriched_b": tgt_enr,
                    "text_jaccard": round(score, 4),
                    "source_doc": src_doc,
                    "target_doc": tgt_doc,
                }))

    # Sort each bucket
    for key in buckets:
        buckets[key].sort(key=lambda x: x[0], reverse=True)

    # Quota allocation: ~1/3 cross-modal (500), ~2/3 same-modal (1000)
    n_cross = min(top_k // 3, 500)
    n_same = top_k - n_cross

    cross_modal_keys = ["figure+table", "figure+formula", "table+formula"]
    same_modal_keys = ["figure+figure", "table+table", "formula+formula"]

    # Allocate cross-modal proportionally to bucket sizes
    cross_sizes = {k: len(buckets.get(k, [])) for k in cross_modal_keys}
    total_cross = sum(cross_sizes.values())
    cross_quotas = {}
    if total_cross > 0:
        for k in cross_modal_keys:
            cross_quotas[k] = min(cross_sizes[k], int(n_cross * cross_sizes[k] / total_cross))
        # Distribute remainder
        remainder = n_cross - sum(cross_quotas.values())
        for k in sorted(cross_modal_keys, key=lambda k: cross_sizes[k], reverse=True):
            if remainder <= 0:
                break
            extra = min(remainder, cross_sizes[k] - cross_quotas[k])
            cross_quotas[k] += extra
            remainder -= extra

    # Allocate same-modal proportionally
    same_sizes = {k: len(buckets.get(k, [])) for k in same_modal_keys}
    total_same = sum(same_sizes.values())
    same_quotas = {}
    if total_same > 0:
        for k in same_modal_keys:
            same_quotas[k] = min(same_sizes[k], int(n_same * same_sizes[k] / total_same))
        remainder = n_same - sum(same_quotas.values())
        for k in sorted(same_modal_keys, key=lambda k: same_sizes[k], reverse=True):
            if remainder <= 0:
                break
            extra = min(remainder, same_sizes[k] - same_quotas[k])
            same_quotas[k] += extra
            remainder -= extra

    # Collect selected pairs
    selected = []
    for bucket_key in cross_modal_keys + same_modal_keys:
        quota = cross_quotas.get(bucket_key, 0) or same_quotas.get(bucket_key, 0)
        bucket_pairs = buckets.get(bucket_key, [])
        selected.extend([p for _, p in bucket_pairs[:quota]])

    # Report
    type_counts = Counter()
    cross_mod = 0
    for p in selected:
        t = f"{p['element_a_type']}+{p['element_b_type']}"
        type_counts[t] += 1
        if p['element_a_type'] != p['element_b_type']:
            cross_mod += 1

    print(f"Element pair pool: {sum(len(v) for v in buckets.values())} total from {len(section_edges)} section edges")
    print(f"Bucket sizes: {dict((k, len(v)) for k, v in buckets.items())}")
    print(f"Selected {len(selected)}: cross-modal={cross_mod}, same-modal={len(selected)-cross_mod}")
    print(f"Type distribution: {type_counts.most_common(10)}")

    return selected


_MODALITY_BUCKETS = {
    "figure+figure", "table+table", "formula+formula",
    "figure+table", "table+figure",
    "figure+formula", "formula+figure",
    "table+formula", "formula+table",
}


# ── Main pipeline ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cross-doc-edges", default="data/01_graphs/cross_doc_sim_edges.json")
    parser.add_argument("--elements", default="data/02_enriched/multimodal_elements_enriched.json")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--limit", type=int, default=1500)
    parser.add_argument("--model", default=os.environ.get("COMPANY_API_MODEL") or DEFAULT_MODEL)
    parser.add_argument("--provider", choices=["company", "anthropic", "openai"], default="anthropic",
                        help="LLM provider (company API is down as of 2026-05-25)")
    parser.add_argument("--delay", type=float, default=0.3)
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM calls, just build pair pool")
    parser.add_argument("--dry-run", type=int, default=0, help="Run only N pairs as dry run")
    args = parser.parse_args()

    # Auto-detect model for provider
    if args.provider == "anthropic":
        args.model = args.model if args.model != DEFAULT_MODEL else "claude-sonnet-4-6"
    elif args.provider == "openai":
        args.model = args.model if args.model != DEFAULT_MODEL else "gpt-4.1"

    # Setup output
    if args.output_dir:
        out_dir = Path(args.output_dir)
        if not out_dir.is_absolute():
            out_dir = ROOT / out_dir
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = ROOT / f"data/05_eval/linguistic_element_level_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Setup API
    client = None
    if args.provider == "company":
        from local_api_logger.logger import APILogger
        import local_api_logger.tracker as tracker
        log_dir = ROOT / "api_logs_cannt_delete"
        log_dir.mkdir(parents=True, exist_ok=True)
        tracker._default_tracker = tracker.APITracker(APILogger(str(log_dir)))
        set_company_credentials(
            os.environ.get("COMPANY_API_URL", ""),
            os.environ.get("COMPANY_API_KEY", ""),
        )
    elif args.provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    elif args.provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Load data
    section_edges = load_section_edges(ROOT / args.cross_doc_edges)
    print(f"Loaded {len(section_edges)} section-level cross-doc edges")
    elements = load_elements_index(ROOT / args.elements)
    print(f"Loaded {len(elements)} elements")

    # Phase 0: Select element pairs
    pairs = select_element_pairs(section_edges, elements, top_k=args.limit)
    pair_pool_path = out_dir / "element_pair_pool.json"
    with open(pair_pool_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)
    print(f"Pair pool saved to {pair_pool_path}")

    if args.skip_llm:
        print("--skip-llm: stopping after pair pool construction")
        return

    if args.dry_run:
        pairs = pairs[:args.dry_run]
        print(f"DRY RUN: processing only {len(pairs)} pairs")

    # Phase 1: Genette+RST + Asymmetry on element text
    total_in = 0
    total_out = 0
    stats = Counter()
    results_path = out_dir / "linguistic_validated_elements.jsonl"

    print(f"\n--- Processing {len(pairs)} element pairs ---")
    budget_est = len(pairs) * 0.0005  # rough $0.0005 per pair for gpt-5.4
    print(f"Estimated budget: ~${budget_est:.2f}")

    for idx, pair in enumerate(pairs, 1):
        try:
            # Phase 2: Genette+RST classification (no decontext needed for elements)
            rst_prompt = build_element_rst_prompt(
                pair["caption_a"], pair["enriched_a"], pair["element_a_type"],
                pair["caption_b"], pair["enriched_b"], pair["element_b_type"],
            )
            raw_rst, tin, tout = call_llm(
                client=client, model=args.model, provider=args.provider,
                prompt=rst_prompt, system_prompt=ELEMENT_RST_SYSTEM,
                max_tokens=400, temperature=0.0, user_tag="element_rst",
            )
            total_in += tin; total_out += tout
            rst = parse_json(raw_rst or "")

            time.sleep(args.delay)

            # Phase 3: Asymmetric verification
            rst_json = json.dumps(rst) if isinstance(rst, dict) else (raw_rst or "")[:300]
            asym_prompt = f"""Original relationship (A→B): {rst_json}

Now test the reverse direction (B→A). Can you construct a meaningful rhetorical relationship?

Element B (now the source, {pair['element_b_type']}):
{(pair['caption_b'] + ' ' + pair['enriched_b'])[:400]}

Element A (now the target, {pair['element_a_type']}):
{(pair['caption_a'] + ' ' + pair['enriched_a'])[:400]}"""

            raw_asym, tin, tout = call_llm(
                client=client, model=args.model, provider=args.provider,
                prompt=asym_prompt, system_prompt=ELEMENT_ASYMMETRIC_SYSTEM,
                max_tokens=300, temperature=0.0, user_tag="element_asym",
            )
            total_in += tin; total_out += tout
            asym = parse_json(raw_asym or "")

            time.sleep(args.delay)

            # Compile result
            genette_type = rst.get("genette_type", "unknown") if isinstance(rst, dict) else "unknown"
            rst_rel = rst.get("rst_relation", "unknown") if isinstance(rst, dict) else "unknown"
            is_causal = rst.get("is_causal_chain", False) if isinstance(rst, dict) else False
            conf = rst.get("confidence", 0.0) if isinstance(rst, dict) else 0.0
            asym_score = asym.get("asymmetry_score", 0.5) if isinstance(asym, dict) else 0.5

            # Quality tier (same rubric as section-level)
            if genette_type == "direct_quotation" and conf >= 0.7:
                tier = "gold"
            elif genette_type == "transformation" and is_causal and conf >= 0.6:
                tier = "strong"
            elif genette_type in ("transformation", "commentary") and conf >= 0.5:
                tier = "weak"
            elif genette_type in ("paratextual", "architextual"):
                tier = "topical"
            else:
                tier = "noise"

            stats[f"genette_{genette_type}"] += 1
            stats[f"tier_{tier}"] += 1
            if is_causal:
                stats["causal_edges"] += 1

            result = {
                **pair,
                "genette_type": genette_type,
                "rst_relation": rst_rel,
                "is_causal_chain": is_causal,
                "confidence": conf,
                "rst_evidence": rst.get("evidence", "") if isinstance(rst, dict) else "",
                "asymmetry_score": asym_score,
                "linguistic_quality_tier": tier,
            }

            with results_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()

            if idx % 10 == 0 or idx == len(pairs):
                usable = stats.get("tier_gold", 0) + stats.get("tier_strong", 0) + stats.get("tier_weak", 0)
                print(
                    f"[{idx:04d}/{len(pairs):04d}] "
                    f"gold={stats['tier_gold']} strong={stats['tier_strong']} "
                    f"weak={stats['tier_weak']} topical={stats['tier_topical']} "
                    f"noise={stats['tier_noise']} usable_rate={usable/max(idx,1):.1%} "
                    f"(tok_in={total_in})"
                )

        except Exception as e:
            stats["exception"] += 1
            print(f"[{idx:04d}/{len(pairs):04d}] EXCEPTION: {e}")

    # Summary
    total_rated = sum(stats.get(f"tier_{t}", 0) for t in
                      ["gold", "strong", "weak", "topical", "noise"])
    usable = stats.get("tier_gold", 0) + stats.get("tier_strong", 0) + stats.get("tier_weak", 0)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "pipeline": "element_level",
        "total_pairs_processed": len(pairs),
        "total_rated": total_rated,
        "genette_distribution": {
            t: stats.get(f"genette_{t}", 0)
            for t in ["direct_quotation", "transformation", "commentary",
                       "paratextual", "architextual", "unknown"]
        },
        "quality_tiers": {
            "gold": stats.get("tier_gold", 0),
            "strong": stats.get("tier_strong", 0),
            "weak": stats.get("tier_weak", 0),
            "topical": stats.get("tier_topical", 0),
            "noise": stats.get("tier_noise", 0),
        },
        "causal_edges": stats.get("causal_edges", 0),
        "usable_rate": round(usable / max(total_rated, 1), 4) if total_rated > 0 else 0,
        "tokens": {"in": total_in, "out": total_out},
        "output_file": str(results_path.relative_to(ROOT)),
    }
    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n=== Element-Level Linguistic Validation ===")
    print(f"  Genette:")
    for t in ["direct_quotation", "transformation", "commentary",
              "paratextual", "architextual", "unknown"]:
        print(f"    {t}: {stats.get('genette_'+t, 0)}")
    print(f"  Quality:")
    for t in ["gold", "strong", "weak", "topical", "noise"]:
        print(f"    {t}: {stats.get('tier_'+t, 0)}")
    print(f"  Usable (gold+strong+weak): {usable}/{total_rated} = {summary['usable_rate']:.1%}")
    print(f"  Causal edges: {stats.get('causal_edges', 0)}")
    print(f"  Tokens: {total_in} in / {total_out} out")
    print(f"\nSummary: {summary_path}")

    log_run(
        script="linguistic_element_level",
        model=f"company:{args.model}",
        purpose=f"Element-level Genette+RST validation of {len(pairs)} cross-doc element pairs",
        input_tokens=total_in,
        output_tokens=total_out,
        extra={
            "pairs_processed": len(pairs),
            "usable": usable,
            "usable_rate": summary["usable_rate"],
            "output": str(out_dir),
        },
    )


if __name__ == "__main__":
    main()
