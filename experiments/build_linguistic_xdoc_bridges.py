#!/usr/bin/env python3
"""Linguistics-informed cross-document bridge validator.

Combines Genette's transtextuality theory + Rhetorical Structure Theory (RST)
to validate and score cross-document edges.

Pipeline (inspired by McManus 2024 + Chen 2025):
  1. Decontextualization: LLM rewrites each element description to be
     independently understandable (no "this figure", "as shown above")
  2. RST relation classification: LLM judges what rhetorical relation
     holds between the two elements (Cause-Effect, Elaboration, Contrast, etc.)
  3. Asymmetric verification: check both A→B and B→A directions
  4. Genette-style scoring: classify into intertextual tiers
     (quotation > hypertextuality > metatextuality > paratextuality)

Output: linguistically-validated cross-doc edges with rhetorical types.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api.llm import call_llm, parse_json, set_company_credentials
from src.utils.token_logger import log_run

DEFAULT_MODEL = "gpt-5.4"

# ── Phase 1: Decontextualization ────────────────────────────────────────────

DECONTEXTUALIZE_SYSTEM = """You are a text normalization expert. Rewrite the given document
excerpt so it is COMPLETELY self-contained and understandable without any surrounding context.

Rules:
1. Replace ALL deictic references (this figure, the above table, as shown, here, etc.)
   with explicit descriptions of what is being referenced.
2. Expand abbreviations and acronyms on first use.
3. If a sentence refers to "the model" or "the method", specify WHICH model/method by name.
4. If a sentence says "it outperforms X", specify WHAT outperforms X.
5. Preserve ALL specific numbers, named entities, technical terms, and factual claims.
6. Do NOT add information that isn't in the original text.
7. Output only the rewritten text, no commentary."""


def build_decontextualize_prompt(text: str, element_type: str, doc_title: str = "") -> str:
    src = f"Source: {element_type}"
    if doc_title:
        src += f" from paper '{doc_title[:200]}'"
    return f"""{src}

Original text:
{text[:800]}

Rewrite this to be fully self-contained and understandable on its own:"""


# ── Phase 2: RST Relation Classification ────────────────────────────────────

RST_SYSTEM = """You are a rhetorical structure analyst. Given two self-contained
descriptions of scientific elements (from potentially different papers), classify the
rhetorical relation between them using Genette's transtextuality framework.

Genette's 5 types, adapted for scientific documents:
1. **direct_quotation** (Intertextuality): Element B explicitly copies, cites, or
   reproduces content from Element A. Evidence: same numbers, same formulas, same figures.
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


def build_rst_classification_prompt(
    elem_a_decontext: str,
    elem_b_decontext: str,
    elem_a_type: str = "unknown",
    elem_b_type: str = "unknown",
) -> str:
    return f"""Classify the cross-document relationship:

Element A ({elem_a_type}):
{elem_a_decontext[:600]}

Element B ({elem_b_type}):
{elem_b_decontext[:600]}

What is the relationship from A to B?"""


# ── Phase 3: Asymmetric Verification ──────────────────────────────────────

ASYMMETRIC_SYSTEM = """You are testing whether a cross-document relationship is
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


# ── Main pipeline ───────────────────────────────────────────────────────────

def load_edges(edges_path: Path) -> List[Dict[str, Any]]:
    with open(edges_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("edges", data if isinstance(data, list) else [])


def load_elements(elements_path: Path) -> Dict[str, Dict[str, Any]]:
    with open(elements_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    elements: Dict[str, Dict[str, Any]] = {}
    docs = data.get("documents", {})
    for doc_id, doc in docs.items():
        for eid, el in doc.get("elements", {}).items():
            elements[eid] = el
    return elements


def get_element_text(el: Dict[str, Any]) -> str:
    """Get best available text for an element."""
    text = el.get("caption", "") or ""
    content = el.get("content", "") or ""
    enriched = el.get("enriched_content", "") or ""
    return f"{text}\n{content}\n{enriched}"[:1200]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cross-doc-edges",
        default="data/01_graphs/cross_doc_sim_edges.json",
    )
    parser.add_argument(
        "--elements",
        default="data/02_enriched/multimodal_elements_enriched.json",
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--model", default=os.environ.get("COMPANY_API_MODEL") or DEFAULT_MODEL)
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()

    # Setup
    if args.output_dir:
        out_dir = Path(args.output_dir)
        if not out_dir.is_absolute():
            out_dir = ROOT / out_dir
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = ROOT / f"data/05_eval/linguistic_xdoc_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    from local_api_logger.logger import APILogger
    import local_api_logger.tracker as tracker
    log_dir = ROOT / "api_logs_cannt_delete"
    log_dir.mkdir(parents=True, exist_ok=True)
    tracker._default_tracker = tracker.APITracker(APILogger(str(log_dir)))
    set_company_credentials(
        os.environ.get("COMPANY_API_URL", ""),
        os.environ.get("COMPANY_API_KEY", ""),
    )

    # Load data
    edges = load_edges(ROOT / args.cross_doc_edges)
    print(f"Loaded {len(edges)} cross-doc edges")
    elements = load_elements(ROOT / args.elements)
    print(f"Loaded {len(elements)} elements")

    if args.limit:
        edges = edges[: args.limit]

    # To get element text, we need to map section nodes → elements.
    # The edges use section summary nodes (e.g., "1104.3913_secsummary_...")
    # We'll use the per-doc element enrichment as the decontextualization source.
    # For each edge, look up elements in both docs.
    doc_elements: Dict[str, List[Tuple[str, Dict]]] = {}
    for eid, el in elements.items():
        doc_id = eid.split("_figure_")[0].split("_table_")[0].split("_formula_")[0]
        if doc_id not in doc_elements:
            doc_elements[doc_id] = []
        doc_elements[doc_id].append((eid, el))

    total_in = 0
    total_out = 0
    stats = Counter()
    results_path = out_dir / "linguistic_validated_edges.jsonl"
    summary_path = out_dir / "summary.json"

    print(f"\n--- Processing {len(edges)} edge pairs ---")

    for idx, edge in enumerate(edges, 1):
        src_doc = edge.get("source_doc", "").strip()
        tgt_doc = edge.get("target_doc", "").strip()
        src_text = edge.get("source_text_preview", "") or ""
        tgt_text = edge.get("target_text_preview", "") or ""

        if not src_text or not tgt_text:
            stats["skipped_no_text"] += 1
            continue

        try:
            # Phase 1: Decontextualize both texts
            src_prompt = build_decontextualize_prompt(src_text, "section", src_doc)
            raw_src, tin, tout = call_llm(
                client=None, model=args.model, provider="company",
                prompt=src_prompt, system_prompt=DECONTEXTUALIZE_SYSTEM,
                max_tokens=600, temperature=0.0, user_tag="decontext",
            )
            total_in += tin; total_out += tout
            src_decontext = raw_src.strip() if raw_src else src_text

            tgt_prompt = build_decontextualize_prompt(tgt_text, "section", tgt_doc)
            raw_tgt, tin, tout = call_llm(
                client=None, model=args.model, provider="company",
                prompt=tgt_prompt, system_prompt=DECONTEXTUALIZE_SYSTEM,
                max_tokens=600, temperature=0.0, user_tag="decontext",
            )
            total_in += tin; total_out += tout
            tgt_decontext = raw_tgt.strip() if raw_tgt else tgt_text

            time.sleep(args.delay)

            # Phase 2: RST relation classification
            rst_prompt = build_rst_classification_prompt(
                src_decontext, tgt_decontext, "section", "section"
            )
            raw_rst, tin, tout = call_llm(
                client=None, model=args.model, provider="company",
                prompt=rst_prompt, system_prompt=RST_SYSTEM,
                max_tokens=400, temperature=0.0, user_tag="rst_classify",
            )
            total_in += tin; total_out += tout
            rst = parse_json(raw_rst or "")

            time.sleep(args.delay)

            # Phase 3: Asymmetric verification
            asym_prompt = f"""Original relationship (A→B): {json.dumps(rst) if isinstance(rst, dict) else raw_rst[:300]}

Now test the reverse direction (B→A). Can you construct a meaningful rhetorical relationship?

Element B (now the source):
{tgt_decontext[:400]}

Element A (now the target):
{src_decontext[:400]}"""
            raw_asym, tin, tout = call_llm(
                client=None, model=args.model, provider="company",
                prompt=asym_prompt, system_prompt=ASYMMETRIC_SYSTEM,
                max_tokens=300, temperature=0.0, user_tag="asymmetric",
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
            reverse_valid = asym.get("reverse_valid", False) if isinstance(asym, dict) else False

            # Genette quality tier
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

            # Composite linguistic score
            lingu_score = (
                conf * 0.4 +
                (1.0 if is_causal else 0.0) * 0.3 +
                asym_score * 0.2 +
                (0.0 if genette_type == "unknown" else 0.1)
            )

            stats[f"genette_{genette_type}"] += 1
            stats[f"tier_{tier}"] += 1
            if is_causal:
                stats["causal_chains"] += 1

            result = {
                **edge,
                "src_decontext": src_decontext[:500],
                "tgt_decontext": tgt_decontext[:500],
                "genette_type": genette_type,
                "rst_relation": rst_rel,
                "is_causal_chain": is_causal,
                "bidirectional": rst.get("bidirectional", False) if isinstance(rst, dict) else False,
                "confidence": conf,
                "rst_evidence": rst.get("evidence", "") if isinstance(rst, dict) else "",
                "asymmetry_score": asym_score,
                "reverse_valid": reverse_valid,
                "linguistic_quality_tier": tier,
                "linguistic_score": round(lingu_score, 4),
            }

            with results_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()

            if idx % 5 == 0 or idx == len(edges):
                print(
                    f"[{idx:03d}/{len(edges):03d}] "
                    f"gold={stats['tier_gold']} "
                    f"strong={stats['tier_strong']} "
                    f"weak={stats['tier_weak']} "
                    f"topical={stats['tier_topical']} "
                    f"noise={stats['tier_noise']} "
                    f"(tok_in={total_in}, tok_out={total_out})"
                )

        except Exception as e:
            stats["exception"] += 1
            print(f"[{idx:03d}/{len(edges):03d}] EXCEPTION: {e}")

    # Summary
    total_processed = sum(1 for t in ["gold", "strong", "weak", "topical", "noise"]
                          for _ in range(stats.get(f"tier_{t}", 0)))
    usable = stats.get("tier_gold", 0) + stats.get("tier_strong", 0) + stats.get("tier_weak", 0)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "total_edges_processed": len(edges),
        "total_rated": total_processed if total_processed > 0 else sum(
            stats[f"genette_{t}"] for t in
            ["direct_quotation", "transformation", "commentary", "paratextual", "architextual", "unknown"]
        ),
        "genette_distribution": {
            t: stats[f"genette_{t}"]
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
        "causal_chains": stats["causal_chains"],
        "usable_rate": round(
            usable / max(total_processed, 1), 4
        ) if total_processed > 0 else 0,
        "tokens": {"in": total_in, "out": total_out},
        "output_file": str(results_path.relative_to(ROOT)),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n=== Linguistic XDoc Validation Results ===")
    print(f"  Genette types:")
    for t in ["direct_quotation", "transformation", "commentary",
              "paratextual", "architextual", "unknown"]:
        print(f"    {t}: {stats['genette_'+t]}")
    print(f"  Quality tiers:")
    for t in ["gold", "strong", "weak", "topical", "noise"]:
        print(f"    {t}: {stats.get('tier_'+t, 0)}")
    print(f"  Causal chains: {stats['causal_chains']}")
    print(f"  Total tokens: {total_in} in / {total_out} out")
    print(f"\nSummary: {summary_path}")

    log_run(
        script="linguistic_xdoc_validator",
        model=f"company:{args.model}",
        purpose=f"Genette+RST linguistic validation of {len(edges)} cross-doc edges",
        input_tokens=total_in,
        output_tokens=total_out,
        extra={
            "edges_processed": len(edges),
            "strong_chains": stats["causal_chains"],
            "output": str(out_dir),
        },
    )


if __name__ == "__main__":
    main()
