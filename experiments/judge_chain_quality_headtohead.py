#!/usr/bin/env python3
"""Chain-level head-to-head: does linguistic filtering retain higher-quality chains?

Setup:
  - Two groups, both drawn from B_raw_matched_100 chains (same 100 raw CLIP edges):
      Group VALIDATED  — chains whose cross-doc bridge passes through a doc-pair
                         that linguistic validator labelled strong|weak
      Group REJECTED   — chains whose cross-doc bridge passes through a doc-pair
                         that linguistic validator labelled topical|noise|unknown
  - Both groups drawn from the *same chain source* (B), so the only difference
    is which doc-pairs they cross. This isolates linguistic-filter effect.

Judge:
  For each chain, present element captions + enriched content snippets to LLM
  and ask for a usable / weak / noise verdict on the chain itself.

Result interpretation:
  - VALIDATED >> REJECTED (on usable rate) → linguistic filter does keep
    semantically higher-quality bridges.
  - VALIDATED ≈ REJECTED                  → filter discards useful bridges
    along with the noise; filter offers no chain-quality improvement.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api.llm import call_llm, parse_json, set_company_credentials  # noqa: E402
from src.utils.token_logger import log_run  # noqa: E402


JUDGE_SYSTEM = (
    "You judge whether a candidate cross-document multi-hop reasoning chain "
    "across academic papers is genuinely meaningful, not just topologically "
    "reachable. Return strict JSON only."
)


JUDGE_TEMPLATE = """A candidate reasoning chain traverses the following elements ({hop_count}-hop, cross-document, cross-modal). Consecutive steps must support stepwise reasoning; one transition crosses the document boundary.

{steps_block}

Cross-doc boundary crossed at: between step {bridge_step_a} ({bridge_doc_a}) and step {bridge_step_b} ({bridge_doc_b}).

Judge on three criteria:
1. SEMANTIC COHERENCE — Do consecutive elements relate so that one informs interpretation of the next?
2. CROSS-DOC BRIDGE — Where the chain crosses documents, is the connection substantive (not just same broad topic)?
3. RESEARCHER UTILITY — Could a domain researcher ask a question whose answer genuinely requires walking this chain in sequence?

Verdicts:
- "usable" — all three criteria pass; chain supports a real multi-hop question
- "weak"   — some connection but needs extra bridging; chain is borderline
- "noise"  — no real reasoning relationship; topology coincidence

Return STRICT JSON:
{{"verdict": "usable" | "weak" | "noise",
  "cross_doc_bridge_quality": "strong" | "weak" | "none",
  "reasoning": "one or two sentences"}}"""


def doc_id_of(eid: str) -> str:
    for tag in ("_figure_", "_table_", "_formula_"):
        if tag in eid:
            return eid.split(tag)[0]
    return eid


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def load_elements(path: Path) -> Dict[str, Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for _, doc in data.get("documents", {}).items():
        for eid, el in doc.get("elements", {}).items():
            out[eid] = el
    return out


def bridge_doc_pair(path: List[str]) -> Tuple[str, str] | None:
    """Find the (doc_a, doc_b) pair where the chain crosses doc boundary."""
    docs = [doc_id_of(e) for e in path]
    for i in range(len(docs) - 1):
        if docs[i] != docs[i + 1]:
            a, b = docs[i], docs[i + 1]
            return (a, b) if a < b else (b, a)
    return None


def bridge_step_indices(path: List[str]) -> Tuple[int, int]:
    """Return 1-indexed step indices where doc-boundary is crossed."""
    docs = [doc_id_of(e) for e in path]
    for i in range(len(docs) - 1):
        if docs[i] != docs[i + 1]:
            return (i + 1, i + 2)
    return (1, 2)


def element_snippet(el: Dict[str, Any], max_chars: int = 280) -> str:
    parts = []
    caption = (el.get("caption") or "").strip()
    if caption:
        parts.append(f"caption: {caption}")
    enriched = (el.get("enriched_content") or el.get("content") or "").strip()
    if enriched:
        parts.append(f"content: {enriched}")
    s = " | ".join(parts)
    return s[:max_chars] + ("…" if len(s) > max_chars else "")


def format_chain_for_prompt(
    path: List[str],
    elements: Dict[str, Dict[str, Any]],
) -> str:
    rows = []
    for i, eid in enumerate(path, 1):
        el = elements.get(eid, {})
        etype = el.get("element_type", "?")
        rows.append(
            f"[Step {i}] doc={doc_id_of(eid)} | type={etype} | id={eid}\n"
            f"         {element_snippet(el)}"
        )
    return "\n".join(rows)


def call_judge(
    chain: Dict[str, Any],
    elements: Dict[str, Dict[str, Any]],
    model: str,
) -> Tuple[Dict[str, Any], int, int]:
    path = chain["path"]
    steps_block = format_chain_for_prompt(path, elements)
    ba, bb = bridge_step_indices(path)
    docs = [doc_id_of(e) for e in path]
    prompt = JUDGE_TEMPLATE.format(
        hop_count=chain["hop_count"],
        steps_block=steps_block,
        bridge_step_a=ba,
        bridge_step_b=bb,
        bridge_doc_a=docs[ba - 1],
        bridge_doc_b=docs[bb - 1],
    )
    raw, tin, tout = call_llm(
        client=None, model=model, provider="company",
        prompt=prompt, system_prompt=JUDGE_SYSTEM,
        max_tokens=400, temperature=0.0, user_tag="chain_judge",
    )
    parsed = parse_json(raw or "") or {}
    if not isinstance(parsed, dict):
        parsed = {"verdict": "noise", "reasoning": "parse_fail", "raw": raw[:200] if raw else ""}
    return parsed, tin, tout


def stratified_sample(
    chains: List[Dict[str, Any]],
    n: int,
    seed: int,
) -> List[Dict[str, Any]]:
    by_hop = defaultdict(list)
    for c in chains:
        by_hop[c["hop_count"]].append(c)
    rng = random.Random(seed)
    keys = sorted(by_hop.keys())
    per_bucket = max(1, n // max(1, len(keys)))
    picked: List[Dict[str, Any]] = []
    for k in keys:
        bucket = by_hop[k][:]
        rng.shuffle(bucket)
        picked.extend(bucket[:per_bucket])
    rng.shuffle(picked)
    return picked[:n]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fair-dir",
                    default="data/05_eval/graph_linguistic_fusion_fair_20260524T134916Z")
    ap.add_argument("--linguistic-edges",
                    default="data/05_eval/linguistic_xdoc_20260524T124648Z/linguistic_validated_edges.jsonl")
    ap.add_argument("--elements",
                    default="data/02_enriched/multimodal_elements_enriched.json")
    ap.add_argument("--output-dir", default="")
    ap.add_argument("--n-per-group", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--model", default="gpt-5.4")
    ap.add_argument("--delay", type=float, default=0.25)
    args = ap.parse_args()

    fair_dir = ROOT / args.fair_dir
    chains_B = json.loads((fair_dir / "chains_B_raw_matched_100.json").read_text())["chains"]
    lingu_edges = load_jsonl(ROOT / args.linguistic_edges)
    elements = load_elements(ROOT / args.elements)

    print(f"B (raw matched 100) chains: {len(chains_B)}")
    print(f"linguistic edges: {len(lingu_edges)}")
    print(f"elements: {len(elements)}")

    validated_doc_pairs: Set[Tuple[str, str]] = set()
    rejected_doc_pairs: Set[Tuple[str, str]] = set()
    for e in lingu_edges:
        tier = e.get("linguistic_quality_tier", "")
        a, b = e.get("source_doc", ""), e.get("target_doc", "")
        if not a or not b:
            continue
        key = (a, b) if a < b else (b, a)
        if tier in ("strong", "weak", "gold"):
            validated_doc_pairs.add(key)
        elif tier in ("topical", "noise"):
            rejected_doc_pairs.add(key)
    print(f"validated doc-pairs (strong/weak/gold): {len(validated_doc_pairs)}")
    print(f"rejected doc-pairs (topical/noise): {len(rejected_doc_pairs)}")

    chains_val: List[Dict[str, Any]] = []
    chains_rej: List[Dict[str, Any]] = []
    for c in chains_B:
        bridge = bridge_doc_pair(c["path"])
        if bridge is None:
            continue
        c2 = dict(c, bridge_doc_pair=list(bridge))
        if bridge in validated_doc_pairs:
            chains_val.append(c2)
        elif bridge in rejected_doc_pairs:
            chains_rej.append(c2)

    print(f"\nB chains crossing VALIDATED doc-pairs: {len(chains_val)}")
    print(f"B chains crossing REJECTED  doc-pairs: {len(chains_rej)}")

    sample_val = stratified_sample(chains_val, args.n_per_group, args.seed)
    sample_rej = stratified_sample(chains_rej, args.n_per_group, args.seed + 1)
    print(f"sampled VALIDATED: {len(sample_val)}, REJECTED: {len(sample_rej)}")

    # API setup
    from local_api_logger.logger import APILogger
    import local_api_logger.tracker as tracker
    log_dir = ROOT / "api_logs_cannt_delete"
    log_dir.mkdir(parents=True, exist_ok=True)
    tracker._default_tracker = tracker.APITracker(APILogger(str(log_dir)))
    set_company_credentials(
        os.environ.get("COMPANY_API_URL", ""),
        os.environ.get("COMPANY_API_KEY", ""),
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) if args.output_dir else \
        ROOT / f"data/05_eval/chain_judge_h2h_{stamp}"
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "verdicts.jsonl"
    fout = out_path.open("w", encoding="utf-8")

    total_in = 0
    total_out = 0
    counts: Dict[str, Counter] = {"VALIDATED": Counter(), "REJECTED": Counter()}

    for group_name, samples in (("VALIDATED", sample_val), ("REJECTED", sample_rej)):
        print(f"\n=== Judging group {group_name} (n={len(samples)}) ===")
        for i, c in enumerate(samples, 1):
            try:
                verdict, tin, tout = call_judge(c, elements, args.model)
                total_in += tin
                total_out += tout
                counts[group_name][verdict.get("verdict", "noise")] += 1
                rec = {
                    "group": group_name,
                    "path": c["path"],
                    "hop_count": c["hop_count"],
                    "modality_pair": c["modality_pair"],
                    "bridge_doc_pair": c["bridge_doc_pair"],
                    "verdict": verdict,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                if i % 5 == 0 or i == len(samples):
                    print(f"  [{i:3d}/{len(samples)}] verdict counts so far: {dict(counts[group_name])}")
            except Exception as e:
                print(f"  [{i:3d}] EXCEPTION: {e}")
                counts[group_name]["error"] += 1
            time.sleep(args.delay)

    fout.close()

    def rate(group: str, key: str) -> float:
        n = sum(counts[group].values()) - counts[group].get("error", 0)
        return counts[group][key] / n if n else 0.0

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_per_group": args.n_per_group,
        "seed": args.seed,
        "model": args.model,
        "validated_doc_pairs_total": len(validated_doc_pairs),
        "rejected_doc_pairs_total": len(rejected_doc_pairs),
        "validated_chains_pool": len(chains_val),
        "rejected_chains_pool": len(chains_rej),
        "verdict_counts": {g: dict(c) for g, c in counts.items()},
        "usable_rate": {g: rate(g, "usable") for g in counts},
        "weak_rate": {g: rate(g, "weak") for g in counts},
        "noise_rate": {g: rate(g, "noise") for g in counts},
        "tokens": {"in": total_in, "out": total_out},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n=== Head-to-head result ===")
    for g in ("VALIDATED", "REJECTED"):
        print(f"  {g}: counts={dict(counts[g])}  "
              f"usable={summary['usable_rate'][g]:.3f}  "
              f"weak={summary['weak_rate'][g]:.3f}  "
              f"noise={summary['noise_rate'][g]:.3f}")
    print(f"\nUsable delta (VALIDATED - REJECTED): "
          f"{summary['usable_rate']['VALIDATED'] - summary['usable_rate']['REJECTED']:+.3f}")
    print(f"Tokens: {total_in} in / {total_out} out")
    print(f"\nWritten: {out_dir}")

    log_run(
        script="judge_chain_quality_headtohead",
        model=f"company:{args.model}",
        purpose=(
            f"Chain-level head-to-head: linguistic-VALIDATED vs REJECTED "
            f"doc-pairs from B raw matched 100 (n={args.n_per_group} each)"
        ),
        input_tokens=total_in,
        output_tokens=total_out,
        extra={
            "n_per_group": args.n_per_group,
            "validated_pool": len(chains_val),
            "rejected_pool": len(chains_rej),
            "usable_validated": summary["usable_rate"]["VALIDATED"],
            "usable_rejected": summary["usable_rate"]["REJECTED"],
            "output": str(out_dir),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
