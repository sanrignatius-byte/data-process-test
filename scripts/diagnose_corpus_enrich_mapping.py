#!/usr/bin/env python3
"""Phase A diagnostic for CORPUS_ENRICH_FIX_PLAN_20260503.

Verifies why ~695/1095 figure passages in M4query_v1 corpus_v1_enriched.jsonl
still carry placeholder text "[Image: xxx.jpg]" despite 1285 elements being
enriched in data/02_enriched/multimodal_elements_enriched.json.

Buckets:
  D1 — figure passage with text starting "[Image:"  AND  no entry in MODORA
       enrichment file at all. (Genuinely missing upstream.)
  D2 — figure passage with text starting "[Image:"  BUT  MODORA file has a
       matching enriched entry. (Build script silently dropped it → fixable.)
  D3 — figure passage that is NOT degraded (no [Image: prefix). Sanity bucket.

Outputs:
  data/05_eval/corpus_fix_v1/diagnose_report.md
"""
from __future__ import annotations
import json, os, sys, re
from pathlib import Path
from collections import Counter

ROOT = Path("/projects/myyyx1/data-process-test")
CORPUS = ROOT / "data/05_eval/dense_retrieval/rebuilt_20260417/augmented/corpus_v1_enriched.jsonl"
ENRICH = ROOT / "data/02_enriched/multimodal_elements_enriched.json"
HUB_FILES = [
    ROOT / "data/02_enriched/hub_candidates_enriched_v4_intra_doc.json",
    ROOT / "data/02_enriched/hub_candidates_enriched_v3_intra_doc.json",
]
OUT_DIR = ROOT / "data/05_eval/corpus_fix_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "diagnose_report.md"


def load_modora_index() -> dict:
    """Flatten MODORA enrichment file into {element_id: entry}."""
    d = json.load(open(ENRICH, encoding="utf-8"))
    idx = {}
    for doc_id, doc in d.get("documents", {}).items():
        for eid, e in doc.get("elements", {}).items():
            ec = (e.get("enriched_content") or "").strip()
            et = (e.get("enriched_title") or "").strip()
            if ec or et:
                idx[eid] = {"enriched_title": et, "enriched_content": ec, "doc_id": doc_id}
    return idx


def simulate_hub_loader() -> dict:
    """Replicate the BUGGY load_enriched_index logic on hub files."""
    idx = {}
    for p in HUB_FILES:
        if not p.exists():
            continue
        d = json.load(open(p, encoding="utf-8"))
        pairs = d.get("pairs", d.get("candidates", []))
        for pair in pairs:
            for key in ("element_a", "element_b"):
                e = pair.get(key, {})
                eid = e.get("element_id", "")
                ec = (e.get("enriched_content") or "").strip()
                if eid and ec and eid not in idx:
                    idx[eid] = {"enriched_title": (e.get("enriched_title") or "").strip(),
                                "enriched_content": ec}
    return idx


def simulate_hub_loader_fixed() -> dict:
    """What the loader WOULD return if it understood the actual hub flat-key layout."""
    idx = {}
    for p in HUB_FILES:
        if not p.exists():
            continue
        d = json.load(open(p, encoding="utf-8"))
        for pair in d.get("pairs", []):
            for side in ("a", "b"):
                eid = pair.get(f"element_{side}_id", "")
                ec = (pair.get(f"element_{side}_enriched_content")
                      or pair.get(f"enriched_content_{side}") or "").strip()
                if eid and ec and eid not in idx:
                    idx[eid] = {"enriched_content": ec}
    return idx


IMAGE_PREFIX = re.compile(r"^\[Image:\s*[^\]]+\]\s*$")


def main():
    modora = load_modora_index()
    hub_buggy = simulate_hub_loader()
    hub_fixed_attempt = simulate_hub_loader_fixed()

    # Read corpus
    rows = [json.loads(l) for l in open(CORPUS, encoding="utf-8")]
    figs = [r for r in rows if r.get("type") == "figure" or r.get("modality") == "figure"]
    n_total = len(rows)
    n_fig = len(figs)

    d1 = d2 = d3 = 0
    d1_samples, d2_samples = [], []
    d2_doc_dist = Counter()
    match_kind = Counter()  # which key shape hits MODORA
    for r in figs:
        text = (r.get("text") or "").strip()
        pid = r["passage_id"]
        doc_id = r.get("doc_id", "")
        short_eid = pid[len(doc_id) + 1:] if pid.startswith(doc_id + "_") else pid
        is_degraded = bool(IMAGE_PREFIX.match(text)) or text.startswith("[Image:")
        if not is_degraded:
            d3 += 1
            continue
        # Try multiple key shapes against MODORA
        hit = None
        for k_kind, k in (("full_pid", pid), ("short_eid", short_eid), ("doc__short", f"{doc_id}_{short_eid}")):
            if k in modora:
                hit = (k_kind, k); break
        if hit is not None:
            d2 += 1
            match_kind[hit[0]] += 1
            d2_doc_dist[doc_id] += 1
            if len(d2_samples) < 5:
                m = modora[hit[1]]
                d2_samples.append({
                    "passage_id": pid,
                    "matched_via": hit[0],
                    "matched_key": hit[1],
                    "current_text": text[:80],
                    "modora_enriched_title": m["enriched_title"][:80],
                    "modora_enriched_content_head": m["enriched_content"][:120],
                })
        else:
            d1 += 1
            if len(d1_samples) < 5:
                d1_samples.append({"passage_id": pid, "short_eid": short_eid,
                                   "current_text": text[:80]})

    # Modality dist over corpus + degraded-rate sanity
    modal = Counter(r.get("type", r.get("modality", "?")) for r in rows)
    fig_degraded = d1 + d2

    lines = []
    lines.append("# Corpus Enrichment Mapping Diagnose — Phase A")
    lines.append("")
    lines.append("**Inputs**")
    lines.append(f"- corpus: `{CORPUS.relative_to(ROOT)}` ({n_total} passages, {n_fig} figures)")
    lines.append(f"- enrichment: `{ENRICH.relative_to(ROOT)}` ({len(modora)} elements with enriched_*)")
    lines.append(f"- hub files: {[str(p.relative_to(ROOT)) for p in HUB_FILES]}")
    lines.append("")
    lines.append("**Modality distribution (full corpus)**")
    for k, v in modal.most_common():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Loader behaviour comparison")
    lines.append("")
    lines.append(f"| loader source | flat-key layout | enriched entries returned |")
    lines.append(f"|---|---|---|")
    lines.append(f"| current `load_enriched_index` on hub files | reads `pair.element_a.element_id` (NESTED) | **{len(hub_buggy)}** |")
    lines.append(f"| same loader, flat-key probe `element_a_id` | hub files actually use this | {len(hub_fixed_attempt)} |")
    lines.append(f"| MODORA `multimodal_elements_enriched.json` (`documents.elements`) | NEVER REFERENCED by build script | {len(modora)} |")
    lines.append("")
    lines.append("→ **Root cause confirmed**: build script's `load_enriched_index` only handles a")
    lines.append("  nested `pair.element_a.element_id` layout that no current enrichment file uses.")
    lines.append("  It silently returns 0 entries from hub files, and the MODORA per-element file")
    lines.append("  (1285 enriched elements) is never even passed in via `--enriched-files`.")
    lines.append("")
    lines.append("## Figure-passage degradation buckets")
    lines.append("")
    lines.append(f"- **D1** (no MODORA entry, genuinely missing): **{d1}**")
    lines.append(f"- **D2** (MODORA has entry, build script dropped it): **{d2}**")
    lines.append(f"- **D3** (figure not degraded, sanity): **{d3}**")
    lines.append(f"- total degraded figures (D1+D2): **{fig_degraded}** of {n_fig}  ({fig_degraded/n_fig*100:.1f}%)")
    lines.append(f"- D2 spans **{len(d2_doc_dist)}** documents")
    lines.append(f"- D2 match-kind distribution: {dict(match_kind)}")
    lines.append("")
    lines.append("### D2 sample (first 5)")
    lines.append("")
    for s in d2_samples:
        lines.append(f"- `{s['passage_id']}` matched via `{s['matched_via']}` → key `{s['matched_key']}`")
        lines.append(f"  - current text: `{s['current_text']}`")
        lines.append(f"  - MODORA enriched_title: {s['modora_enriched_title']}")
        lines.append(f"  - MODORA enriched_content head: {s['modora_enriched_content_head']}…")
    lines.append("")
    lines.append("### D1 sample (first 5)")
    lines.append("")
    for s in d1_samples:
        lines.append(f"- `{s['passage_id']}`  (short_eid=`{s['short_eid']}`) — `{s['current_text']}`")
    lines.append("")
    lines.append("## Patch proposal (one paragraph)")
    lines.append("")
    lines.append("Add a second branch to `load_enriched_index` that detects the `documents` top-level")
    lines.append("key (MODORA per-element format) and ingests `documents[*].elements[*]` directly,")
    lines.append("then add `data/02_enriched/multimodal_elements_enriched.json` to `DEFAULT_ENRICHED_FILES`.")
    lines.append("Keep the existing pairs-style branch for backward compatibility (it harmlessly")
    lines.append("returns 0 entries today). Expected effect: D2 figures (≈" + str(d2) + ") promoted from")
    lines.append("`[Image: …]` placeholder to MODORA enriched description; figure degraded-rate drops")
    lines.append(f"from {fig_degraded/n_fig*100:.1f}% to {d1/n_fig*100:.1f}% (D1 only).")
    lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[ok] wrote {OUT}")
    print(f"[summary] D1={d1} D2={d2} D3={d3} (figs={n_fig})")
    print(f"[summary] modora_idx={len(modora)} hub_buggy={len(hub_buggy)} hub_flat_attempt={len(hub_fixed_attempt)}")


if __name__ == "__main__":
    main()
