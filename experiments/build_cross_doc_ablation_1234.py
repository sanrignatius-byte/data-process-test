#!/usr/bin/env python3
"""Build four cross-document chain ablation inputs.

Strategies:
1. path_baseline_fixed: existing fixed 2-hop path chains (already judged).
2. entity_cluster: same entity anchors 3 papers.
3. gated_path: fixed 2-hop paths with named-entity and middle-relay gates.
4. entity_cluster_enriched: entity_cluster plus coarse entity context.
"""

from __future__ import annotations

import itertools
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/05_eval/cross_doc_ablation_1234"

PAIR_PACK = ROOT / "data/05_eval/entity_bridge_candidates_v2/judge_pack.jsonl"
FIXED_CHAINS = ROOT / "data/05_eval/cross_doc_chains_final_fixed.json"
RAW_FIXED_CHAINS = ROOT / "data/05_eval/entity_bridge_chains_53_fixed_20260522T0910Z/chains.jsonl"
ENRICHED = ROOT / "data/02_enriched/multimodal_elements_enriched.json"

GENERIC_ENTITIES = {
    "linear model", "outcome", "predictor", "adversarial training",
    "clustering", "2d embedding", "reconstruction", "autoencoder",
    "invariance", "discriminators", "overlap", "distribution comparison",
    "structural equation", "empirical risk", "hypothesis class",
    "marginalization", "exogenous noise", "mediator",
}

BAD_ARTIFACT_TERMS = {
    "legend", "square marker", "typographic", "glyph", "icon",
    "marker", "axis", "plot label",
}

NAMED_ENTITIES = {
    "winobias", "ontonotes", "coreference resolution", "compas",
    "recidivism", "demographic parity", "equal opportunity",
    "positive prediction rate", "false positive rate", "g-formula",
    "path-specific fairness", "counterfactual", "causal dag",
    "mediation", "protected attribute a",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, chains: list[dict[str, Any]], strategy: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    papers = sorted({p for c in chains for p in c.get("papers", [])})
    payload = {
        "strategy": strategy,
        "chains": chains,
        "stats": {
            "total": len(chains),
            "papers": len(papers),
            "element_types": dict(Counter(t for c in chains for t in c.get("element_types", []))),
            "top_entities": Counter(e for c in chains for e in c.get("shared_entities", [])).most_common(20),
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with path.with_suffix(".jsonl").open("w", encoding="utf-8") as f:
        for chain in chains:
            f.write(json.dumps(chain, ensure_ascii=False) + "\n")


def clean_entity(entity: str) -> str:
    return " ".join(str(entity or "").lower().strip().split())


def is_bad_entity(entity: str) -> bool:
    e = clean_entity(entity)
    return any(term in e for term in BAD_ARTIFACT_TERMS)


def is_generic(entity: str) -> bool:
    return clean_entity(entity) in GENERIC_ENTITIES


def is_named(entity: str) -> bool:
    e = clean_entity(entity)
    return e in NAMED_ENTITIES or any(token in e for token in ("winobias", "ontonotes", "compas"))


def entity_specificity(entity: str) -> float:
    e = clean_entity(entity)
    if is_bad_entity(e):
        return -10.0
    score = 0.0
    if is_named(e):
        score += 8.0
    if not is_generic(e):
        score += 3.0
    words = e.split()
    score += min(len(words), 4) * 0.8
    if re.search(r"\d|[A-Z]", entity):
        score += 1.5
    if len(e) < 4:
        score -= 2.0
    return score


def load_enriched_index() -> dict[str, dict[str, Any]]:
    data = load_json(ENRICHED)
    idx: dict[str, dict[str, Any]] = {}
    for doc_id, doc in data.get("documents", {}).items():
        for eid, el in doc.get("elements", {}).items():
            idx[eid] = {
                "doc_id": doc_id,
                "element_id": eid,
                "element_type": el.get("element_type", ""),
                "caption": el.get("caption", "") or el.get("content", ""),
                "enriched_title": el.get("enriched_title", ""),
                "enriched_content": el.get("enriched_content", ""),
            }
    return idx


def element_from_id(eid: str, doc_id: str, enriched: dict[str, dict[str, Any]]) -> dict[str, Any]:
    el = dict(enriched.get(eid, {}))
    el.setdefault("doc_id", doc_id)
    el.setdefault("element_id", eid)
    el.setdefault("element_type", "")
    el.setdefault("caption", "")
    el.setdefault("enriched_title", "")
    el.setdefault("enriched_content", "")
    return {
        "doc_id": el.get("doc_id") or doc_id,
        "element_id": eid,
        "element_type": el.get("element_type", ""),
        "caption": el.get("caption", ""),
        "enriched_title": el.get("enriched_title", ""),
        "role": "",
    }


def aggregate_entity_context(entity: str, eids: list[str], enriched: dict[str, dict[str, Any]]) -> str:
    snippets = []
    seen = set()
    for eid in eids:
        if eid in seen:
            continue
        seen.add(eid)
        el = enriched.get(eid, {})
        title = el.get("enriched_title", "")
        content = el.get("enriched_content", "")
        doc = el.get("doc_id", "")
        if title or content:
            snippets.append(f"[{doc}] {title}. {content[:500]}")
        if len(snippets) >= 8:
            break
    if not snippets:
        return f"No enriched context found for {entity}."
    return " ".join(snippets)[:1800]


def build_pair_entity_index(pairs: list[dict[str, Any]], enriched: dict[str, dict[str, Any]]) -> tuple[dict, dict]:
    mentions: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    pair_scores: dict[tuple[str, str, str], float] = {}
    for row in pairs:
        ents = [clean_entity(e) for e in row.get("_meta", {}).get("shared_entities", [])]
        ents = [e for e in ents if e and not is_bad_entity(e)]
        src_doc = row.get("source_doc", "")
        tgt_doc = row.get("target_doc", "")
        src_eid = row.get("source_element_id", "")
        tgt_eid = row.get("target_element_id", "")
        score = float(row.get("_meta", {}).get("entity_bridge_score", 0) or 0)
        for ent in ents:
            for doc_id, eid in ((src_doc, src_eid), (tgt_doc, tgt_eid)):
                if not doc_id or not eid:
                    continue
                el = element_from_id(eid, doc_id, enriched)
                prev = mentions[ent].get(doc_id)
                if prev is None or score > prev.get("_score", 0):
                    el["_score"] = score
                    mentions[ent][doc_id] = el
            key = (ent, *sorted([src_doc, tgt_doc]))
            pair_scores[key] = max(pair_scores.get(key, 0.0), score)
    return mentions, pair_scores


def chain_from_entity_cluster(
    chain_id: str,
    entity: str,
    docs: tuple[str, str, str],
    mentions: dict[str, dict[str, dict[str, Any]]],
    pair_scores: dict[tuple[str, str, str], float],
    enriched: dict[str, dict[str, Any]],
    with_context: bool,
) -> dict[str, Any]:
    elements = []
    for idx, doc in enumerate(docs):
        el = dict(mentions[entity][doc])
        el.pop("_score", None)
        el["role"] = ["chain_start", "chain_joint", "chain_end"][idx]
        elements.append(el)
    bridges = []
    for a, b in ((docs[0], docs[1]), (docs[1], docs[2])):
        score = pair_scores.get((entity, *sorted([a, b])), 0.0)
        bridges.append({
            "type": "cross_doc_entity",
            "from_doc": a,
            "to_doc": b,
            "from_element_id": mentions[entity][a]["element_id"],
            "to_element_id": mentions[entity][b]["element_id"],
            "shared_entities": [entity],
            "clean_shared_entities": [entity],
            "bridge_score": round(score, 3),
            "bridge_description": f"Papers [{a}] and [{b}] both discuss: {entity}",
        })
    context = {}
    if with_context:
        context[entity] = aggregate_entity_context(
            entity, [e["element_id"] for e in elements], enriched
        )
    base_score = sum(b["bridge_score"] for b in bridges) + entity_specificity(entity)
    return {
        "chain_id": chain_id,
        "source_chain_id": "",
        "strategy": "entity_cluster_enriched" if with_context else "entity_cluster",
        "papers": list(docs),
        "n_papers": 3,
        "n_elements": len(elements),
        "n_bridges": 2,
        "score": round(base_score, 3),
        "shared_entities": [entity],
        "joint_entities": [entity],
        "entity_context": context,
        "element_types": [e.get("element_type", "") for e in elements],
        "bridge_types": ["cross_doc_entity", "cross_doc_entity"],
        "elements": elements,
        "bridges": bridges,
    }


def build_entity_cluster_chains(with_context: bool) -> list[dict[str, Any]]:
    enriched = load_enriched_index()
    pairs = load_jsonl(PAIR_PACK)
    mentions, pair_scores = build_pair_entity_index(pairs, enriched)
    raw = []
    for ent, by_doc in mentions.items():
        if len(by_doc) < 3:
            continue
        if entity_specificity(ent) <= 0:
            continue
        docs = sorted(by_doc, key=lambda d: by_doc[d].get("_score", 0), reverse=True)[:8]
        for combo in itertools.combinations(docs, 3):
            chain = chain_from_entity_cluster("tmp", ent, combo, mentions, pair_scores, enriched, with_context)
            raw.append(chain)
    raw.sort(key=lambda c: c["score"], reverse=True)

    final = []
    seen = set()
    per_entity: Counter[str] = Counter()
    for chain in raw:
        ent = chain["shared_entities"][0]
        if per_entity[ent] >= 8:
            continue
        key = (ent, tuple(sorted(chain["papers"])))
        if key in seen:
            continue
        seen.add(key)
        per_entity[ent] += 1
        chain["chain_id"] = (
            f"xdoc_cluster_enriched_{len(final):04d}" if with_context
            else f"xdoc_cluster_{len(final):04d}"
        )
        final.append(chain)
        if len(final) >= 50:
            break
    return final


def normalize_fixed_chain(c: dict[str, Any], strategy: str) -> dict[str, Any]:
    out = dict(c)
    out["strategy"] = strategy
    return out


def build_gated_path_chains() -> list[dict[str, Any]]:
    data = load_json(FIXED_CHAINS)
    candidates = []
    for c in data.get("chains", []):
        bridges = c.get("bridges", [])
        if len(bridges) != 2 or len(c.get("papers", [])) != 3:
            continue
        ents1 = {clean_entity(e) for e in bridges[0].get("clean_shared_entities", bridges[0].get("shared_entities", []))}
        ents2 = {clean_entity(e) for e in bridges[1].get("clean_shared_entities", bridges[1].get("shared_entities", []))}
        all_ents = ents1 | ents2
        if any(is_bad_entity(e) for e in all_ents):
            continue
        named = {e for e in all_ents if is_named(e)}
        overlap = (ents1 & ents2) - GENERIC_ENTITIES
        middle_doc = c["papers"][1]
        middle_eids = []
        for b in bridges:
            if b.get("from_doc") == middle_doc:
                middle_eids.append(b.get("from_element_id", ""))
            if b.get("to_doc") == middle_doc:
                middle_eids.append(b.get("to_element_id", ""))
        same_middle = len(set(e for e in middle_eids if e)) == 1
        if not (named or overlap or same_middle):
            continue
        non_generic_count = sum(1 for e in all_ents if not is_generic(e))
        out = normalize_fixed_chain(c, "gated_path")
        out["gate_features"] = {
            "named_entities": sorted(named),
            "bridge_entity_overlap": sorted(overlap),
            "same_middle_element": same_middle,
            "non_generic_count": non_generic_count,
        }
        out["score"] = round(float(c.get("score", 0)) + len(named) * 5 + len(overlap) * 3 + (4 if same_middle else 0), 3)
        candidates.append(out)
    candidates.sort(key=lambda c: c["score"], reverse=True)
    final = []
    seen = set()
    for c in candidates:
        key = frozenset(e["element_id"] for e in c.get("elements", []))
        if key in seen:
            continue
        seen.add(key)
        c["chain_id"] = f"xdoc_gated_path_{len(final):04d}"
        final.append(c)
        if len(final) >= 50:
            break
    return final


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    baseline = [normalize_fixed_chain(c, "path_baseline_fixed") for c in load_json(FIXED_CHAINS)["chains"]]
    clusters = build_entity_cluster_chains(with_context=False)
    gated = build_gated_path_chains()
    clusters_enriched = build_entity_cluster_chains(with_context=True)

    write_json(OUT / "path_baseline_fixed.json", baseline, "path_baseline_fixed")
    write_json(OUT / "entity_cluster_chains.json", clusters, "entity_cluster")
    write_json(OUT / "gated_path_chains.json", gated, "gated_path")
    write_json(OUT / "entity_cluster_enriched_chains.json", clusters_enriched, "entity_cluster_enriched")

    summary = {
        "path_baseline_fixed": len(baseline),
        "entity_cluster": len(clusters),
        "gated_path": len(gated),
        "entity_cluster_enriched": len(clusters_enriched),
    }
    (OUT / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
