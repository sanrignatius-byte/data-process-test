#!/usr/bin/env python3
"""Judge chunk-bridge cross-document chains with the company VLM.

Unlike entity-bridge (shared entities = keywords), chunk-bridge uses TF-IDF
matched paragraph text as the bridge. The judge evaluates whether the bridge
paragraphs meaningfully connect the linked elements across documents.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api import call_llm, parse_json, set_company_credentials  # noqa: E402
from src.utils.token_logger import log_run  # noqa: E402

VERDICTS = {
    "strong_chain",
    "weak_but_related",
    "topic_only",
    "wrong_target",
    "wrong_source",
    "insufficient_context",
}


def load_jsonl(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def clip(text: Any, limit: int) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= limit else s[: limit - 3] + "..."


def image_to_b64(path: str) -> tuple[str, str] | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists() or not p.is_file():
        return None
    mime = mimetypes.guess_type(str(p))[0] or "image/jpeg"
    return base64.b64encode(p.read_bytes()).decode("ascii"), mime


class JudgePackBuilder:
    """Build judge items from chunk-bridge chains + element metadata."""

    def __init__(self, root: Path):
        self.root = root
        self._enriched: dict | None = None
        self._topo_nodes: dict | None = None  # element_id -> node
        self._mineru_elements: dict | None = None  # element_id -> element data
        self._chains: list[dict] | None = None

    # ---- loaders ----

    def _load_enriched(self) -> dict:
        if self._enriched is None:
            path = self.root / "data/02_enriched/multimodal_elements_enriched.json"
            with open(path) as f:
                self._enriched = json.load(f)
        return self._enriched

    def _load_topo_nodes(self) -> dict[str, dict]:
        if self._topo_nodes is None:
            path = self.root / "data/05_eval/mineru_topology_graph_v1_latest/mineru_topology_graph_v1.json"
            with open(path) as f:
                topo = json.load(f)
            self._topo_nodes = {n["element_id"]: n for n in topo.get("nodes", [])}
        return self._topo_nodes

    def _load_mineru_elements(self) -> dict[str, dict]:
        """Load mineru raw elements (paragraph text etc)."""
        if self._mineru_elements is None:
            path = self.root / "data/05_eval/mineru_only_graph_v1_latest/mineru_elements_v1.json"
            idx: dict[str, dict] = {}
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                for doc_id, doc_els in data.get("documents", {}).items():
                    if isinstance(doc_els, dict):
                        for el_id, el_data in doc_els.items():
                            el_data["_doc_id"] = doc_id
                            idx[el_id] = el_data
            self._mineru_elements = idx
        return self._mineru_elements

    def _load_chains(self, chains_path: str = "") -> list[dict]:
        if self._chains is None:
            if chains_path:
                path = Path(chains_path)
                if not path.is_absolute():
                    path = self.root / path
            else:
                # Default: find latest chunk_bridge_chains_53_* directory
                base = self.root / "data/05_eval"
                candidates = sorted(base.glob("chunk_bridge_chains_53_*/chains.jsonl"))
                if not candidates:
                    raise FileNotFoundError("No chunk_bridge_chains_53_*/chains.jsonl found")
                path = candidates[-1]  # latest by name (timestamp)
            self._chains = load_jsonl(path)
        return self._chains

    # ---- element accessors ----

    def _get_element_data(self, element_id: str) -> dict[str, Any]:
        """Get the best available data for an element from all sources."""
        result: dict[str, Any] = {"element_id": element_id}

        # 1. Try enriched (figure/table/formula)
        enriched = self._load_enriched()
        for doc_id, doc_data in enriched.get("documents", {}).items():
            els = doc_data.get("elements", {})
            if element_id in els:
                el = els[element_id]
                result.update({
                    "element_type": el.get("element_type", ""),
                    "caption": el.get("caption", "") or el.get("content", ""),
                    "enriched_title": el.get("enriched_title", ""),
                    "enriched_content": el.get("enriched_content", ""),
                    "image_path": el.get("image_path", ""),
                })
                return result

        # 2. Try topology graph nodes (section/text)
        topo = self._load_topo_nodes()
        if element_id in topo:
            node = topo[element_id]
            result.update({
                "element_type": node.get("node_type", ""),
                "caption": node.get("section_title", "") or node.get("label", ""),
                "enriched_title": "",
                "enriched_content": node.get("text_snippet", ""),
                "image_path": "",
            })
            return result

        # 3. Try mineru raw elements
        mineru = self._load_mineru_elements()
        if element_id in mineru:
            el = mineru[element_id]
            result.update({
                "element_type": el.get("element_type", ""),
                "caption": el.get("caption", "") or el.get("content_preview", ""),
                "enriched_title": "",
                "enriched_content": el.get("content_preview", ""),
                "image_path": el.get("image_path", ""),
            })
            return result

        result["element_type"] = "unknown"
        return result

    # ---- build ----

    def build(self, limit: int = 0, chains_path: str = "") -> list[dict[str, Any]]:
        chains = self._load_chains(chains_path)
        if limit:
            chains = chains[:limit]

        items = []
        for idx, chain in enumerate(chains):
            item = self._build_one(idx, chain)
            items.append(item)
        return items

    def _build_one(self, idx: int, chain: dict) -> dict[str, Any]:
        hop = chain["hops"][0]  # 1-hop chains, take the bridge hop

        from_els = hop.get("from_linked_elements", [])
        to_els = hop.get("to_linked_elements", [])

        # Get element data for up to 3 elements per side
        from_data = [self._get_element_data(e) for e in from_els[:3]]
        to_data = [self._get_element_data(e) for e in to_els[:3]]

        # Build rich descriptions
        from_desc = self._format_elements(from_data)
        to_desc = self._format_elements(to_data)

        bridge_src = hop.get("bridge_text_source", "")
        bridge_tgt = hop.get("bridge_text_target", "")

        # Determine element types
        from_types = [d.get("element_type", "") for d in from_data if d.get("element_type")]
        to_types = [d.get("element_type", "") for d in to_data if d.get("element_type")]
        from_type = from_types[0] if from_types else "unknown"
        to_type = to_types[0] if to_types else "unknown"

        # Image paths (first figure element only)
        from_images = [d.get("image_path", "") for d in from_data
                       if d.get("element_type") in ("figure", "table") and d.get("image_path")]
        to_images = [d.get("image_path", "") for d in to_data
                     if d.get("element_type") in ("figure", "table") and d.get("image_path")]

        return {
            "candidate_id": chain.get("chain_id", f"chunk_bridge_{idx:05d}"),
            "judge_index": idx + 1,
            "target_stratum": "chunk_bridge",
            "source_doc": hop.get("from_doc", ""),
            "target_doc": hop.get("to_doc", ""),
            "source_element_ids": from_els[:5],
            "target_element_ids": to_els[:5],
            "source_element_type": from_type,
            "target_element_type": to_type,
            "pair_type": f"{from_type}+{to_type}",
            "source_elements_desc": from_desc,
            "target_elements_desc": to_desc,
            "source_element_images": from_images,
            "target_element_images": to_images,
            "bridge_text_source": bridge_src,
            "bridge_text_target": bridge_tgt,
            "similarity": hop.get("similarity", 0),
            "cross_doc_hops": chain.get("cross_doc_hops", 1),
            "paper_path": chain.get("paper_path", []),
            "total_score": chain.get("total_score", 0),
        }

    def _format_elements(self, el_data: list[dict]) -> str:
        parts = []
        for el in el_data:
            if not el.get("element_type"):
                continue
            header = f"[{el['element_id']}] ({el.get('element_type', '')})"
            lines = [header]
            if el.get("caption"):
                lines.append(f"  caption: {clip(el['caption'], 400)}")
            if el.get("enriched_title"):
                lines.append(f"  title: {clip(el['enriched_title'], 300)}")
            if el.get("enriched_content"):
                lines.append(f"  content: {clip(el['enriched_content'], 500)}")
            parts.append("\n".join(lines))
        return "\n\n".join(parts) if parts else "(no linked elements)"


# ---- prompt ----

def build_prompt(row: dict[str, Any]) -> str:
    source_desc = row.get("source_elements_desc", "(none)")
    target_desc = row.get("target_elements_desc", "(none)")
    bridge_src = clip(row.get("bridge_text_source", ""), 500)
    bridge_tgt = clip(row.get("bridge_text_target", ""), 500)

    return f"""You are judging whether a cross-document paragraph bridge meaningfully connects elements from TWO DIFFERENT scientific papers into a valid multi-hop evidence chain.

The "bridge" is a pair of paragraphs (one from each paper) that TF-IDF matched as discussing similar content. Your task is to determine whether this bridge + the linked elements form a semantically valid cross-document chain suitable for generating M4 (Multi-hop, Multi-document) training queries.

BRIDGE PARAGRAPH from Paper A ({row.get("source_doc")}):
---
{bridge_src}
---

BRIDGE PARAGRAPH from Paper B ({row.get("target_doc")}):
---
{bridge_tgt}
---

TF-IDF similarity score: {row.get("similarity", 0):.4f}

ELEMENTS in Paper A linked through this bridge:
{source_desc}

ELEMENTS in Paper B linked through this bridge:
{target_desc}

EVALUATION STANDARD (conservative):
- strong_chain: The bridge paragraphs discuss the SAME specific concept/method/dataset/theory in a substantive way, AND the linked elements on both sides are topically connected to that shared concept. A human annotator could write a meaningful multi-hop query that requires evidence from elements in BOTH papers.
- weak_but_related: The paragraphs are on related topics and the elements share some thematic connection, but the link is loose or the bridge text is mostly boilerplate. Could possibly write a query, but it would be forced.
- topic_only: Both papers are in the same research area, but the specific paragraphs discuss different things, or the matched text is generic/boilerplate (author affiliations, arXiv IDs, generic introductory sentences). The TF-IDF match is superficial.
- wrong_target: The elements in Paper B clearly do NOT relate to what the bridge paragraph is about.
- wrong_source: The elements in Paper A clearly do NOT relate to what the bridge paragraph is about.
- insufficient_context: Not enough paragraph text or element information to decide.

CRITICAL RULES:
- Author affiliations and arXiv URL matches are NOT scientific bridges → topic_only.
- Generic phrases like "we propose a novel method" or "experimental results show" without specific entity names → topic_only at best.
- If the bridge paragraphs name the SAME dataset (e.g. COMPAS, ImageNet, UCI Adult), same metric (e.g. demographic parity, equalized odds), same method variant, or same theorem → likely strong_chain.
- If the linked elements are figures/tables/formulas that visualize or report results for the shared concept → strong_chain.
- If only ONE side has meaningful linked elements → weak_but_related at best.

Return valid JSON only:
{{
  "verdict": "strong_chain | weak_but_related | topic_only | wrong_target | wrong_source | insufficient_context",
  "confidence": 0.0,
  "bridge_is_scientific": true,
  "both_sides_have_elements": true,
  "bridge_specificity": "exact_entity_match | closely_related_entity | same_broad_topic | spurious_text_match | boilerplate | none",
  "main_failure": "none | bridge_is_boilerplate_or_author_block | elements_dont_match_bridge | only_one_side_has_elements | topic_too_different | missing_context | other",
  "rationale": "2-4 concise sentences explaining whether the bridge paragraphs and linked elements support a cross-document evidence chain",
  "evidence": {{
    "shared_concept": "the core concept that the bridge paragraphs both discuss",
    "element_relevance_source": "how the source elements relate to the shared concept",
    "element_relevance_target": "how the target elements relate to the shared concept",
    "bridge_quality": "specific_match | broad_overlap | spurious | boilerplate | insufficient"
  }}
}}

Candidate: {row.get("candidate_id")}
Papers: {row.get("source_doc")} -> {row.get("target_doc")}
Chain score: {row.get("total_score", 0):.4f}
"""


# ---- validation ----

def validate(obj: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(obj, dict):
        return {"valid": False, "reason": "not_object"}
    missing = [
        key for key in (
            "verdict", "confidence", "bridge_is_scientific",
            "both_sides_have_elements", "bridge_specificity",
            "main_failure", "rationale", "evidence",
        ) if key not in obj
    ]
    if missing:
        return {"valid": False, "reason": "missing:" + ",".join(missing)}
    if obj.get("verdict") not in VERDICTS:
        return {"valid": False, "reason": f"bad_verdict:{obj.get('verdict')}"}
    try:
        conf = float(obj.get("confidence"))
    except (TypeError, ValueError):
        return {"valid": False, "reason": "bad_confidence"}
    if not 0 <= conf <= 1:
        return {"valid": False, "reason": "confidence_oob"}
    return {"valid": True, "reason": "ok"}


def write_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def summarize(judgments: list[dict[str, Any]]) -> dict[str, Any]:
    verdicts: Counter[str] = Counter()
    by_pair_type: dict[str, Counter[str]] = defaultdict(Counter)
    by_score_bucket: dict[str, Counter[str]] = defaultdict(Counter)
    for j in judgments:
        v = j["judgment"].get("verdict", "parse_failed") if isinstance(j.get("judgment"), dict) else "parse_failed"
        verdicts[v] += 1
        pt = j.get("pair_type", "unknown")
        score = j.get("total_score", 0)
        bucket = f"[{int(score*10)/10:.1f}-{int(score*10+1)/10:.1f})"
        by_pair_type[pt][v] += 1
        by_score_bucket[bucket][v] += 1
    total = len(judgments)
    strong = verdicts.get("strong_chain", 0)
    weak = verdicts.get("weak_but_related", 0)
    return {
        "total": total,
        "strong_chain": strong,
        "strong_rate": round(strong / total, 4) if total else 0,
        "weak_but_related": weak,
        "usable_rate": round((strong + weak) / total, 4) if total else 0,
        "verdict_counts": {k: v for k, v in verdicts.most_common()},
        "validation_counts": dict(Counter(
            j.get("validation", {}).get("reason", "unknown") for j in judgments
        )),
        "by_pair_type": {pt: dict(c) for pt, c in sorted(by_pair_type.items())},
        "by_score_bucket": {b: dict(c) for b, c in sorted(by_score_bucket.items())},
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Chunk-Bridge Judge Summary",
        "",
        f"- total: **{summary['total']}**",
        f"- strong_chain: **{summary['strong_chain']}** ({summary['strong_rate']:.1%})",
        f"- weak_but_related: **{summary.get('weak_but_related', 0)}**",
        f"- usable (strong+weak): **{summary.get('usable_rate', 0):.1%}**",
        f"- model: `{summary['model']}`",
    ]
    lines.append("\n## Verdict Counts\n")
    for v, c in summary.get("verdict_counts", {}).items():
        lines.append(f"- `{v}`: {c}")
    if summary.get("by_pair_type"):
        lines.append("\n## By Pair Type\n")
        for pt, vc in summary["by_pair_type"].items():
            inner = ", ".join(f"{k}={v}" for k, v in vc.items())
            lines.append(f"- `{pt}`: {inner}")
    if summary.get("by_score_bucket"):
        lines.append("\n## By Score Bucket\n")
        for b, vc in summary["by_score_bucket"].items():
            inner = ", ".join(f"{k}={v}" for k, v in vc.items())
            lines.append(f"- `{b}`: {inner}")
    return "\n".join(lines)


def make_out_dir(base: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if base:
        out_dir = ROOT / base
    else:
        out_dir = ROOT / f"data/05_eval/chunk_bridge_judge_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def main():
    parser = argparse.ArgumentParser(
        description="Judge chunk-bridge cross-document chains"
    )
    parser.add_argument("--chains", default="", help="Path to chains.jsonl (default: latest chunk_bridge_chains_53_*/)")
    parser.add_argument("--limit", type=int, default=0, help="Max chains to judge (0=all)")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--no-images", action="store_true")
    parser.add_argument("--company-api-url", default=os.environ.get("COMPANY_API_URL", ""))
    parser.add_argument("--company-api-key", default=os.environ.get("COMPANY_API_KEY", ""))
    args = parser.parse_args()

    set_company_credentials(args.company_api_url, args.company_api_key)

    builder = JudgePackBuilder(ROOT)
    rows = builder.build(limit=args.limit, chains_path=args.chains)
    out_dir = make_out_dir(args.output_dir)

    prompts_path = out_dir / "prompts.jsonl"
    responses_path = out_dir / "responses.jsonl"
    judgments_path = out_dir / "judgments.jsonl"
    failures_path = out_dir / "failures.jsonl"

    for p in [prompts_path, responses_path, judgments_path, failures_path]:
        if p.exists():
            p.unlink()

    total_in = 0
    total_out = 0
    judgments: list[dict[str, Any]] = []

    print(f"Loaded {len(rows)} chains to judge")
    print(f"Output: {out_dir}")
    print(f"Model: {args.model}")
    print(f"Images: {not args.no_images}")

    for idx, row in enumerate(rows, 1):
        prompt = build_prompt(row)

        images: list[tuple[str, str] | None] = []
        if not args.no_images:
            for img_path in row.get("source_element_images", [])[:1]:
                b64 = image_to_b64(img_path)
                if b64:
                    images.append(b64)
            for img_path in row.get("target_element_images", [])[:1]:
                b64 = image_to_b64(img_path)
                if b64:
                    images.append(b64)

        write_jsonl(prompts_path, {
            "candidate_id": row.get("candidate_id"),
            "prompt": prompt,
            "image_count": sum(1 for x in images if x is not None),
        })

        raw, tin, tout = call_llm(
            client=None,
            model=args.model,
            provider="company",
            prompt=prompt,
            images=images,
            system_prompt=(
                "You are a conservative scientific evidence-chain judge. "
                "Return valid JSON only. Verify that the bridge paragraphs "
                "substantively connect elements across documents — do not "
                "accept author blocks or generic phrases as bridges."
            ),
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            user_tag="chunk_bridge_judge",
        )
        total_in += tin
        total_out += tout
        parsed = parse_json(raw or "")
        validation = validate(parsed)

        out = {
            "candidate_id": row.get("candidate_id"),
            "judge_index": row.get("judge_index"),
            "source_doc": row.get("source_doc"),
            "target_doc": row.get("target_doc"),
            "source_element_ids": row.get("source_element_ids", []),
            "target_element_ids": row.get("target_element_ids", []),
            "pair_type": row.get("pair_type"),
            "total_score": row.get("total_score", 0),
            "similarity": row.get("similarity", 0),
            "judgment": parsed if isinstance(parsed, dict) else None,
            "validation": validation,
            "tokens": {"in": tin, "out": tout},
        }
        judgments.append(out)
        write_jsonl(judgments_path, out)
        write_jsonl(responses_path, {
            "candidate_id": row.get("candidate_id"),
            "raw": raw,
            "tokens": {"in": tin, "out": tout},
        })
        if not validation["valid"]:
            write_jsonl(failures_path, out | {"raw": raw})

        verdict = out["judgment"].get("verdict") if out["judgment"] else "parse_failed"
        print(
            f"[{idx:03d}/{len(rows):03d}] {row.get('source_doc')}->{row.get('target_doc')} "
            f"{row.get('candidate_id','')[:50]} -> {verdict}"
        )

    summary = {
        "status": "ok",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": args.model,
        "output_dir": str(out_dir.relative_to(ROOT)),
        "company_api_url_set": bool(args.company_api_url),
        "company_api_key_set": bool(args.company_api_key),
        "files": {
            "prompts": str(prompts_path.relative_to(ROOT)),
            "responses": str(responses_path.relative_to(ROOT)),
            "judgments": str(judgments_path.relative_to(ROOT)),
            "failures": str(failures_path.relative_to(ROOT)),
        },
        "tokens": {"in": total_in, "out": total_out},
        **summarize(judgments),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")

    log_run(
        script="experiments/judge_chunk_bridge_pack.py",
        model=f"company:{args.model}",
        purpose="Judge chunk-bridge cross-document chains for M4 query viability",
        input_tokens=total_in,
        output_tokens=total_out,
        extra={
            "chains_judged": len(judgments),
            "strong_chain": summary["strong_chain"],
            "output": str(out_dir.relative_to(ROOT)),
        },
    )

    print(f"\n=== Judge Complete ===")
    print(f"Total: {summary['total']}")
    print(f"Strong chain: {summary['strong_chain']} ({summary['strong_rate']:.1%})")
    print(f"Weak but related: {summary.get('weak_but_related', 0)}")
    print(f"Usable (strong+weak): {summary.get('usable_rate', 0):.1%}")
    print(f"Verdicts: {summary['verdict_counts']}")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
