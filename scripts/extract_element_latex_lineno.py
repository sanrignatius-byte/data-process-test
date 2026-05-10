#!/usr/bin/env python3
"""extract_element_latex_lineno.py
==================================
Annotate each mineru element in `multimodal_elements.json` with its LaTeX
source `line_no` and `label_key` (where derivable). Closes the formula
0% match-rate gap from [B2 audit](../refine-logs/B2_MINERU_LATEX_MATCH_AUDIT_20260510.md)
by adding LaTeX-content matching for equations.

Match priority:
  1. Number + type (existing logic in build_latex_reference_graph._match_labels_to_elements)
  2. Caption Jaccard (existing fallback, for figure/table)
  3. **NEW**: For formulas, parse `.tex` source; extract `\begin{equation}…\end{equation}`
     blocks and match by LaTeX-content Jaccard against mineru `content` field

Output: `data/01_graphs/element_latex_lineno_map.json` keyed by element_id:
  {
    "<element_id>": {
        "latex_line_no": int,
        "label_key": str | null,
        "match_method": "number" | "caption" | "content_overlap" | "label_proximity",
        "confidence": float
    }
  }

Usage:
  python scripts/extract_element_latex_lineno.py \\
    --elements data/03_queries/M4query_v1/graphs/multimodal_elements.json \\
    --latex-graph data/01_graphs/latex_reference_graph.json \\
    --latex-source-root data/00_raw/latex_sources_delivery/extracted \\
    --output data/01_graphs/element_latex_lineno_map.json
"""
from __future__ import annotations
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

EQ_BEGIN_RE = re.compile(r"\\begin\{(equation\*?|align\*?|gather\*?|multline\*?|displaymath)\}")
EQ_END_RE = re.compile(r"\\end\{(equation\*?|align\*?|gather\*?|multline\*?|displaymath)\}")
LABEL_RE = re.compile(r"\\label\{([^}]+)\}")


def find_main_tex(doc_dir: Path) -> Path | None:
    """Locate the main .tex file (heuristic: largest .tex under doc dir)."""
    if not doc_dir.exists():
        return None
    candidates = sorted(doc_dir.rglob("*.tex"), key=lambda p: p.stat().st_size, reverse=True)
    return candidates[0] if candidates else None


def extract_equation_blocks(tex_path: Path) -> list[dict]:
    """Parse .tex file → list of equation blocks.
    Each block: {start_line, end_line, content, label_key | None}"""
    if not tex_path.exists():
        return []
    text = tex_path.read_text(errors="replace")
    lines = text.splitlines()
    blocks = []
    i = 0
    while i < len(lines):
        m = EQ_BEGIN_RE.search(lines[i])
        if not m:
            i += 1
            continue
        start = i
        # Find matching end
        depth = 1
        j = i + 1
        env = m.group(1).rstrip("*")
        while j < len(lines) and depth > 0:
            if EQ_BEGIN_RE.search(lines[j]):
                depth += 1
            if EQ_END_RE.search(lines[j]):
                depth -= 1
            if depth == 0:
                break
            j += 1
        if j >= len(lines):
            break
        block_text = "\n".join(lines[start:j+1])
        # Extract label inside block (if any)
        label_match = LABEL_RE.search(block_text)
        label_key = label_match.group(1) if label_match else None
        blocks.append({
            "start_line": start + 1,  # 1-indexed
            "end_line": j + 1,
            "content": block_text,
            "label_key": label_key,
            "env": env,
        })
        i = j + 1
    return blocks


def normalize_latex_content(text: str) -> set[str]:
    """Tokenize LaTeX content for Jaccard. Drops comments and whitespace.

    Mineru-extracted formula content has spaces between every char
    (`o p t`, `s u b j e c t`), so we collapse single-letter sequences first.
    Single-char tokens (math symbols x, y, d) ARE kept — they carry signal
    in formulas where word-level tokens are sparse."""
    text = re.sub(r"%.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\\(begin|end|label|tag|hline)\{[^}]*\}", " ", text)
    text = re.sub(r"\\[a-zA-Z]+", " ", text)  # strip latex commands
    # Collapse char-spacing: "o p t" -> "opt" (only when ≥3 single letters chained)
    text = re.sub(r"\b((?:[A-Za-z] ){2,}[A-Za-z])\b", lambda m: m.group(1).replace(" ", ""), text)
    tokens = re.findall(r"[A-Za-z0-9]+", text)
    return set(t.lower() for t in tokens)


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--elements", default="data/03_queries/M4query_v1/graphs/multimodal_elements.json")
    parser.add_argument("--latex-graph", default="data/01_graphs/latex_reference_graph.json")
    parser.add_argument("--latex-source-root", default="data/00_raw/latex_sources_delivery/extracted")
    parser.add_argument("--output", default="data/01_graphs/element_latex_lineno_map.json")
    parser.add_argument("--min-content-jaccard", type=float, default=0.30)
    args = parser.parse_args()

    me = json.load(open(args.elements))
    docs = me["documents"]
    lg = json.load(open(args.latex_graph))["documents"]
    src_root = Path(args.latex_source_root)

    # Stats
    stats = defaultdict(lambda: defaultdict(int))
    out: dict = {}

    for doc_id, doc in docs.items():
        elements = doc["elements"]
        # Existing latex_labels (from build_latex_reference_graph): use as ground truth for figure/table
        labels = (lg.get(doc_id, {}) or {}).get("labels", {})
        if isinstance(labels, list):
            labels = {l["key"]: l for l in labels}

        # 1) Use existing matching for figure/table (the build_latex_reference_graph result)
        # Replicate the match: prefer number, then caption.
        elements_by_type = defaultdict(list)
        for eid, e in elements.items():
            elements_by_type[e["element_type"]].append((eid, e))
        type_map = {"figure": "figure", "table": "table", "equation": "formula"}
        used_eids: set[str] = set()
        for lkey, linfo in labels.items():
            ltype = linfo.get("label_type")
            target_type = type_map.get(ltype)
            if not target_type:
                continue
            cands = elements_by_type.get(target_type, [])
            chosen = None
            method = None
            confidence = 0.0
            # Match by trailing number
            num_m = re.search(r"(\d+)\s*$", lkey)
            if num_m:
                want = int(num_m.group(1))
                for eid, e in cands:
                    if eid in used_eids:
                        continue
                    if e.get("number") == want:
                        chosen, method, confidence = eid, "number", 1.0
                        break
            # Caption Jaccard
            if not chosen and linfo.get("caption"):
                cap_t = set(linfo["caption"].lower().split())
                best, best_o = None, 0.0
                for eid, e in cands:
                    if eid in used_eids:
                        continue
                    ec = (e.get("caption", "") or "").lower()
                    et = set(ec.split())
                    if not et:
                        continue
                    o = len(cap_t & et) / max(1, len(cap_t | et))
                    if o > best_o and o > 0.3:
                        best_o, best = o, eid
                if best:
                    chosen, method, confidence = best, "caption", best_o
            if chosen:
                used_eids.add(chosen)
                out[chosen] = {
                    "latex_line_no": linfo.get("line_no"),
                    "label_key": lkey,
                    "match_method": method,
                    "confidence": round(confidence, 3),
                }
                stats[target_type][method] += 1

        # 2) NEW: Formula content matching via .tex source
        formula_cands = [(eid, e) for eid, e in elements_by_type.get("formula", []) if eid not in used_eids]
        if not formula_cands:
            continue
        doc_dir = src_root / doc_id
        tex_path = find_main_tex(doc_dir)
        if not tex_path:
            stats["formula"]["no_tex_source"] += len(formula_cands)
            continue
        eq_blocks = extract_equation_blocks(tex_path)
        if not eq_blocks:
            stats["formula"]["no_blocks_in_tex"] += len(formula_cands)
            continue
        # For each unmatched formula, find best content-jaccard block
        block_used: set[int] = set()
        for eid, e in formula_cands:
            mineru_text = e.get("content", "") or ""
            mtok = normalize_latex_content(mineru_text)
            if not mtok:
                stats["formula"]["empty_content"] += 1
                continue
            best_idx, best_j = None, 0.0
            for idx, blk in enumerate(eq_blocks):
                if idx in block_used:
                    continue
                btok = normalize_latex_content(blk["content"])
                j = jaccard(mtok, btok)
                if j > best_j:
                    best_j, best_idx = j, idx
            if best_idx is not None and best_j >= args.min_content_jaccard:
                blk = eq_blocks[best_idx]
                block_used.add(best_idx)
                out[eid] = {
                    "latex_line_no": blk["start_line"],
                    "label_key": blk["label_key"],
                    "match_method": "content_overlap",
                    "confidence": round(best_j, 3),
                }
                stats["formula"]["content_overlap"] += 1
            else:
                stats["formula"]["unmatched"] += 1

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))

    # Report
    total_elements = {t: sum(1 for d in docs.values() for e in d["elements"].values() if e["element_type"] == t)
                      for t in ("figure", "table", "formula")}
    print(f"Wrote {args.output}")
    print(f"\n{'Type':<10}{'total':>8}{'by_num':>8}{'by_cap':>8}{'content':>9}{'no_tex':>8}{'unmatch':>9}{'match%':>9}")
    for t in ("figure", "table", "formula"):
        s = stats[t]
        n = total_elements[t]
        matched = s["number"] + s["caption"] + s["content_overlap"]
        pct = matched / n * 100 if n else 0
        print(f"{t:<10}{n:>8}{s['number']:>8}{s['caption']:>8}{s['content_overlap']:>9}"
              f"{s['no_tex_source']:>8}{s['unmatched']:>9}{pct:>8.1f}%")


if __name__ == "__main__":
    main()
