#!/usr/bin/env python3
"""Audit readiness for migrating from LaTeX-centric graph logic to MinerU-only artifacts.

Experimental lane only. This script does not call any API and does not write
production artifacts. It summarizes what pure MinerU can currently provide and
where existing graph/query logic still depends on LaTeX-derived signals.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MINERU_DIR = ROOT / "data/00_raw/mineru_output"
DEFAULT_ELEMENTS = ROOT / "data/01_graphs/multimodal_elements.json"
LATEX_PATTERNS = [
    "latex_reference_graph",
    "latex_long",
    "LaTeX",
    "latex",
    "line_no",
    "source_line",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_raw_mineru(mineru_dir: Path) -> dict[str, Any]:
    doc_dirs = sorted(path for path in mineru_dir.iterdir() if path.is_dir()) if mineru_dir.exists() else []
    structure_paths = sorted(mineru_dir.glob("*/structure.json"))
    formula_paths = sorted(mineru_dir.glob("*/formulas.jsonl"))

    type_counts: Counter[str] = Counter()
    docs_with_type: dict[str, set[str]] = defaultdict(set)
    key_counts: Counter[str] = Counter()
    nonempty_counts: Counter[str] = Counter()
    image_like_total = 0
    image_like_with_path = 0
    examples: list[dict[str, Any]] = []

    for structure_path in structure_paths:
        doc_id = structure_path.parent.name
        try:
            obj = read_json(structure_path)
        except Exception as exc:  # noqa: BLE001 - audit should continue
            examples.append({"doc_id": doc_id, "error": repr(exc)})
            continue
        elements = obj.get("elements", []) if isinstance(obj, dict) else []
        for element in elements:
            if not isinstance(element, dict):
                continue
            element_type = str(element.get("type") or element.get("element_type") or "unknown")
            type_counts[element_type] += 1
            docs_with_type[element_type].add(doc_id)
            for key, value in element.items():
                key_counts[key] += 1
                if value not in (None, "", [], {}):
                    nonempty_counts[key] += 1
            if element_type in {"figure", "image", "table"}:
                image_like_total += 1
                if element.get("image_path"):
                    image_like_with_path += 1
            if len(examples) < 8:
                examples.append(
                    {
                        "doc_id": doc_id,
                        "element_id": element.get("element_id"),
                        "type": element_type,
                        "page_idx": element.get("page_idx"),
                        "has_image_path": bool(element.get("image_path")),
                        "content_preview": str(element.get("content") or "")[:240],
                        "metadata_keys": sorted((element.get("metadata") or {}).keys())
                        if isinstance(element.get("metadata"), dict)
                        else [],
                    }
                )

    return {
        "doc_dirs": len(doc_dirs),
        "structure_json_docs": len(structure_paths),
        "formulas_jsonl_docs": len(formula_paths),
        "raw_elements": sum(type_counts.values()),
        "raw_element_type_counts": dict(type_counts),
        "docs_with_type": {key: len(value) for key, value in docs_with_type.items()},
        "top_raw_keys": key_counts.most_common(40),
        "top_nonempty_raw_keys": nonempty_counts.most_common(40),
        "image_like_with_image_path": image_like_with_path,
        "image_like_total": image_like_total,
        "examples": examples,
    }


def iter_graph_elements(elements_path: Path):
    data = read_json(elements_path)
    for doc_id, doc in data.get("documents", {}).items():
        raw = doc.get("elements", {}) if isinstance(doc, dict) else {}
        values = raw.values() if isinstance(raw, dict) else raw if isinstance(raw, list) else []
        for element in values:
            if isinstance(element, dict):
                yield doc_id, element


def audit_multimodal_graph(elements_path: Path) -> dict[str, Any]:
    type_counts: Counter[str] = Counter()
    key_counts: Counter[str] = Counter()
    nonempty_counts: Counter[str] = Counter()
    docs: set[str] = set()
    examples: list[dict[str, Any]] = []
    for doc_id, element in iter_graph_elements(elements_path):
        docs.add(doc_id)
        element_type = str(element.get("element_type") or element.get("type") or "unknown")
        type_counts[element_type] += 1
        for key, value in element.items():
            key_counts[key] += 1
            if value not in (None, "", [], {}):
                nonempty_counts[key] += 1
        if len(examples) < 8:
            examples.append(
                {
                    "doc_id": doc_id,
                    "element_id": element.get("element_id"),
                    "element_type": element_type,
                    "page_idx": element.get("page_idx"),
                    "position_idx": element.get("position_idx"),
                    "has_caption": bool(element.get("caption")),
                    "has_content": bool(element.get("content")),
                    "has_context": bool(element.get("context_before") or element.get("context_after")),
                    "has_image_path": bool(element.get("image_path")),
                }
            )
    return {
        "graph_docs": len(docs),
        "graph_elements": sum(type_counts.values()),
        "graph_element_type_counts": dict(type_counts),
        "top_graph_keys": key_counts.most_common(40),
        "top_nonempty_graph_keys": nonempty_counts.most_common(40),
        "examples": examples,
    }


def audit_latex_dependency(paths: list[Path]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    files: Counter[str] = Counter()
    pattern = re.compile("|".join(re.escape(item) for item in LATEX_PATTERNS), re.IGNORECASE)
    for root in paths:
        if not root.exists():
            continue
        for file_path in root.rglob("*.py"):
            if file_path.resolve() == Path(__file__).resolve():
                continue
            try:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            matches = pattern.findall(text)
            if not matches:
                continue
            rel = str(file_path.relative_to(ROOT))
            files[rel] = len(matches)
            for match in matches:
                counts[match.lower()] += 1
    return {
        "pattern_counts": dict(counts),
        "top_files": files.most_common(40),
    }


def write_report(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    raw = summary["raw_mineru"]
    graph = summary["multimodal_graph"]
    latex = summary["latex_dependency"]
    lines = [
        "# MinerU-only Migration Audit",
        "",
        "## Current Coverage",
        f"- MinerU doc dirs: **{raw['doc_dirs']}**",
        f"- docs with `structure.json`: **{raw['structure_json_docs']}**",
        f"- docs with `formulas.jsonl`: **{raw['formulas_jsonl_docs']}**",
        f"- raw MinerU elements: **{raw['raw_elements']}** `{raw['raw_element_type_counts']}`",
        f"- current `multimodal_elements.json` docs/elements: **{graph['graph_docs']} / {graph['graph_elements']}** `{graph['graph_element_type_counts']}`",
        f"- image-like raw elements with image paths: **{raw['image_like_with_image_path']} / {raw['image_like_total']}**",
        "",
        "## LaTeX Dependency Hotspots",
        f"- pattern counts: `{latex['pattern_counts']}`",
        "- top files:",
    ]
    for file_name, count in latex["top_files"][:15]:
        lines.append(f"  - `{file_name}`: {count}")
    lines.extend(
        [
            "",
            "## Migration Implication",
            "- The viable unit is not LaTeX label/path anymore; it is a MinerU reading-order element plus PDF-local context.",
            "- `structure.json` is currently underused by the relationship builder; a MinerU-only v2 builder should parse it directly and treat text paragraphs as first-class elements.",
            "- Pure MinerU should replace LaTeX line numbers with `(page_idx, position_idx, bbox/image_path, local context window)` and replace LaTeX reference paths with local co-reference/layout/semantic edges.",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit MinerU-only migration readiness")
    parser.add_argument("--mineru-dir", default=str(DEFAULT_MINERU_DIR))
    parser.add_argument("--elements", default=str(DEFAULT_ELEMENTS))
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else ROOT / f"data/05_eval/mineru_only_migration_audit_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "raw_mineru": audit_raw_mineru(Path(args.mineru_dir)),
        "multimodal_graph": audit_multimodal_graph(Path(args.elements)),
        "latex_dependency": audit_latex_dependency([ROOT / "scripts", ROOT / "experiments"]),
    }
    write_report(out_dir, summary)
    latest = ROOT / "data/05_eval/mineru_only_migration_audit_latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(out_dir.resolve())
    except OSError:
        pass
    print(f"[ok] wrote {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()