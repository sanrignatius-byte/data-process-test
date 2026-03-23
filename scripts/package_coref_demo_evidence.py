#!/usr/bin/env python3
"""
将 co-reference bridge paragraph demo 5 条 query 的 evidence 按 query 打包到独立文件夹。

每个 query 文件夹包含:
  - query_info.json          : query/answer/reasoning_steps/latex_bridge 等元信息
  - element_a_<type>.<ext>   : 第一个 element 的截图（从 MinerU images/ 复制）
  - element_b_<type>.<ext>   : 第二个 element 的截图
  - element_a_context.md     : 第一个 element 的 caption + 上下文
  - element_b_context.md     : 第二个 element 的 caption + 上下文
  - bridge_paragraphs.md     : LaTeX bridge 原文 + MinerU markdown 截取的桥接段落

数据来源:
  - data/coref_demo_selection_5.json     : 5 条 co-ref demo query
  - data/multimodal_elements.json        : element 详情
  - <mineru_output>/<doc_id>/...         : 截图 + markdown

用法:
  # Windows 本地:
  python scripts/package_coref_demo_evidence.py --mineru-dir D:/Code_store/data-process-test/data/mineru_output

  # 集群:
  python scripts/package_coref_demo_evidence.py --mineru-dir /projects/_hdd/myyyx1/data-process-test/data/mineru_output
"""
import argparse
import json
import pathlib
import re
import shutil
import sys
from typing import Dict, List, Optional

ROOT = pathlib.Path(__file__).resolve().parent.parent

DEMO_FILE = ROOT / "data/coref_demo_selection_5.json"
ELEMENTS  = ROOT / "data/multimodal_elements.json"


# ─────────────────────────── helpers (from package_l3_demo_evidence.py) ───────────────────────────

def extract_formula_variables(content: str) -> str:
    if not content:
        return "(none)"
    regions = re.findall(r"\$\$?(.+?)\$\$?", content, flags=re.DOTALL)
    math_text = " ".join(regions) if regions else content
    funcs = re.findall(r"\\([A-Za-z]{2,})", math_text)
    funcs = [f for f in funcs if f not in {
        "begin", "end", "left", "right", "frac", "cdot", "times",
        "sum", "prod", "int", "mid", "tag", "qquad", "quad", "text",
        "mathbf", "mathrm", "mathbb", "operatorname", "sim", "leq", "geq",
    }]
    vars_sub = re.findall(r"\b([A-Za-z]+_[A-Za-z0-9]+)\b", math_text)
    all_vars = list(dict.fromkeys(funcs + vars_sub))
    return ", ".join(all_vars[:15]) if all_vars else "(no variables detected)"


def load_elements(path: pathlib.Path) -> Dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    if isinstance(data, dict) and "documents" in data:
        for doc_id, doc in data["documents"].items():
            elems = doc.get("elements", {})
            if isinstance(elems, dict):
                out.update(elems)
            elif isinstance(elems, list):
                for e in elems:
                    out[e["element_id"]] = e
    elif isinstance(data, list):
        out = {e["element_id"]: e for e in data}
    return out


def find_markdown(mineru_dir: pathlib.Path, doc_id: str) -> Optional[pathlib.Path]:
    base = mineru_dir / doc_id / doc_id / "hybrid_auto"
    candidates = [base / f"{doc_id}.md", base / "content.md"]
    for p in candidates:
        if p.exists():
            return p
    if base.exists():
        for md in base.glob("*.md"):
            if "formula" not in md.name.lower():
                return md
    return None


def resolve_element_image(elem: dict, mineru_dir: pathlib.Path) -> Optional[pathlib.Path]:
    doc_id = elem.get("doc_id", "")
    img_dir = mineru_dir / doc_id / doc_id / "hybrid_auto" / "images"

    meta = elem.get("metadata", {}) or {}
    fname = meta.get("image_filename", "")
    if fname:
        p = img_dir / fname
        if p.exists():
            return p

    raw_path = elem.get("image_path", "")
    if raw_path:
        fname2 = pathlib.PurePosixPath(raw_path).name or pathlib.PureWindowsPath(raw_path).name
        if fname2:
            p = img_dir / fname2
            if p.exists():
                return p

    etype = elem.get("element_type", "")
    number = elem.get("number")
    type_prefix_map = {"figure": "fig", "table": "tbl", "formula": "eq"}
    prefix = type_prefix_map.get(etype, etype)
    if number is not None and img_dir.exists():
        pattern = f"{doc_id}_page*_{prefix}{number}.*"
        matches = list(img_dir.glob(pattern))
        if matches:
            return matches[0]

    return None


# ─────────────────────────── bridge paragraph extractor ───────────────────────────

def extract_bridge_from_markdown(
    md_path: pathlib.Path,
    elem_a: dict,
    elem_b: dict,
) -> str:
    md_text = md_path.read_text(encoding="utf-8", errors="ignore")
    lines = md_text.split("\n")

    def find_element_line(elem: dict) -> int:
        label = elem.get("label", "")
        caption = (elem.get("caption") or "")[:60]
        for i, line in enumerate(lines):
            if label and label in line:
                return i
            if caption and len(caption) > 15 and caption[:30] in line:
                return i
        return -1

    line_a = find_element_line(elem_a)
    line_b = find_element_line(elem_b)
    if line_a >= 0 and line_b >= 0:
        start = max(0, min(line_a, line_b) - 2)
        end = min(len(lines), max(line_a, line_b) + 3)
        return "\n".join(lines[start:end])

    return ""


def extract_bridge_from_context(elem_a: dict, elem_b: dict) -> str:
    parts = []
    for elem, tag in [(elem_a, "Element A"), (elem_b, "Element B")]:
        eid = elem.get("element_id", "?")
        ctx_before = (elem.get("context_before") or "").strip()
        ctx_after = (elem.get("context_after") or "").strip()
        if ctx_before or ctx_after:
            parts.append(f"## {tag}: {eid}\n")
            if ctx_before:
                parts.append(f"### Context Before\n\n{ctx_before}\n")
            if ctx_after:
                parts.append(f"### Context After\n\n{ctx_after}\n")
    return "\n".join(parts)


# ─────────────────────────── per-query folder builder ───────────────────────────

def build_query_folder(
    folder: pathlib.Path,
    item: dict,
    elem_db: Dict[str, dict],
    mineru_dir: pathlib.Path,
):
    qid = item["query_id"]
    eids = item.get("element_ids", [])
    tags = ["a", "b"][:len(eids)]
    elems = [elem_db.get(eid, {}) for eid in eids]

    # ── 1) query_info.json ──
    info = {
        "query_id": qid,
        "pair_type": item.get("pair_type"),
        "bridge_type": item.get("bridge_type"),
        "quality_tier": item.get("quality_tier"),
        "quality_score": item.get("quality_score"),
        "is_cross_doc": item.get("is_cross_doc", False),
        "hop_distance": item.get("hop_distance"),
        "reasoning_depth": item.get("reasoning_depth"),
        "element_ids": eids,
        "path": item.get("path", []),
        "query": item["query"],
        "answer": item["answer"],
        "reasoning_steps": item.get("reasoning_steps", []),
        "latex_bridge": item.get("latex_bridge", {}),
    }
    (folder / "query_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ── 2) element 截图 / formula LaTeX ──
    img_results = {}
    for tag, eid, elem in zip(tags, eids, elems):
        if not elem:
            img_results[tag] = False
            continue

        etype = elem.get("element_type", "unknown")

        if etype == "formula":
            latex_content = (elem.get("content") or "").strip()
            formula_vars = extract_formula_variables(latex_content)
            formula_md = [
                f"# Formula: {eid}",
                f"- label: {elem.get('label', '')}",
                f"- doc_id: {elem.get('doc_id', '')}",
                "",
                "## LaTeX",
                "",
                latex_content if latex_content else "(empty)",
                "",
                "## Key Variables",
                "",
                formula_vars,
                "",
            ]
            (folder / f"element_{tag}_formula.md").write_text(
                "\n".join(formula_md), encoding="utf-8"
            )
            img_results[tag] = True
            continue

        src_img = resolve_element_image(elem, mineru_dir)
        if src_img and src_img.exists():
            dst = folder / f"element_{tag}_{etype}{src_img.suffix}"
            shutil.copy2(src_img, dst)
            img_results[tag] = True
        else:
            img_results[tag] = False
            (folder / f"element_{tag}_IMAGE_NOT_FOUND.txt").write_text(
                f"element_id: {eid}\n"
                f"image_path (from elements json): {elem.get('image_path', '?')}\n"
                f"在 mineru_dir 下未找到对应文件。\n",
                encoding="utf-8",
            )

    # ── 3) element 上下文 ──
    for tag, eid, elem in zip(tags, eids, elems):
        if not elem:
            continue
        etype = elem.get("element_type", "")
        lines_out = [
            f"# {eid}",
            f"- type: {etype}",
            f"- doc_id: {elem.get('doc_id', '')}",
            f"- label: {elem.get('label', '')}",
            f"- page_idx: {elem.get('page_idx', '?')}",
            "",
        ]
        caption = (elem.get("caption") or "").strip()
        if caption:
            lines_out += ["## Caption", "", caption, ""]
        content = (elem.get("content") or "").strip()
        if content and content != caption:
            lines_out += ["## Content", "", content, ""]
        ctx_b = (elem.get("context_before") or "").strip()
        ctx_a = (elem.get("context_after") or "").strip()
        if ctx_b:
            lines_out += ["## Context Before", "", ctx_b, ""]
        if ctx_a:
            lines_out += ["## Context After", "", ctx_a, ""]

        (folder / f"element_{tag}_context.md").write_text(
            "\n".join(lines_out), encoding="utf-8"
        )

    # ── 4) 桥接段落 ──
    bridge_parts = []

    # 4a) LaTeX bridge 原文（co-reference 的核心证据）
    latex_bridge = item.get("latex_bridge", {})
    if latex_bridge:
        bridge_parts.append("## LaTeX Bridge (Co-reference Source)\n")
        bridge_parts.append(f"- **Bridge type**: {item.get('bridge_type', '?')}")
        bridge_parts.append(f"- **Strategy**: {latex_bridge.get('strategy', '?')}")
        bridge_parts.append(f"- **Char distance**: {latex_bridge.get('char_dist', '?')}")
        bridge_parts.append(f"- **Label A**: `{latex_bridge.get('label_a', '')}`")
        bridge_parts.append(f"- **Label B**: `{latex_bridge.get('label_b', '')}`")
        bridge_parts.append("")
        bridge_parts.append("### Raw LaTeX Text")
        bridge_parts.append("")
        bridge_parts.append(f"```latex\n{latex_bridge.get('bridge_text', '')}\n```")
        bridge_parts.append("")

    # 4b) MinerU markdown 截取
    if elems and len(elems) >= 2:
        elem_a, elem_b = elems[0], elems[1]
        doc_id = elem_a.get("doc_id", "")
        md_path = find_markdown(mineru_dir, doc_id)

        md_bridge = ""
        if md_path:
            md_bridge = extract_bridge_from_markdown(md_path, elem_a, elem_b)

        if md_bridge.strip():
            bridge_parts.append("## MinerU Markdown Excerpt (Between Two Elements)\n")
            bridge_parts.append(md_bridge)
            bridge_parts.append("")

        # fallback: context_before/after
        if not md_bridge.strip():
            ctx_bridge = extract_bridge_from_context(elem_a, elem_b)
            if ctx_bridge.strip():
                bridge_parts.append("## Context Fallback (from multimodal_elements.json)\n")
                bridge_parts.append(ctx_bridge)
                bridge_parts.append("")

    # 4c) reasoning_steps 中 bridge_paragraph 的 evidence_span
    bridge_spans = []
    for step in item.get("reasoning_steps", []):
        if step.get("evidence_element_id") == "bridge_paragraph":
            span = step.get("evidence_span", "")
            if span:
                bridge_spans.append(span)
    if bridge_spans:
        bridge_parts.append("## Bridge Evidence Spans from Reasoning Steps\n")
        for i, span in enumerate(bridge_spans, 1):
            bridge_parts.append(f"**Bridge {i}:** {span}\n")

    if bridge_parts:
        (folder / "bridge_paragraphs.md").write_text(
            "\n".join(bridge_parts).strip(), encoding="utf-8"
        )

    return img_results


# ─────────────────────────── main ───────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="打包 co-reference bridge paragraph demo evidence 到独立文件夹"
    )
    parser.add_argument(
        "--mineru-dir",
        type=pathlib.Path,
        required=True,
        help="MinerU 输出根目录，如 D:/Code_store/data-process-test/data/mineru_output",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=ROOT / "data/coref_demo_evidence",
        help="输出文件夹（默认 data/coref_demo_evidence）",
    )
    parser.add_argument(
        "--demo-file",
        type=pathlib.Path,
        default=DEMO_FILE,
        help="demo selection JSON 文件",
    )
    args = parser.parse_args()

    mineru_dir = args.mineru_dir
    out_dir = args.output_dir

    if not mineru_dir.exists():
        print(f"ERROR: mineru_dir 不存在: {mineru_dir}", file=sys.stderr)
        sys.exit(1)

    demo = json.loads(args.demo_file.read_text(encoding="utf-8"))
    elem_db = load_elements(ELEMENTS)

    print(f"Demo queries:    {len(demo)}")
    print(f"Elements DB:     {len(elem_db)}")
    print(f"MinerU dir:      {mineru_dir}")
    print(f"Output dir:      {out_dir}")
    print()

    if out_dir.exists():
        tmp_name = out_dir.with_name(out_dir.name + "_old_tmp")
        if tmp_name.exists():
            shutil.rmtree(tmp_name, ignore_errors=True)
        try:
            out_dir.rename(tmp_name)
            shutil.rmtree(tmp_name, ignore_errors=True)
        except (PermissionError, OSError):
            for child in out_dir.rglob("*"):
                if child.is_file():
                    try:
                        child.unlink()
                    except PermissionError:
                        pass
            shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_imgs = 0
    found_imgs = 0

    for idx, item in enumerate(demo, 1):
        qid = item["query_id"]
        btype = item.get("bridge_type", "?")
        folder = out_dir / qid
        folder.mkdir()

        img_results = build_query_folder(folder, item, elem_db, mineru_dir)

        n_files = len(list(folder.iterdir()))
        total_imgs += len(img_results)
        found_imgs += sum(1 for v in img_results.values() if v)

        status = "  ".join(
            f"elem_{tag}: {'OK' if ok else 'MISSING'}"
            for tag, ok in img_results.items()
        )
        print(f"  [{idx}] {qid} [{btype}]  ->  {n_files} files  ({status})")

    print(f"\nImages found: {found_imgs}/{total_imgs}")
    print(f"Output: {out_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
