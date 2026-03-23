#!/usr/bin/env python3
"""
将 L3 demo 10 条 query 的 evidence 按 query 打包到独立文件夹。

每个 query 文件夹包含:
  - query_info.json          : query/answer/reasoning_steps 等元信息
  - element_a_<type>.<ext>   : 第一个 element 的截图（从 MinerU images/ 复制）
  - element_b_<type>.<ext>   : 第二个 element 的截图
  - element_a_context.md     : 第一个 element 的 caption + 上下文（从 content_list 提取）
  - element_b_context.md     : 第二个 element 的 caption + 上下文
  - bridge_paragraphs.md     : 桥接段落的 markdown 文本（从 MinerU .md 文件截取）

数据来源:
  - data/m2/l3_demo_selection_10.json        : 10 条 demo query
  - data/multimodal_elements.json            : element 详情（image_path, position_idx 等）
  - <mineru_output>/<doc_id>/<doc_id>/hybrid_auto/images/  : 截图
  - <mineru_output>/<doc_id>/<doc_id>/hybrid_auto/<doc_id>.md 或类似 : markdown 全文

用法:
  # 在 Windows 本地运行（指定你的 mineru_output 目录）:
  python scripts/package_l3_demo_evidence.py --mineru-dir D:/Code_store/data-process-test/data/mineru_output

  # 在集群运行:
  python scripts/package_l3_demo_evidence.py --mineru-dir /projects/_hdd/myyyx1/data-process-test/data/mineru_output

  # 自定义输出目录:
  python scripts/package_l3_demo_evidence.py --mineru-dir ... --output-dir data/m2/l3_evidence_folders
"""
import argparse
import json
import pathlib
import re
import shutil
import sys
from typing import Any, Dict, List, Optional, Tuple


def extract_formula_variables(content: str) -> str:
    """从 LaTeX 公式文本中提取变量/函数名（简化版）"""
    if not content:
        return "(none)"
    # 提取 $...$ 或 $$...$$ 内的数学区域
    regions = re.findall(r"\$\$?(.+?)\$\$?", content, flags=re.DOTALL)
    math_text = " ".join(regions) if regions else content
    # LaTeX 命令（函数名）
    funcs = re.findall(r"\\([A-Za-z]{2,})", math_text)
    funcs = [f for f in funcs if f not in {
        "begin", "end", "left", "right", "frac", "cdot", "times",
        "sum", "prod", "int", "mid", "tag", "qquad", "quad", "text",
        "mathbf", "mathrm", "mathbb", "operatorname", "sim", "leq", "geq",
    }]
    # 下标变量 (e.g. x_i, P_C)
    vars_sub = re.findall(r"\b([A-Za-z]+_[A-Za-z0-9]+)\b", math_text)
    all_vars = list(dict.fromkeys(funcs + vars_sub))  # 去重保序
    return ", ".join(all_vars[:15]) if all_vars else "(no variables detected)"

ROOT = pathlib.Path(__file__).resolve().parent.parent

DEMO_FILE  = ROOT / "data/m2/l3_demo_selection_10.json"
ELEMENTS   = ROOT / "data/multimodal_elements.json"
L3_QUERIES = ROOT / "data/m2/l3_reasoning_chain_queries.jsonl"


# ─────────────────────────── loaders ───────────────────────────

def load_elements(path: pathlib.Path) -> Dict[str, dict]:
    """element_id -> element dict"""
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


def load_l3_queries(path: pathlib.Path) -> Dict[str, dict]:
    """query_id -> full query record"""
    out = {}
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["query_id"]] = rec
    return out


def load_content_list(mineru_dir: pathlib.Path, doc_id: str) -> Optional[List[dict]]:
    """加载 content_list（v2 优先）并返回有序 item 列表"""
    base = mineru_dir / doc_id / doc_id / "hybrid_auto"
    candidates = [
        base / f"{doc_id}_content_list_v2.json",
        base / f"{doc_id}_content_list.json",
        base / "content_list.json",
    ]
    for p in candidates:
        if p.exists():
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            # content_list 可能是 list 或 dict with pages
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                # 有的格式是 {"pages": [{"items": [...]}]}
                pages = data.get("pages", [])
                flat = []
                for page in pages:
                    items = page.get("items", page.get("content", []))
                    for item in items:
                        item.setdefault("page_idx", page.get("page_idx", 0))
                        flat.append(item)
                return flat if flat else None
    return None


def find_markdown(mineru_dir: pathlib.Path, doc_id: str) -> Optional[pathlib.Path]:
    """找到 MinerU 生成的 .md 文件"""
    base = mineru_dir / doc_id / doc_id / "hybrid_auto"
    candidates = [
        base / f"{doc_id}.md",
        base / "content.md",
    ]
    for p in candidates:
        if p.exists():
            return p
    # fallback: 任意 .md（排除 formulas.md）
    if base.exists():
        for md in base.glob("*.md"):
            if "formula" not in md.name.lower():
                return md
    return None


# ─────────────────────────── element image locator ───────────────────────────

def resolve_element_image(elem: dict, mineru_dir: pathlib.Path) -> Optional[pathlib.Path]:
    """
    根据 element 信息定位截图文件。
    优先用 metadata.image_filename，再 fallback 到 image_path 末段。
    """
    doc_id = elem.get("doc_id", "")
    img_dir = mineru_dir / doc_id / doc_id / "hybrid_auto" / "images"

    # 1) 从 metadata 拿 filename
    meta = elem.get("metadata", {}) or {}
    fname = meta.get("image_filename", "")
    if fname:
        p = img_dir / fname
        if p.exists():
            return p

    # 2) 从 image_path 拿末段
    raw_path = elem.get("image_path", "")
    if raw_path:
        fname2 = pathlib.PurePosixPath(raw_path).name or pathlib.PureWindowsPath(raw_path).name
        if fname2:
            p = img_dir / fname2
            if p.exists():
                return p

    # 3) 按命名规则猜测: doc_id_pageX_figN.jpg / doc_id_pageX_tblN.jpg
    etype = elem.get("element_type", "")
    page = elem.get("page_idx", 0)
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
    bridge_ids: List[str],
) -> str:
    """
    从 MinerU markdown 中截取桥接段落。

    策略:
    - 找到两个 element 在 markdown 中的位置（通过 caption/label 匹配）
    - 两个 element 之间的文本即为桥接段落
    - 如果两个 element 在不同文档（跨文档），则分别提取各自上下文
    """
    md_text = md_path.read_text(encoding="utf-8", errors="ignore")
    lines = md_text.split("\n")

    # 用 caption 或 label 在 md 中定位 element
    def find_element_line(elem: dict) -> int:
        """返回 element 在 markdown 中的行号（0-indexed），找不到返回 -1"""
        label = elem.get("label", "")
        caption = (elem.get("caption") or "")[:60]  # 前 60 字符
        for i, line in enumerate(lines):
            if label and label in line:
                return i
            if caption and len(caption) > 15 and caption[:30] in line:
                return i
        return -1

    doc_a = elem_a.get("doc_id", "")
    doc_b = elem_b.get("doc_id", "")

    if doc_a == doc_b:
        # 同一文档：截取两个 element 之间的文本
        line_a = find_element_line(elem_a)
        line_b = find_element_line(elem_b)
        if line_a >= 0 and line_b >= 0:
            start = min(line_a, line_b)
            end = max(line_a, line_b)
            # 包含两端各扩展 2 行上下文
            start = max(0, start - 2)
            end = min(len(lines), end + 3)
            return "\n".join(lines[start:end])

    # fallback：用 element 的 context_before/after 拼接
    parts = []
    for elem, tag in [(elem_a, "Element A"), (elem_b, "Element B")]:
        ctx_before = (elem.get("context_before") or "").strip()
        ctx_after = (elem.get("context_after") or "").strip()
        if ctx_before or ctx_after:
            parts.append(f"### {tag} ({elem.get('element_id', '?')}) 上下文\n")
            if ctx_before:
                parts.append(f"**[前文]**\n{ctx_before}\n")
            if ctx_after:
                parts.append(f"**[后文]**\n{ctx_after}\n")

    return "\n".join(parts) if parts else ""


def extract_bridge_from_context(elem_a: dict, elem_b: dict) -> str:
    """
    当 markdown 文件不可用时，用 multimodal_elements.json 中的
    context_before/context_after 作为桥接段落的近似。
    """
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
    l3_db: Dict[str, dict],
    mineru_dir: pathlib.Path,
):
    """为单条 query 创建 evidence 文件夹"""
    qid = item["query_id"]
    l3_rec = l3_db.get(qid, {})

    eids = item.get("element_ids", [])
    tags = ["a", "b"][:len(eids)]
    elems = []
    for eid in eids:
        elems.append(elem_db.get(eid, {}))

    # ── 1) query_info.json ──
    info = {
        "query_id": qid,
        "pair_type": item.get("pair_type"),
        "is_cross_doc": item.get("is_cross_doc", False),
        "hop_distance": item.get("hop_distance"),
        "reasoning_depth": item.get("reasoning_depth"),
        "element_ids": eids,
        "path": item.get("path", []),
        "query": item["query"],
        "answer": item["answer"],
        "reasoning_steps": item.get("reasoning_steps", []),
        "required_evidence_spans": l3_rec.get("required_evidence_spans", []),
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

        # Formula 元素没有截图，只有 LaTeX 文本 —— 保存为 .md
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
            img_results[tag] = True  # formula 算 "found"（只是形式不同）
            continue

        # Figure / Table —— 找截图
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

    # ── 3) element 上下文 (从 content_list 或 fallback 到 elements json) ──
    for tag, eid, elem in zip(tags, eids, elems):
        if not elem:
            continue
        doc_id = elem.get("doc_id", "")
        etype = elem.get("element_type", "")
        lines_out = [
            f"# {eid}",
            f"- type: {etype}",
            f"- doc_id: {doc_id}",
            f"- label: {elem.get('label', '')}",
            f"- page_idx: {elem.get('page_idx', '?')}",
            "",
        ]

        # caption
        caption = (elem.get("caption") or "").strip()
        if caption:
            lines_out += ["## Caption", "", caption, ""]

        # content（formula 的 LaTeX 等）
        content = (elem.get("content") or "").strip()
        if content and content != caption:
            lines_out += ["## Content", "", content, ""]

        # context_before / context_after
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
    bridge_ids = [p for p in item.get("path", []) if "::p::" in p]
    bridge_text = ""

    if elems and len(elems) >= 2:
        elem_a, elem_b = elems[0], elems[1]
        doc_a = elem_a.get("doc_id", "")

        # 尝试从 markdown 截取
        md_path = find_markdown(mineru_dir, doc_a)
        if md_path:
            bridge_text = extract_bridge_from_markdown(md_path, elem_a, elem_b, bridge_ids)

        # 跨文档时，也截取 doc_b 的 markdown
        doc_b = elem_b.get("doc_id", "")
        if doc_b != doc_a:
            md_path_b = find_markdown(mineru_dir, doc_b)
            if md_path_b:
                bridge_b = extract_bridge_from_markdown(md_path_b, elem_b, elem_a, bridge_ids)
                if bridge_b:
                    bridge_text += f"\n\n---\n\n## Doc B ({doc_b}) 相关片段\n\n{bridge_b}"

        # fallback：用 context_before/after
        if not bridge_text.strip():
            bridge_text = extract_bridge_from_context(elem_a, elem_b)

    # 追加 reasoning_steps 中 bridge_paragraph 的 evidence_span
    bridge_spans = []
    for step in item.get("reasoning_steps", []):
        if step.get("evidence_element_id") == "bridge_paragraph":
            span = step.get("evidence_span", "")
            if span:
                bridge_spans.append(span)

    if bridge_spans:
        bridge_text += "\n\n---\n\n## Reasoning Steps 中的桥接 Evidence Span\n\n"
        for i, span in enumerate(bridge_spans, 1):
            bridge_text += f"**Bridge {i}:** {span}\n\n"

    if bridge_text.strip():
        (folder / "bridge_paragraphs.md").write_text(bridge_text.strip(), encoding="utf-8")

    return img_results


# ─────────────────────────── main ───────────────────────────

def main():
    parser = argparse.ArgumentParser(description="打包 L3 demo query evidence 到独立文件夹")
    parser.add_argument(
        "--mineru-dir",
        type=pathlib.Path,
        required=True,
        help="MinerU 输出根目录，如 D:/Code_store/data-process-test/data/mineru_output",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=ROOT / "data/m2/l3_demo_evidence",
        help="输出文件夹（默认 data/m2/l3_demo_evidence）",
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
    l3_db = load_l3_queries(L3_QUERIES)

    print(f"Demo queries:    {len(demo)}")
    print(f"Elements DB:     {len(elem_db)}")
    print(f"L3 query DB:     {len(l3_db)}")
    print(f"MinerU dir:      {mineru_dir}")
    print(f"Output dir:      {out_dir}")
    print()

    # 清理旧输出（Windows 下文件可能被占用，先 rename 再删）
    if out_dir.exists():
        tmp_name = out_dir.with_name(out_dir.name + "_old_tmp")
        if tmp_name.exists():
            shutil.rmtree(tmp_name, ignore_errors=True)
        try:
            out_dir.rename(tmp_name)
            shutil.rmtree(tmp_name, ignore_errors=True)
        except (PermissionError, OSError):
            # 如果 rename 也失败，逐文件清理
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
        folder = out_dir / qid
        folder.mkdir()

        img_results = build_query_folder(folder, item, elem_db, l3_db, mineru_dir)

        n_files = len(list(folder.iterdir()))
        total_imgs += len(img_results)
        found_imgs += sum(1 for v in img_results.values() if v)

        status = "  ".join(
            f"elem_{tag}: {'OK' if ok else 'MISSING'}"
            for tag, ok in img_results.items()
        )
        print(f"  [{idx:02d}] {qid}  ->  {n_files} files  ({status})")

    print(f"\nImages found: {found_imgs}/{total_imgs}")
    print(f"Output: {out_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
