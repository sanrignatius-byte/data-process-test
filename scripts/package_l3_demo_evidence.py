#!/usr/bin/env python3
"""
将 L3 demo 10 条 query 的 evidence 按 query 打包到独立文件夹：

输出结构:
  data/m2/l3_demo_evidence/
  ├── 01_l3_de_1703.06856_0021/
  │   ├── query_info.md          # query + answer + 推理链
  │   ├── element_a_figure.jpg   # 图片（如果本地可访问）
  │   ├── element_a_context.md   # caption + context_before/after
  │   ├── element_b_context.md
  │   ├── bridge_evidence.md     # 桥接段落 evidence
  │   └── reasoning_chain.md     # 三步推理链详情
  ├── 02_l3_de_1706.02409_0081/
  │   └── ...
  └── ...

用法: python scripts/package_l3_demo_evidence.py
"""
import json, pathlib, shutil, textwrap

ROOT = pathlib.Path(__file__).resolve().parent.parent

DEMO_FILE  = ROOT / "data/m2/l3_demo_selection_10.json"
ELEMENTS   = ROOT / "data/multimodal_elements.json"
L3_QUERIES = ROOT / "data/m2/l3_reasoning_chain_queries.jsonl"
OUT_DIR    = ROOT / "data/m2/l3_demo_evidence"


# --------------- loaders ---------------

def load_elements(path):
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


def load_l3_queries(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["query_id"]] = rec
    return out


# --------------- writers ---------------

def write_query_info(folder: pathlib.Path, item: dict):
    """query_info.md — 问答本体 + 元信息"""
    lines = [
        f"# {item['query_id']}",
        "",
        f"- **pair_type**: {item.get('pair_type', '?')}",
        f"- **cross_doc**: {item.get('is_cross_doc', '?')}",
        f"- **hop_distance**: {item.get('hop_distance', '?')}",
        f"- **reasoning_depth**: {item.get('reasoning_depth', '?')}",
        f"- **element_ids**: {item.get('element_ids', [])}",
        "",
        "## Query",
        "",
        item["query"],
        "",
        "## Answer",
        "",
        item["answer"],
        "",
    ]
    (folder / "query_info.md").write_text("\n".join(lines), encoding="utf-8")


def write_element_context(folder: pathlib.Path, tag: str, eid: str, elem: dict):
    """element_{a|b}_context.md — caption + content + 上下文"""
    etype = elem.get("element_type", "unknown")
    lines = [
        f"# Element: {eid}",
        "",
        f"- **type**: {etype}",
        f"- **doc_id**: {elem.get('doc_id', '?')}",
        f"- **label**: {elem.get('label', '')}",
        f"- **page_idx**: {elem.get('page_idx', '?')}",
        "",
        "## Caption",
        "",
        elem.get("caption") or "(无)",
        "",
        "## Content",
        "",
        elem.get("content") or "(无)",
        "",
        "## Context Before",
        "",
        elem.get("context_before") or "(无)",
        "",
        "## Context After",
        "",
        elem.get("context_after") or "(无)",
        "",
    ]
    # referring_paragraphs
    refs = elem.get("referring_paragraphs") or []
    if refs:
        lines += ["## Referring Paragraphs", ""]
        for i, rp in enumerate(refs, 1):
            if isinstance(rp, dict):
                lines.append(f"### Ref {i}")
                lines.append("")
                lines.append(rp.get("text", str(rp)))
            else:
                lines.append(f"{i}. {rp}")
            lines.append("")

    (folder / f"element_{tag}_context.md").write_text("\n".join(lines), encoding="utf-8")


def try_copy_image(folder: pathlib.Path, tag: str, elem: dict):
    """尝试复制图片文件到文件夹，返回是否成功"""
    img = elem.get("image_path", "")
    if not img:
        return False
    src = pathlib.Path(img)
    if not src.exists():
        # 尝试相对于项目根目录
        src = ROOT / img
    if not src.exists():
        # 记录路径供手动复制
        (folder / f"element_{tag}_image_NOT_FOUND.txt").write_text(
            f"原始路径: {img}\n请从集群手动复制此文件。\n", encoding="utf-8"
        )
        return False
    suffix = src.suffix or ".jpg"
    dst = folder / f"element_{tag}_{elem.get('element_type', 'img')}{suffix}"
    shutil.copy2(src, dst)
    return True


def write_reasoning_chain(folder: pathlib.Path, steps: list):
    """reasoning_chain.md — 三步推理链"""
    lines = ["# Reasoning Chain", ""]
    for step in steps:
        sid = step.get("step_id", "?")
        lines += [
            f"## Step {sid}: {step.get('reasoning_role', '')}",
            "",
            f"- **evidence_element**: {step.get('evidence_element_id', '?')}",
            f"- **evidence_type**: {step.get('evidence_type', '?')}",
            f"- **depends_on**: {step.get('depends_on_steps', [])}",
            "",
            "### Evidence Span",
            "",
            step.get("evidence_span", "(无)"),
            "",
            "### Produces Claim",
            "",
            step.get("produces_claim", "(无)"),
            "",
        ]
    (folder / "reasoning_chain.md").write_text("\n".join(lines), encoding="utf-8")


def write_bridge_evidence(folder: pathlib.Path, steps: list, l3_rec: dict):
    """bridge_evidence.md — 桥接段落 evidence（从 reasoning_steps 中提取）"""
    bridge_spans = []
    for step in steps:
        if step.get("evidence_element_id") == "bridge_paragraph":
            bridge_spans.append(step.get("evidence_span", ""))

    # 也从 required_evidence_spans 取
    for span_rec in l3_rec.get("required_evidence_spans", []):
        if span_rec.get("element_id") == "bridge_paragraph":
            s = span_rec.get("span", "")
            if s and s not in bridge_spans:
                bridge_spans.append(s)

    if not bridge_spans:
        return

    lines = ["# Bridge Evidence (桥接段落)", ""]
    for i, span in enumerate(bridge_spans, 1):
        lines += [f"## Bridge {i}", "", span, ""]

    (folder / "bridge_evidence.md").write_text("\n".join(lines), encoding="utf-8")


# --------------- main ---------------

def main():
    demo    = json.loads(DEMO_FILE.read_text(encoding="utf-8"))
    elem_db = load_elements(ELEMENTS)
    l3_db   = load_l3_queries(L3_QUERIES)

    print(f"Demo queries:  {len(demo)}")
    print(f"Elements DB:   {len(elem_db)}")
    print(f"L3 query DB:   {len(l3_db)}")

    # 清理旧输出
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    img_found = 0
    img_total = 0

    for idx, item in enumerate(demo, 1):
        qid = item["query_id"]
        l3_rec = l3_db.get(qid, {})
        folder_name = f"{idx:02d}_{qid}"
        folder = OUT_DIR / folder_name
        folder.mkdir()

        # 1) query_info.md
        write_query_info(folder, item)

        # 2) 每个 element 的 context + 图片
        eids = item.get("element_ids", [])
        tags = ["a", "b"] if len(eids) == 2 else [chr(ord("a") + i) for i in range(len(eids))]
        for tag, eid in zip(tags, eids):
            elem = elem_db.get(eid)
            if not elem:
                (folder / f"element_{tag}_MISSING.txt").write_text(
                    f"element_id: {eid}\n未在 multimodal_elements.json 中找到\n",
                    encoding="utf-8",
                )
                continue
            write_element_context(folder, tag, eid, elem)
            img_total += 1
            if try_copy_image(folder, tag, elem):
                img_found += 1

        # 3) reasoning_chain.md
        steps = item.get("reasoning_steps", [])
        if steps:
            write_reasoning_chain(folder, steps)

        # 4) bridge_evidence.md
        write_bridge_evidence(folder, steps, l3_rec)

        # 统计
        n_files = len(list(folder.iterdir()))
        print(f"  [{idx:02d}] {qid}  →  {n_files} files")

    print(f"\nImages copied: {img_found}/{img_total} (其余留 NOT_FOUND 标记，需从集群手动复制)")
    print(f"Output: {OUT_DIR}/")
    print("Done!")


if __name__ == "__main__":
    main()
