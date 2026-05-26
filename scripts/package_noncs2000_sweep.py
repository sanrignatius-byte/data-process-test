#!/usr/bin/env python3
"""Package noncs2000 sweep outputs into M4query_v2_clean_chunk_aug-style delivery.

Inputs:
  - 3 per-config merged query files (acad / acad_persona / mixed_persona)
  - noncs2000_elements_enriched_2111.json   (figure/table/formula + enriched desc)
  - noncs2000_latex_reference_graph_2111.json (paragraphs)
  - noncs2000_hierarchical_chunks_2111_enriched.json (chunks)

Outputs (to data/03_queries/M4query_noncs2000_v1/):
  - corpus.jsonl.gz       : figure/table/formula/paragraph/chunk passages
  - train_triplets.jsonl  : per-query 3-4 positives + 5 hard_neg + 5 random_neg
  - README.md             : stats

Schema matches M4query_v2_clean_chunk_aug:
  passage = {passage_id, type, text, caption, image_path, description}
  triplet = {query_id, query, positive_passages[], hard_negative_passages[], random_negative_passages[]}
"""

from __future__ import annotations
import argparse
import collections
import gzip
import json
import random
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUERY_DIR = ROOT / "data" / "03_queries"
GRAPH_DIR = ROOT / "data" / "01_graphs"
ENRICHED_DIR = ROOT / "data" / "02_enriched"

# ── Inputs ─────────────────────────────────────────────────────────────
QUERY_FILES = {
    "acad":           QUERY_DIR / "noncs2000_sweep_l3_acad_final_pass.jsonl",
    "acad_persona":   QUERY_DIR / "noncs2000_sweep_l3_acad_persona_final_pass.jsonl",
    "mixed_persona":  QUERY_DIR / "noncs2000_sweep_l3_mixed_persona_final_pass.jsonl",
}
ELEMENTS_PATH = ENRICHED_DIR / "noncs2000_elements_enriched_2111.json"
SECTIONS_PATH = ENRICHED_DIR / "noncs2000_section_nodes_enriched_2111.json"
CHUNKS_PATH   = GRAPH_DIR / "noncs2000_hierarchical_chunks_2111_enriched.json"

# ── Output pack ───────────────────────────────────────────────────────
PACK_NAME = "M4query_noncs2000_v1"
PACK_DIR  = QUERY_DIR / PACK_NAME

# Negative-sampling targets (mirror M4query_v2_clean_chunk_aug)
HARD_NEG_TARGET   = 5
RANDOM_NEG_TARGET = 5
TEXT_SLOTS_MIN    = 2
TEXT_SLOTS_MAX    = 3   # 50% pick 2, 50% pick 3 — gives ~50% text in negs

random.seed(42)


# ── Helpers ───────────────────────────────────────────────────────────
def normalize_eid(eid: str) -> str:
    return (eid or "").replace("_fig_", "_figure_").replace("_equation_", "_formula_")


def doc_of(eid: str) -> str:
    eid = normalize_eid(eid)
    return eid.split("_", 1)[0] if "_" in eid else eid


def infer_type(eid: str) -> str:
    eid = normalize_eid(eid)
    if "_figure_" in eid: return "figure"
    if "_table_"  in eid: return "table"
    if "_formula_" in eid: return "formula"
    if "_section_" in eid: return "section"
    if "_chunk_" in eid: return "chunk"
    return "section"


# ── Step 1 : Load + merge queries ─────────────────────────────────────
def load_queries():
    all_q = []
    seen = set()
    per_config = collections.Counter()
    for cfg, path in QUERY_FILES.items():
        if not path.exists():
            print(f"  WARN: missing {path}")
            continue
        for line in open(path):
            o = json.loads(line)
            qid = o["query_id"]
            if qid in seen: continue
            seen.add(qid)
            o["_source_config"] = cfg
            all_q.append(o)
            per_config[cfg] += 1
    print(f"[merge] {len(all_q)} unique pass queries")
    for cfg, n in per_config.most_common():
        print(f"   {cfg}: {n}")
    return all_q


# ── Step 2 : Build corpus ─────────────────────────────────────────────
def build_corpus(elements_doc_filter):
    """Build corpus dict {passage_id → passage}. Restrict to doc_ids in filter."""
    corpus = {}

    # figure / table / formula (+ enriched description)
    print("[corpus] loading elements …")
    elem_data = json.load(open(ELEMENTS_PATH))
    docs = elem_data.get("documents", {})
    n_visual = 0
    for doc_id, doc in docs.items():
        if doc_id not in elements_doc_filter: continue
        for eid, el in (doc.get("elements") or {}).items():
            eid_n = normalize_eid(eid)
            t = el.get("element_type") or infer_type(eid_n)
            if t not in {"figure", "table", "formula"}: continue
            caption = (el.get("caption") or "").strip()
            content = (el.get("content") or "").strip()
            desc = (el.get("enriched_content") or "").strip()
            img = (el.get("image_path") or "").strip()
            corpus[eid_n] = {
                "passage_id": eid_n,
                "type": t,
                "text": content[:2000],
                "caption": caption,
                "image_path": img,
                "description": desc,
            }
            n_visual += 1
    print(f"  visual passages: {n_visual}")
    del elem_data

    # section (from section_nodes_enriched — section-level passages, type=section)
    print("[corpus] loading section nodes …")
    sec_data = json.load(open(SECTIONS_PATH))
    n_sec = 0
    for sec in sec_data.get("sections", []):
        doc_id = sec.get("doc_id")
        if doc_id not in elements_doc_filter: continue
        sid = sec.get("section_id") or ""
        # "0704.0212::sec::0001" → "0704.0212_section_0001"
        idx = sid.rsplit("::", 1)[-1] if "::" in sid else sid
        pid_n = f"{doc_id}_section_{idx}"
        text = (sec.get("section_text") or "").strip() or (sec.get("enriched_content") or "").strip()
        if not text: continue
        corpus[pid_n] = {
            "passage_id": pid_n,
            "type": "section",
            "text": text[:3000],
            "caption": (sec.get("section_title") or "").strip(),
            "image_path": "",
            "description": (sec.get("enriched_content") or "").strip()[:1500],
        }
        n_sec += 1
    print(f"  sections: {n_sec}")
    del sec_data

    # chunks
    print("[corpus] loading hierarchical chunks …")
    chunk_data = json.load(open(CHUNKS_PATH))
    chunks_by_doc = chunk_data.get("documents", chunk_data)
    n_chunk = 0
    for doc_id, doc in chunks_by_doc.items():
        if doc_id not in elements_doc_filter: continue
        fines = doc.get("fine_chunks") or doc.get("chunks") or []
        for ch in fines:
            cid = ch.get("chunk_id") or ch.get("id") or ""
            if not cid: continue
            cid_n = cid if cid.startswith(doc_id) else f"{doc_id}_chunk_{cid}"
            text = (ch.get("text") or ch.get("content") or "").strip()
            if not text: continue
            corpus[cid_n] = {
                "passage_id": cid_n,
                "type": "chunk",
                "text": text[:4000],
                "caption": "",
                "image_path": "",
                "description": (ch.get("enriched_description") or "").strip(),
            }
            n_chunk += 1
    print(f"  chunks: {n_chunk}")

    # Build elem→chunk index for positive enrichment
    elem_to_chunks = collections.defaultdict(list)
    for doc_id, doc in chunks_by_doc.items():
        if doc_id not in elements_doc_filter: continue
        for ch in (doc.get("fine_chunks") or doc.get("chunks") or []):
            cid = ch.get("chunk_id") or ""
            if not cid: continue
            cid_n = cid if cid.startswith(doc_id) else f"{doc_id}_chunk_{cid}"
            for eid in (ch.get("element_ids") or []):
                elem_to_chunks[normalize_eid(eid)].append(cid_n)
    print(f"[index] elem→chunks: {len(elem_to_chunks)} elements covered")
    del chunk_data
    return corpus, elem_to_chunks


# ── Step 3 : Triplet construction ─────────────────────────────────────
_LATEX_REF_RE = re.compile(r"(?:Sec|Table|Tab|Fig|Eq|Equation|Section|Appendix)\.?~?\[[^\]]+\]")

def _bridge_probe(text: str) -> str:
    """Strip LaTeX refs, return a clean probe substring for section matching."""
    cleaned = _LATEX_REF_RE.sub("", text or "").strip()
    return cleaned[:60]


def build_positives(query, corpus, elem_to_chunks, doc_to_chunks, doc_to_sections):
    """Collect positives: query.element_ids + bridge section + chunks (with doc-level fallback)."""
    pos = []
    seen = set()
    def add(pid):
        if pid and pid in corpus and pid not in seen:
            pos.append(pid); seen.add(pid)

    doc_id = query.get("doc_id")
    for eid in (query.get("element_ids") or []):
        add(normalize_eid(eid))

    # bridge paragraph → section text match (LaTeX-ref-cleaned probe)
    for span in (query.get("required_evidence_spans") or []):
        if span.get("element_id") == "bridge_paragraph":
            content = (span.get("content") or "").strip()
            if not content: continue
            probe = _bridge_probe(content)
            if not probe or len(probe) < 12: continue
            for sec_pid in doc_to_sections.get(doc_id, []):
                if probe in corpus[sec_pid]["text"]:
                    add(sec_pid); break

    # path nodes → section heuristic match
    for node in (query.get("path") or []):
        if isinstance(node, dict): node = node.get("node_id") or node.get("id")
        if isinstance(node, str) and ("paragraph" in node or "section" in node or "_p_" in node):
            tail = node.split("::")[-1]
            for sec_pid in doc_to_sections.get(doc_id, []):
                if sec_pid.endswith(tail): add(sec_pid); break

    # chunks containing positive elements (direct elem→chunk)
    for eid in list(seen):
        for cid in elem_to_chunks.get(eid, [])[:2]:
            add(cid)
            if len(pos) >= 4: break
        if len(pos) >= 4: break

    # Fallback: if still <3 positives, pad with same-doc chunks (then sections) so every triplet has ≥3
    if len(pos) < 3:
        for cid in doc_to_chunks.get(doc_id, []):
            add(cid)
            if len(pos) >= 3: break
    if len(pos) < 3:
        for sec_pid in doc_to_sections.get(doc_id, []):
            add(sec_pid)
            if len(pos) >= 3: break

    return pos[:4]   # cap at 4 to match M4query_v2_clean_chunk_aug


def sample_negatives(query, positives, corpus, doc_to_passages, all_passage_ids, n=5):
    """Sample hard (same doc) and random (other doc) negatives."""
    doc_id = query.get("doc_id")
    same_doc = [p for p in doc_to_passages.get(doc_id, []) if p not in positives]
    other_doc = [p for p in all_passage_ids if not p.startswith(doc_id) and p not in positives]

    text_slots = random.choice([TEXT_SLOTS_MIN, TEXT_SLOTS_MAX])
    visual_slots = n - text_slots

    def pick(pool, k, type_filter=None):
        if type_filter:
            pool = [p for p in pool if corpus[p]["type"] in type_filter]
        return random.sample(pool, min(k, len(pool)))

    hard_text   = pick(same_doc, text_slots,  type_filter={"section", "chunk"})
    hard_visual = pick([p for p in same_doc if p not in hard_text], visual_slots,
                       type_filter={"figure", "table", "formula"})
    hard = (hard_text + hard_visual)[:n]
    while len(hard) < n and same_doc:
        rest = [p for p in same_doc if p not in hard]
        if not rest: break
        hard.append(random.choice(rest))

    rand_text   = pick(other_doc, text_slots,  type_filter={"section", "chunk"})
    rand_visual = pick([p for p in other_doc if p not in rand_text], visual_slots,
                       type_filter={"figure", "table", "formula"})
    rand = (rand_text + rand_visual)[:n]
    while len(rand) < n and other_doc:
        rest = [p for p in other_doc if p not in rand]
        if not rest: break
        rand.append(random.choice(rest))

    return hard, rand


def build_triplets(queries, corpus, elem_to_chunks):
    print("[triplets] indexing corpus by doc …")
    doc_to_passages = collections.defaultdict(list)
    doc_to_chunks   = collections.defaultdict(list)
    doc_to_sections = collections.defaultdict(list)
    for pid, p in corpus.items():
        d = doc_of(pid)
        doc_to_passages[d].append(pid)
        if p["type"] == "chunk":   doc_to_chunks[d].append(pid)
        if p["type"] == "section": doc_to_sections[d].append(pid)
    all_passage_ids = list(corpus.keys())
    print(f"  docs: {len(doc_to_passages)} | total passages: {len(all_passage_ids)}")

    triplets = []
    pos_size_dist = collections.Counter()
    skipped = 0
    for q in queries:
        pos = build_positives(q, corpus, elem_to_chunks, doc_to_chunks, doc_to_sections)
        if len(pos) < 2:
            skipped += 1
            continue
        hard, rand = sample_negatives(q, pos, corpus, doc_to_passages, all_passage_ids)
        triplets.append({
            "query_id": q["query_id"],
            "query": q.get("query", ""),
            "positive_passages": pos,
            "hard_negative_passages": hard,
            "random_negative_passages": rand,
        })
        pos_size_dist[len(pos)] += 1
    print(f"[triplets] built {len(triplets)} | skipped (insufficient positives) {skipped}")
    print(f"  positive count distribution: {dict(pos_size_dist)}")
    return triplets


# ── Step 4 : Write delivery ───────────────────────────────────────────
def write_corpus_gz(corpus, path: Path):
    print(f"[write] corpus → {path}")
    type_count = collections.Counter()
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for p in corpus.values():
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
            type_count[p["type"]] += 1
    print(f"  type counts: {dict(type_count)}")
    return type_count


def write_triplets(triplets, path: Path):
    print(f"[write] triplets → {path}")
    with open(path, "w", encoding="utf-8") as f:
        for t in triplets:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")


def write_readme(queries, triplets, corpus_type_count, neg_type_count, path: Path):
    pos_size_dist = collections.Counter(len(t["positive_passages"]) for t in triplets)
    total = sum(neg_type_count.values()) or 1
    md = []
    md.append(f"# {PACK_NAME}\n")
    md.append(f"noncs2000 (2111 LaTeX-sourced arXiv papers, hop≥3 L3 sweep) 多粒度交付包。\n")
    md.append(f"Built {datetime.now().isoformat(timespec='seconds')}\n")
    md.append("## 文件\n")
    md.append("| 路径 | 数量 | 作用 |")
    md.append("| --- | ---: | --- |")
    md.append(f"| `corpus.jsonl.gz` | {sum(corpus_type_count.values())} | figure/table/formula/section/chunk 多粒度。解压：`gzip -cd corpus.jsonl.gz > corpus.jsonl`。 |")
    pos_min = min(pos_size_dist) if pos_size_dist else 0
    pos_max = max(pos_size_dist) if pos_size_dist else 0
    md.append(f"| `train_triplets.jsonl` | {len(triplets)} | {pos_min}-{pos_max} positive + {HARD_NEG_TARGET} hard_neg + {RANDOM_NEG_TARGET} random_neg。 |\n")
    md.append("## corpus type 分布\n")
    md.append("| type | 数量 |")
    md.append("| --- | ---: |")
    for t in ("section", "chunk", "figure", "table", "formula"):
        md.append(f"| `{t}` | {corpus_type_count.get(t, 0)} |")
    md.append("")
    md.append("## positive 数量分布\n")
    md.append("| 数量 | query 数 |")
    md.append("| --- | ---: |")
    for k in sorted(pos_size_dist):
        md.append(f"| {k} | {pos_size_dist[k]} |")
    md.append("")
    md.append("## negative type 占比\n")
    md.append("| type | 占比 |")
    md.append("| --- | ---: |")
    for t in ("section", "chunk", "figure", "table", "formula"):
        pct = 100 * neg_type_count.get(t, 0) / total
        md.append(f"| `{t}` | {pct:.1f}% |")
    md.append("")
    md.append("## query 来源\n")
    src = collections.Counter(q.get("_source_config", "?") for q in queries)
    md.append("| config | unique pass |")
    md.append("| --- | ---: |")
    for cfg, n in src.most_common():
        md.append(f"| {cfg} | {n} |")
    md.append("")
    md.append("## 生成脚本\n`scripts/package_noncs2000_sweep.py`\n")
    path.write_text("\n".join(md), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack-dir", default=str(PACK_DIR))
    args = ap.parse_args()

    pack_dir = Path(args.pack_dir)
    pack_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. queries
    queries = load_queries()
    doc_ids = {q["doc_id"] for q in queries if q.get("doc_id")}
    print(f"[docs] queries cover {len(doc_ids)} unique docs")

    # ── 2. corpus
    corpus, elem_to_chunks = build_corpus(doc_ids)
    print(f"[corpus] total passages: {len(corpus)}")

    # ── 3. triplets
    triplets = build_triplets(queries, corpus, elem_to_chunks)

    # ── 4. write
    corpus_type_count = write_corpus_gz(corpus, pack_dir / "corpus.jsonl.gz")
    write_triplets(triplets, pack_dir / "train_triplets.jsonl")

    # Compute negative type distribution post-hoc
    neg_type_count = collections.Counter()
    for t in triplets:
        for p in t["hard_negative_passages"] + t["random_negative_passages"]:
            if p in corpus: neg_type_count[corpus[p]["type"]] += 1
    write_readme(queries, triplets, corpus_type_count, neg_type_count, pack_dir / "README.md")

    print(f"\n✅ DONE → {pack_dir}")
    print(f"   {pack_dir}/corpus.jsonl.gz")
    print(f"   {pack_dir}/train_triplets.jsonl")
    print(f"   {pack_dir}/README.md")


if __name__ == "__main__":
    main()
