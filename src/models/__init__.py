"""图 + 检索用到的基础数据结构，全项目共享。

Node / Edge  → 文档引用图的节点和边（analyze_latex_graph_topology 建图，run_phase0_eval_ab 评测用）
Chunk        → 一个检索单元（一个多模态元素 = 一个 chunk，BM25 打分的最小粒度）

这几个 dataclass 故意很轻量 —— 不引任何外部依赖，纯标准库。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Node:
    """图里的一个节点 —— 可以是 figure/table/equation/paragraph/section 等。

    mapped_element_id 是从 LaTeX label 映射到 MinerU element 的结果，
    匹配率目前约 49.8%（Jaccard + 数字后缀 fallback），不高但够用。
    """

    node_id: str
    doc_id: str
    node_type: str  # "figure", "table", "equation", "paragraph", "section", ...
    label: str = ""
    mapped_element_id: Optional[str] = None
    page_idx: Optional[int] = None
    position_idx: Optional[int] = None
    line_no: Optional[int] = None
    line_no_end: Optional[int] = None
    source_file: Optional[str] = None
    text_snippet: Optional[str] = None
    section_level: Optional[int] = None
    section_title: Optional[str] = None
    paragraph_order: Optional[int] = None


@dataclass
class Edge:
    """图里的一条有向边 —— paragraph_ref / backbone / element_ref / cross_doc_cite 等。

    key() 返回 (source, target, type) 三元组，用来 set 去重 ——
    建图过程中重复边巨多，这个 O(1) 去重很关键。
    """

    source_id: str
    target_id: str
    doc_id: str = ""
    edge_type: str = ""  # "paragraph_ref", "backbone", "element_ref", ...
    weight: float = 1.0

    def key(self) -> tuple:
        """Deduplication key: (source_id, target_id, edge_type)."""
        return (self.source_id, self.target_id, self.edge_type)


@dataclass
class Chunk:
    """一个检索单元 = 一个多模态元素（figure/table/formula）。

    text 字段是拼接 enriched_content + caption + content + context_before 的
    检索友好文本，给 BM25 打分用。enriched_* 字段是 MoDora enrichment 后加的。
    """

    chunk_id: str
    doc_id: str
    text: str
    caption: str = ""
    content: str = ""
    context: str = ""
    enriched_title: str = ""
    enriched_content: str = ""
