"""
M4 Evidence Format - 标准化的多跳问答数据格式

参考格式来自金融领域多文档QA数据集，支持：
- 多跳推理问题
- 精确的证据定位（page, bbox, layout_type）
- 搜索子句分解
- 带citation的答案

Example:
{
    "qid": 1,
    "domain": "research",
    "question": "...",
    "doc_list": ["paper1.pdf", "paper2.pdf", ...],
    "doc_relevant": ["paper1.pdf", "paper2.pdf"],
    "search_clause": ["子查询1", "子查询2", ...],
    "evidence_chunk_list": [
        {
            "doc": "paper1.pdf",
            "type": "chunk",
            "id": 1,
            "page": 5,
            "page_size": [1080, 1920],
            "layout_bbox": [[x1,y1,x2,y2], ...],
            "layout_type": ["text", "table", "image"],
            "layout_info": ["文本内容", "images/xxx.jpg", ...]
        }
    ],
    "answer_short": "简短答案",
    "answer_long": "详细答案，包含[citation:1]引用标记"
}
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Set
from enum import Enum
import json


class LayoutType(Enum):
    """布局元素类型"""
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    FORMULA = "formula"
    CHART = "chart"


@dataclass
class EvidenceChunk:
    """证据块 - 包含精确位置信息"""
    doc: str                              # 文档名
    type: str = "chunk"                   # 类型
    id: int = 0                           # 证据ID
    page: int = 0                         # 页码
    page_size: List[int] = field(default_factory=lambda: [1080, 1920])  # 页面尺寸
    layout_bbox: List[List[int]] = field(default_factory=list)  # 边界框列表
    layout_type: List[str] = field(default_factory=list)        # 类型列表
    layout_info: List[str] = field(default_factory=list)        # 内容列表

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_passage(cls, passage: Any, chunk_id: int = 0) -> "EvidenceChunk":
        """从Passage对象创建EvidenceChunk"""
        doc_id = getattr(passage, 'doc_id', 'unknown')
        page_idx = getattr(passage, 'page_idx', 0)
        modal_type = passage.modal_type.value if hasattr(passage.modal_type, 'value') else str(passage.modal_type)
        content = getattr(passage, 'content', '')
        bbox = getattr(passage, 'bbox', None)
        image_path = getattr(passage, 'image_path', None)

        # 构建layout信息
        layout_bbox = [bbox] if bbox else []
        layout_type = [modal_type]
        layout_info = [image_path if image_path else content[:500]]

        return cls(
            doc=f"{doc_id}.pdf",
            type="chunk",
            id=chunk_id,
            page=page_idx,
            layout_bbox=layout_bbox,
            layout_type=layout_type,
            layout_info=layout_info
        )


@dataclass
class M4QueryData:
    """M4问答数据格式"""
    qid: int
    domain: str
    question: str
    doc_list: List[str]                   # 候选文档池
    doc_relevant: List[str]               # 相关文档
    search_clause: List[str]              # 搜索子句分解
    evidence_chunk_list: List[EvidenceChunk]
    evidence_meta_list: List[Dict] = field(default_factory=list)
    answer_short: str = ""
    answer_long: str = ""
    annotator: str = "auto"
    history: List[str] = field(default_factory=list)  # 对话历史

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "qid": self.qid,
            "domain": self.domain,
            "annotator": self.annotator,
            "history": self.history,
            "question": self.question,
            "doc_list": self.doc_list,
            "doc_relevant": self.doc_relevant,
            "search_clause": self.search_clause,
            "evidence_chunk_list": [e.to_dict() for e in self.evidence_chunk_list],
            "evidence_meta_list": self.evidence_meta_list,
            "answer_short": self.answer_short,
            "answer_long": self.answer_long
        }

    def to_json(self, indent: int = 2) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def validate(self) -> List[str]:
        """验证数据完整性"""
        issues = []

        if not self.question:
            issues.append("Missing question")

        if not self.doc_relevant:
            issues.append("No relevant documents specified")

        if not self.evidence_chunk_list:
            issues.append("No evidence chunks provided")

        if not self.answer_short and not self.answer_long:
            issues.append("No answer provided")

        # 检查citation标记
        if self.answer_long:
            import re
            citations = re.findall(r'\[citation:(\d+)\]', self.answer_long)
            evidence_ids = {e.id for e in self.evidence_chunk_list}
            for cit in citations:
                if int(cit) not in evidence_ids:
                    issues.append(f"Citation {cit} references non-existent evidence")

        return issues

    @property
    def is_multi_hop(self) -> bool:
        """是否为多跳问题"""
        return len(self.evidence_chunk_list) >= 2

    @property
    def is_multi_doc(self) -> bool:
        """是否跨多文档"""
        docs = {e.doc for e in self.evidence_chunk_list}
        return len(docs) >= 2

    @property
    def is_multi_modal(self) -> bool:
        """是否多模态"""
        all_types = set()
        for e in self.evidence_chunk_list:
            all_types.update(e.layout_type)
        return len(all_types) >= 2


# Prompt模板 - 用于生成符合此格式的M4问答
M4_EVIDENCE_GENERATION_PROMPT = """你是一个专业的多跳问答数据标注专家。

给定以下来自多个文档的证据片段，生成一个需要综合多个证据才能回答的复杂问题。

文档证据：
{evidence_context}

要求：
1. 问题必须需要至少{min_hops}个证据片段才能回答
2. 问题应该自然、具体，像真实用户会问的问题
3. 分解问题为多个搜索子句（search_clause）
4. 提供简短答案（answer_short）和详细答案（answer_long）
5. 详细答案必须使用[citation:N]标记引用证据

返回JSON格式：
{{
    "question": "完整问题",
    "search_clause": ["子查询1", "子查询2", ...],
    "answer_short": "简短答案（50字以内）",
    "answer_long": "详细答案，使用[citation:1]、[citation:2]等标记引用证据"
}}

只返回JSON，不要其他内容。"""


M4_MULTI_TURN_PROMPT = """将以下单轮问题转换为多轮对话形式。

原问题：{question}
原答案：{answer}

要求：
1. 拆分为2-3轮自然对话
2. 后续轮次使用代词（它、这个、那些）而非重复实体名
3. 每轮问题应该逐步深入

返回JSON格式：
{{
    "turns": [
        {{"role": "user", "content": "第一轮问题"}},
        {{"role": "assistant", "content": "第一轮回答"}},
        {{"role": "user", "content": "第二轮问题（使用代词）"}},
        {{"role": "assistant", "content": "最终回答"}}
    ]
}}

只返回JSON，不要其他内容。"""
