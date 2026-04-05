"""训练数据的统一 Schema —— 用 Pydantic 硬卡格式，不合规直接报错。

这是整个 training pipeline 的 "宪法"，所有 L1/L2/L3 的 query 最终都要
转成这里定义的 StandardQuery。下游的 triplet 构建、数据集导出、负样本采样
全都依赖这套 schema。

关键设计：
  - EvidenceSpan 必须有 element_id（不能为空，validator 强制）
  - StandardQuery 必须 ≥1 个 evidence_span（没证据的 query 没有意义）
  - L3 必须有 reasoning_steps（model_validator 强制 —— L3 没有推理链直接炸）
  - schema_version 用于检测格式变更

历史 JSONL 通过 scripts/normalize_queries.py 转换成这个格式。
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Constants ─────────────────────────────────────────────────────────────────

SCHEMA_VERSION = "1.0.0"


# ── Enums ─────────────────────────────────────────────────────────────────────

class DifficultyLevel(str, Enum):
    """查询难度：L1 看一个元素就能答，L2 要两个证据/跨文档，L3 要多跳推理链。"""

    L1 = "L1"  # single-element intra-doc
    L2 = "L2"  # dual-evidence / cross-doc
    L3 = "L3"  # multi-hop reasoning chain


# ── Evidence ──────────────────────────────────────────────────────────────────

class EvidenceSpan(BaseModel):
    """一条证据 —— 指向某个元素（figure/table/formula）的某段文本。

    element_id 不能为空（validator 硬卡），doc_id 用于下游的 doc-level split 防泄漏。
    别从 element_id 字符串里猜 doc_id！直接用 doc_id 字段！（血泪教训）
    """

    element_id: str
    doc_id: str = ""
    element_type: str = ""          # "figure" / "table" / "formula" / "text"
    span_text: str = ""             # the textual excerpt used as evidence
    evidence_type: str = ""         # "observation" / "data" / "attribution" / …
    image_path: Optional[str] = None

    @field_validator("element_id")
    @classmethod
    def element_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("element_id must be a non-empty string")
        return v


# ── Reasoning Step ────────────────────────────────────────────────────────────

class ReasoningStep(BaseModel):
    """多跳推理链中的一步（L3 专属）。

    depends_on_steps 建立步骤间依赖（比如 step 2 依赖 step 1 的结论），
    reasoning_role 分 premise / bridge / conclusion，形成完整推理弧。
    """

    step_id: int
    evidence_element_id: str = ""
    evidence_type: str = ""
    evidence_span: str = ""
    reasoning_role: str = ""        # "premise" / "bridge" / "conclusion"
    depends_on_steps: List[int] = Field(default_factory=list)
    produces_claim: str = ""


# ── Standard Query ────────────────────────────────────────────────────────────

class StandardQuery(BaseModel):
    """L1/L2/L3 通吃的统一 query schema —— training pipeline 的宪法。

    所有产出或消费 query 的环节都必须能用这个 schema 表示。
    历史遗留的各种 JSONL 格式通过 normalize_queries.py 转成这个。

    注意 L3 的硬约束：没有 reasoning_steps 的 L3 query 会被 model_validator 直接拒绝。
    """

    schema_version: str = SCHEMA_VERSION

    # ── identity
    query_id: str
    query_text: str
    answer_text: str

    # ── difficulty
    difficulty_level: DifficultyLevel

    # ── evidence
    evidence_spans: List[EvidenceSpan] = Field(default_factory=list)

    # ── multi-hop (L3)
    reasoning_steps: List[ReasoningStep] = Field(default_factory=list)

    # ── generation metadata (optional, preserved from upstream)
    doc_id: str = ""
    pair_id: str = ""
    pair_type: str = ""             # "figure+table", "figure+formula", …
    element_ids: List[str] = Field(default_factory=list)
    image_paths: List[str] = Field(default_factory=list)
    query_style: str = "academic"
    query_type: str = ""
    persona: str = ""
    hop_distance: int = 0

    # ── QC metadata
    qc_pass: bool = True
    qc_issues: List[str] = Field(default_factory=list)

    # ── catch-all for extra upstream fields
    extra: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("evidence_spans")
    @classmethod
    def at_least_one_evidence(cls, v: list) -> list:
        if len(v) == 0:
            raise ValueError("must have ≥1 evidence span")
        return v

    @model_validator(mode="after")
    def reasoning_steps_required_for_l3(self) -> "StandardQuery":
        if self.difficulty_level == DifficultyLevel.L3 and not self.reasoning_steps:
            raise ValueError("L3 queries must have at least one reasoning step")
        return self


# ── Triplet ───────────────────────────────────────────────────────────────────

class Triplet(BaseModel):
    """对比学习三元组：query + 正样本（对的证据） + 负样本（故意选的干扰项）。

    positive 和 hard_negatives 都是 EvidenceSpan 列表。
    negative_strategy 记录用了什么采样策略（in_doc_swap / random / graph_aware）。
    """

    schema_version: str = SCHEMA_VERSION

    query_id: str
    query_text: str
    difficulty_level: DifficultyLevel

    positive: List[EvidenceSpan]
    hard_negatives: List[EvidenceSpan] = Field(default_factory=list)

    negative_strategy: str = ""     # "in_doc_swap" / "same_type_hard" / …
    difficulty_score: float = 0.0

    # ── provenance
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("positive")
    @classmethod
    def at_least_one_positive(cls, v: list) -> list:
        if len(v) == 0:
            raise ValueError("triplet must have ≥1 positive evidence")
        return v
