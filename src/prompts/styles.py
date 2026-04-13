"""Query 风格路由 —— 决定每个 pair 用什么模板。

select_template() 是模板路由器：
  0. L3（reasoning_chain_target=True）**始终**用 "3step_reasoning_chain"，
     不受 query_style 影响（保证 reasoning_steps[] 结构化输出）
  1. 先按 style 分轨（academic / real_user / mixed）
  2. academic: 按模态组合 + hop 距离选模板
  3. real_user: 5 种子类型轮换（md5 哈希确定性）
  4. mixed: 50% academic + 50% real_user（同一个 pair 每次跑结果一样）
"""
from __future__ import annotations
import hashlib
from typing import Dict
from .templates import REAL_USER_STYLE_CYCLE


# mixed 模式：按 pair_id md5 稳定 50/50 分配 academic / real_user
def resolve_query_style(query_style: str, pair_id: str) -> str:
    """Resolve effective style per pair.

    For `mixed`, use a deterministic 50/50 split by stable md5 hash so the
    same pair gets the same style across runs/machines.
    """
    if query_style != "mixed":
        return query_style
    stable_hash = int(hashlib.md5(pair_id.encode("utf-8")).hexdigest()[:8], 16) if pair_id else 0
    return "academic" if (stable_hash % 2 == 0) else "real_user"


# 模板路由：先按 style 分轨（academic/real_user），再按模态组合选 prompt
def select_template(pair: Dict, query_style: str = "academic") -> str:
    """Choose the right prompt template based on modality combo, hop distance, and style.

    query_style:
      "academic"  — original dual-evidence PhD persona templates (default, backward-compat)
      "real_user" — natural-language reader templates (5 rotating sub-types)
      "mixed"     — 50 % academic / 50 % real_user, chosen deterministically by pair hash
    """
    if query_style == "mixed":
        query_style = resolve_query_style(query_style, str(pair.get("pair_id", "")))

    # Level 3: 3-step reasoning chain ALWAYS takes priority over query style.
    # L3 queries require the structured reasoning_steps[] output that only the
    # 3step_reasoning_chain template provides.  Without it, L3 entries are
    # labelled "reasoning_chain" but carry empty reasoning_steps — see GH issue.
    if pair.get("reasoning_chain_target"):
        return "3step_reasoning_chain"

    if query_style == "real_user":
        # Rotate through 5 real-user sub-types deterministically by stable pair_id hash
        pid = str(pair.get("pair_id", ""))
        stable_hash = int(hashlib.md5(pid.encode("utf-8")).hexdigest()[:8], 16) if pid else 0
        return f"real_user_{REAL_USER_STYLE_CYCLE[stable_hash % len(REAL_USER_STYLE_CYCLE)]}"

    # academic (default)
    a_type = pair["element_a_type"]
    b_type = pair["element_b_type"]
    hop = pair["hop_distance"]
    types = {a_type, b_type}

    if types == {"figure", "table"}:
        if hop <= 1:
            return "figure_table_1hop"
        else:
            return "figure_table_2hop"
    elif types == {"figure", "formula"}:
        return "figure_formula"
    elif types == {"formula", "table"}:
        return "formula_table"
    else:
        return "figure_table_1hop"  # fallback
