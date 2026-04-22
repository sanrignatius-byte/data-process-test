"""QC 质量闸门 —— 不合格的 query 一个都不放过。

对外的 API：
  - qc_multihop_query  → 严格学术风格 QC（15+ 个检查组合）
  - qc_real_user_query  → 宽松版（不查 template shortcuts，yes/no 只是 issue 不硬 fail）
  - qc_reasoning_depth  → L3 推理深度分析（M4 Schema 1 支持）

每个检查函数是独立的原子操作（在 checks.py），pipeline 把它们串起来（在 pipelines.py）。
这样你可以单独测试每个检查，也可以自由组合。
"""

from src.qc.pipelines import qc_multihop_query, qc_real_user_query
from src.qc.reasoning import (
    classify_query_intent,
    classify_reasoning_structure,
    qc_reasoning_depth,
)
from src.qc.llm_judge import (
    run_llm_qc,
    run_ablation_qc,
    judge_single_element_batch,
    judge_answer_grounding,
)

__all__ = [
    "qc_multihop_query",
    "qc_real_user_query",
    "qc_reasoning_depth",
    "classify_query_intent",
    "classify_reasoning_structure",
    "run_llm_qc",
    "run_ablation_qc",
    "judge_single_element_batch",
    "judge_answer_grounding",
]
