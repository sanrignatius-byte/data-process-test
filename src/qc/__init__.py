"""Quality control system for query validation.

Public API:
  - ``qc_multihop_query``  — strict academic-style QC
  - ``qc_real_user_query``  — relaxed real-user style QC
  - ``qc_reasoning_depth``  — M4 Schema 1 reasoning depth analysis
  - Individual check functions from ``checks`` sub-module
"""

from src.qc.pipelines import qc_multihop_query, qc_real_user_query
from src.qc.reasoning import (
    classify_query_intent,
    classify_reasoning_structure,
    qc_reasoning_depth,
)

__all__ = [
    "qc_multihop_query",
    "qc_real_user_query",
    "qc_reasoning_depth",
    "classify_query_intent",
    "classify_reasoning_structure",
]
