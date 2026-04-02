"""Prompt templates, personas, and query-style logic for cross-modal query generation.

Extracted from scripts/generate_multihop_l1_queries.py to enable reuse.
"""

from .templates import (  # noqa: F401
    SYSTEM_PROMPT,
    PROMPT_FIGURE_TABLE_1HOP,
    PROMPT_FIGURE_TABLE_2HOP,
    PROMPT_FIGURE_FORMULA,
    PROMPT_FORMULA_TABLE,
    PROMPT_3STEP_REASONING_CHAIN,
    PROMPT_REAL_USER_FACTUAL,
    PROMPT_REAL_USER_SUMMARY,
    PROMPT_REAL_USER_COMPARISON,
    PROMPT_REAL_USER_HOW_WORKS,
    PROMPT_REAL_USER_WHAT_IF,
    REAL_USER_TEMPLATES,
    REAL_USER_STYLE_CYCLE,
)
from .personas import (  # noqa: F401
    load_personahub_personas,
    resolve_persona_entry,
    resolve_persona,
    resolve_persona_id,
    inject_persona_prefix,
)
from .styles import (  # noqa: F401
    resolve_query_style,
    select_template,
)

__all__ = [
    # templates
    "SYSTEM_PROMPT",
    "PROMPT_FIGURE_TABLE_1HOP",
    "PROMPT_FIGURE_TABLE_2HOP",
    "PROMPT_FIGURE_FORMULA",
    "PROMPT_FORMULA_TABLE",
    "PROMPT_3STEP_REASONING_CHAIN",
    "PROMPT_REAL_USER_FACTUAL",
    "PROMPT_REAL_USER_SUMMARY",
    "PROMPT_REAL_USER_COMPARISON",
    "PROMPT_REAL_USER_HOW_WORKS",
    "PROMPT_REAL_USER_WHAT_IF",
    "REAL_USER_TEMPLATES",
    "REAL_USER_STYLE_CYCLE",
    # personas
    "load_personahub_personas",
    "resolve_persona_entry",
    "resolve_persona",
    "resolve_persona_id",
    "inject_persona_prefix",
    # styles
    "resolve_query_style",
    "select_template",
]
