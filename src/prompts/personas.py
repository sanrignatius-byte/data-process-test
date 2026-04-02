"""PersonaHub integration: diverse reader personas for query generation.

Based on PersonaHub methodology (Ge et al., 2024):
  "Scaling Synthetic Data Creation with 1,000,000,000 Personas"
  arXiv:2406.20094 | Dataset: proj-persona/PersonaHub (HuggingFace)
"""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

# src/prompts/ → src/ → project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_PERSONAHUB_PERSONAS: Optional[List[Dict[str, str]]] = None  # lazy-loaded cache


# 懒加载 PersonaHub 人设库（50 类学术读者），缺文件时退回 5 条内置人设
def load_personahub_personas(path: Optional[str] = None) -> List[Dict[str, str]]:
    """Load PersonaHub-format personas from JSON file.

    Returns a list of dicts with at least 'id' and 'persona' keys.
    Falls back to a minimal built-in set if the file is unavailable.
    """
    global _PERSONAHUB_PERSONAS
    if _PERSONAHUB_PERSONAS is not None:
        return _PERSONAHUB_PERSONAS

    if path is None:
        path = str(PROJECT_ROOT / "data" / "personahub_academic_personas.json")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        personas = data.get("personas", [])
        if personas:
            _PERSONAHUB_PERSONAS = personas
            return _PERSONAHUB_PERSONAS
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"  WARNING: Failed to load PersonaHub personas from {path}: {e}")
        print("  Falling back to built-in minimal persona set.")

    # Minimal fallback (should not be needed in normal operation)
    _PERSONAHUB_PERSONAS = [
        {"id": "phd_ml_fairness", "persona": "A third-year PhD student researching algorithmic fairness who is preparing for their qualifying exam and needs to deeply understand how different fairness metrics interact with model performance across demographic groups."},
        {"id": "ml_engineer_production", "persona": "A machine learning engineer at a mid-size tech company evaluating whether to adopt a new method in production, primarily concerned with compute cost, data requirements, latency constraints, and whether reported gains hold on real-world benchmarks."},
        {"id": "reviewer_harsh", "persona": "An experienced conference reviewer known for thorough and critical reviews, who systematically checks whether baselines are fair, experiments are reproducible, and limitations are honestly disclosed."},
        {"id": "busy_professor", "persona": "A tenured professor with heavy administrative duties who only has 15 minutes to skim a paper before a committee meeting and needs to quickly grasp the main contribution and its significance."},
        {"id": "undergrad_first_paper", "persona": "An undergraduate computer science student reading their first machine learning research paper, struggling with mathematical notation and needing concrete examples to understand abstract concepts."},
    ]
    return _PERSONAHUB_PERSONAS


# 按 pair_id 的 md5 稳定哈希分配人设，保证跨机器可复现
def resolve_persona_entry(pair_id: str, persona_file: Optional[str] = None) -> Dict[str, str]:
    """Shared lookup: deterministically pick a persona entry by pair_id hash."""
    personas = load_personahub_personas(persona_file)
    if not pair_id or not personas:
        return personas[0] if personas else {"id": "unknown", "persona": ""}
    stable = int(hashlib.md5(pair_id.encode("utf-8")).hexdigest()[:8], 16)
    return personas[stable % len(personas)]


def resolve_persona(pair_id: str, persona_file: Optional[str] = None) -> str:
    """Deterministically assign a PersonaHub persona to a pair via stable hash.
    Returns the persona description text (not the id).
    """
    return resolve_persona_entry(pair_id, persona_file)["persona"]


# 同上，返回短标签用于日志 / 切片分析
def resolve_persona_id(pair_id: str, persona_file: Optional[str] = None) -> str:
    """Return the persona id (short label) for logging/slicing."""
    return resolve_persona_entry(pair_id, persona_file)["id"]


# 用人设描述替换 prompt 首句 "You are a …"，避免双重角色定义
def inject_persona_prefix(prompt: str, persona: str) -> str:
    """Prepend a PersonaHub persona description to an existing prompt template.

    The persona replaces the first "You are a ..." sentence in the prompt
    so we don't double-define the role.  Falls back to simple prepend.
    """
    if not persona:
        return prompt
    # Replace first "You are a ..." sentence with the persona
    first_sentence_pat = re.compile(r"^(You are [^.]+\.)", re.MULTILINE)
    if first_sentence_pat.search(prompt):
        return first_sentence_pat.sub(persona, prompt, count=1)
    return persona + "\n\n" + prompt
