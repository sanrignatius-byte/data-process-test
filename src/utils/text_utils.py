"""文本处理工具包 —— 分词 / Jaccard / 公式符号提取等。

之前 5+ 个脚本各自实现了一遍分词，现在统一收到这里。
注意有三种分词函数，别搞混了：
  - tokenize()：≥3 字母小写化，用于拓扑分析和 hub enrichment
  - tokenize_words()：\\w+ 小写化含数字，用于 enrich_hub_candidates
  - tokenize_for_retrieval()：字母开头 2+ 字符，返回 list（保序+保重复），给 BM25 用

content_tokens() 是 QC 专用的：去停用词后的内容 token，用于 anchor 泄漏检测。
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set

# ── Tokenisation ──────────────────────────────────────────────────────────────

# Regex used by retrieval / eval scripts (matches alpha-prefixed 2+ char tokens)
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{1,}")


def tokenize(text: str) -> Set[str]:
    """粗分词：≥3 字母小写化，不含数字不含短词。拓扑分析和 hub enrichment 用。"""
    return {t for t in re.findall(r"[a-zA-Z]{3,}", (text or "").lower())}


def tokenize_words(text: str) -> Set[str]:
    r"""全量分词：``\w+`` 小写化含数字。enrich_hub_candidates 用。"""
    return set(re.findall(r"\w+", (text or "").lower()))


def tokenize_for_retrieval(text: str) -> List[str]:
    """检索用分词：字母开头 2+ 字符，返回 list（保序+保重复）。BM25 打分用。"""
    return [t.lower() for t in TOKEN_RE.findall(text or "")]


# ── Set similarity ────────────────────────────────────────────────────────────

def jaccard(a: Set[str], b: Set[str]) -> float:
    """Jaccard 相似度：|A∩B| / |A∪B|，两个空集返回 0。"""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def overlap_ratio(a: Set[str], b: Set[str]) -> float:
    """Asymmetric overlap: ``|a ∩ b| / |a|``.  Returns 0 when *a* is empty."""
    if not a:
        return 0.0
    return len(a & b) / len(a)


# ── Label / number helpers ────────────────────────────────────────────────────

# Canonical mapping used by topology analysis
LABEL_TO_MODALITY: Dict[str, str] = {
    "figure": "figure",
    "table": "table",
    "equation": "equation",
    "formula": "equation",
}

# Extended mapping used by hub enrichment (includes common LaTeX aliases)
LABEL_TYPE_MAP: Dict[str, str] = {
    "figure": "figure",
    "fig": "figure",
    "subfigure": "figure",
    "table": "table",
    "tab": "table",
    "equation": "equation",
    "eq": "equation",
    "align": "equation",
    "eqnarray": "equation",
    "formula": "equation",
}


def normalize_label_type(raw: str) -> Optional[str]:
    """Map a raw LaTeX label type to one of ``figure | table | equation``.

    Uses :data:`LABEL_TYPE_MAP` (the superset of :data:`LABEL_TO_MODALITY`).
    Returns ``None`` when *raw* is empty or unknown.
    """
    if not raw:
        return None
    return LABEL_TYPE_MAP.get(str(raw).lower().strip())


def parse_number(text: str) -> Optional[int]:
    """Extract the first integer from *text* (tries trailing digits first)."""
    if not text:
        return None
    m = re.search(r"(\d+)\s*$", text)
    if not m:
        m = re.search(r"(\d+)", text)
    if not m:
        return None
    return int(m.group(1))


# ── Statistics ────────────────────────────────────────────────────────────────

def variance(values: List[float]) -> float:
    """Population variance."""
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((x - mean) ** 2 for x in values) / len(values)


def summarize_distribution(values: List[float]) -> Dict[str, float]:
    """Return count / mean / variance / min / max / p50 / p90 for *values*."""
    if not values:
        return {"count": 0, "mean": 0.0, "variance": 0.0,
                "min": 0.0, "max": 0.0, "p50": 0.0, "p90": 0.0}
    arr = sorted(values)
    p50 = arr[int(round(0.50 * (len(arr) - 1)))]
    p90 = arr[int(round(0.90 * (len(arr) - 1)))]
    return {
        "count": len(values),
        "mean": round(sum(values) / len(values), 4),
        "variance": round(variance(values), 4),
        "min": round(arr[0], 4),
        "max": round(arr[-1], 4),
        "p50": round(p50, 4),
        "p90": round(p90, 4),
    }


# ── Content-token helpers (used by QC) ────────────────────────────────────────

LEAK_STOPWORDS: Set[str] = {
    "the", "a", "an", "of", "in", "to", "for", "on", "at", "by", "and", "or",
    "is", "are", "was", "were", "be", "been", "with", "from", "as", "that",
    "this", "it", "its", "how", "what", "which", "when", "where", "does", "do",
    "between", "across", "than", "both", "each", "all", "into", "over",
}


def content_tokens(text: str) -> Set[str]:
    """QC 专用分词：≥3 字母去停用词后的纯内容 token，用于 anchor/evidence 泄漏检测。"""
    words = set(re.findall(r"\b[a-zA-Z]{3,}\b", (text or "").lower()))
    return words - LEAK_STOPWORDS


def number_tokens(text: str) -> Set[str]:
    """Normalized numeric tokens (e.g. ``"1,234.5"`` → ``"1234.5"``)."""
    nums: Set[str] = set()
    for raw in re.findall(r"\b\d+(?:[.,]\d+)?%?\b", text or ""):
        tok = raw.replace(",", "").rstrip("%")
        if tok:
            nums.add(tok)
    return nums


# ── LaTeX math-region extraction ──────────────────────────────────────────────

_LATEX_IGNORE_CMDS = frozenset({
    "begin", "end", "left", "right", "frac", "cdot", "times",
    "sum", "prod", "int", "mid", "tag",
})

_RE_GREEK = re.compile(
    r"\\(alpha|beta|gamma|delta|epsilon|lambda|mu|sigma|tau|theta|phi|psi|omega)"
)


def extract_math_regions(text: str) -> str:
    """从 LaTeX 文本中提取行内/行间公式内容。没找到公式就返回原文。"""
    regions: List[str] = []
    regions += re.findall(r"\$(.+?)\$", text, flags=re.DOTALL)
    regions += re.findall(r"\\\((.+?)\\\)", text, flags=re.DOTALL)
    regions += re.findall(r"\\\[(.+?)\\\]", text, flags=re.DOTALL)
    return " ".join(regions) if regions else text


def extract_formula_variables(content: str) -> str:
    """从公式文本中提取变量和函数名，给 prompt 用。

    返回类似 "Variables: x, y, alpha; Functions/terms: softmax, argmax" 的字符串。
    """
    if not content:
        return "(none)"

    math_text = extract_math_regions(content)

    funcs = re.findall(r"\\([A-Za-z]{2,})", math_text)
    funcs = [f for f in funcs if f not in _LATEX_IGNORE_CMDS]

    vars_sub = re.findall(r"\b([A-Za-z]+_[A-Za-z0-9]+)\b", math_text)
    vars_single = re.findall(
        r"(?:^|[=+\-*/(,\s{])([A-Za-z])(?:$|[=+\-*/),\s}_^])", math_text,
    )
    greek = _RE_GREEK.findall(math_text)
    tokens = list(dict.fromkeys(vars_sub + vars_single + greek))

    functions = ", ".join(list(dict.fromkeys(funcs))[:8]) if funcs else ""
    variables = ", ".join(tokens[:12]) if tokens else ""
    parts = []
    if variables:
        parts.append(f"Variables: {variables}")
    if functions:
        parts.append(f"Functions/terms: {functions}")
    return "; ".join(parts) if parts else "(no clear variables found)"


def extract_formula_symbol_terms(content: str) -> Set[str]:
    """从公式中提取符号术语 —— QC 用，检查答案有没有引用公式符号。"""
    if not content:
        return set()
    math_text = extract_math_regions(content)

    terms: Set[str] = set()
    for g in _RE_GREEK.findall(math_text):
        terms.add(g.lower())
    for t in re.findall(r"\b([A-Za-z]+_[A-Za-z0-9]+)\b", math_text):
        terms.add(t.lower())
    for t in re.findall(r"\b([A-Za-z]+\([A-Za-z0-9,\s]+\))", math_text):
        terms.add(re.sub(r"\s+", "", t.lower()))
    for cmd in re.findall(r"\\([A-Za-z]{2,})", math_text):
        c = cmd.lower()
        if c not in _LATEX_IGNORE_CMDS:
            terms.add(c)
    return {t for t in terms if t and len(t) >= 2}


def extract_table_headers(content: str, max_chars: int = 150) -> str:
    """从 HTML/markdown table 中提取表头标签，跳过纯数字单元格。"""
    if not content:
        return "(none)"

    headers: List[str] = []

    if "<td" in content.lower() or "<th" in content.lower():
        cells = re.findall(
            r"<t[dh][^>]*>(.*?)</t[dh]>", content,
            flags=re.IGNORECASE | re.DOTALL,
        )
        for c in cells[:24]:
            txt = re.sub(r"<[^>]+>", " ", c)
            txt = re.sub(r"\s+", " ", txt).strip()
            if not txt:
                continue
            if re.search(r"\d", txt) and not re.search(r"[A-Za-z]", txt):
                continue
            headers.append(txt)
            if len(" ; ".join(headers)) >= max_chars:
                break
    else:
        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
        for ln in lines[:20]:
            if "|" in ln:
                cells = [c.strip() for c in ln.split("|") if c.strip()]
                for c in cells:
                    if re.search(r"\d", c) and not re.search(r"[A-Za-z]", c):
                        continue
                    headers.append(c)
            else:
                left = ln.split(":")[0].strip()
                if left and re.search(r"[A-Za-z]", left):
                    headers.append(left)
            if len(" ; ".join(headers)) >= max_chars:
                break

    deduped: List[str] = []
    seen: Set[str] = set()
    for h in headers:
        norm = re.sub(r"\s+", " ", h.lower()).strip()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        cleaned = re.sub(r"\b\d+(?:[.,]\d+)?%?\b", "", h).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        if cleaned:
            deduped.append(cleaned)

    out = " ; ".join(deduped)
    return out[:max_chars] if out else "(headers unavailable)"
