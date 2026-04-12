"""QC 常量池 —— 阈值、正则、词表全在这。

从 generate_multihop_l1_queries.py 的 3900 行大单体里拆出来的，
让 qc/checks.py、qc/reasoning.py 和任何未来的 QC 消费者都能用。

关键阈值（别乱改，这些都是实验调出来的）：
  - ANCHOR_LEAK_THRESHOLD = 0.20：query 和 anchor 的 Jaccard 超 20% 就算泄漏
  - ANSWER_BALANCE_THRESHOLD = 0.20：答案不能只靠一个元素
  - MAX_QUERY_WORDS = 40：query 超 40 词就太长了 (2026-04-11: 从 30 提升，多跳 query 需要更多空间)
"""

from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple

# ── Meta-language patterns (hard fail) ────────────────────────────────────────

BAD_META_PATTERNS = [
    r"\bfigure\b",
    r"\btable\b",
    r"\bequation\b",
    r"\bformula\b",
    r"according to",
    r"as (?:shown|mentioned|stated|described|depicted|illustrated) in",
    r"the (?:text|caption|paper|section|paragraph)",
    r"(?:this|the) (?:figure|table|chart|plot|graph|diagram)",
]

# ── Yes/no starters ──────────────────────────────────────────────────────────

YES_NO_STARTERS = [
    "do ", "does ", "did ", "can ", "could ", "is ", "are ",
    "would ", "has ", "have ", "will ", "was ", "were ",
    "had ", "should ", "may ", "might ",
]

# ── QC thresholds ─────────────────────────────────────────────────────────────

ANCHOR_LEAK_THRESHOLD = 0.20
ANSWER_BALANCE_THRESHOLD = 0.20
MIN_OVERLAP_BY_TYPE: Dict[str, int] = {
    "figure": 1,
    "table": 2,
    "formula": 2,
}
SHORT_QUERY_MIN_WORDS = 8
SHORT_QUERY_MAX_WORDS = 14
LONG_QUERY_MIN_WORDS = 18
MAX_QUERY_WORDS = 40  # Raised from 30: multi-hop queries need more words
TEXT_EVIDENCE_OVERLAP_WARN_THRESHOLD = 0.4

# ── Template shortcut patterns ────────────────────────────────────────────────

QUERY_SHORTCUT_PATTERNS = [
    r"^which\s+component\b",
    r"^which\s+method\b",
    r"^which\s+approach\b",
    r"^which\s+variable\b",
    r"^which\s+pair\b",
    r"^how\s+does\s+.+\s+relate\s+to\s+.+",
    r"^how\s+do\s+.+\s+relate\s+to\s+.+",
    r"^what\s+relationship\s+exists\b",
]

TEMPLATED_QUERY_OPENINGS: Tuple[str, ...] = (
    "given that",
    "what causes",
    "i notice",
)

TEMPLATE_COLLAPSE_PATTERNS: Tuple[str, ...] = (
    r"^under\s+what\b",
    r"^why\s+is\b.+\bdifferent from\b",
)

# ── Entity amnesty terms ──────────────────────────────────────────────────────

ENTITY_AMNESTY_TERMS: Set[str] = {
    "f1", "auc", "rmse", "accuracy", "precision", "recall", "pvalue", "p-value",
    "lambda", "alpha", "beta", "baseline", "attention", "adversary", "adversarial",
    "fairness", "bias", "parity", "statistical", "significance", "classification",
    "error", "score", "scores", "metric", "metrics", "dataset", "datasets",
}

# ── Concept term sets ─────────────────────────────────────────────────────────

STRUCTURAL_CONCEPT_TERMS: Set[str] = {
    "separation", "cluster", "topology", "network", "node", "nodes",
    "graph", "curve", "curves", "heatmap", "bar", "bars", "layout",
}

METRIC_CONCEPT_TERMS: Set[str] = {
    "performance", "score", "scores", "accuracy", "error", "errors",
    "precision", "recall", "auc", "f1", "rmse", "metric", "metrics",
    "rate", "rates", "p-value", "pvalue", "statistical",
}

ARCHITECTURE_KEYWORDS: Set[str] = {
    "architecture", "framework", "pipeline", "overview", "module", "modules",
    "component", "components", "encoder", "decoder", "branch", "branches",
    "backbone", "head", "heads", "block", "blocks", "graph", "topology",
    "fusion", "attention", "stage", "stages", "stream", "pathway",
}

ARCHITECTURE_INTENT_TERMS: Set[str] = {
    "summarize", "summary", "design", "structure", "innovation", "novel",
    "component", "module", "branch", "encoder", "decoder", "pipeline",
    "objective", "constraint", "loss", "penalty", "regularization",
    "ablation", "experiment", "effect", "improvement", "performance",
}

# ── Relation connectors ──────────────────────────────────────────────────────

RELATION_CONNECTORS: Set[str] = {
    "because", "due to", "therefore", "thus", "hence",
    "leads to", "results in", "explains", "matches", "corresponds to",
    "driven by", "caused by", "consistent with", "deviates from",
    "constrained by", "compared with", "whereas", "despite", "under",
}

# ── Cross-modal operator vocabulary ──────────────────────────────────────────

CROSS_MODAL_OPERATORS: Set[str] = {
    "verify", "verified", "verifies",
    "derive", "derived", "derives",
    "map", "maps", "mapped",
    "align", "aligns", "aligned",
    "contradict", "contradicts", "contradicted",
    "explain", "explains", "explained",
    "instantiate", "instantiates", "instantiated",
    "calibrate", "calibrates", "calibrated",
    "attribute", "attributes", "attributed",
    "quantify", "quantifies", "quantified",
    "predict", "predicts", "predicted",
    "justify", "justifies", "justified",
    "enforce", "enforces", "enforced",
    "converge", "converges", "converged",
    "reveal", "reveals", "revealed",
    "validate", "validates", "validated",
    "expose", "exposes", "exposed",
    "constrain", "constrains", "constrained",
    "decompose", "decomposes", "decomposed",
    "propagate", "propagates", "propagated",
    "regulate", "regulates", "regulated",
    "bound", "bounds", "bounded",
    "separate", "separates", "separated",
    "correspond", "corresponds", "corresponded",
    "account", "accounts", "accounted",
    "limit", "limits", "limited",
    "associate", "associates", "associated",
    "depend", "depends", "depended",
    "link", "links", "linked",
    "relate", "relates", "related",
    "appear", "appears", "appeared",
    "decline", "declines", "declined", "declining",
    "inconsistent",
    "show", "shows", "showed",
    "cause", "causes", "caused",
    "drop", "drops", "dropped",
    "exceed", "exceeds", "exceeded",
    "mismatch", "mismatches", "mismatched",
    "require", "requires", "required",
    "affect", "affects", "affected",
    "differ", "differs", "differed",
    "increase", "increases", "increased",
    "decrease", "decreases", "decreased",
    "change", "changes", "changed",
    "improve", "improves", "improved",
    "reduce", "reduces", "reduced",
    "lead", "leads",
    "result", "results", "resulted",
    "indicate", "indicates", "indicated",
    "suggest", "suggests", "suggested",
    "support", "supports", "supported",
    "reflect", "reflects", "reflected",
    "produce", "produces", "produced",
    "impact", "impacts", "impacted",
    "correlate", "correlates", "correlated",
    "matter", "matters", "mattered",
    "achieve", "achieves", "achieved",
    "occur", "occurs", "occurred",
    "remain", "remains", "remained",
    "shift", "shifts", "shifted",
    "fail", "fails", "failed",
    "vary", "varies", "varied",
    "scale", "scales", "scaled",
    "difference", "differences",
    "better", "worse",
    "emerge", "emerges", "emerged",
    "outperform", "outperforms", "outperformed",
}

CROSS_MODAL_OPERATOR_PATTERNS: Tuple[str, ...] = (
    r"\bdifferent(?:\s+from|\s+than)?\b",
    r"\bcompar(?:e|es|ed|ing|ison)\b",
    r"\b(higher|lower|greater|less)\b.{0,20}\bthan\b",
    r"\bmaintain(?:s|ed|ing)?\b",
)

# ── Enrichment noise patterns ─────────────────────────────────────────────────

_ENRICHMENT_NOISE_PATTERNS = [
    r"[\u2460-\u2473\u25a0-\u25ff\u2600-\u26ff\u2700-\u27bf]",
    r"\b(glyph|icon|marker|symbol|bullet|arrow|checkmark|watermark)\b",
    r"[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮]",
    r"\b(ocr error|illegible|unreadable|corrupted text|extraction failed)\b",
    r"^[\W\d\s]{0,20}$",
]

ENRICHMENT_NOISE_RE = re.compile(
    "|".join(_ENRICHMENT_NOISE_PATTERNS), re.IGNORECASE | re.UNICODE
)

# ── Reasoning structure patterns ──────────────────────────────────────────────

SERIAL_REASONING_MARKERS = re.compile(
    r"\b(because|therefore|thus|hence|consequently|as a result|this (?:means|implies|suggests|explains|shows)|"
    r"which (?:means|implies|suggests|explains|leads|causes)|"
    r"due to|owing to|given that|since .{5,30}then|"
    r"if .{5,40}then|from .{5,40}(?:follows|we (?:can|see|know))|"
    r"(?:step|first|second|third|next|finally|building on|combining))\b",
    re.IGNORECASE,
)

PARALLEL_EVIDENCE_MARKERS = re.compile(
    r"\b((?:both|together|combined|alongside|in addition|additionally|also|moreover|furthermore|"
    r"similarly|likewise|and .{0,10}respectively|while .{5,30}also))\b",
    re.IGNORECASE,
)

EVIDENCE_TYPE_PATTERNS: Dict[str, re.Pattern] = {
    "observation": re.compile(r"\b(show|display|illustrate|reveal|depict|present|indicate|demonstrate)\b", re.I),
    "attribution": re.compile(r"\b(cause|due to|because|result from|attribute|ablation|removing|without)\b", re.I),
    "explanation": re.compile(r"\b(prove|equation|formula|inequality|theorem|bound|derive|constraint)\b", re.I),
    "verification": re.compile(r"\b(confirm|validate|verify|consistent with|align|match|support)\b", re.I),
    "prediction": re.compile(r"\b(predict|expect|would|should|if .{5,30} then|hypothesize|imply)\b", re.I),
}

# ── Query intent classification patterns ──────────────────────────────────────

SUBJECTIVE_QUERY_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(summarize|summarise|summary|overview|describe|explain|discuss)\b", re.I),
    re.compile(r"\bwhat (is|are|does|do|did)\b.{0,30}\b(about|mean|trying|aim|purpose|motivation|contribution|novel|limit)\b", re.I),
    re.compile(r"\b(briefly|concisely|in (one|a few) (sentence|word|paragraph))\b", re.I),
    re.compile(r"\b(tell me|give me|walk me|help me)\b", re.I),
    re.compile(r"\b(what (distinguishes|makes|sets|defines))\b", re.I),
    re.compile(r"\b(pros and cons|trade.?off|advantage|disadvantage|strength|weakness)\b", re.I),
]

OBJECTIVE_QUERY_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(how (many|much|often|long|far|large|small|fast|accurate))\b", re.I),
    re.compile(r"\b(what (percentage|fraction|ratio|number|value|score|result|accuracy|f1|bleu|rouge))\b", re.I),
    re.compile(r"\b(which (model|method|approach|baseline|dataset|benchmark))\b", re.I),
    re.compile(r"\b(is there|does .+? show|does .+? outperform|is .+? better|is .+? worse)\b", re.I),
    re.compile(r"\b(what (specific|exact|precise))\b", re.I),
]

# ── Anchor specificity patterns ───────────────────────────────────────────────

GENERIC_ANCHOR_PATTERNS = re.compile(
    r'^(?:the (?:table|figure|formula|chart|graph|plot|image|equation)|'
    r'(?:table|figure|formula) (?:content|data|information)|'
    r'overall (?:structure|layout|content))$',
    re.IGNORECASE,
)

SPECIFIC_ANCHOR_MARKERS = re.compile(
    r'(?:row|column|col|cell|axis|line|bar|marker|label|legend|'
    r'panel|subplot|left|right|top|bottom|color|dashed|solid|'
    r'\d+(?:st|nd|rd|th)|x-axis|y-axis|variable|term|coefficient|'
    r'numerator|denominator|subscript|superscript)',
    re.IGNORECASE,
)
