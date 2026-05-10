#!/usr/bin/env python3
"""build_math_normalized_corpus.py
===================================
F-formula Phase 2 (Option A): LaTeX symbol normalisation before encoding.

Hypothesis (C11): raw-LaTeX formula passages hit a tokenizer ceiling —
  ``\\operatorname``, ``\\mathbb``, ``\\leq`` are tokenised as nonsense
  subwords. A text encoder never saw them during pretraining.

This script normalises LaTeX math markup into readable tokens before
the embedding step, WITHOUT calling any LLM:

  \\operatorname{opt}  →  opt
  \\mathbb{E}          →  E
  \\leq                →  <=
  \\mathcal{L}         →  L
  \\mathbf{x}          →  x (bold vector)
  \\text{subject}      →  subject
  \\frac{a}{b}         →  a / b
  \\sum_{i=1}^{n}      →  sum over i=1 to n
  \\int_{0}^{∞}        →  integral from 0 to ∞
  \\sqrt{x}            →  sqrt(x)
  \\alpha, \\beta ...  →  alpha, beta ...

Only formula-type passages are normalised; figure/table/text passages
pass through unchanged.

Input:
  data/03_queries/M4query_v1/corpus.jsonl  (or augmented corpus)

Output:
  data/05_eval/dense_retrieval/rebuilt_20260417/augmented/corpus_v1_math_norm.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── LaTeX → readable symbol map ──────────────────────────────────────────────

# Single-command replacements (longest first to avoid partial matches)
LATEX_CMD_MAP: list[tuple[str, str]] = [
    # ── font / style commands (strip, keep content) ──
    (r"\\operatorname\s*\{\s*([^}]*?)\s*\}", r"\1"),
    (r"\\mathrm\s*\{\s*([^}]*?)\s*\}", r"\1"),
    (r"\\mathit\s*\{\s*([^}]*?)\s*\}", r"\1"),
    (r"\\mathbf\s*\{\s*([^}]*?)\s*\}", r"\1 (bold vector)"),
    (r"\\mathsf\s*\{\s*([^}]*?)\s*\}", r"\1"),
    (r"\\mathtt\s*\{\s*([^}]*?)\s*\}", r"\1"),
    (r"\\mathcal\s*\{\s*([^}]*?)\s*\}", r"\1"),
    (r"\\mathbb\s*\{\s*([^}]*?)\s*\}", r"\1"),
    (r"\\mathfrak\s*\{\s*([^}]*?)\s*\}", r"\1"),
    (r"\\text\s*\{\s*([^}]*?)\s*\}", r"\1"),
    (r"\\textbf\s*\{\s*([^}]*?)\s*\}", r"\1"),
    (r"\\textit\s*\{\s*([^}]*?)\s*\}", r"\1"),
    (r"\\emph\s*\{\s*([^}]*?)\s*\}", r"\1"),
    (r"\\underline\s*\{\s*([^}]*?)\s*\}", r"\1"),
    (r"\\overline\s*\{\s*([^}]*?)\s*\}", r"conjugate(\1)"),
    (r"\\hat\s*\{\s*([^}]*?)\s*\}", r"\1-hat"),
    (r"\\tilde\s*\{\s*([^}]*?)\s*\}", r"tilde-\1"),
    (r"\\bar\s*\{\s*([^}]*?)\s*\}", r"\1-bar"),
    (r"\\vec\s*\{\s*([^}]*?)\s*\}", r"vector \1"),
    (r"\\dot\s*\{\s*([^}]*?)\s*\}", r"\1-dot"),
    (r"\\ddot\s*\{\s*([^}]*?)\s*\}", r"\1-ddot"),

    # ── binary operators / relations ──
    (r"\\leq", " <= "),
    (r"\\geq", " >= "),
    (r"\\ll", " << "),
    (r"\\gg", " >> "),
    (r"\\neq", " != "),
    (r"\\equiv", " == "),
    (r"\\approx", " approximately "),
    (r"\\sim", " ~ "),
    (r"\\propto", " proportional to "),
    (r"\\times", " x "),
    (r"\\cdot", " * "),
    (r"\\circ", " composed with "),
    (r"\\oplus", " xor "),
    (r"\\otimes", " tensor product "),
    (r"\\odot", " elementwise product "),
    (r"\\pm", " +/- "),
    (r"\\mp", " -/+ "),
    (r"\\subset", " subset of "),
    (r"\\supset", " superset of "),
    (r"\\subseteq", " subset or equal "),
    (r"\\supseteq", " superset or equal "),
    (r"\\in", " in "),
    (r"\\notin", " not in "),
    (r"\\ni", " contains "),
    (r"\\forall", " for all "),
    (r"\\exists", " there exists "),
    (r"\\emptyset", " empty set "),
    (r"\\varnothing", " empty set "),

    # ── arrows ──
    (r"\\to", " -> "),
    (r"\\rightarrow", " -> "),
    (r"\\leftarrow", " <- "),
    (r"\\Rightarrow", " implies "),
    (r"\\Leftrightarrow", " iff "),
    (r"\\mapsto", " maps to "),
    (r"\\longrightarrow", " long right arrow "),
    (r"\\Longrightarrow", " implies "),

    # ── brackets / delimiters ──
    (r"\\left\s*[\[\(\{|]", " "),
    (r"\\right\s*[\]\)\}|]", " "),
    (r"\\langle", " <"),
    (r"\\rangle", "> "),

    # ── Greek letters ──
    (r"\\alpha", " alpha "),
    (r"\\beta", " beta "),
    (r"\\gamma", " gamma "),
    (r"\\delta", " delta "),
    (r"\\epsilon", " epsilon "),
    (r"\\varepsilon", " epsilon "),
    (r"\\zeta", " zeta "),
    (r"\\eta", " eta "),
    (r"\\theta", " theta "),
    (r"\\vartheta", " theta "),
    (r"\\iota", " iota "),
    (r"\\kappa", " kappa "),
    (r"\\lambda", " lambda "),
    (r"\\mu", " mu "),
    (r"\\nu", " nu "),
    (r"\\xi", " xi "),
    (r"\\pi", " pi "),
    (r"\\varpi", " pi "),
    (r"\\rho", " rho "),
    (r"\\varrho", " rho "),
    (r"\\sigma", " sigma "),
    (r"\\varsigma", " sigma "),
    (r"\\tau", " tau "),
    (r"\\upsilon", " upsilon "),
    (r"\\phi", " phi "),
    (r"\\varphi", " phi "),
    (r"\\chi", " chi "),
    (r"\\psi", " psi "),
    (r"\\omega", " omega "),
    # uppercase Greek
    (r"\\Gamma", " Gamma "),
    (r"\\Delta", " Delta "),
    (r"\\Theta", " Theta "),
    (r"\\Lambda", " Lambda "),
    (r"\\Xi", " Xi "),
    (r"\\Pi", " Pi "),
    (r"\\Sigma", " Sigma "),
    (r"\\Upsilon", " Upsilon "),
    (r"\\Phi", " Phi "),
    (r"\\Psi", " Psi "),
    (r"\\Omega", " Omega "),

    # ── named operators ──
    (r"\\min", " min "),
    (r"\\max", " max "),
    (r"\\sup", " sup "),
    (r"\\inf", " inf "),
    (r"\\lim", " limit "),
    (r"\\log", " log "),
    (r"\\ln", " ln "),
    (r"\\exp", " exp "),
    (r"\\det", " det "),
    (r"\\dim", " dim "),
    (r"\\arg", " arg "),
    (r"\\Pr", " Pr "),
    (r"\\gcd", " gcd "),
    (r"\\hom", " hom "),
    (r"\\ker", " ker "),
    (r"\\deg", " deg "),

    # ── big operators ──
    (r"\\infty", " infinity "),
    (r"\\partial", " partial derivative "),
    (r"\\nabla", " gradient "),
    (r"\\ell", " l "),

    # ── miscellaneous ──
    (r"\\ldots", " ... "),
    (r"\\cdots", " ... "),
    (r"\\vdots", " vertical dots "),
    (r"\\ddots", " diagonal dots "),
    (r"\\colon", " : "),
    (r"\\mid", " | "),
    (r"\\parallel", " parallel "),
    (r"\\perp", " perpendicular "),
    (r"\\angle", " angle "),
    (r"\\triangle", " triangle "),
    (r"\\square", " square "),
    (r"\\diamond", " diamond "),
    (r"\\clubsuit", " club "),
    (r"\\heartsuit", " heart "),
    (r"\\spadesuit", " spade "),
    (r"\\dagger", " dagger "),
    (r"\\ddagger", " double dagger "),
    (r"\\S", " section symbol "),
    (r"\\P", " paragraph symbol "),

    # ── spacing / breaks ──
    (r"\\qquad", "  "),
    (r"\\quad", " "),
    (r"\\,", " "),
    (r"\\;", " "),
    (r"\\:", " "),
    (r"\\!", ""),
    (r"\\\\", " ; "),
]

# Compile the map: apply longest-first
_LATEX_REPLACE: list[tuple[re.Pattern, str]] = [
    (re.compile(p), r) for p, r in LATEX_CMD_MAP
]

# ── Structural LaTeX → readable form ─────────────────────────────────────────

def _frac_replacer(m: re.Match) -> str:
    num = m.group(1).strip()
    den = m.group(2).strip()
    return f" ({num}) / ({den}) "

FRAC_RE = re.compile(r"\\frac\s*\{\s*([^{}]*(?:\{[^{}]*\}[^{}]*)*)\s*\}\s*\{\s*([^{}]*(?:\{[^{}]*\}[^{}]*)*)\s*\}")

def _sqrt_replacer(m: re.Match) -> str:
    inner = m.group(1) or m.group(2) or ""
    return f" sqrt({inner}) "

SQRT_RE = re.compile(r"\\sqrt\s*\{\s*([^{}]*(?:\{[^{}]*\}[^{}]*)*)\s*\}")
SQRT_OPT_RE = re.compile(r"\\sqrt\s*\[([^\]]*)\]\s*\{\s*([^{}]*(?:\{[^{}]*\}[^{}]*)*)\s*\}")

def _sum_replacer(m: re.Match) -> str:
    sub = m.group(1).strip()
    sup = m.group(2).strip() if m.group(2) else ""
    body = m.group(3).strip() if m.group(3) else ""
    if sup:
        return f" sum over {sub} from {sup} of ({body}) "
    return f" sum over {sub} of ({body}) "

SUM_RE = re.compile(r"\\sum\s*_\s*\{\s*([^{}]*)\s*\}(?:\s*\^\s*\{\s*([^{}]*)\s*\})?\s*(.*?)(?=\\sum|\\int|\\prod|$|\}\)|\\end)")

def _int_replacer(m: re.Match) -> str:
    sub = m.group(1).strip() if m.group(1) else ""
    sup = m.group(2).strip() if m.group(2) else ""
    body = m.group(3).strip() if m.group(3) else ""
    parts = [" integral "]
    if sub:
        parts.append(f" from {sub} ")
    if sup:
        parts.append(f" to {sup} ")
    if body:
        parts.append(f" of ({body}) ")
    return " ".join(parts)

INT_RE = re.compile(r"\\int\s*(?:_\s*\{\s*([^{}]*)\s*\})?\s*(?:\^\s*\{\s*([^{}]*)\s*\})?\s*(.*?)(?=\\sum|\\int|\\prod|$|\}\)|\\end)")

def _prod_replacer(m: re.Match) -> str:
    sub = m.group(1).strip()
    sup = m.group(2).strip() if m.group(2) else ""
    body = m.group(3).strip() if m.group(3) else ""
    if sup:
        return f" product over {sub} from {sup} of ({body}) "
    return f" product over {sub} of ({body}) "

PROD_RE = re.compile(r"\\prod\s*_\s*\{\s*([^{}]*)\s*\}(?:\s*\^\s*\{\s*([^{}]*)\s*\})?\s*(.*?)(?=\\sum|\\int|\\prod|$|\}\)|\\end)")

def _lim_replacer(m: re.Match) -> str:
    sub = m.group(1).strip()
    return f" limit as {sub} "

LIM_RE = re.compile(r"\\lim\s*_\s*\{\s*([^{}]*)\s*\}")

# ── Normalisation entry point ─────────────────────────────────────────────────

def normalise_latex(text: str) -> str:
    """Convert a LaTeX formula string into readable text for embedding."""
    t = text

    # Remove [FORMULA] prefix — we know it's a formula
    t = re.sub(r"\[FORMULA\]\s*", "", t)

    # Remove $$ delimiters
    t = t.replace("$$", " ")

    # Remove \begin{array}...\end{array} blocks (keep content)
    t = re.sub(r"\\begin\{array\}\{[^}]*\}", " ", t)
    t = re.sub(r"\\end\{array\}", " ", t)
    t = re.sub(r"\\begin\{aligned\}\s*", " ", t)
    t = re.sub(r"\\end\{aligned\}", " ", t)
    t = re.sub(r"\\begin\{cases\}\s*", " ", t)
    t = re.sub(r"\\end\{cases\}", " ", t)
    t = re.sub(r"\\begin\{matrix\}\{[^}]*\}", " ", t)
    t = re.sub(r"\\end\{matrix\}", " ", t)
    t = re.sub(r"\\begin\{pmatrix\}\{[^}]*\}", " ", t)
    t = re.sub(r"\\end\{pmatrix\}", " ", t)
    t = re.sub(r"\\begin\{bmatrix\}\{[^}]*\}", " ", t)
    t = re.sub(r"\\end\{bmatrix\}", " ", t)

    # Structural transforms: \frac, \sqrt, \sum, \int, \prod, \lim
    t = FRAC_RE.sub(_frac_replacer, t)
    t = SQRT_OPT_RE.sub(_sqrt_replacer, t)
    t = SQRT_RE.sub(_sqrt_replacer, t)
    t = SUM_RE.sub(_sum_replacer, t)
    t = INT_RE.sub(_int_replacer, t)
    t = PROD_RE.sub(_prod_replacer, t)
    t = LIM_RE.sub(_lim_replacer, t)

    # Apply command map (longest-first regex replacements)
    for pat, repl in _LATEX_REPLACE:
        t = pat.sub(repl, t)

    # Remove remaining LaTeX commands we don't recognise
    # (strip \command and \command{args} but keep \command* variants too)
    t = re.sub(r"\\[a-zA-Z]+\*?\s*\{[^}]*\}", " ", t)
    t = re.sub(r"\\[a-zA-Z]+\*?", " ", t)

    # Remove \tag{...} remnants
    t = re.sub(r"tag\s*\{[^}]*\}", " ", t)
    t = re.sub(r"\\tag\s*\{[^}]*\}", " ", t)

    # Clean up braces, dollars, backslashes
    t = t.replace("{", " ").replace("}", " ").replace("\\", " ")

    # Clean up: collapse whitespace, remove excessive punctuation
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\s*_\s*", " ", t)  # stray subscripts
    t = re.sub(r"\s*\^\s*", " ", t)  # stray superscripts

    # Put the [FORMULA] prefix back for corpus consistency
    t = f"[FORMULA] {t}"

    return t


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Normalise LaTeX formulas in corpus for math-aware embedding")
    ap.add_argument("--corpus", default="data/03_queries/M4query_v1/corpus.jsonl")
    ap.add_argument("--output",
                    default="data/05_eval/dense_retrieval/rebuilt_20260417/augmented/corpus_v1_math_norm.jsonl")
    ap.add_argument("--dry-run", type=int, default=3,
                    help="Print first N normalised formulas and exit (0 = run full)")
    args = ap.parse_args()

    corpus_path = PROJECT_ROOT / args.corpus
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    normalised = 0
    total = 0
    with open(corpus_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            rec = json.loads(line)
            total += 1
            if rec.get("type") == "formula":
                orig = rec["text"]
                norm = normalise_latex(orig)
                if args.dry_run > 0 and normalised < args.dry_run:
                    print(f"[{rec['passage_id']}]")
                    print(f"  ORIG: {orig[:150]!r}")
                    print(f"  NORM: {norm[:150]!r}")
                    print()
                rec = dict(rec)  # shallow copy
                rec["text"] = norm
                normalised += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if args.dry_run > 0:
        print(f"[dry-run] Normalised {normalised} formulas. "
              f"Run with --dry-run 0 to build full corpus ({total} total passages).")
        return

    print(f"Wrote {output_path}")
    print(f"  Total passages: {total}")
    print(f"  Normalised formulas: {normalised}")


if __name__ == "__main__":
    main()
