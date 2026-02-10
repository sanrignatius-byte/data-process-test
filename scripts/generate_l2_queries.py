#!/usr/bin/env python3
"""Generate L2 (cross-document) queries from candidate document pairs.

Sends both figures as images to Claude Vision API so the model can
generate queries that reference actual visual elements from both docs.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

SYSTEM_PROMPT = (
    "You are an expert research analyst creating cross-document reasoning questions "
    "for training multimodal retrieval systems. "
    "Output valid JSON only, no other text, no markdown fences."
)

BAD_META_PATTERNS = [
    r"according to (?:the )?(?:document|paper|figure|text|study)",
    r"in document [ab12]",
    r"from doc(?:ument)? [ab12]",
    r"(?:the|this) paper (?:states?|shows?|mentions?|describes?)",
    r"(?:the|this) figure (?:shows?|depicts?|illustrates?)",
    r"as (?:shown|mentioned|stated|described) in",
    r"document [ab12] (?:shows?|states?|reports?)",
]

# Resolve image paths relative to the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


SPECULATIVE_PHRASES = [
    "could potentially", "theoretically", "might enable", "could enable",
    "could be used", "may suggest", "possibly", "presumably",
    "it is conceivable", "one could argue",
]

YES_NO_STARTERS = ["do ", "does ", "can ", "could ", "is ", "are ", "would ", "has ", "have "]

TEMPLATE_VERBS = ["align with", "relate to", "reflect the", "illustrate the"]


ANCHOR_LEAK_THRESHOLD = 0.15  # Jaccard overlap between query and anchor tokens
EVIDENCE_CLOSURE_THRESHOLD = 0.35  # Coverage of answer claims by evidence_refs
# Tokens to ignore when computing leakage (common function words + numbers)
LEAK_STOPWORDS = {
    "the", "a", "an", "of", "in", "to", "for", "on", "at", "by", "and", "or",
    "is", "are", "was", "were", "be", "been", "with", "from", "as", "that",
    "this", "it", "its", "how", "what", "which", "when", "where", "does", "do",
    "between", "across", "than", "both", "each", "all", "into", "over",
}
METRIC_TERMS = {
    "accuracy", "f1", "fpr", "fnr", "auc", "mrr", "ndcg", "recall", "precision",
    "rmse", "mae", "parity", "fairness", "utility", "di", "disparate", "impact",
    "false positive", "false negative", "balanced accuracy",
}


def _content_tokens(text: str) -> Set[str]:
    """Extract content tokens (lowercase, 3+ chars, no stopwords/numbers)."""
    words = set(re.findall(r"\b[a-zA-Z]{3,}\b", text.lower()))
    return words - LEAK_STOPWORDS


def anchor_leak_jaccard(query: str, evidence_refs: List[Dict[str, Any]]) -> float:
    """Compute max Jaccard overlap between query tokens and any anchor tokens."""
    q_tokens = _content_tokens(query)
    if not q_tokens:
        return 0.0
    max_jacc = 0.0
    for ref in evidence_refs:
        a_tokens = _content_tokens(ref.get("anchor", ""))
        if not a_tokens:
            continue
        intersection = q_tokens & a_tokens
        union = q_tokens | a_tokens
        jacc = len(intersection) / len(union) if union else 0.0
        max_jacc = max(max_jacc, jacc)
    return max_jacc


def evidence_closure_score(answer: str, evidence_refs: List[Dict[str, Any]]) -> float:
    """Estimate how much answer content can be traced back to evidence refs."""
    evidence_chunks: List[str] = []
    for ref in evidence_refs:
        anchor = ref.get("anchor", "")
        text_evidence = ref.get("text_evidence", "")
        if isinstance(anchor, str) and anchor.strip():
            evidence_chunks.append(anchor.lower())
        if isinstance(text_evidence, str) and text_evidence.strip():
            evidence_chunks.append(text_evidence.lower())
    evidence_text = " ".join(evidence_chunks)
    if not evidence_text.strip():
        return 0.0

    ans = answer.lower()
    answer_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", ans))
    answer_metrics = {m for m in METRIC_TERMS if m in ans}

    total_claims = len(answer_numbers) + len(answer_metrics)
    supported_claims = 0
    for num in answer_numbers:
        if num in evidence_text:
            supported_claims += 1
    for metric in answer_metrics:
        if metric in evidence_text:
            supported_claims += 1

    if total_claims > 0:
        return supported_claims / total_claims

    # If no explicit metric/number claims, fall back to lexical support ratio.
    ans_tokens = _content_tokens(answer)
    ev_tokens = _content_tokens(evidence_text)
    if not ans_tokens:
        return 0.0
    return len(ans_tokens & ev_tokens) / len(ans_tokens)


def qc_l2_query(
    obj: Dict[str, Any],
    anchor_leak_threshold: float = ANCHOR_LEAK_THRESHOLD,
    evidence_closure_threshold: float = EVIDENCE_CLOSURE_THRESHOLD,
) -> Tuple[List[str], Dict[str, float]]:
    """Run QC checks. Returns (issues, metrics)."""
    issues: List[str] = []
    metrics: Dict[str, float] = {}
    q = obj.get("query", "").lower()
    a = obj.get("answer", "").lower()

    # Meta-language
    if any(re.search(p, q, re.IGNORECASE) for p in BAD_META_PATTERNS):
        issues.append("meta_language_query")
    if any(re.search(p, a, re.IGNORECASE) for p in BAD_META_PATTERNS):
        issues.append("meta_language_answer")

    # Dual-doc evidence
    refs = obj.get("evidence_refs", [])
    docs = {x.get("doc_id") for x in refs if isinstance(x, dict)}
    if len(docs) < 2:
        issues.append("missing_dual_doc_evidence")

    # Basic completeness
    if len(obj.get("turns", [])) < 1:
        issues.append("empty_turns")
    if not obj.get("query"):
        issues.append("empty_query")
    if len(obj.get("answer", "")) < 20:
        issues.append("short_answer")

    # Anchor leakage: query must NOT copy visual tokens from evidence anchors
    leak = anchor_leak_jaccard(obj.get("query", ""), refs)
    metrics["anchor_leak_jaccard"] = round(leak, 4)
    if leak > anchor_leak_threshold:
        issues.append("anchor_leakage")

    closure = evidence_closure_score(obj.get("answer", ""), refs)
    metrics["evidence_closure"] = round(closure, 4)
    if closure < evidence_closure_threshold:
        issues.append("low_evidence_closure")

    # No yes/no questions
    if any(q.startswith(s) for s in YES_NO_STARTERS):
        issues.append("yes_no_question")

    # No speculative answers
    if any(p in a for p in SPECULATIVE_PHRASES):
        issues.append("speculative_answer")

    # Template verb + vagueness
    if any(v in q for v in TEMPLATE_VERBS):
        issues.append("template_verb")

    return issues, metrics


def build_prompt(pair: Dict[str, Any]) -> str:
    shared = ", ".join(pair.get("shared_entities", [])[:10])
    doc_a = pair["doc_a"]
    doc_b = pair["doc_b"]

    return f"""Generate ONE cross-document reasoning query that REQUIRES both figures to answer.

## Document A: {doc_a}
Figure: {pair.get('doc_a_figure_id', 'unknown')} ({pair.get('doc_a_figure_type', 'unknown')})
Caption: {pair.get('doc_a_caption', '(none)')[:400]}
Context from L1: {pair.get('doc_a_query', '')}
Finding: {pair.get('doc_a_answer', '')}

## Document B: {doc_b}
Figure: {pair.get('doc_b_figure_id', 'unknown')} ({pair.get('doc_b_figure_type', 'unknown')})
Caption: {pair.get('doc_b_caption', '(none)')[:400]}
Context from L1: {pair.get('doc_b_query', '')}
Finding: {pair.get('doc_b_answer', '')}

## Shared concepts: {shared}

## YOUR TASK: Create a reasoning question, NOT a comparison question.

Think about what REASONING OPERATION connects these two documents:
- Can Doc B's framework/method EXPLAIN an observation in Doc A's figure?
- Can Doc A's empirical result PREDICT what Doc B's approach would show?
- Does Doc A's finding CONTRADICT or SUPPORT Doc B's theoretical claim?
- What would happen if you APPLIED Doc B's technique to Doc A's data?

## STRICT RULES:

1) **INFORMATION GAP**: The query must describe ONE document's context (method, dataset, phenomenon)
   and ask a question whose answer requires looking at the OTHER document's figure.
   The retrieval model should need to FIND the second document, not just confirm something obvious.

2) **NO ANCHOR COPYING**: Do NOT copy specific visual descriptions into the query.
   The query uses CONCEPTUAL language (method names, metric names, phenomena).
   Visual details (colors, coordinates, bar heights, curve shapes) belong ONLY in evidence_refs.anchor.
   WRONG: "How does the blue dashed curve at 0.8 compare to the red bar at 0.75?"
   RIGHT: "What accuracy does VFAE achieve on Adult when the compositional adversary shows AUC collapse at strong regularization?"

3) **NO META-LANGUAGE**: Never say "document A/B", "the paper", "the figure shows", "according to".
   Use method names, dataset names, or concept names.

4) **NO YES/NO QUESTIONS**: Start with "What", "How much", "Which", "At what point", or "Why".

5) **FACTUAL ANSWERS ONLY**: Answer with concrete numbers, directions, specific relationships.
   NEVER use "could potentially", "theoretically", "might", "suggests that".
   If you cannot give a factual answer, output NULL.

6) **SEMANTIC RELEVANCE CHECK**: If the shared concepts are homonyms or the documents address
   genuinely unrelated problems, output:
   {{"status": "NULL", "reason": "<specific reason>"}}

7) **NO FORCED BRIDGES**: If the only connection is a generic concept (accuracy, fairness, bias)
   used in completely different experimental contexts, output NULL.
   The query must involve a genuine intellectual connection, not just shared vocabulary.

## BAD (anchor leakage / forced comparison):
- "How does the blue dashed line at ~1.0 compare to the gray bar at ~0.87?" (copies visual tokens)
- "How does X relate to Y?" (vague essay prompt)
- "How does the FPR pattern in COMPAS relate to the causal arrow from A to Y?" (forced bridge)

## GOOD (reasoning with information gap):
- "What accuracy penalty does the Equity fairness approach incur on Adult data when enforcing the same level of demographic parity that COMPAS's optimal fair policy achieves for African-Americans?"
  (Query names methods and metrics conceptually; answerer must find specific values in BOTH figures)
- "At the regularization strength where the compositional adversary's Gender AUC drops below 0.6 on MovieLens, what Δ_CP constraint level produces equivalent accuracy degradation in FFVAE on CelebA?"
  (Requires reading precise values from both figures; no visual tokens in query)
- "Why does within-category FPR disparity persist in COMPAS even when overall rates are equalized, and what does the equal-mean beta distribution simulation reveal about the structural cause?"
  (Genuine reasoning connection; query is conceptual, not visual)

## Output format (JSON only):
{{
  "query": "Reasoning question using method/metric/dataset names, max 40 words, NO visual description tokens",
  "answer": "Factual answer with specific values from both figures (2-3 sentences, no speculation)",
  "query_type": "cross_application|cross_prediction|cross_diagnosis|cross_comparison",
  "reasoning_direction": "A_explains_B|B_explains_A|mutual",
  "turns": ["<the main query>"],
  "evidence_refs": [
    {{"doc_id": "{doc_a}", "anchor": "specific visual element with colors/shapes/values (detail goes HERE not in query)", "text_evidence": "direct quote"}},
    {{"doc_id": "{doc_b}", "anchor": "specific visual element with colors/shapes/values (detail goes HERE not in query)", "text_evidence": "direct quote"}}
  ]
}}"""


def encode_image(path: str) -> Optional[Tuple[str, str]]:
    """Return (base64_data, mime_type) or None if file missing."""
    # Try both absolute and project-root-relative
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / path
    if not p.exists() or p.stat().st_size < 500:
        return None
    ext = p.suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(
        ext, "image/jpeg"
    )
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8"), mime


def call_llm_anthropic(
    client: Any,
    model: str,
    prompt: str,
    images: List[Optional[Tuple[str, str]]],
) -> Tuple[Optional[str], int, int]:
    """Call Anthropic API with optional images. Returns (text, input_tokens, output_tokens)."""
    content: List[Dict[str, Any]] = []

    for img in images:
        if img is not None:
            b64, mime = img
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": mime, "data": b64},
            })

    content.append({"type": "text", "text": prompt})

    r = client.messages.create(
        model=model,
        system=SYSTEM_PROMPT,
        max_tokens=1024,
        temperature=0.5,
        messages=[{"role": "user", "content": content}],
    )
    return (
        r.content[0].text,
        r.usage.input_tokens,
        r.usage.output_tokens,
    )


def parse_json(txt: Optional[str]) -> Optional[Dict[str, Any]]:
    if not txt:
        return None
    t = txt.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t).strip()
        t = re.sub(r"\s*```$", "", t).strip()
    try:
        return json.loads(t)
    except Exception:
        # Try to find JSON object in mixed text
        m = re.search(r"\{.*\}", t, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate L2 queries from candidate pairs")
    ap.add_argument("--pairs", default="data/l2_candidate_pairs_v2.json")
    ap.add_argument("--output", default="data/l2_queries_v3.jsonl")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    ap.add_argument("--model", default="claude-sonnet-4-5-20250929")
    ap.add_argument("--delay", type=float, default=0.5, help="Seconds between API calls")
    ap.add_argument(
        "--anchor-leak-threshold",
        type=float,
        default=ANCHOR_LEAK_THRESHOLD,
        help="Max allowed query/anchor Jaccard overlap",
    )
    ap.add_argument(
        "--evidence-closure-threshold",
        type=float,
        default=EVIDENCE_CLOSURE_THRESHOLD,
        help="Min answer-to-evidence support ratio",
    )
    ap.add_argument("--no-images", action="store_true", help="Skip sending images (text-only)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key and not args.dry_run:
            print("ERROR: ANTHROPIC_API_KEY not set. Run: export $(grep -v '^#' .env | xargs)")
            sys.exit(1)

    pair_data = json.loads(Path(args.pairs).read_text(encoding="utf-8"))
    pairs = pair_data.get("pairs", [])[: args.limit]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"L2 Query Generation")
    print(f"  Pairs: {len(pairs)}")
    print(f"  Model: {args.model}")
    print(f"  Images: {'disabled' if args.no_images else 'enabled'}")
    print(f"  Anchor leak threshold: {args.anchor_leak_threshold:.2f}")
    print(f"  Evidence closure threshold: {args.evidence_closure_threshold:.2f}")
    print(f"  Output: {out}")
    print()

    # Initialize client once
    client = None
    if not args.dry_run and args.provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    total_input_tokens = 0
    total_output_tokens = 0
    kept = 0
    failed_parse = 0
    null_pairs = 0
    qc_failed = 0

    with out.open("w", encoding="utf-8") as f:
        for i, p in enumerate(pairs):
            doc_a, doc_b = p["doc_a"], p["doc_b"]
            prompt = build_prompt(p)

            if args.dry_run:
                print(f"\n--- pair {i+1}/{len(pairs)}: {doc_a} vs {doc_b} (score={p.get('score',0)}) ---")
                print(f"Shared: {', '.join(p.get('shared_entities', []))}")
                # Check images
                for side in ["doc_a", "doc_b"]:
                    img_path = p.get(f"{side}_image_path", "")
                    img = encode_image(img_path) if not args.no_images else None
                    status = f"OK ({len(img[0])//1024}KB b64)" if img else "MISSING"
                    print(f"  {side} image: {status} — {img_path}")
                print(prompt[:400] + "\n...")
                continue

            print(f"  [{i+1}/{len(pairs)}] {doc_a} x {doc_b}...", end=" ", flush=True)

            # Load images
            images: List[Optional[Tuple[str, str]]] = []
            if not args.no_images:
                images.append(encode_image(p.get("doc_a_image_path", "")))
                images.append(encode_image(p.get("doc_b_image_path", "")))
                img_count = sum(1 for x in images if x is not None)
            else:
                img_count = 0

            try:
                raw, in_tok, out_tok = call_llm_anthropic(client, args.model, prompt, images)
                total_input_tokens += in_tok
                total_output_tokens += out_tok
            except Exception as e:
                print(f"API ERROR: {e}")
                if "rate" in str(e).lower() or "429" in str(e):
                    print("  Rate limited, waiting 30s...")
                    time.sleep(30)
                continue

            obj = parse_json(raw)
            if not obj:
                print(f"PARSE FAIL")
                failed_parse += 1
                continue

            if obj.get("status") == "NULL":
                print(f"NULL ({obj.get('reason', 'no reason')[:60]})")
                null_pairs += 1
                continue

            # Attach metadata
            obj["l2_id"] = f"l2_v3_{i:03d}"
            obj["pair"] = {"doc_a": doc_a, "doc_b": doc_b}
            obj["shared_entities"] = p.get("shared_entities", [])
            obj["pair_score"] = p.get("score", 0)
            obj["images_sent"] = img_count
            obj["doc_a_image_path"] = p.get("doc_a_image_path", "")
            obj["doc_b_image_path"] = p.get("doc_b_image_path", "")
            obj["doc_a_figure_id"] = p.get("doc_a_figure_id", "")
            obj["doc_b_figure_id"] = p.get("doc_b_figure_id", "")

            issues, qc_metrics = qc_l2_query(
                obj,
                anchor_leak_threshold=args.anchor_leak_threshold,
                evidence_closure_threshold=args.evidence_closure_threshold,
            )
            obj["qc_issues"] = issues
            obj["qc_pass"] = len(issues) == 0
            obj["qc_metrics"] = qc_metrics

            if obj["qc_pass"]:
                kept += 1
                print(
                    "OK "
                    f"({obj.get('query_type', '?')}, "
                    f"leak={qc_metrics.get('anchor_leak_jaccard', 0):.2f}, "
                    f"closure={qc_metrics.get('evidence_closure', 0):.2f})"
                )
            else:
                qc_failed += 1
                print(
                    f"QC FAIL: {issues} "
                    f"(leak={qc_metrics.get('anchor_leak_jaccard', 0):.2f}, "
                    f"closure={qc_metrics.get('evidence_closure', 0):.2f})"
                )

            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

            if args.delay > 0 and i < len(pairs) - 1:
                time.sleep(args.delay)

    if args.dry_run:
        print(f"\nDry-run complete for {len(pairs)} pairs")
        return

    # Cost estimate: Sonnet 4.5 = $3/M input, $15/M output
    est_cost = total_input_tokens * 3 / 1e6 + total_output_tokens * 15 / 1e6

    print(f"\n{'='*60}")
    print(f"L2 Generation Summary")
    print(f"{'='*60}")
    print(f"  Total pairs:       {len(pairs)}")
    print(f"  QC passed:         {kept}")
    print(f"  QC failed:         {qc_failed}")
    print(f"  NULL (no query):   {null_pairs}")
    print(f"  Parse failures:    {failed_parse}")
    print(f"  Input tokens:      {total_input_tokens:,}")
    print(f"  Output tokens:     {total_output_tokens:,}")
    print(f"  Est. cost:         ${est_cost:.2f}")
    print(f"  Output:            {out}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
