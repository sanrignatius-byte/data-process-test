---
type: claim
node_id: claim:C16
status: supported
created_at: 2026-05-20T00:00:00Z
updated_at: 2026-05-20T00:00:00Z
---

# Claim

Pure MinerU + CLIP visual similarity + caption/context/enriched TF-IDF rerank produces
an effective cross-document **recall** layer but not **strong** edges. The text-rerank
signal is bottlenecked by parser-degraded captions, so high-confidence cross-doc
semantic edges are rare and the edges should not enter a multi-hop graph as hard edges
without a stronger semantic rerank.

## Evidence

- `exp:20260520_mineru_clip_xdoc_pipeline`, rerank audit on 3238 xdoc visual edges.
- **87.2%** of xdoc edges have `caption_sim=0` (zero token overlap between the two
  figures' captions); only **5.1%** have `enriched_sim>0.15` (real text support).
- The `strong_text_supported` tier (587) is optimistically named: sampled context
  support median ~0.07; it is mostly vis≈0.88 lifted by a sliver of context, with the
  far end often a bare "Figure 16".
- Root cause: of 937 figure/table nodes, 64.2% have real captions but **35.8% are
  unusable** for text matching (20% subfig "(a)(b)" labels, 10% too short, 5.5% OCR'd
  HTML fragments). Even real captions rarely share tokens across documents.
- Genuinely good xdoc edges exist but are rare (e.g. gender-bias-template tables
  1910.10872↔1809.02208, enr=0.27) and are not surfaced via caption.
- `visual_only_risky` (186, default-dropped) sampled as correct drops: bare-number /
  HTML-fragment pure layout matches.

## Scope

53-doc corpus, cross-doc figure/table visual edges. Does not apply to intra-doc edges
(see `claim:C15`) or to formula edges (`claim:C17`).

## Why It Matters

Sets the honest ceiling of the current cross-doc method and motivates `gap:G10`:
to make cross-doc edges strong, the rerank must bypass degraded captions
(VLM direct link judgment, recaption enrichment, or LLM rerank), not lean on caption
token overlap.

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
