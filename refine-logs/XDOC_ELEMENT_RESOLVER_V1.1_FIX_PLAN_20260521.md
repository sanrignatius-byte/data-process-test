# Cross-Doc Element Resolver v1.1 Fix Plan

**Date**: 2026-05-21
**Context**: v1 executed, pipeline runs, but reviewer audit found three precision-blocking issues.
**Principle**: don't rollback v1; fix forward with a sharper judge pack and corrected reporting.

## Issues Confirmed (with exact counts)

### Issue 1: anchored 判定太宽

182 `target_explicit_number_anchored` 的 anchor_reason 分布：

| anchor_reason | count | 可靠性 |
|---|---|---|
| `title_words_in_window` | 19 | **硬锚定** — target paper title token 出现在 numbered ref 窗口内 |
| `title_match_ge_0.2` | 115 | 中等 — C18 title_match 特征，但 title_match 是 chunk 级别的，不等于 "Figure 2" 指向 target paper |
| `low_fanout` | 24 | 弱 — source chunk 只引了 ≤2 篇论文，但 "Figure 2" 仍可能是 source paper 自己的图 |
| `single_ref_high_prob` | 24 | 弱 — chunk 只有一个 numbered ref + C18 prob≥0.95，同样不保证指向 target |

**风险**：后两类（48/182 = 26%）很容易把 source chunk 里的 "Figure 2" 错映射到 target paper 的 Figure 2。样本中已经能观察到：`xdoc_resolver_v1_000012` 的 "TABLE 5" 被 `low_fanout` 锚定，但 citing chunk 可能在讨论自己论文的 Table 5 而非 target paper 的。

### Issue 2: L3 recovery 没验证 explicit target 路线

3 个 endpoint 命中全是 `target_caption_overlap`，没有一个是 explicit-number。G2 的正确表述是 "v1 > v0 on L3 doc-pair recovery (14.8% vs 0% on recoverable subset)，但 explicit target route 尚未被 L3 独立验证"。

### Issue 3: target_score_buckets 是 raw attempt 统计

`summary.json` 里 `target_score_buckets` sum = 39,691（构造循环 raw count，在 dedup/chunk-cap/top-5000 之前）。报告里读成 final distribution 会误导。需要加 `post_filter_target_score_buckets`。

### Issue 4: judge pack 分层不够锋利

当前 25 个 `anchored_explicit` 里只有 5 个是 `title_words_in_window`，其余是软锚定。G3 的 explicit precision 会混在一起，看不出到底是哪种机制有效。

## Fix Plan

### F1: Add `post_filter_target_score_buckets` to summary

**File**: `experiments/build_xdoc_element_resolver_v1.py`

After line ~455 (after `pairs = pairs[:args.max_pairs]`), add:

```python
post_filter_buckets: Counter = Counter()
for p in pairs:
    score = p["hub_metadata"]["target_resolution_score"]
    method = p["hub_metadata"]["target_resolution_method"]
    if "anchored" in method:
        post_filter_buckets[">=0.90 (explicit anchored)"] += 1
    elif "unanchored" in method:
        post_filter_buckets["0.70-0.90 (explicit unanchored)"] += 1
    elif score >= 0.20:
        post_filter_buckets[">=0.20 (overlap)"] += 1
    else:
        post_filter_buckets["0.12-0.20 (overlap)"] += 1
summary["post_filter_target_score_buckets"] = dict(post_filter_buckets)
```

Also add `post_filter_anchor_reasons`:

```python
post_filter_anchor_reasons: Counter = Counter()
for p in pairs:
    detail = p["hub_metadata"].get("target_resolution_detail") or {}
    reason = detail.get("anchor_reason", "N/A")
    post_filter_anchor_reasons[reason] += 1
summary["post_filter_anchor_reasons"] = dict(post_filter_anchor_reasons)
```

Re-run full v1 to regenerate `summary.json` with corrected buckets.

### F2: Rebuild judge pack with anchor_reason as primary stratum

**File**: `experiments/build_xdoc_resolver_judge_pack.py`

Replace `bucket_target_method` with:

```python
def judge_stratum(pair: dict[str, Any]) -> str:
    """Primary stratification by anchor_reason for explicit, score tier for overlap."""
    method = pair["hub_metadata"]["target_resolution_method"]
    detail = pair["hub_metadata"].get("target_resolution_detail") or {}
    reason = detail.get("anchor_reason", "")
    score = pair["hub_metadata"]["target_resolution_score"]

    if reason == "title_words_in_window":
        return "A_hard_title_window"
    if reason == "title_match_ge_0.2":
        return "B_edge_title_match"
    if reason in ("low_fanout", "single_ref_high_prob"):
        return "C_soft_fanout_or_single_ref"
    if "unanchored" in method:
        return "D_unanchored_explicit"
    if score >= 0.20:
        return "E_overlap_high"
    return "F_overlap_low"
```

Target budget for n=120 (6 strata × 20):

| Stratum | Label | Target | Available (est.) |
|---|---|---|---|
| A | hard_title_window | 20 | 19 (may be <20) |
| B | edge_title_match | 20 | 115 |
| C | soft_fanout_or_single_ref | 20 | 48 |
| D | unanchored_explicit | 20 | 30 |
| E | overlap_high (≥0.20) | 20 | 1844 |
| F | overlap_low (0.12-0.20) | 20 | 2944 |

Redistribution rule: if A < 20, fill from B. If A+B+C+D < 80, fill from E. Keep the same pair_type balancing within each stratum.

Judge item schema: unchanged from v1, but add `anchor_reason` and `stratum` fields to each item.

### F3: Add anchor_reason to report

In `build_report()`, add a section:

```markdown
## Anchor Reason Breakdown (post-filter, top-5000)

| anchor_reason | count |
|---|---|
| title_words_in_window | 19 |
| title_match_ge_0.2 | 115 |
| low_fanout | 24 |
| single_ref_high_prob | 24 |
```

### F4: Update L3 recovery interpretation

In `evaluate_xdoc_resolver_l3_recovery.py`, add a method-stratified breakdown at K=5000:

```python
# Per gold hit, record which target method resolved it
method_hits_detail = []
for g in gold:
    g_ep = (g["element_ids"][0], g["element_ids"][1])
    if g_ep in tk_ep_set:
        p = tk_ep_lookup.get(g_ep) or tk_ep_lookup.get((g_ep[1], g_ep[0]))
        if p:
            method_hits_detail.append({
                "query_id": g["query_id"],
                "target_method": p["hub_metadata"]["target_resolution_method"],
                "anchor_reason": (p["hub_metadata"].get("target_resolution_detail") or {}).get("anchor_reason", "N/A"),
            })
```

Report this as "L3 endpoint hits by resolution method" — make it visible that all 3 hits are overlap, not explicit.

### F5: Tests

Add to `tests/test_xdoc_element_resolver_v1.py`:

```python
def test_anchor_reasons_are_exclusive():
    """Each anchored pair should have exactly one anchor_reason category."""
    ...

def test_post_filter_buckets_sum_to_max_pairs():
    """post_filter_target_score_buckets should sum to max_pairs."""
    ...
```

## Execution Order

| Step | What | File | Cost |
|---|---|---|---|
| S0 | Confirm baseline (v1 artifacts still in place) | read summary.json | 1 min |
| S1 | Add post_filter buckets to v1 builder | `build_xdoc_element_resolver_v1.py` | 15 min |
| S2 | Re-run full v1 with corrected summary | `python3 experiments/build_xdoc_element_resolver_v1.py --max-pairs 5000` | 5 min |
| S3 | Rebuild judge pack with 6 strata | `build_xdoc_resolver_judge_pack.py` | 15 min |
| S4 | Add method-stratified L3 detail | `evaluate_xdoc_resolver_l3_recovery.py` | 10 min |
| S5 | Re-run L3 evaluation, confirm explicit=0 hits still | `evaluate_xdoc_resolver_l3_recovery.py` | 1 min |
| S6 | Tests | `pytest -q tests/test_xdoc_citation_filter.py tests/test_xdoc_element_resolver_v1.py tests/test_intra_doc_pairing.py` | 1 min |
| S7 | Update wiki/log with corrected interpretation | `20260519_xdoc_pairing_module.md`, `log.md` | 10 min |

## Key Rules for Next Assistant

1. **Do NOT delete v1 artifacts.** Fix forward — overwrite latest symlink with corrected version.
2. **Do NOT promote to `src/pairing/cross_doc_pairs.py`.** Still experimental lane.
3. **The judge pack is the next gate, not a deliverable.** Don't run LLM judging without the user's explicit request — just build the pack and report the stratification.
4. **Anchor reason naming**: use `title_words_in_window` (not `hard_title_window`) in code to match existing field. The stratum labels A-F are for the judge pack only.
5. **G3 interpretation**: only stratum A (`title_words_in_window`) should be counted as "hard explicit precision". Strata B and C are probing whether softer signals also work — that's the research question, not a given.
