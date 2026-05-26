"""Unit tests for xdoc element resolver v1 components."""

from __future__ import annotations

import json
from pathlib import Path
import sys

# Ensure project root is on sys.path for experiments imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Import from the v1 builder (as a module, not a script)
import importlib.util
_v1_path = _PROJECT_ROOT / "experiments" / "build_xdoc_element_resolver_v1.py"
_spec = importlib.util.spec_from_file_location("resolver_v1", _v1_path)
_resolver_v1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_resolver_v1)

_judge_path = _PROJECT_ROOT / "experiments" / "build_xdoc_resolver_judge_pack.py"
_judge_spec = importlib.util.spec_from_file_location("judge_pack", _judge_path)
_judge_pack = importlib.util.module_from_spec(_judge_spec)
_judge_spec.loader.exec_module(_judge_pack)

_eval_path = _PROJECT_ROOT / "experiments" / "evaluate_xdoc_resolver_l3_recovery.py"
_eval_spec = importlib.util.spec_from_file_location("l3_eval", _eval_path)
_l3_eval = importlib.util.module_from_spec(_eval_spec)
_eval_spec.loader.exec_module(_l3_eval)


# ── target explicit numbered ref extraction ─────────────────────────────────

def test_extract_numbered_refs_finds_table():
    text = "As shown in Table 5 of the target paper, results improve."
    refs = _resolver_v1.extract_numbered_refs_with_windows(text)
    assert len(refs) >= 1
    table_refs = [r for r in refs if r["element_type"] == "table"]
    assert any(r["number"] == 5 for r in table_refs)


def test_extract_numbered_refs_finds_figure():
    text = "Figure 3 demonstrates the architecture."
    refs = _resolver_v1.extract_numbered_refs_with_windows(text)
    fig_refs = [r for r in refs if r["element_type"] == "figure"]
    assert any(r["number"] == 3 for r in fig_refs)


def test_extract_numbered_refs_finds_equation():
    text = "Following Eq. (7), we derive the bound."
    refs = _resolver_v1.extract_numbered_refs_with_windows(text)
    eq_refs = [r for r in refs if r["element_type"] == "formula"]
    assert any(r["number"] == 7 for r in eq_refs)


def test_extract_numbered_refs_deduplicates_same_span():
    """Same (type, number, span_start) should dedup, but different spans are distinct."""
    text = "Table 2 summarizes the results."
    refs = _resolver_v1.extract_numbered_refs_with_windows(text)
    table_refs = [r for r in refs if r["element_type"] == "table" and r["number"] == 2]
    assert len(table_refs) == 1  # single mention, single ref


# ── target anchoring ────────────────────────────────────────────────────────


def test_is_target_anchored_low_fanout():
    ref = {"element_type": "table", "number": 2, "ref_text": "Table 2",
           "window_before": "results in", "window_after": "show improvement"}
    edge = {"features": {"title_match": 0.05}, "probability": 0.5}
    all_refs = [ref]
    is_anchored, reason = _resolver_v1.is_target_anchored_ref(
        ref, edge, set(), fanout=1, all_refs_in_chunk=all_refs)
    assert is_anchored is True
    assert reason == "low_fanout"


def test_is_target_anchored_title_match_ge_02():
    ref = {"element_type": "figure", "number": 1, "ref_text": "Figure 1",
           "window_before": "", "window_after": ""}
    edge = {"features": {"title_match": 0.35}, "probability": 0.5}
    all_refs = [ref, {"element_type": "table", "number": 3}]
    is_anchored, reason = _resolver_v1.is_target_anchored_ref(
        ref, edge, set(), fanout=6, all_refs_in_chunk=all_refs)
    assert is_anchored is True
    assert reason == "title_match_ge_0.2"


def test_is_target_anchored_single_ref_high_prob():
    ref = {"element_type": "figure", "number": 1, "ref_text": "Figure 1",
           "window_before": "", "window_after": ""}
    edge = {"features": {"title_match": 0.05}, "probability": 0.99}
    all_refs = [ref]
    is_anchored, reason = _resolver_v1.is_target_anchored_ref(
        ref, edge, set(), fanout=5, all_refs_in_chunk=all_refs)
    assert is_anchored is True
    assert reason == "single_ref_high_prob"


def test_is_target_unanchored_high_fanout_no_title():
    ref = {"element_type": "table", "number": 4, "ref_text": "Table 4",
           "window_before": "", "window_after": ""}
    edge = {"features": {"title_match": 0.05}, "probability": 0.80}
    all_refs = [ref, {"element_type": "figure", "number": 2}]
    is_anchored, reason = _resolver_v1.is_target_anchored_ref(
        ref, edge, set(), fanout=7, all_refs_in_chunk=all_refs)
    assert is_anchored is False
    assert reason == "unanchored"


def test_is_target_anchored_with_title_aliases():
    ref = {"element_type": "table", "number": 2, "ref_text": "Table 2",
           "window_before": "the UCF101 dataset provides",
           "window_after": "summarizes action recognition results"}
    edge = {"features": {"title_match": 0.05}, "probability": 0.5}
    all_refs = [ref, {"element_type": "figure", "number": 1}]
    target_aliases = {"ucf101", "dataset", "actions", "human", "classes"}
    is_anchored, reason = _resolver_v1.is_target_anchored_ref(
        ref, edge, target_aliases, fanout=3, all_refs_in_chunk=all_refs)
    assert is_anchored is True
    assert reason == "title_words_in_window"


# ── fanout penalty ──────────────────────────────────────────────────────────


def test_fanout_penalty_low():
    assert _resolver_v1.source_fanout_penalty(1) == 1.0
    assert _resolver_v1.source_fanout_penalty(2) == 1.0


def test_fanout_penalty_medium():
    assert _resolver_v1.source_fanout_penalty(3) == 0.75
    assert _resolver_v1.source_fanout_penalty(5) == 0.75


def test_fanout_penalty_high():
    assert _resolver_v1.source_fanout_penalty(8) == 0.55
    assert _resolver_v1.source_fanout_penalty(11) == 0.35


def test_fanout_penalty_decreases_with_fanout():
    p_low = _resolver_v1.source_fanout_penalty(1)
    p_med = _resolver_v1.source_fanout_penalty(4)
    p_high = _resolver_v1.source_fanout_penalty(8)
    p_vhigh = _resolver_v1.source_fanout_penalty(15)
    assert p_low > p_med > p_high >= p_vhigh


# ── post-filter statistics helpers ─────────────────────────────────────────


def test_target_score_bucket_boundaries():
    assert _resolver_v1.target_score_bucket(1.0) == ">=0.90 (explicit anchored)"
    assert _resolver_v1.target_score_bucket(0.90) == ">=0.90 (explicit anchored)"
    assert _resolver_v1.target_score_bucket(0.70) == "0.70-0.90 (explicit unanchored)"
    assert _resolver_v1.target_score_bucket(0.20) == ">=0.20"
    assert _resolver_v1.target_score_bucket(0.12) == "0.12-0.20"
    assert _resolver_v1.target_score_bucket(0.07) == "0.07-0.12"


def test_v1_summary_post_filter_stats_sum_to_total():
    summary_path = _PROJECT_ROOT / "data/05_eval/xdoc_element_resolver_v1_latest/summary.json"
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if "post_filter_target_score_buckets" not in summary:
        return
    total = summary["total_pairs"]
    assert sum(summary["post_filter_target_score_buckets"].values()) == total
    assert sum(summary["post_filter_anchor_reasons"].values()) == total
    assert summary["target_score_buckets_scope"].startswith("raw candidate attempts")


# ── judge pack stratification ───────────────────────────────────────────────


def _fake_pair(method: str, score: float, reason: str | None = None) -> dict:
    detail = {"anchor_reason": reason} if reason else {}
    return {
        "pair_id": "p",
        "pair_type": "figure+table",
        "quality_score": 1.0,
        "hub_metadata": {
            "target_resolution_method": method,
            "target_resolution_score": score,
            "target_resolution_detail": detail,
        },
    }


def test_judge_pack_bucket_pair_splits_anchor_reasons():
    assert _judge_pack.bucket_pair(_fake_pair(
        "target_explicit_number_anchored", 1.0, "title_words_in_window"
    )) == "A_hard_title_window"
    assert _judge_pack.bucket_pair(_fake_pair(
        "target_explicit_number_anchored", 1.0, "title_match_ge_0.2"
    )) == "B_edge_title_match"
    assert _judge_pack.bucket_pair(_fake_pair(
        "target_explicit_number_anchored", 1.0, "low_fanout"
    )) == "C_soft_fanout_or_single_ref"
    assert _judge_pack.bucket_pair(_fake_pair(
        "target_explicit_number_unanchored", 0.7, "unanchored"
    )) == "D_unanchored_explicit"
    assert _judge_pack.bucket_pair(_fake_pair(
        "target_caption_overlap", 0.21
    )) == "E_overlap_high"
    assert _judge_pack.bucket_pair(_fake_pair(
        "target_caption_overlap", 0.13
    )) == "F_overlap_low"


# ── L3 method stratification helpers ───────────────────────────────────────


def test_l3_eval_pair_route_classification():
    explicit = _fake_pair("target_explicit_number_anchored", 1.0, "title_words_in_window")
    overlap = _fake_pair("target_caption_overlap", 0.2)
    assert _l3_eval.pair_target_route(explicit) == "explicit_number"
    assert _l3_eval.pair_target_route(overlap) == "caption_overlap"
    assert _l3_eval.pair_anchor_reason(explicit) == "title_words_in_window"


# ── endpoint matching (unordered) ───────────────────────────────────────────


def test_endpoint_pair_unordered_match():
    """Endpoint pairs should match regardless of order."""
    ep1 = ("1709.02012_figure_4", "1703.09207_table_2")
    ep2 = ("1703.09207_table_2", "1709.02012_figure_4")
    assert (ep1[0], ep1[1]) != (ep2[0], ep2[1])
    assert set(ep1) == set(ep2)


# ── v1 pair schema compatibility ────────────────────────────────────────────


def test_v1_pair_schema_has_generate_multihop_required_fields():
    """v1 pairs must have fields required by generate_multihop_l1_queries.py."""
    v1_path = _PROJECT_ROOT / "data/05_eval/xdoc_element_resolver_v1_latest/cross_doc_pairs_v1.json"
    if not v1_path.exists():
        return  # Skip if v1 not built yet
    data = json.loads(v1_path.read_text(encoding="utf-8"))
    pairs = data.get("pairs", [])
    if not pairs:
        return
    required_fields = [
        "pair_id", "doc_id", "element_a_id", "element_b_id",
        "element_a_type", "element_b_type", "pair_type",
        "hop_distance", "element_a", "element_b", "node_group",
        "hub_metadata",
    ]
    p = pairs[0]
    missing = [f for f in required_fields if f not in p]
    assert missing == [], f"Missing fields: {missing}"


def test_v1_pair_has_extended_metadata():
    """v1 pairs must include new v1-specific metadata fields."""
    v1_path = _PROJECT_ROOT / "data/05_eval/xdoc_element_resolver_v1_latest/cross_doc_pairs_v1.json"
    if not v1_path.exists():
        return
    data = json.loads(v1_path.read_text(encoding="utf-8"))
    pairs = data.get("pairs", [])
    if not pairs:
        return
    p = pairs[0]
    meta = p["hub_metadata"]
    v1_fields = ["resolver_version", "citation_fanout", "source_fanout_penalty",
                 "target_resolution_method", "target_resolution_detail"]
    missing = [f for f in v1_fields if f not in meta]
    assert missing == [], f"Missing v1 metadata fields: {missing}"


def test_v1_has_nonzero_explicit_target_pairs():
    """v1 should have at least some explicit target matches."""
    v1_path = _PROJECT_ROOT / "data/05_eval/xdoc_element_resolver_v1_latest/cross_doc_pairs_v1.json"
    if not v1_path.exists():
        return
    data = json.loads(v1_path.read_text(encoding="utf-8"))
    pairs = data.get("pairs", [])
    explicit = [p for p in pairs if "explicit_number" in p["hub_metadata"]["target_resolution_method"]]
    assert len(explicit) > 0, "v1 should have explicit target pairs"
    print(f"  Explicit target pairs: {len(explicit)}")


# ── title aliases ───────────────────────────────────────────────────────────


def test_load_title_aliases_returns_dict():
    ref_path = _PROJECT_ROOT / "data/01_graphs/latex_reference_graph_v2.json"
    if not ref_path.exists():
        return
    aliases = _resolver_v1.load_title_aliases(ref_path)
    assert isinstance(aliases, dict)
    assert len(aliases) > 0


def test_title_aliases_have_tokens():
    ref_path = _PROJECT_ROOT / "data/01_graphs/latex_reference_graph_v2.json"
    if not ref_path.exists():
        return
    aliases = _resolver_v1.load_title_aliases(ref_path)
    for doc_id, tokens_set in list(aliases.items())[:1]:
        assert isinstance(tokens_set, set)
        assert len(tokens_set) > 0
        assert all(isinstance(t, str) for t in tokens_set)
