"""端点锚定式多跳 Query 生成 - 使用预计算的图数据。

核心思路：
- 直接复用 analyze_latex_graph_topology.py 已经建好的图结构
- 使用 latex_hub_multihop_candidates.json 中预计算的路径
- 只需 1 次 LLM 调用生成自然语言问答

数据来源：
- data/01_graphs/latex_graph_hubs.json (traffic_hubs, adjacent_backbone_bridges)
- data/01_graphs/latex_hub_multihop_candidates.json (预计算的 500 个多跳候选)

使用示例:
    from src.pairing.endpoint_anchor import (
        load_precomputed_candidates,
        filter_candidates_by_hop,
        build_query_prompt_from_candidate,
    )
    
    # 加载预计算候选
    candidates = load_precomputed_candidates()
    
    # 筛选 4-6 hop 的
    filtered = filter_candidates_by_hop(candidates, min_hop=4, max_hop=6)
    
    # 为候选生成 prompt
    prompt = build_query_prompt_from_candidate(candidate, bridge_texts)
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union


# ── Load Precomputed Data ────────────────────────────────────────────────────

def _resolve_data_dir(data_dir: Optional[Union[str, Path]] = None) -> Path:
    """解析数据目录路径。
    
    支持:
    - None: 使用环境变量 DATA_PROCESS_DATA_DIR 或当前目录的 data/01_graphs
    - 相对路径: 相对于当前工作目录
    - 绝对路径: 直接使用
    """
    import os
    
    if data_dir is not None:
        return Path(data_dir)
    
    # 优先用环境变量
    env_dir = os.environ.get("DATA_PROCESS_DATA_DIR")
    if env_dir:
        return Path(env_dir) / "01_graphs"
    
    # 回退到相对于脚本的路径（兼容旧代码）
    script_based = Path(__file__).parent.parent.parent / "data" / "01_graphs"
    if script_based.exists():
        return script_based
    
    # 最后尝试当前目录
    return Path("data/01_graphs")


def load_precomputed_candidates(
    data_dir: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """加载预计算的多跳候选。
    
    来源: latex_hub_multihop_candidates.json
    
    每个候选包含:
    - path_node_ids: 路径节点 ID 列表
    - path_node_types: 节点类型列表
    - hop_distance: 跳数
    - modalities_in_path: 涉及的模态
    - is_cross_doc: 是否跨文档
    - long_query_seed: 预生成的长 query 种子
    - short_query_seed: 预生成的短 query 种子
    """
    data_dir = _resolve_data_dir(data_dir)
    
    path = data_dir / "latex_hub_multihop_candidates.json"
    if not path.exists():
        raise FileNotFoundError(f"Candidates file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return data.get("candidates", [])


def load_hub_bridges(
    data_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """加载 hub 和 bridge 数据。
    
    来源: latex_graph_hubs.json
    
    Returns:
        Dict with keys:
        - traffic_hubs: 流量枢纽节点
        - bridge_hubs: 桥接枢纽
        - adjacent_backbone_bridges: 相邻骨干桥
    """
    data_dir = _resolve_data_dir(data_dir)
    
    path = data_dir / "latex_graph_hubs.json"
    if not path.exists():
        raise FileNotFoundError(f"Hubs file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return {
        "traffic_hubs": data.get("traffic_hubs", []),
        "bridge_hubs": data.get("bridge_hubs", []),
        "adjacent_backbone_bridges": data.get("adjacent_backbone_bridges", []),
    }


def load_reference_graph(
    data_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """加载 LaTeX reference graph。
    
    来源: latex_reference_graph.json 或 enrich 后的版本
    """
    data_dir = _resolve_data_dir(data_dir)
    
    # 优先用 enriched 版本
    enriched_path = data_dir.parent / "02_enriched" / "enriched_latex_graph.json"
    if enriched_path.exists():
        with open(enriched_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    # 回退到原始版本
    raw_path = data_dir.parent / "00_raw" / "latex_reference_graph.json"
    if raw_path.exists():
        with open(raw_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    raise FileNotFoundError(f"Reference graph not found in {data_dir.parent}")


# ── Filtering & Selection ────────────────────────────────────────────────────

def filter_candidates_by_hop(
    candidates: List[Dict[str, Any]],
    min_hop: int = 3,
    max_hop: int = 8,
) -> List[Dict[str, Any]]:
    """按 hop 数筛选候选。"""
    return [
        c for c in candidates
        if min_hop <= c.get("hop_distance", 0) <= max_hop
    ]


def filter_candidates_by_modality(
    candidates: List[Dict[str, Any]],
    required_modalities: Optional[Set[str]] = None,
    min_modality_count: int = 2,
) -> List[Dict[str, Any]]:
    """按模态筛选候选。
    
    Args:
        required_modalities: 必须包含的模态集合（如 {"figure", "table"}）
        min_modality_count: 最少模态数
    """
    result = []
    for c in candidates:
        mods = set(c.get("modalities_in_path", []))
        
        # 检查数量
        if len(mods) < min_modality_count:
            continue
        
        # 检查必须包含
        if required_modalities and not required_modalities.issubset(mods):
            continue
        
        result.append(c)
    
    return result


def filter_cross_doc_only(
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """只保留跨文档的候选。"""
    return [c for c in candidates if c.get("is_cross_doc", False)]


def sample_diverse_candidates(
    candidates: List[Dict[str, Any]],
    n: int = 100,
    per_doc_limit: int = 5,
) -> List[Dict[str, Any]]:
    """多样性采样，限制每个文档的候选数量。
    
    确保不会被单一文档主导。
    """
    # 按文档分组
    by_doc: Dict[str, List[Dict[str, Any]]] = {}
    for c in candidates:
        doc_id = c.get("doc_id", "unknown")
        by_doc.setdefault(doc_id, []).append(c)
    
    # 从每个文档采样
    sampled = []
    for doc_id, doc_candidates in by_doc.items():
        random.shuffle(doc_candidates)
        sampled.extend(doc_candidates[:per_doc_limit])
    
    # 全局打乱并截断
    random.shuffle(sampled)
    return sampled[:n]


# ── Bridge Text Resolution ───────────────────────────────────────────────────

def resolve_path_bridge_texts(
    candidate: Dict[str, Any],
    reference_graph: Dict[str, Any],
    bridge_text_cache: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """为候选路径解析每个节点的 bridge_text。
    
    Args:
        candidate: 候选 dict（含 path_node_ids）
        reference_graph: LaTeX reference graph
        bridge_text_cache: 可选的缓存（element_id → bridge_text）
    
    Returns:
        List of node dicts with resolved bridge_text
    """
    path_node_ids = candidate.get("path_node_ids", [])
    if not path_node_ids:
        return []
    
    # 懒加载缓存
    if bridge_text_cache is None:
        from src.utils.bridge_utils import get_bridge_text_cache
        bridge_text_cache = get_bridge_text_cache(reference_graph)
    
    nodes = []
    for node_id in path_node_ids:
        # node_id 格式: "1904.03310::el::tab:lm_cor" 或 "1904.03310::p::00001"
        parts = node_id.split("::")
        if len(parts) < 3:
            nodes.append({
                "node_id": node_id,
                "node_type": "unknown",
                "bridge_text": "",
            })
            continue
        
        doc_id = parts[0]
        node_type = parts[1]  # "el" or "p"
        label = parts[2]
        
        if node_type == "el":
            # 元素节点，查找 bridge_text
            # element_id 格式可能是 "1904.03310_table_1" 或直接用 label
            mapped_eid = candidate.get("mapped_element_id") if node_id == path_node_ids[0] else None
            
            # 尝试多种 key
            bridge_text = ""
            for key in [node_id, f"{doc_id}_{label}", label]:
                if key in bridge_text_cache:
                    bridge_text = bridge_text_cache[key]
                    break
            
            nodes.append({
                "node_id": node_id,
                "doc_id": doc_id,
                "node_type": "element",
                "label": label,
                "bridge_text": bridge_text,
            })
        
        else:  # paragraph
            # 段落节点，暂时没有 bridge_text
            nodes.append({
                "node_id": node_id,
                "doc_id": doc_id,
                "node_type": "paragraph",
                "label": label,
                "bridge_text": "",
            })
    
    return nodes


# ── Prompt Building ──────────────────────────────────────────────────────────

def extract_element_description(elem: Dict[str, Any], max_len: int = 150) -> str:
    """从元素中提取简洁的语义描述。"""
    # 优先用 bridge_text
    bridge = elem.get("bridge_text", "") or ""
    if bridge:
        return bridge.strip()[:max_len]
    
    # 其次用 caption
    caption = elem.get("caption", "") or ""
    if caption:
        return caption.strip()[:max_len]
    
    # 最后用 label
    label = elem.get("label", elem.get("node_id", "unknown"))
    node_type = elem.get("node_type", "element")
    return f"{node_type}: {label}"


def build_query_prompt_from_candidate(
    candidate: Dict[str, Any],
    path_nodes: Optional[List[Dict[str, Any]]] = None,
    include_path_hint: bool = True,
    prior_context: Optional[str] = None,
    turn_number: int = 1,
) -> str:
    """从预计算候选构建 query prompt。
    
    Args:
        candidate: 预计算的候选 dict
        path_nodes: 已解析的路径节点（可选，如果没有会尝试从候选提取）
        include_path_hint: 是否在 prompt 中包含路径提示
        prior_context: 多轮对话的上下文
        turn_number: 当前轮数
    """
    path_node_ids = candidate.get("path_node_ids", [])
    if len(path_node_ids) < 2:
        raise ValueError("Candidate must have at least 2 nodes in path")
    
    # 使用预计算的 query seed 作为参考
    long_seed = candidate.get("long_query_seed", "")
    short_seed = candidate.get("short_query_seed", "")
    
    # 起点终点描述
    start_id = path_node_ids[0]
    end_id = path_node_ids[-1]
    
    if path_nodes:
        start_desc = extract_element_description(path_nodes[0])
        end_desc = extract_element_description(path_nodes[-1])
        mid_descriptions = [
            f"Step {i+1} ({n.get('node_type', 'node')}): {extract_element_description(n, 80)}"
            for i, n in enumerate(path_nodes[1:-1], start=1)
        ]
    else:
        # 没有解析节点，用 ID 和 seed
        start_desc = f"Element {start_id}"
        end_desc = f"Element {end_id}"
        mid_descriptions = []
    
    # 路径提示
    path_context = ""
    if include_path_hint and mid_descriptions:
        path_context = f"""
REASONING PATH (for answer generation, NOT to be mentioned in query):
Step 1: START - {start_desc[:80]}...
{chr(10).join(mid_descriptions)}
Step {len(path_node_ids)}: END - {end_desc[:80]}...

CRITICAL: Each step's conclusion MUST be used by the next step.
The answer must show SERIAL dependency, not parallel facts.
"""

    # 多轮上下文
    prior_section = ""
    if prior_context and turn_number > 1:
        prior_section = f"""
PRIOR CONVERSATION (Turn {turn_number - 1} concluded):
{prior_context}

This turn's question should BUILD ON the prior conclusion.
"""

    # 参考 seed
    seed_hint = ""
    if short_seed:
        seed_hint = f"""
REFERENCE (you may rephrase, DO NOT copy verbatim):
"{short_seed[:100]}..."
"""

    hop_count = candidate.get("hop_distance", len(path_node_ids) - 1)
    modalities = candidate.get("modalities_in_path", [])
    is_cross_doc = candidate.get("is_cross_doc", False)

    prompt = f"""Generate a natural multi-hop question and answer pair.

=== ENDPOINTS ===
START: {start_desc}
END: {end_desc}

=== METADATA ===
Hop count: {hop_count}
Modalities: {', '.join(modalities)}
Cross-document: {is_cross_doc}
{prior_section}{path_context}{seed_hint}
=== REQUIREMENTS ===

QUESTION (< 35 words):
- Ask about the CONNECTION between START and END
- Do NOT mention intermediate steps explicitly
- Use natural, curious language

ANSWER (SERIAL CHAIN - this is critical):
- Follow this pattern exactly:
  "Step 1 establishes [fact_A]. Because of [fact_A], Step 2 reveals [fact_B]. 
   Building on [fact_B], Step 3 shows [fact_C]. Therefore, [conclusion]."
- Each step MUST reference the previous step's finding
- If any intermediate step is removed, the answer becomes incomplete

=== OUTPUT FORMAT (JSON) ===
{{
  "query": "natural question under 35 words",
  "answer": "serial chain reasoning with explicit dependencies",
  "reasoning_chain": [
    {{"step": 1, "element": "{start_id}", "finding": "...", "depends_on": null}},
    {{"step": 2, "element": "...", "finding": "...", "depends_on": 1}},
    ...
  ],
  "hop_count": {hop_count},
  "mentions_both_endpoints": true
}}
"""
    return prompt.strip()


# ── Multi-Turn Session ───────────────────────────────────────────────────────

def build_multiturn_session_prompts(
    candidates: List[Dict[str, Any]],
    num_turns: int = 3,
) -> List[Dict[str, Any]]:
    """构建多轮对话 session 的 prompts。
    
    思路：选择 num_turns 个相关联的候选，每轮 query 建立在前一轮的结论上。
    
    Returns:
        List of turn dicts, each with:
        - turn_number: 轮数
        - candidate: 当前轮的候选
        - prompt: 生成用的 prompt
        - prior_summary_placeholder: 上一轮结论的占位符
    """
    if len(candidates) < num_turns:
        # 候选不够，重复使用
        candidates = candidates * ((num_turns // len(candidates)) + 1)
    
    # 尝试选择来自相同或相关文档的候选
    by_doc: Dict[str, List[Dict[str, Any]]] = {}
    for c in candidates:
        doc_id = c.get("doc_id", "unknown")
        by_doc.setdefault(doc_id, []).append(c)
    
    # 找一个有足够候选的文档
    session_candidates = []
    for doc_id, doc_cands in by_doc.items():
        if len(doc_cands) >= num_turns:
            session_candidates = random.sample(doc_cands, num_turns)
            break
    
    # 如果没有足够的单文档候选，混合使用
    if not session_candidates:
        session_candidates = random.sample(candidates[:num_turns * 2], num_turns)
    
    turns = []
    for i, cand in enumerate(session_candidates):
        turn = {
            "turn_number": i + 1,
            "candidate": cand,
            "prior_summary_placeholder": f"[TURN_{i}_CONCLUSION]" if i > 0 else None,
        }
        
        # 第一轮没有上下文
        if i == 0:
            turn["prompt"] = build_query_prompt_from_candidate(
                cand, 
                include_path_hint=True,
                turn_number=1,
            )
        else:
            # 后续轮次引用前一轮
            turn["prompt"] = build_query_prompt_from_candidate(
                cand,
                include_path_hint=True,
                prior_context=f"[TURN_{i}_CONCLUSION]",  # 占位符
                turn_number=i + 1,
            )
        
        turns.append(turn)
    
    return turns


# ── Convenience Functions ────────────────────────────────────────────────────

def get_ready_candidates(
    data_dir: Optional[Union[str, Path]] = None,
    min_hop: int = 3,
    max_hop: int = 6,
    min_modalities: int = 2,
    sample_n: int = 100,
) -> List[Dict[str, Any]]:
    """一键获取准备好的候选。
    
    组合了加载、筛选、采样等步骤。
    """
    candidates = load_precomputed_candidates(data_dir)
    
    # 筛选 hop 数
    candidates = filter_candidates_by_hop(candidates, min_hop, max_hop)
    
    # 筛选模态数
    candidates = filter_candidates_by_modality(
        candidates, 
        min_modality_count=min_modalities,
    )
    
    # 多样性采样
    candidates = sample_diverse_candidates(candidates, n=sample_n)
    
    return candidates


def quick_generate_prompt(
    candidate_index: int = 0,
    data_dir: Optional[Union[str, Path]] = None,
) -> str:
    """快速生成一个 prompt（用于测试）。"""
    candidates = get_ready_candidates(data_dir, sample_n=10)
    if not candidates:
        return "No candidates found"
    
    idx = candidate_index % len(candidates)
    return build_query_prompt_from_candidate(candidates[idx])


# ── Main (for testing) ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    
    print("=== Endpoint Anchor Module Test ===\n")
    
    # 1. 加载候选
    print("Loading precomputed candidates...")
    try:
        candidates = load_precomputed_candidates()
        print(f"  Loaded {len(candidates)} candidates")
    except FileNotFoundError as e:
        print(f"  Error: {e}")
        sys.exit(1)
    
    # 2. 筛选
    print("\nFiltering (3-6 hops, 2+ modalities)...")
    filtered = filter_candidates_by_hop(candidates, min_hop=3, max_hop=6)
    filtered = filter_candidates_by_modality(filtered, min_modality_count=2)
    print(f"  After filtering: {len(filtered)} candidates")
    
    # 3. 采样
    print("\nSampling 5 diverse candidates...")
    sampled = sample_diverse_candidates(filtered, n=5, per_doc_limit=2)
    
    # 4. 展示一个 prompt
    if sampled:
        print("\n=== Sample Prompt ===")
        print(build_query_prompt_from_candidate(sampled[0]))
    
    print("\n=== Done ===")
