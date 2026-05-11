"""训练数据导出器 —— 从 query 到模型可用数据的最后一公里。

整个流程：
  1. 加载 normalize 后的 StandardQuery
  2. 把 evidence span 对应到 Chunk（文本 + 可选图片路径）
  3. 通过可插拔的 NegativeSampler 采负样本，构建对比学习 Triplet
  4. doc-level hash split（同文档的 query 只进同一个 split，防泄漏）
  5. 写 JSONL + manifest.json

用法很简单：
    builder = DatasetBuilder(DatasetConfig(train_ratio=0.9))
    builder.build("queries.jsonl", "elements.json", "output/")
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.models import Chunk
from src.models.training import (
    DifficultyLevel,
    EvidenceSpan,
    StandardQuery,
    Triplet,
)
from src.sampling.negative_sampler import NegativeSampler, build_sampler

logger = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class DatasetConfig:
    """训练导出的配置 —— train/val/test 比例 + 负样本数量 + 采样策略。"""

    train_ratio: float = 0.9
    val_ratio: float = 0.05
    # test_ratio = 1 - train_ratio - val_ratio

    num_negatives: int = 3
    negative_strategy: str = "random"
    negative_config: Dict[str, Any] = field(default_factory=dict)

    seed: int = 42
    output_format: str = "jsonl"    # "jsonl" | "parquet"
    include_images: bool = True

    @property
    def test_ratio(self) -> float:
        return max(0.0, 1.0 - self.train_ratio - self.val_ratio)


# ── Helper: element index ─────────────────────────────────────────────────────

def _build_element_index(elements_path: Path) -> Dict[str, Dict[str, Any]]:
    """把 multimodal_elements.json 加载成 {element_id: 元素字典} 的索引。"""
    if not elements_path.exists():
        logger.warning("Elements file not found: %s", elements_path)
        return {}
    with open(elements_path, encoding="utf-8") as fh:
        data = json.load(fh)

    idx: Dict[str, Dict[str, Any]] = {}
    if isinstance(data, list):
        iterable = data
    elif isinstance(data, dict) and "documents" in data:
        iterable = []
        for doc in data["documents"].values():
            els = doc.get("elements", {})
            if isinstance(els, dict):
                iterable.extend(els.values())
            elif isinstance(els, list):
                iterable.extend(els)
    else:
        iterable = data.get("elements", []) if isinstance(data, dict) else []
    for el in iterable:
        eid = el.get("element_id", "")
        if eid:
            idx[eid] = el
    logger.info("Element index: %d entries", len(idx))
    return idx


def _element_to_chunk(el: Dict[str, Any]) -> Chunk:
    """把原始元素字典转成 Chunk —— 顺便拼好检索用的 text 字段。"""
    return Chunk(
        chunk_id=el.get("element_id", ""),
        doc_id=el.get("doc_id", ""),
        text=_assemble_chunk_text(el),
        caption=el.get("caption", ""),
        content=el.get("content", ""),
        context=el.get("context_before", ""),
        enriched_title=el.get("enriched_title", ""),
        enriched_content=el.get("enriched_content", ""),
    )


def _assemble_chunk_text(el: Dict[str, Any]) -> str:
    """拼检索文本：优先 enriched_content > caption > content > context_before。"""
    parts: list[str] = []
    for key in ("enriched_content", "caption", "content", "context_before"):
        val = (el.get(key) or "").strip()
        if val:
            parts.append(val)
    return " ".join(parts) if parts else el.get("element_id", "")


# ── Core builder ──────────────────────────────────────────────────────────────

class DatasetBuilder:
    """总指挥：queries → triplets → train/val/test split → 写盘。

    核心思路：每个 query 配上它的正样本证据，再用 NegativeSampler 采负样本，
    组成 (query, positive, negatives) 三元组。最后按 doc_id 哈希分组，
    保证同一篇论文的所有 query 只出现在同一个 split 里（防泄漏）。
    """

    def __init__(self, config: DatasetConfig | None = None) -> None:
        self.config = config or DatasetConfig()
        neg_cfg = dict(self.config.negative_config)
        neg_cfg.setdefault("strategy", self.config.negative_strategy)
        neg_cfg.setdefault("seed", self.config.seed)
        self.sampler: NegativeSampler = build_sampler(neg_cfg)

    # ── public entry point ────────────────────────────────────────────────

    def build(
        self,
        queries_path: str | Path,
        elements_path: str | Path,
        output_dir: str | Path,
    ) -> Dict[str, Any]:
        """跑完整个导出流程，返回统计信息。一把梭！"""
        queries_path = Path(queries_path)
        elements_path = Path(elements_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1) load
        queries = self._load_queries(queries_path)
        logger.info("Loaded %d queries", len(queries))

        element_idx = _build_element_index(elements_path)
        all_chunks = [_element_to_chunk(el) for el in element_idx.values()]
        logger.info("Loaded %d element chunks", len(all_chunks))

        # 2) build triplets
        triplets = self._build_triplets(queries, element_idx, all_chunks)
        logger.info("Built %d triplets", len(triplets))

        # 3) doc-level split
        splits = self._doc_stratified_split(triplets)

        # 4) write
        stats: Dict[str, Any] = {"total_triplets": len(triplets)}
        for split_name, split_data in splits.items():
            out_path = output_dir / f"{split_name}.jsonl"
            self._write_jsonl(split_data, out_path)
            stats[f"{split_name}_count"] = len(split_data)
            logger.info("Wrote %d records to %s", len(split_data), out_path)

        # 5) manifest
        manifest = {
            "schema_version": "1.0.0",
            "config": {
                "train_ratio": self.config.train_ratio,
                "val_ratio": self.config.val_ratio,
                "num_negatives": self.config.num_negatives,
                "negative_strategy": self.config.negative_strategy,
                "seed": self.config.seed,
            },
            "stats": stats,
        }
        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, ensure_ascii=False)

        return stats

    # ── internals ─────────────────────────────────────────────────────────

    @staticmethod
    def _load_queries(path: Path) -> List[StandardQuery]:
        queries: list[StandardQuery] = []
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    queries.append(StandardQuery.model_validate_json(line))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Skipping line %d in %s: %s", lineno, path, exc)
        return queries

    def _build_triplets(
        self,
        queries: List[StandardQuery],
        element_idx: Dict[str, Dict[str, Any]],
        all_chunks: List[Chunk],
    ) -> List[Triplet]:
        triplets: list[Triplet] = []
        for q in queries:
            pos_spans: list[EvidenceSpan] = []
            pos_ids: list[str] = []
            for es in q.evidence_spans:
                pos_spans.append(es)
                pos_ids.append(es.element_id)

            if not pos_spans:
                continue

            negs_chunks = self.sampler.sample(
                query_text=q.query_text,
                positive_ids=pos_ids,
                candidates=all_chunks,
                n=self.config.num_negatives,
            )
            neg_spans = [
                EvidenceSpan(
                    element_id=c.chunk_id,
                    doc_id=c.doc_id,
                    span_text=c.text[:200],
                )
                for c in negs_chunks
            ]

            triplets.append(
                Triplet(
                    query_id=q.query_id,
                    query_text=q.query_text,
                    difficulty_level=q.difficulty_level,
                    positive=pos_spans,
                    hard_negatives=neg_spans,
                    negative_strategy=self.config.negative_strategy,
                )
            )
        return triplets

    def _doc_stratified_split(
        self,
        triplets: List[Triplet],
    ) -> Dict[str, List[Triplet]]:
        """按 doc_id 哈希分 train/val/test —— 防泄漏的关键。

        md5(doc_id) % 1000 决定去哪个 split，确定性、可复现、零泄漏。
        同一篇论文的所有 query 只会出现在同一个 split 里。
        """

        def _doc_key(t: Triplet) -> str:
            if t.positive and t.positive[0].doc_id:
                return t.positive[0].doc_id
            return "unknown"

        # group by doc
        by_doc: Dict[str, List[Triplet]] = defaultdict(list)
        for t in triplets:
            by_doc[_doc_key(t)].append(t)

        # deterministic doc ordering
        doc_ids = sorted(by_doc.keys())
        # hash-based assignment for reproducibility
        train: list[Triplet] = []
        val: list[Triplet] = []
        test: list[Triplet] = []

        for doc_id in doc_ids:
            h = int(hashlib.md5(doc_id.encode()).hexdigest(), 16) % 1000
            threshold_train = int(self.config.train_ratio * 1000)
            threshold_val = threshold_train + int(self.config.val_ratio * 1000)
            if h < threshold_train:
                train.extend(by_doc[doc_id])
            elif h < threshold_val:
                val.extend(by_doc[doc_id])
            else:
                test.extend(by_doc[doc_id])

        return {"train": train, "val": val, "test": test}

    @staticmethod
    def _write_jsonl(triplets: Sequence[Triplet], path: Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for t in triplets:
                fh.write(t.model_dump_json() + "\n")
