"""Training dataset builder — the bridge from queries to model-ready data.

Responsibilities
----------------
1. Load normalised ``StandardQuery`` records.
2. Resolve evidence spans to ``Chunk`` objects (text + optional image path).
3. Build contrastive triplets via pluggable :class:`NegativeSampler`.
4. Perform a **doc-level** train / val / test split (prevents data leakage).
5. Write output as JSONL (and optionally Parquet).

Usage
-----
::

    from src.export import DatasetBuilder, DatasetConfig

    cfg = DatasetConfig(train_ratio=0.9, val_ratio=0.05)
    builder = DatasetBuilder(cfg)
    stats = builder.build(
        queries_path="data/queries_normalized.jsonl",
        elements_path="data/multimodal_elements.json",
        output_dir="data/contrastive_data/v1",
    )
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
    """Configuration for :class:`DatasetBuilder`."""

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
    """Load ``multimodal_elements.json`` into a ``{element_id: record}`` map."""
    if not elements_path.exists():
        logger.warning("Elements file not found: %s", elements_path)
        return {}
    with open(elements_path, encoding="utf-8") as fh:
        data = json.load(fh)

    idx: Dict[str, Dict[str, Any]] = {}
    elements = data if isinstance(data, list) else data.get("elements", [])
    for el in elements:
        eid = el.get("element_id", "")
        if eid:
            idx[eid] = el
    logger.info("Element index: %d entries", len(idx))
    return idx


def _element_to_chunk(el: Dict[str, Any]) -> Chunk:
    """Convert a raw element dict to a :class:`Chunk`."""
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
    """Build a retrieval-friendly text representation of an element."""
    parts: list[str] = []
    for key in ("enriched_content", "caption", "content", "context_before"):
        val = (el.get(key) or "").strip()
        if val:
            parts.append(val)
    return " ".join(parts) if parts else el.get("element_id", "")


# ── Core builder ──────────────────────────────────────────────────────────────

class DatasetBuilder:
    """Orchestrates: queries → triplets → train/val/test → disk."""

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
        """Run the full build pipeline.  Returns summary statistics."""
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
                    logger.warning("Skipping line %d: %s", lineno, exc)
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
        """Split by doc_id (from positive evidence) to prevent data leakage."""

        def _doc_key(t: Triplet) -> str:
            if t.positive:
                eid = t.positive[0].element_id
                # convention: element_id = "docid_type_N"
                parts = eid.rsplit("_", 2)
                if len(parts) >= 2:
                    return parts[0]
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
