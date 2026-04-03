"""Training dataset export layer.

Provides :class:`DatasetBuilder` — the bridge between the query generation
pipeline and model training.  It reads normalised ``StandardQuery`` records,
builds contrastive triplets via pluggable negative-sampling strategies,
performs a doc-level train/val/test split, and writes the result to JSONL
(and optionally Parquet / HuggingFace Dataset format).
"""

from src.export.dataset_builder import DatasetBuilder, DatasetConfig

__all__ = ["DatasetBuilder", "DatasetConfig"]
