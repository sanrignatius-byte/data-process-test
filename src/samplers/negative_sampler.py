"""
Negative Sampler for Contrastive Learning

Implements strategic negative sampling strategies for high-quality
contrastive learning triplets.
"""

import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

# Optional embedding support
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False


@dataclass
class ContrastiveTriplet:
    """
    A contrastive learning triplet: (query, positive, negatives).
    """
    triplet_id: str
    query: Dict[str, Any]  # Query information
    positive: Dict[str, Any]  # Positive passage
    negatives: List[Dict[str, Any]]  # Negative passages
    difficulty_score: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "triplet_id": self.triplet_id,
            "query": self.query,
            "positive": self.positive,
            "negatives": self.negatives,
            "difficulty_score": self.difficulty_score,
            "metadata": self.metadata
        }

    def to_training_format(self) -> Dict[str, Any]:
        """Convert to standard training format."""
        return {
            "query": self.query.get("query_text", self.query.get("text", "")),
            "query_type": self.query.get("query_type", "unknown"),
            "positive": {
                "text": self.positive.get("content", ""),
                "modal_type": self.positive.get("modal_type", "text"),
                "image_path": self.positive.get("image_path"),
                "metadata": self.positive.get("metadata", {})
            },
            "negatives": [
                {
                    "text": neg.get("content", ""),
                    "modal_type": neg.get("modal_type", "text"),
                    "image_path": neg.get("image_path"),
                    "negative_type": neg.get("negative_type", "random"),
                    "metadata": neg.get("metadata", {})
                }
                for neg in self.negatives
            ],
            "difficulty_score": self.difficulty_score
        }


class NegativeSampler:
    """
    Base negative sampler with random sampling strategy.
    """

    def __init__(
        self,
        num_negatives: int = 3,
        seed: Optional[int] = None
    ):
        """
        Initialize sampler.

        Args:
            num_negatives: Number of negatives per query
            seed: Random seed for reproducibility
        """
        self.num_negatives = num_negatives
        self.random = random.Random(seed)

    def sample(
        self,
        query: Any,
        positive: Any,
        candidate_pool: List[Any]
    ) -> List[Any]:
        """
        Sample negative passages.

        Args:
            query: Query object
            positive: Positive passage
            candidate_pool: Pool of candidate negatives

        Returns:
            List of negative passages
        """
        # Filter out the positive
        candidates = [c for c in candidate_pool if c.passage_id != positive.passage_id]

        if not candidates:
            return []

        # Random sampling
        k = min(self.num_negatives, len(candidates))
        negatives = self.random.sample(candidates, k)

        # Add negative type annotation
        for neg in negatives:
            neg.metadata["negative_type"] = "random"

        return negatives


class HardNegativeSampler(NegativeSampler):
    """
    Hard negative sampler with multiple strategies:
    - Same modality (hardest)
    - Cross-modal confusion
    - Semantic similarity based (requires embeddings)
    - Random (baseline)
    """

    def __init__(
        self,
        num_negatives: int = 3,
        strategy: str = "modal_mixed",
        distribution: Dict[str, float] = None,
        use_embeddings: bool = False,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        seed: Optional[int] = None
    ):
        """
        Initialize hard negative sampler.

        Args:
            num_negatives: Number of negatives per query
            strategy: Sampling strategy ("random", "modal_same", "modal_mixed", "semantic_hard")
            distribution: Distribution for modal_mixed strategy
            use_embeddings: Whether to use embeddings for hard negatives
            embedding_model: Sentence transformer model name
            seed: Random seed
        """
        super().__init__(num_negatives, seed)

        self.strategy = strategy
        self.distribution = distribution or {
            "hard_same_modal": 0.6,
            "cross_modal": 0.3,
            "random": 0.1
        }
        self.use_embeddings = use_embeddings and EMBEDDING_AVAILABLE

        # Initialize embedding model if needed
        self._embedder = None
        self._embedding_cache: Dict[str, np.ndarray] = {}
        if self.use_embeddings:
            try:
                self._embedder = SentenceTransformer(embedding_model)
            except Exception as e:
                print(f"Failed to load embedding model: {e}")
                self.use_embeddings = False

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get or compute embedding for text."""
        if not self.use_embeddings or not self._embedder:
            return None

        if text not in self._embedding_cache:
            try:
                embedding = self._embedder.encode(text[:512])  # Truncate long text
                self._embedding_cache[text] = embedding
            except Exception:
                return None

        return self._embedding_cache.get(text)

    def _compute_similarity(
        self,
        query_text: str,
        candidate: Any
    ) -> float:
        """Compute semantic similarity between query and candidate."""
        if not self.use_embeddings:
            return 0.0

        query_emb = self._get_embedding(query_text)
        cand_emb = self._get_embedding(candidate.content[:512])

        if query_emb is None or cand_emb is None:
            return 0.0

        # Cosine similarity
        similarity = np.dot(query_emb, cand_emb) / (
            np.linalg.norm(query_emb) * np.linalg.norm(cand_emb) + 1e-8
        )
        return float(similarity)

    def _group_by_modality(
        self,
        candidates: List[Any]
    ) -> Dict[str, List[Any]]:
        """Group candidates by modality."""
        groups = defaultdict(list)
        for cand in candidates:
            modal = cand.modal_type.value if hasattr(cand.modal_type, 'value') else cand.modal_type
            groups[modal].append(cand)
        return dict(groups)

    def sample(
        self,
        query: Any,
        positive: Any,
        candidate_pool: List[Any]
    ) -> List[Any]:
        """
        Sample hard negatives using configured strategy.

        Args:
            query: Query object (GeneratedQuery or dict)
            positive: Positive passage
            candidate_pool: Pool of candidate negatives

        Returns:
            List of negative passages with type annotations
        """
        # Filter out positive
        candidates = [c for c in candidate_pool if c.passage_id != positive.passage_id]

        if not candidates:
            return []

        # Get positive modality
        pos_modal = positive.modal_type.value if hasattr(positive.modal_type, 'value') else positive.modal_type

        # Group by modality
        by_modality = self._group_by_modality(candidates)

        negatives = []

        if self.strategy == "random":
            negatives = self._sample_random(candidates, self.num_negatives)

        elif self.strategy == "modal_same":
            negatives = self._sample_same_modal(by_modality, pos_modal, self.num_negatives)

        elif self.strategy == "modal_mixed":
            negatives = self._sample_mixed(
                by_modality, pos_modal, candidates, self.num_negatives
            )

        elif self.strategy == "semantic_hard":
            query_text = query.query_text if hasattr(query, 'query_text') else query.get("query_text", "")
            negatives = self._sample_semantic_hard(
                query_text, candidates, self.num_negatives
            )

        return negatives

    def _sample_random(
        self,
        candidates: List[Any],
        k: int
    ) -> List[Any]:
        """Random sampling."""
        k = min(k, len(candidates))
        selected = self.random.sample(candidates, k)
        for neg in selected:
            neg.metadata["negative_type"] = "random"
        return selected

    def _sample_same_modal(
        self,
        by_modality: Dict[str, List[Any]],
        target_modal: str,
        k: int
    ) -> List[Any]:
        """Sample from same modality (hardest negatives)."""
        same_modal = by_modality.get(target_modal, [])

        if not same_modal:
            # Fallback to random
            all_cands = [c for cands in by_modality.values() for c in cands]
            return self._sample_random(all_cands, k)

        k = min(k, len(same_modal))
        selected = self.random.sample(same_modal, k)
        for neg in selected:
            neg.metadata["negative_type"] = "hard_same_modal"
        return selected

    def _sample_mixed(
        self,
        by_modality: Dict[str, List[Any]],
        pos_modal: str,
        all_candidates: List[Any],
        k: int
    ) -> List[Any]:
        """
        Mixed sampling strategy:
        - 60% same modality (hard)
        - 30% different modality (cross-modal confusion)
        - 10% random
        """
        negatives = []

        # Calculate counts based on distribution
        n_hard = int(k * self.distribution.get("hard_same_modal", 0.6))
        n_cross = int(k * self.distribution.get("cross_modal", 0.3))
        n_random = k - n_hard - n_cross

        # Same modality negatives
        same_modal = by_modality.get(pos_modal, [])
        if same_modal:
            n_same = min(n_hard, len(same_modal))
            selected = self.random.sample(same_modal, n_same)
            for neg in selected:
                neg.metadata["negative_type"] = "hard_same_modal"
            negatives.extend(selected)

        # Cross-modal negatives
        other_modals = [c for modal, cands in by_modality.items()
                       if modal != pos_modal for c in cands]
        if other_modals:
            n_other = min(n_cross, len(other_modals))
            selected = self.random.sample(other_modals, n_other)
            for neg in selected:
                neg.metadata["negative_type"] = "cross_modal"
            negatives.extend(selected)

        # Random negatives (fill remaining)
        remaining = k - len(negatives)
        if remaining > 0:
            used_ids = {n.passage_id for n in negatives}
            available = [c for c in all_candidates if c.passage_id not in used_ids]
            if available:
                n_rand = min(remaining, len(available))
                selected = self.random.sample(available, n_rand)
                for neg in selected:
                    neg.metadata["negative_type"] = "random"
                negatives.extend(selected)

        return negatives

    def _sample_semantic_hard(
        self,
        query_text: str,
        candidates: List[Any],
        k: int
    ) -> List[Any]:
        """Sample hard negatives based on semantic similarity."""
        if not self.use_embeddings:
            return self._sample_random(candidates, k)

        # Compute similarities
        scored = []
        for cand in candidates:
            sim = self._compute_similarity(query_text, cand)
            scored.append((cand, sim))

        # Sort by similarity (descending) - higher similarity = harder negative
        scored.sort(key=lambda x: x[1], reverse=True)

        # Take top-k most similar (hardest)
        negatives = []
        for cand, sim in scored[:k]:
            cand.metadata["negative_type"] = "semantic_hard"
            cand.metadata["similarity_score"] = sim
            negatives.append(cand)

        return negatives

    def construct_triplets(
        self,
        query_data: Dict[str, List[Any]],
        passage_pool: List[Any],
        passages_by_id: Dict[str, Any]
    ) -> List[ContrastiveTriplet]:
        """
        Construct contrastive triplets from queries and passages.

        Args:
            query_data: Dict mapping passage_id to list of queries
            passage_pool: Pool of all passages for negative sampling
            passages_by_id: Dict mapping passage_id to passage object

        Returns:
            List of ContrastiveTriplet objects
        """
        triplets = []

        for passage_id, queries in query_data.items():
            positive = passages_by_id.get(passage_id)
            if not positive:
                continue

            for query in queries:
                # Sample negatives
                negatives = self.sample(query, positive, passage_pool)

                if not negatives:
                    continue

                # Create triplet
                triplet_id = f"{query.query_id}_triplet"

                triplet = ContrastiveTriplet(
                    triplet_id=triplet_id,
                    query={
                        "query_id": query.query_id,
                        "query_text": query.query_text,
                        "query_type": query.query_type,
                        "target_modality": query.target_modality,
                        "difficulty": query.difficulty
                    },
                    positive=positive.to_dict(),
                    negatives=[neg.to_dict() for neg in negatives],
                    difficulty_score=self._estimate_triplet_difficulty(query, positive, negatives),
                    metadata={
                        "doc_id": positive.doc_id,
                        "strategy": self.strategy,
                        "num_negatives": len(negatives)
                    }
                )

                triplets.append(triplet)

        return triplets

    def _estimate_triplet_difficulty(
        self,
        query: Any,
        positive: Any,
        negatives: List[Any]
    ) -> float:
        """Estimate overall triplet difficulty."""
        difficulty = query.difficulty if hasattr(query, 'difficulty') else 0.5

        # Higher proportion of hard negatives = higher difficulty
        hard_count = sum(
            1 for neg in negatives
            if neg.metadata.get("negative_type") in ["hard_same_modal", "semantic_hard"]
        )
        if negatives:
            difficulty += 0.2 * (hard_count / len(negatives))

        # Positive quality affects difficulty
        if hasattr(positive, 'quality_score'):
            # Lower quality positive = harder task
            difficulty += 0.1 * (1 - positive.quality_score)

        return min(1.0, max(0.0, difficulty))

    def get_sampling_stats(self) -> Dict[str, Any]:
        """Get statistics about sampling."""
        return {
            "strategy": self.strategy,
            "num_negatives": self.num_negatives,
            "distribution": self.distribution,
            "embeddings_enabled": self.use_embeddings,
            "embedding_cache_size": len(self._embedding_cache)
        }

    def clear_embedding_cache(self) -> None:
        """Clear embedding cache to free memory."""
        self._embedding_cache.clear()
