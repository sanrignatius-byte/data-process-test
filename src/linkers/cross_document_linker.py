"""
Cross-Document Linker for M4 Query Generation

This module implements entity extraction, cross-document linking, and evidence chain
construction based on research from:
- M4DocBench (Dong et al., 2025): Multi-hop, multi-modal document research benchmark
- TRACE (EMNLP 2024): Knowledge-grounded reasoning chains construction
- SciCo-Radar: Cross-document coreference with dynamic definitions
- HiRAG: Hierarchical retrieval for multi-hop QA

Key concepts:
1. Entity Extraction: Extract key entities, concepts, methods from passages
2. Cross-Document Linking: Build entity co-occurrence graph across documents
3. Evidence Chain: Construct reasoning paths connecting multiple evidence sources
4. Bridge Logic: Ensure multi-hop queries have valid reasoning steps
"""

import re
import json
import hashlib
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum


class EntityType(Enum):
    """Types of entities that can be extracted from scientific documents."""
    METHOD = "method"           # Algorithm, approach, technique
    DATASET = "dataset"         # Benchmark, corpus, dataset
    METRIC = "metric"           # Evaluation metric (F1, accuracy, BLEU)
    MODEL = "model"             # Neural network, architecture
    TASK = "task"               # NLP/CV task (QA, NER, classification)
    CONCEPT = "concept"         # Scientific concept, theory
    RESULT = "result"           # Experimental finding, number
    COMPONENT = "component"     # Module, layer, block
    FORMULA = "formula"         # Mathematical expression
    FIGURE_REF = "figure_ref"   # Reference to figure/table


@dataclass
class EntityMention:
    """A mention of an entity in a specific passage."""
    entity_id: str
    mention_text: str
    entity_type: EntityType
    passage_id: str
    doc_id: str
    page_idx: Optional[int] = None
    char_offset: Optional[Tuple[int, int]] = None  # (start, end) in passage
    context: Optional[str] = None  # Surrounding text
    confidence: float = 1.0


@dataclass
class DocumentEntity:
    """An entity with all its mentions across a document."""
    entity_id: str
    canonical_name: str
    entity_type: EntityType
    doc_id: str
    mentions: List[EntityMention] = field(default_factory=list)
    aliases: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CrossDocumentLink:
    """A link between entities across different documents."""
    link_id: str
    entity_a: DocumentEntity
    entity_b: DocumentEntity
    link_type: str  # "same_entity", "related", "contrasted", "extends"
    confidence: float
    evidence: List[str] = field(default_factory=list)  # Supporting text snippets


@dataclass
class EvidenceNode:
    """A node in the evidence chain."""
    node_id: str
    passage_id: str
    doc_id: str
    modal_type: str
    content_snippet: str
    entities: List[str]  # Entity IDs mentioned
    page_idx: Optional[int] = None
    bbox: Optional[List[float]] = None


@dataclass
class EvidenceChain:
    """A chain of evidence connecting multiple sources for multi-hop reasoning."""
    chain_id: str
    nodes: List[EvidenceNode]
    reasoning_steps: List[str]
    bridge_entities: List[str]  # Entities that connect the chain
    query_type: str  # "multi_hop", "multi_doc", "multi_modal"
    modalities_involved: Set[str] = field(default_factory=set)
    docs_involved: Set[str] = field(default_factory=set)

    def is_valid_multi_hop(self) -> bool:
        """Check if chain satisfies multi-hop requirements."""
        return len(self.nodes) >= 2 and len(self.reasoning_steps) >= 2

    def is_valid_multi_doc(self) -> bool:
        """Check if chain spans multiple documents."""
        return len(self.docs_involved) >= 2

    def is_valid_multi_modal(self) -> bool:
        """Check if chain involves multiple modalities."""
        return len(self.modalities_involved) >= 2


class CrossDocumentLinker:
    """
    Cross-document entity linking and evidence chain construction.

    Based on research showing that effective multi-hop QA requires:
    1. Explicit entity extraction and linking
    2. Graph-based reasoning over document relationships
    3. Validated evidence chains with clear reasoning steps
    """

    # Patterns for extracting different entity types
    ENTITY_PATTERNS = {
        EntityType.METHOD: [
            r'\b([A-Z][a-zA-Z]*(?:Net|BERT|GPT|LLM|Transformer|GAN|VAE|RNN|LSTM|CNN|GNN))\b',
            r'\b([A-Z]{2,}(?:-[A-Z]+)*)\b',  # Acronyms like BERT, GPT-4
            r'\bour (?:proposed )?(?:method|approach|model|framework|system)\b',
            r'\b(?:we )?(?:propose|introduce|present) ([^.]+?)(?:\.|,)',
        ],
        EntityType.DATASET: [
            r'\b([A-Z][a-zA-Z]*(?:QA|NLI|RE|NER|Bench|Set|Corpus))\b',
            r'\b(SQuAD|GLUE|SuperGLUE|ImageNet|COCO|MNIST|CIFAR)\b',
            r'\bdataset (?:called|named) ([^.]+?)(?:\.|,)',
        ],
        EntityType.METRIC: [
            r'\b(F1(?:-score)?|accuracy|precision|recall|BLEU|ROUGE|perplexity)\b',
            r'\b(AUC|mAP|IoU|EM|MRR|NDCG)\b',
            r'(\d+\.?\d*%)',  # Percentage values
        ],
        EntityType.TASK: [
            r'\b(question answering|QA|NER|named entity recognition)\b',
            r'\b(text classification|sentiment analysis|machine translation)\b',
            r'\b(object detection|image classification|semantic segmentation)\b',
        ],
        EntityType.CONCEPT: [
            r'\b(attention mechanism|self-attention|cross-attention)\b',
            r'\b(embedding|representation|encoding|latent space)\b',
            r'\b(fine-tuning|pre-training|transfer learning)\b',
        ],
        EntityType.RESULT: [
            r'(?:achieve|obtain|reach)s? (?:an? )?(?:accuracy|F1|score) of (\d+\.?\d*%?)',
            r'(\d+\.?\d*%?) (?:accuracy|F1|improvement)',
            r'outperforms? .+ by (\d+\.?\d*%?)',
        ],
        EntityType.FIGURE_REF: [
            r'(?:Figure|Fig\.?|Table|Tab\.?) ?(\d+[a-z]?)',
            r'(?:Equation|Eq\.?) ?(?:\()?(\d+)(?:\))?',
        ],
    }

    def __init__(
        self,
        similarity_threshold: float = 0.7,
        min_entity_frequency: int = 1,
        use_llm_extraction: bool = False,
        llm_client: Optional[Any] = None
    ):
        """
        Initialize the cross-document linker.

        Args:
            similarity_threshold: Threshold for entity matching (0.0-1.0)
            min_entity_frequency: Minimum mentions for an entity to be considered
            use_llm_extraction: Whether to use LLM for entity extraction
            llm_client: LLM client for advanced extraction
        """
        self.similarity_threshold = similarity_threshold
        self.min_entity_frequency = min_entity_frequency
        self.use_llm_extraction = use_llm_extraction
        self.llm_client = llm_client

        # Storage
        self.entities_by_doc: Dict[str, List[DocumentEntity]] = defaultdict(list)
        self.cross_doc_links: List[CrossDocumentLink] = []
        self.entity_index: Dict[str, DocumentEntity] = {}  # entity_id -> entity

    def extract_entities_from_passage(
        self,
        passage: Any,
        doc_id: str
    ) -> List[EntityMention]:
        """
        Extract entity mentions from a single passage.

        Uses pattern matching for scientific entities common in research papers.
        """
        mentions = []
        content = passage.content if hasattr(passage, 'content') else str(passage)
        passage_id = passage.passage_id if hasattr(passage, 'passage_id') else "unknown"
        page_idx = passage.page_idx if hasattr(passage, 'page_idx') else None

        for entity_type, patterns in self.ENTITY_PATTERNS.items():
            for pattern in patterns:
                try:
                    for match in re.finditer(pattern, content, re.IGNORECASE):
                        mention_text = match.group(1) if match.groups() else match.group(0)
                        mention_text = mention_text.strip()

                        if len(mention_text) < 2 or len(mention_text) > 100:
                            continue

                        # Generate entity ID from normalized text
                        normalized = self._normalize_entity_name(mention_text)
                        entity_id = hashlib.md5(
                            f"{doc_id}:{entity_type.value}:{normalized}".encode()
                        ).hexdigest()[:12]

                        # Get context window
                        start = max(0, match.start() - 50)
                        end = min(len(content), match.end() + 50)
                        context = content[start:end]

                        mentions.append(EntityMention(
                            entity_id=entity_id,
                            mention_text=mention_text,
                            entity_type=entity_type,
                            passage_id=passage_id,
                            doc_id=doc_id,
                            page_idx=page_idx,
                            char_offset=(match.start(), match.end()),
                            context=context,
                            confidence=0.8 if entity_type in [EntityType.METHOD, EntityType.DATASET] else 0.6
                        ))
                except re.error:
                    continue

        return mentions

    def _normalize_entity_name(self, name: str) -> str:
        """Normalize entity name for matching."""
        # Lowercase, remove extra spaces, basic normalization
        normalized = name.lower().strip()
        normalized = re.sub(r'\s+', ' ', normalized)
        normalized = re.sub(r'[^\w\s-]', '', normalized)
        return normalized

    def build_document_entities(
        self,
        passages: List[Any],
        doc_id: str
    ) -> List[DocumentEntity]:
        """
        Build document-level entities from all passages.

        Groups mentions into entities and resolves aliases.
        """
        # Extract all mentions
        all_mentions = []
        for passage in passages:
            mentions = self.extract_entities_from_passage(passage, doc_id)
            all_mentions.extend(mentions)

        # Group mentions by normalized name and type
        mention_groups: Dict[str, List[EntityMention]] = defaultdict(list)
        for mention in all_mentions:
            key = f"{mention.entity_type.value}:{self._normalize_entity_name(mention.mention_text)}"
            mention_groups[key].append(mention)

        # Create DocumentEntity for each group
        doc_entities = []
        for key, mentions in mention_groups.items():
            if len(mentions) < self.min_entity_frequency:
                continue

            # Use most common mention text as canonical name
            text_counts = defaultdict(int)
            for m in mentions:
                text_counts[m.mention_text] += 1
            canonical_name = max(text_counts.keys(), key=lambda x: text_counts[x])

            # Collect aliases
            aliases = set(m.mention_text for m in mentions if m.mention_text != canonical_name)

            entity = DocumentEntity(
                entity_id=mentions[0].entity_id,
                canonical_name=canonical_name,
                entity_type=mentions[0].entity_type,
                doc_id=doc_id,
                mentions=mentions,
                aliases=aliases,
                metadata={
                    "frequency": len(mentions),
                    "pages": list(set(m.page_idx for m in mentions if m.page_idx is not None))
                }
            )
            doc_entities.append(entity)
            self.entity_index[entity.entity_id] = entity

        self.entities_by_doc[doc_id] = doc_entities
        return doc_entities

    def find_cross_document_links(
        self,
        doc_ids: Optional[List[str]] = None
    ) -> List[CrossDocumentLink]:
        """
        Find links between entities across different documents.

        Uses normalized name matching and context similarity.
        """
        doc_ids = doc_ids or list(self.entities_by_doc.keys())

        if len(doc_ids) < 2:
            return []

        links = []

        # Compare entities across document pairs
        for i, doc_a in enumerate(doc_ids):
            for doc_b in doc_ids[i+1:]:
                entities_a = self.entities_by_doc.get(doc_a, [])
                entities_b = self.entities_by_doc.get(doc_b, [])

                for ent_a in entities_a:
                    for ent_b in entities_b:
                        # Must be same type
                        if ent_a.entity_type != ent_b.entity_type:
                            continue

                        # Check name similarity
                        similarity = self._compute_entity_similarity(ent_a, ent_b)

                        if similarity >= self.similarity_threshold:
                            link_id = hashlib.md5(
                                f"{ent_a.entity_id}:{ent_b.entity_id}".encode()
                            ).hexdigest()[:12]

                            link = CrossDocumentLink(
                                link_id=link_id,
                                entity_a=ent_a,
                                entity_b=ent_b,
                                link_type="same_entity" if similarity > 0.9 else "related",
                                confidence=similarity,
                                evidence=[
                                    ent_a.mentions[0].context if ent_a.mentions else "",
                                    ent_b.mentions[0].context if ent_b.mentions else ""
                                ]
                            )
                            links.append(link)

        self.cross_doc_links = links
        return links

    def _compute_entity_similarity(
        self,
        entity_a: DocumentEntity,
        entity_b: DocumentEntity
    ) -> float:
        """Compute similarity between two entities."""
        # Exact match
        name_a = self._normalize_entity_name(entity_a.canonical_name)
        name_b = self._normalize_entity_name(entity_b.canonical_name)

        if name_a == name_b:
            return 1.0

        # Check aliases
        all_names_a = {name_a} | {self._normalize_entity_name(a) for a in entity_a.aliases}
        all_names_b = {name_b} | {self._normalize_entity_name(a) for a in entity_b.aliases}

        if all_names_a & all_names_b:
            return 0.95

        # Jaccard similarity on character n-grams
        def ngrams(text, n=3):
            return set(text[i:i+n] for i in range(len(text) - n + 1))

        ngrams_a = ngrams(name_a)
        ngrams_b = ngrams(name_b)

        if not ngrams_a or not ngrams_b:
            return 0.0

        intersection = len(ngrams_a & ngrams_b)
        union = len(ngrams_a | ngrams_b)

        return intersection / union if union > 0 else 0.0

    def build_evidence_chain(
        self,
        passages: List[Any],
        bridge_entities: List[str],
        query_type: str = "multi_hop"
    ) -> Optional[EvidenceChain]:
        """
        Build an evidence chain connecting multiple passages through bridge entities.

        This implements the Bridge Logic from M4 research:
        - Each hop must be connected by a shared entity
        - Reasoning steps must be explicit and logical

        Args:
            passages: List of passages to form the chain
            bridge_entities: Entity IDs that connect the passages
            query_type: Type of query this chain supports

        Returns:
            EvidenceChain if valid chain can be built, None otherwise
        """
        if len(passages) < 2:
            return None

        nodes = []
        modalities = set()
        docs = set()

        for i, passage in enumerate(passages):
            doc_id = passage.doc_id if hasattr(passage, 'doc_id') else "unknown"
            modal_type = passage.modal_type.value if hasattr(passage.modal_type, 'value') else str(getattr(passage, 'modal_type', 'text'))

            # Find entities in this passage
            passage_entities = []
            for ent_id in bridge_entities:
                if ent_id in self.entity_index:
                    entity = self.entity_index[ent_id]
                    for mention in entity.mentions:
                        if mention.passage_id == passage.passage_id:
                            passage_entities.append(ent_id)
                            break

            node = EvidenceNode(
                node_id=f"node_{i}_{passage.passage_id[:8]}",
                passage_id=passage.passage_id,
                doc_id=doc_id,
                modal_type=modal_type,
                content_snippet=passage.content[:300] if hasattr(passage, 'content') else "",
                entities=passage_entities,
                page_idx=getattr(passage, 'page_idx', None),
                bbox=getattr(passage, 'bbox', None)
            )
            nodes.append(node)
            modalities.add(modal_type)
            docs.add(doc_id)

        # Generate reasoning steps based on entity connections
        reasoning_steps = self._generate_reasoning_steps(nodes, bridge_entities)

        if len(reasoning_steps) < 2:
            return None

        chain_id = hashlib.md5(
            ":".join(n.passage_id for n in nodes).encode()
        ).hexdigest()[:12]

        chain = EvidenceChain(
            chain_id=chain_id,
            nodes=nodes,
            reasoning_steps=reasoning_steps,
            bridge_entities=bridge_entities,
            query_type=query_type,
            modalities_involved=modalities,
            docs_involved=docs
        )

        return chain

    def _generate_reasoning_steps(
        self,
        nodes: List[EvidenceNode],
        bridge_entities: List[str]
    ) -> List[str]:
        """Generate reasoning steps that connect evidence nodes."""
        steps = []

        def _entity_concepts(entity_ids: List[str]) -> Set[str]:
            """Map entity IDs to normalized concept names for cross-doc bridging.

            Different documents usually assign different entity_ids even when they
            refer to the same concept (e.g., "BERT" in doc A vs doc B). Using
            normalized canonical names preserves bridge semantics in reasoning steps.
            """
            concepts: Set[str] = set()
            for ent_id in entity_ids:
                entity = self.entity_index.get(ent_id)
                if not entity:
                    continue
                concepts.add(self._normalize_entity_name(entity.canonical_name))
            return concepts

        for i in range(len(nodes) - 1):
            node_a = nodes[i]
            node_b = nodes[i + 1]

            # Find shared bridge concepts between consecutive nodes.
            # NOTE: using raw entity IDs would miss cross-document links because
            # IDs are doc-scoped. We compare normalized concept names instead.
            shared = _entity_concepts(node_a.entities) & _entity_concepts(node_b.entities)

            if shared:
                entity_names = sorted(shared)

                step = f"From {node_a.modal_type} evidence about {', '.join(entity_names[:2])}, " \
                       f"connect to {node_b.modal_type} evidence in {'same' if node_a.doc_id == node_b.doc_id else 'different'} document"
                steps.append(step)
            else:
                # Generic step for unconnected nodes
                step = f"Bridge {node_a.modal_type} (doc: {node_a.doc_id[:8]}) " \
                       f"to {node_b.modal_type} (doc: {node_b.doc_id[:8]})"
                steps.append(step)

        # Add final synthesis step
        if len(nodes) >= 2:
            steps.append(f"Synthesize evidence from {len(nodes)} sources across {len(set(n.doc_id for n in nodes))} documents")

        return steps

    def find_linkable_passage_groups(
        self,
        passages: List[Any],
        min_group_size: int = 2,
        max_group_size: int = 4,
        require_multi_doc: bool = True,
        require_multi_modal: bool = True
    ) -> List[Tuple[List[Any], List[str]]]:
        """
        Find groups of passages that can form valid evidence chains.

        This is the key method for selecting passages for M4 query generation.

        Args:
            passages: All available passages
            min_group_size: Minimum passages in a group
            max_group_size: Maximum passages in a group
            require_multi_doc: Require passages from multiple documents
            require_multi_modal: Require multiple modalities

        Returns:
            List of (passage_group, bridge_entity_ids) tuples
        """
        # First, ensure entities are extracted for all documents
        passages_by_doc = defaultdict(list)
        for p in passages:
            doc_id = p.doc_id if hasattr(p, 'doc_id') else "unknown"
            passages_by_doc[doc_id].append(p)

        for doc_id, doc_passages in passages_by_doc.items():
            if doc_id not in self.entities_by_doc:
                self.build_document_entities(doc_passages, doc_id)

        # Find cross-document links if not already done
        if not self.cross_doc_links:
            self.find_cross_document_links()

        # Build passage groups based on shared entities
        groups = []

        for link in self.cross_doc_links:
            # Get passages that mention entities in this link
            passages_a = self._get_passages_with_entity(link.entity_a, passages)
            passages_b = self._get_passages_with_entity(link.entity_b, passages)

            if not passages_a or not passages_b:
                continue

            # Create groups combining passages from both sides
            for p_a in passages_a[:2]:  # Limit to avoid explosion
                for p_b in passages_b[:2]:
                    group = [p_a, p_b]
                    bridge_entities = [link.entity_a.entity_id, link.entity_b.entity_id]

                    # Check constraints
                    docs = set(p.doc_id for p in group)
                    modals = set(
                        p.modal_type.value if hasattr(p.modal_type, 'value')
                        else str(p.modal_type) for p in group
                    )

                    if require_multi_doc and len(docs) < 2:
                        continue
                    if require_multi_modal and len(modals) < 2:
                        continue

                    groups.append((group, bridge_entities))

        return groups[:50]  # Limit total groups

    def _get_passages_with_entity(
        self,
        entity: DocumentEntity,
        passages: List[Any]
    ) -> List[Any]:
        """Get passages that contain mentions of an entity."""
        passage_ids = set(m.passage_id for m in entity.mentions)
        return [p for p in passages if p.passage_id in passage_ids]

    def get_entity_statistics(self) -> Dict[str, Any]:
        """Get statistics about extracted entities and links."""
        total_entities = sum(len(ents) for ents in self.entities_by_doc.values())

        type_counts = defaultdict(int)
        for entities in self.entities_by_doc.values():
            for ent in entities:
                type_counts[ent.entity_type.value] += 1

        link_type_counts = defaultdict(int)
        for link in self.cross_doc_links:
            link_type_counts[link.link_type] += 1

        return {
            "total_documents": len(self.entities_by_doc),
            "total_entities": total_entities,
            "entities_by_type": dict(type_counts),
            "total_cross_doc_links": len(self.cross_doc_links),
            "links_by_type": dict(link_type_counts),
            "avg_entities_per_doc": total_entities / len(self.entities_by_doc) if self.entities_by_doc else 0
        }
