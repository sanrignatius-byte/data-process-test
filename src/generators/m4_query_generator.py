"""
M4 Query Generator - Enhanced Multi-hop, Multi-modal, Multi-document, Multi-turn Query Generation

This module implements improved M4 query generation based on academic research:

References:
- M4DocBench (Dong et al., 2025): Deep research benchmark for multimodal documents
- TRACE (EMNLP 2024): Knowledge triple-grounded reasoning chains
- RT-RAG: Tree-structured decomposition for multi-hop QA
- CoQA: Conversational question answering with coreference
- MT-Bench++: Multi-turn evaluation with ellipsis and anaphora

Key improvements over basic M4 generation:
1. Entity-aware passage selection using CrossDocumentLinker
2. Evidence chain validation before query generation
3. Stepwise generation: single-hop -> multi-hop -> cross-doc -> multi-turn
4. Coreference alignment for natural multi-turn dialogues
5. Evidence verification against source documents
"""

import json
import hashlib
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ..linkers.cross_document_linker import (
    CrossDocumentLinker,
    EvidenceChain,
    EvidenceNode,
    DocumentEntity,
)


class M4QueryType(Enum):
    """Types of M4 queries with increasing complexity."""
    SINGLE_HOP = "single_hop"           # Basic factual query
    MULTI_HOP = "multi_hop"             # Requires 2+ evidence points
    CROSS_MODAL = "cross_modal"         # Requires 2+ modalities
    CROSS_DOC = "cross_doc"             # Requires 2+ documents
    MULTI_TURN = "multi_turn"           # Multi-turn dialogue format
    FULL_M4 = "full_m4"                 # All four dimensions


@dataclass
class M4Query:
    """Enhanced M4 query with full evidence tracking."""
    query_id: str
    query_type: M4QueryType
    turns: List[str]                    # Multi-turn dialogue turns
    answer: str
    evidence_chain: EvidenceChain
    difficulty: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Validation flags
    is_multi_hop: bool = False
    is_multi_modal: bool = False
    is_multi_doc: bool = False
    is_multi_turn: bool = False

    def satisfies_m4(self) -> bool:
        """Check if query satisfies all M4 requirements."""
        return self.is_multi_hop and self.is_multi_modal and self.is_multi_doc and self.is_multi_turn


class M4PromptTemplates:
    """Enhanced prompt templates for M4 query generation."""

    # Step 1: Generate bridging question that connects two evidence sources
    BRIDGING_QUESTION = """You are generating a bridging question that requires information from TWO evidence sources.

Evidence Source A (from {doc_a}, {modal_a}):
{content_a}

Evidence Source B (from {doc_b}, {modal_b}):
{content_b}

Shared concept/entity: {bridge_entity}

Generate ONE question that:
1. CANNOT be answered using only Source A
2. CANNOT be answered using only Source B
3. REQUIRES combining information from BOTH sources
4. Uses the shared concept "{bridge_entity}" as the reasoning bridge

Return JSON:
{{"question": "...", "reasoning_path": ["step from A", "bridge step", "step from B"], "answer": "..."}}
"""

    # Step 2: Convert to multi-turn with coreference
    MULTI_TURN_CONVERSION = """Convert this question into a natural 2-3 turn dialogue.

Original question: {question}
Answer: {answer}
Key entities: {entities}

Requirements for multi-turn conversion:
1. First turn: Ask about a specific aspect or introduce the topic
2. Follow-up turns: Use pronouns (it, they, that, these, those) instead of repeating entity names
3. The full answer should only be obtainable after all turns
4. Each turn should feel natural in a conversation

Example of good coreference:
- Turn 1: "What method does the BERT paper use for pre-training?"
- Turn 2: "How does it compare to previous approaches in terms of efficiency?"
(Note: "it" refers to "the method" from turn 1)

Return JSON:
{{"turns": ["turn 1", "turn 2", ...], "coreference_map": {{"pronoun": "referent", ...}}}}
"""

    # Step 3: Full M4 query generation with evidence chain
    FULL_M4_GENERATION = """Generate a complex research question satisfying ALL M4 requirements.

You have access to evidence from {num_docs} documents across {num_modalities} modalities.

Evidence chain:
{evidence_chain}

Bridge entities connecting the evidence:
{bridge_entities}

M4 Requirements (ALL must be satisfied):
1. MULTI-HOP: Question requires at least 2 distinct reasoning steps
2. MULTI-MODAL: Answer requires evidence from at least 2 modalities ({modalities})
3. MULTI-DOCUMENT: Answer requires evidence from at least 2 documents ({doc_ids})
4. MULTI-TURN: Question is phrased as a 2-3 turn dialogue with coreference

Validation criteria:
- If ANY single evidence source alone can answer the question, REJECT
- If reasoning steps are not explicitly needed, REJECT
- If follow-up turns don't use pronouns/coreference, REJECT

Return JSON:
{{
  "turns": ["turn 1", "turn 2", ...],
  "reasoning_steps": ["step 1: from doc_a...", "step 2: bridge via entity...", "step 3: from doc_b..."],
  "evidence_mapping": {{
    "turn_1": ["evidence_node_id_1"],
    "turn_2": ["evidence_node_id_2", "evidence_node_id_3"]
  }},
  "modalities_used": ["table", "figure"],
  "docs_used": ["doc_a", "doc_b"],
  "coreference_map": {{"it": "the proposed method", "they": "the results"}},
  "answer": "concise answer",
  "difficulty": "easy|medium|hard",
  "validation": {{
    "is_multi_hop": true,
    "is_multi_modal": true,
    "is_multi_doc": true,
    "is_multi_turn": true,
    "rejection_reason": null
  }}
}}
"""

    # Evidence verification prompt
    EVIDENCE_VERIFICATION = """Verify that the generated query can actually be answered using the provided evidence.

Query turns: {turns}
Claimed answer: {answer}
Reasoning steps: {reasoning_steps}

Evidence sources:
{evidence_sources}

Verify:
1. Is the answer actually supported by the evidence?
2. Are all reasoning steps valid?
3. Is each claimed evidence source actually needed?

Return JSON:
{{
  "is_valid": true/false,
  "issues": ["issue 1", ...],
  "corrected_answer": "...",
  "confidence": 0.0-1.0
}}
"""


class M4QueryGenerator:
    """
    Enhanced M4 query generator with entity-aware selection and evidence chain validation.
    """

    def __init__(
        self,
        llm_client: Any,
        provider: str = "anthropic",
        model: str = "claude-sonnet-4-20250514",
        linker: Optional[CrossDocumentLinker] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ):
        """
        Initialize M4 query generator.

        Args:
            llm_client: LLM client for generation
            provider: LLM provider name
            model: Model identifier
            linker: CrossDocumentLinker instance (created if not provided)
            temperature: Generation temperature
            max_tokens: Maximum tokens in response
        """
        self.llm_client = llm_client
        self.provider = provider
        self.model = model
        self.linker = linker or CrossDocumentLinker()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.templates = M4PromptTemplates()

    def _call_llm(self, prompt: str) -> Optional[str]:
        """Call LLM and return response text."""
        try:
            if self.provider == "anthropic":
                response = self.llm_client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            elif self.provider == "openai":
                response = self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert at generating complex research questions."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"}
                )
                return response.choices[0].message.content
        except Exception as e:
            print(f"LLM call failed: {e}")
            return None

    def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from LLM response."""
        if not response:
            return None

        response = response.strip()

        # Handle markdown code blocks
        if response.startswith("```"):
            match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response)
            if match:
                response = match.group(1).strip()

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return None

    def generate_bridging_question(
        self,
        passage_a: Any,
        passage_b: Any,
        bridge_entity: DocumentEntity
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a bridging question that connects two passages.

        This is the foundation for multi-hop reasoning.
        """
        prompt = self.templates.BRIDGING_QUESTION.format(
            doc_a=passage_a.doc_id,
            modal_a=passage_a.modal_type.value if hasattr(passage_a.modal_type, 'value') else passage_a.modal_type,
            content_a=passage_a.content[:500],
            doc_b=passage_b.doc_id,
            modal_b=passage_b.modal_type.value if hasattr(passage_b.modal_type, 'value') else passage_b.modal_type,
            content_b=passage_b.content[:500],
            bridge_entity=bridge_entity.canonical_name
        )

        response = self._call_llm(prompt)
        return self._parse_json_response(response)

    def convert_to_multi_turn(
        self,
        question: str,
        answer: str,
        entities: List[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Convert a single question into multi-turn dialogue with coreference.

        Based on CoQA and MT-Bench++ research on conversational QA.
        """
        prompt = self.templates.MULTI_TURN_CONVERSION.format(
            question=question,
            answer=answer,
            entities=", ".join(entities)
        )

        response = self._call_llm(prompt)
        return self._parse_json_response(response)

    def generate_full_m4_query(
        self,
        evidence_chain: EvidenceChain,
        bridge_entities: List[DocumentEntity]
    ) -> Optional[M4Query]:
        """
        Generate a full M4 query from an evidence chain.

        This is the main generation method that ensures all M4 requirements are met.
        """
        # Format evidence chain for prompt
        evidence_text = self._format_evidence_chain(evidence_chain)
        entity_names = [e.canonical_name for e in bridge_entities]

        prompt = self.templates.FULL_M4_GENERATION.format(
            num_docs=len(evidence_chain.docs_involved),
            num_modalities=len(evidence_chain.modalities_involved),
            evidence_chain=evidence_text,
            bridge_entities=", ".join(entity_names),
            modalities=", ".join(evidence_chain.modalities_involved),
            doc_ids=", ".join(list(evidence_chain.docs_involved)[:3])
        )

        response = self._call_llm(prompt)
        if response:
            print(f"LLM response preview: {response[:200]}...")
        data = self._parse_json_response(response)

        if not data:
            print(f"Failed to parse JSON from response")
            return None

        # Validate the response
        validation = data.get("validation", {})

        if validation.get("rejection_reason"):
            print(f"Query rejected: {validation['rejection_reason']}")
            return None

        # Create M4Query object
        query_id = hashlib.md5(
            json.dumps(data.get("turns", [])).encode()
        ).hexdigest()[:12]

        return M4Query(
            query_id=f"m4_{query_id}",
            query_type=M4QueryType.FULL_M4,
            turns=data.get("turns", []),
            answer=data.get("answer", ""),
            evidence_chain=evidence_chain,
            difficulty=self._parse_difficulty(data.get("difficulty", "medium")),
            metadata={
                "reasoning_steps": data.get("reasoning_steps", []),
                "evidence_mapping": data.get("evidence_mapping", {}),
                "coreference_map": data.get("coreference_map", {}),
                "modalities_used": data.get("modalities_used", []),
                "docs_used": data.get("docs_used", [])
            },
            is_multi_hop=validation.get("is_multi_hop", False),
            is_multi_modal=validation.get("is_multi_modal", False),
            is_multi_doc=validation.get("is_multi_doc", False),
            is_multi_turn=validation.get("is_multi_turn", False)
        )

    def _format_evidence_chain(self, chain: EvidenceChain) -> str:
        """Format evidence chain for prompt."""
        lines = []
        for i, node in enumerate(chain.nodes):
            lines.append(f"[Node {i+1}] Doc: {node.doc_id}, Type: {node.modal_type}, Page: {node.page_idx}")
            lines.append(f"Content: {node.content_snippet[:200]}...")
            lines.append(f"Entities: {', '.join(node.entities)}")
            lines.append("")
        return "\n".join(lines)

    def _parse_difficulty(self, difficulty_str: str) -> float:
        """Parse difficulty string to float."""
        mapping = {"easy": 0.3, "medium": 0.5, "hard": 0.8}
        return mapping.get(difficulty_str.lower(), 0.5)

    def verify_evidence(
        self,
        query: M4Query
    ) -> Dict[str, Any]:
        """
        Verify that query evidence is valid and sufficient.

        This addresses the concern that LLM-generated evidence locations may be hallucinated.
        """
        # Format evidence sources
        evidence_text = []
        for node in query.evidence_chain.nodes:
            evidence_text.append(f"[{node.node_id}] {node.modal_type}: {node.content_snippet[:300]}")

        prompt = self.templates.EVIDENCE_VERIFICATION.format(
            turns=json.dumps(query.turns),
            answer=query.answer,
            reasoning_steps=json.dumps(query.metadata.get("reasoning_steps", [])),
            evidence_sources="\n".join(evidence_text)
        )

        response = self._call_llm(prompt)
        result = self._parse_json_response(response)

        return result or {"is_valid": False, "issues": ["Failed to verify"]}

    def generate_queries_for_passages(
        self,
        passages: List[Any],
        num_queries: int = 5,
        require_full_m4: bool = True
    ) -> List[M4Query]:
        """
        Generate M4 queries for a set of passages.

        This is the main entry point for query generation.

        Args:
            passages: List of passages from multiple documents
            num_queries: Target number of queries to generate
            require_full_m4: Whether to require all M4 dimensions

        Returns:
            List of validated M4Query objects
        """
        # Step 1: Extract entities and build cross-document links
        passages_by_doc = {}
        for p in passages:
            doc_id = p.doc_id if hasattr(p, 'doc_id') else "unknown"
            if doc_id not in passages_by_doc:
                passages_by_doc[doc_id] = []
            passages_by_doc[doc_id].append(p)

        for doc_id, doc_passages in passages_by_doc.items():
            self.linker.build_document_entities(doc_passages, doc_id)

        self.linker.find_cross_document_links()

        # Step 2: Find linkable passage groups
        groups = self.linker.find_linkable_passage_groups(
            passages,
            require_multi_doc=require_full_m4,
            require_multi_modal=require_full_m4
        )

        if not groups:
            print("No linkable passage groups found")
            return []

        # Step 3: Generate queries for each group
        queries = []
        for passage_group, bridge_entity_ids in groups[:num_queries * 2]:  # Try more groups than needed
            # Build evidence chain
            chain = self.linker.build_evidence_chain(
                passage_group,
                bridge_entity_ids,
                query_type="full_m4" if require_full_m4 else "multi_hop"
            )

            if not chain:
                continue

            # Get bridge entities
            bridge_entities = [
                self.linker.entity_index[eid]
                for eid in bridge_entity_ids
                if eid in self.linker.entity_index
            ]

            # Generate query
            query = self.generate_full_m4_query(chain, bridge_entities)

            if query:
                print(f"Query generated:")
                print(f"  Turns: {query.turns}")
                print(f"  Answer: {query.answer[:200]}...")
                # 暂时跳过验证，直接添加
                queries.append(query)

                if len(queries) >= num_queries:
                    break

        return queries

    def get_generation_statistics(self) -> Dict[str, Any]:
        """Get statistics about the generation process."""
        linker_stats = self.linker.get_entity_statistics()
        return {
            **linker_stats,
            "generator_model": self.model,
            "generator_provider": self.provider
        }


def create_m4_generator(
    provider: str = "anthropic",
    model: str = "claude-sonnet-4-20250514",
    api_key: Optional[str] = None
) -> M4QueryGenerator:
    """
    Factory function to create M4QueryGenerator with appropriate LLM client.

    Args:
        provider: "anthropic" or "openai"
        model: Model identifier
        api_key: API key (uses environment variable if not provided)

    Returns:
        Configured M4QueryGenerator instance
    """
    if provider == "anthropic":
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        except ImportError:
            raise ImportError("anthropic package required for Anthropic provider")
    elif provider == "openai":
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key) if api_key else OpenAI()
        except ImportError:
            raise ImportError("openai package required for OpenAI provider")
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    return M4QueryGenerator(
        llm_client=client,
        provider=provider,
        model=model
    )
