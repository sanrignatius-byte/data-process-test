"""
Query Generator for Multimodal Contrastive Learning

Generates diverse queries for different modalities (table, figure, formula, text)
using LLM/VLM with specialized prompts.
"""

import json
import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import asyncio
from concurrent.futures import ThreadPoolExecutor
import re

# Import OpenAI client (will be available in user's environment)
try:
    from openai import OpenAI, AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


@dataclass
class GeneratedQuery:
    """Represents a generated query."""
    query_id: str
    query_text: str
    query_type: str  # factual, comparative, computational, descriptive, reasoning
    target_modality: str
    passage_id: str
    difficulty: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


class PromptTemplates:
    """Prompt templates for different modalities."""

    TABLE = """You are an expert at analyzing scientific tables and data.

Given a table (in markdown format) extracted from a research paper, generate {num_queries} diverse questions.

Requirements:
1. Questions should require actually reading the table data to answer
2. Include different types:
   - Factual lookup (find specific values)
   - Comparative (compare rows/columns)
   - Computational (calculate averages, find max/min)
   - Trend analysis (identify patterns)
3. Questions should be natural and varied in complexity
4. Do NOT generate questions that can be answered without the table

Table Content:
```
{content}
```

{context_section}

Generate exactly {num_queries} questions in the following JSON format:
{{"questions": [
    {{"text": "question text", "type": "factual|comparative|computational|trend"}},
    ...
]}}

Output only valid JSON, no additional text."""

    FIGURE = """You are an expert at analyzing scientific figures and visualizations.

Given information about a figure from a research paper, generate {num_queries} diverse questions.

Figure Information:
- Caption/Description: {content}
{context_section}

Requirements:
1. Questions should require visual understanding of the figure
2. Include different types:
   - Descriptive (describe main patterns/trends)
   - Identification (identify specific features like peaks, intersections)
   - Interpretive (what does the figure prove/demonstrate)
   - Comparative (compare different elements in the figure)
3. Questions should be specific enough that random guessing won't work

Generate exactly {num_queries} questions in the following JSON format:
{{"questions": [
    {{"text": "question text", "type": "descriptive|identification|interpretive|comparative"}},
    ...
]}}

Output only valid JSON, no additional text."""

    FORMULA = """You are an expert at analyzing mathematical equations and formulas.

Given a mathematical formula from a research paper, generate {num_queries} diverse questions.

Formula:
```
{content}
```

{context_section}

Requirements:
1. Questions should test understanding of the formula
2. Include different types:
   - Semantic (what does the formula calculate/represent)
   - Variable meaning (what does variable X represent)
   - Application (how would you use this formula)
   - Derivation (how does term A relate to term B)
3. Assume the reader can see the formula

Generate exactly {num_queries} questions in the following JSON format:
{{"questions": [
    {{"text": "question text", "type": "semantic|variable|application|derivation"}},
    ...
]}}

Output only valid JSON, no additional text."""

    INFOGRAPHIC = """You are an expert at analyzing diagrams, flowcharts, and infographics.

Given an infographic/diagram from a research paper, generate {num_queries} diverse questions.

Infographic Description: {content}
{context_section}

Requirements:
1. Questions should require understanding the visual structure
2. Include different types:
   - Structural (how are components connected)
   - Process (what are the steps/flow)
   - Component (what does component X do)
   - Relationship (how does A relate to B)
3. Questions should test diagram comprehension

Generate exactly {num_queries} questions in the following JSON format:
{{"questions": [
    {{"text": "question text", "type": "structural|process|component|relationship"}},
    ...
]}}

Output only valid JSON, no additional text."""

    TEXT = """You are an expert at analyzing scientific text passages.

Given a text passage from a research paper, generate {num_queries} diverse questions.

Text Passage:
```
{content}
```

{context_section}

Requirements:
1. Questions should require reading the passage to answer
2. Include different types:
   - Factual (extract specific information)
   - Conceptual (understand the main idea)
   - Inferential (draw conclusions from the text)
   - Definitional (understand terminology used)
3. Questions should not be answerable through general knowledge alone

Generate exactly {num_queries} questions in the following JSON format:
{{"questions": [
    {{"text": "question text", "type": "factual|conceptual|inferential|definitional"}},
    ...
]}}

Output only valid JSON, no additional text."""

    CROSS_MODAL = """You are an expert at analyzing multimodal scientific content.

Given content from multiple modalities, generate {num_queries} questions that require integrating information from multiple sources.

Content:
{content}

Requirements:
1. Questions MUST require information from multiple modalities to answer
2. Questions should test ability to connect information across modalities
3. Include reasoning across text and visual elements

Generate exactly {num_queries} questions in the following JSON format:
{{"questions": [
    {{"text": "question text", "type": "cross_modal_reasoning", "modalities_required": ["modality1", "modality2"]}},
    ...
]}}

Output only valid JSON, no additional text."""

    M4 = """You are an expert at crafting complex research questions for multimodal AI training.

You will be given content excerpts from multiple documents and multiple modalities.
Generate {num_queries} questions that satisfy M4 requirements:

1) Multi-hop reasoning: questions require at least two distinct evidence points.
2) Multi-modal integration: questions require at least two different modalities.
3) Multi-document synthesis: questions require evidence from at least two documents.
4) Multi-turn interaction: questions must be phrased as a short multi-turn dialogue.

Key design logic (Why this works):
- Bridge Logic (multi-hop): enforce explicit reasoning_steps that connect evidence from Chunk A -> Chunk B -> Answer.
  If reasoning_steps cannot be written, the question is invalid.
- Dependency Check (multi-modal): reject questions answerable from only text or only visual.
  Require evidence from at least two modalities, otherwise REJECT.
- Coreference Injection (multi-turn): follow-up turns must use coreferences like "it", "they", "that"
  instead of repeating entity names.

Content (each block includes document id, modality, passage id, and excerpt):
{content}

Return exactly {num_queries} items in the following JSON format:
{{"questions": [
  {{
    "turns": ["question turn 1", "question turn 2", "..."],
    "type": "multi_hop_reasoning|multi_modal_integration|multi_doc_synthesis|multi_turn_interaction",
    "modalities_required": ["text", "table", "figure", "formula"],
    "doc_ids": ["doc_a", "doc_b"],
    "evidence_passage_ids": ["passage_id_1", "passage_id_2"],
    "reasoning_steps": ["step 1", "step 2"],
    "gold_answer": "concise answer"
  }}
]}}

Output only valid JSON, no additional text."""

    M4_TRAINING = """You are a scholarly query generation expert. Generate high-quality training queries
from parsed paper content with precise evidence references.

Input data:
- arXiv ID: {arxiv_id}
- Markdown text: {text_content}
- JSONL structure: {structure_content}
- Assets folder: {assets_dir}

Generate:
1) Single-hop factual queries (5)
2) Multi-hop reasoning queries (3)
3) Multi-modal understanding queries (2)

Requirements:
- Each query must include evidence with page + section + bbox + text span.
- Multi-hop queries must list secondary evidence sources.
- Multi-modal queries must reference required_images and text evidence.

Return JSON only, following this schema:
{{"arxiv_id": "{arxiv_id}", "queries": [
  {{
    "id": "q1",
    "text": "query text",
    "type": "single_hop|multi_hop|multi_modal",
    "difficulty": "easy|medium|hard",
    "evidence": {{
      "primary_source": {{
        "page": 1,
        "section": "2.1",
        "bbox": [0, 0, 100, 100],
        "text_span": "evidence excerpt"
      }},
      "secondary_sources": [],
      "required_images": ["fig_2_b.png"],
      "answer_span": "exact answer"
    }},
    "estimated_tokens": 150
  }}
]}}
Output only valid JSON, no additional text."""


class QueryGenerator(ABC):
    """Abstract base class for query generators."""

    @abstractmethod
    def generate(
        self,
        passage: Any,
        num_queries: int = 3
    ) -> List[GeneratedQuery]:
        """Generate queries for a passage."""
        pass

    @abstractmethod
    def generate_batch(
        self,
        passages: List[Any],
        num_queries: int = 3
    ) -> Dict[str, List[GeneratedQuery]]:
        """Generate queries for multiple passages."""
        pass


class MultimodalQueryGenerator(QueryGenerator):
    """
    LLM-based query generator for multimodal content.

    Supports OpenAI and Anthropic APIs with rate limiting and retry logic.
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
        rate_limit: int = 60,
        max_retries: int = 3,
        retry_delay: float = 2.0
    ):
        """
        Initialize query generator.

        Args:
            provider: LLM provider ("openai" or "anthropic")
            model: Model name
            api_key: API key (uses env var if not provided)
            temperature: Generation temperature
            max_tokens: Maximum tokens in response
            rate_limit: Requests per minute limit
            max_retries: Maximum retry attempts
            retry_delay: Base delay between retries
        """
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.rate_limit = rate_limit
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Initialize client
        if provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("openai package not installed")
            self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        elif provider == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("anthropic package not installed")
            self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        # Rate limiting
        self._last_request_time = 0
        self._min_interval = 60.0 / rate_limit

        # Templates
        self.templates = PromptTemplates()

    def _get_template(self, modal_type: str) -> str:
        """Get prompt template for modality."""
        templates = {
            "table": self.templates.TABLE,
            "figure": self.templates.FIGURE,
            "formula": self.templates.FORMULA,
            "infographic": self.templates.INFOGRAPHIC,
            "text": self.templates.TEXT,
            "cross_modal": self.templates.CROSS_MODAL,
            "m4": self.templates.M4,
            "m4_training": self.templates.M4_TRAINING
        }
        return templates.get(modal_type, self.templates.TEXT)

    def _build_prompt(
        self,
        passage: Any,
        num_queries: int = 3
    ) -> str:
        """Build prompt for a passage."""
        modal_type = passage.modal_type.value if hasattr(passage.modal_type, 'value') else passage.modal_type
        template = self._get_template(modal_type)

        # Build context section
        context_section = ""
        if passage.context:
            context_section = f"Context from surrounding text:\n{passage.context}"

        # Truncate very long content
        content = passage.content
        if len(content) > 2000:
            content = content[:2000] + "...[truncated]"

        return template.format(
            content=content,
            context_section=context_section,
            num_queries=num_queries
        )

    def _build_m4_prompt(
        self,
        passages: List[Any],
        num_queries: int = 4,
        max_docs: int = 3,
        max_passages_per_doc: int = 4
    ) -> Tuple[str, List[str]]:
        """Build M4 prompt from multiple documents and modalities."""
        docs = {}
        for passage in passages:
            doc_id = getattr(passage, "doc_id", None) or "unknown_doc"
            docs.setdefault(doc_id, []).append(passage)

        selected_doc_ids = list(docs.keys())[:max_docs]
        content_blocks = []
        for doc_id in selected_doc_ids:
            doc_passages = docs[doc_id]
            by_modality = {}
            for p in doc_passages:
                modal = p.modal_type.value if hasattr(p.modal_type, "value") else p.modal_type
                by_modality.setdefault(modal, []).append(p)

            used = 0
            for modal_type, modal_passages in by_modality.items():
                if used >= max_passages_per_doc:
                    break
                p = modal_passages[0]
                snippet = p.content[:500] + ("...[truncated]" if len(p.content) > 500 else "")
                content_blocks.append(
                    f"[DOC {doc_id} | {modal_type.upper()} | {p.passage_id}]\n{snippet}"
                )
                used += 1

        template = self._get_template("m4")
        content = "\n\n".join(content_blocks)
        return template.format(content=content, num_queries=num_queries), selected_doc_ids

    def _build_m4_training_prompt(
        self,
        arxiv_id: str,
        text_content: str,
        structure_content: str,
        assets_dir: str
    ) -> str:
        """Build M4 training prompt for evidence-rich query generation."""
        template = self._get_template("m4_training")
        return template.format(
            arxiv_id=arxiv_id,
            text_content=text_content,
            structure_content=structure_content,
            assets_dir=assets_dir
        )

    def _rate_limit_wait(self) -> None:
        """Wait if needed to respect rate limit."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()

    def _call_llm(self, prompt: str) -> Optional[str]:
        """Call LLM API with retry logic."""
        for attempt in range(self.max_retries):
            try:
                self._rate_limit_wait()

                if self.provider == "openai":
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that generates high-quality questions for training AI systems."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        response_format={"type": "json_object"}
                    )
                    return response.choices[0].message.content

                elif self.provider == "anthropic":
                    response = self.client.messages.create(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )
                    return response.content[0].text

            except Exception as e:
                print(f"LLM call attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))

        return None

    def _parse_response(
        self,
        response: str,
        passage: Any
    ) -> List[GeneratedQuery]:
        """Parse LLM response into GeneratedQuery objects."""
        queries = []

        try:
            # Try to extract JSON from response
            response = response.strip()

            # Handle potential markdown code blocks
            if response.startswith("```"):
                # Extract content between code blocks
                match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response)
                if match:
                    response = match.group(1).strip()

            data = json.loads(response)
            questions = data.get("questions", data.get("queries", []))

            modal_type = passage.modal_type.value if hasattr(passage.modal_type, 'value') else passage.modal_type

            for idx, q in enumerate(questions):
                q_dict = q if isinstance(q, dict) else {}
                if isinstance(q, dict):
                    turns = q.get("turns")
                    if turns:
                        query_text = " / ".join([t for t in turns if t])
                    else:
                        query_text = q.get("text", q.get("question", ""))
                    query_type = q.get("type", "factual")
                elif isinstance(q, str):
                    query_text = q
                    query_type = "factual"
                else:
                    continue

                if not query_text:
                    continue
                if modal_type == "m4":
                    if not isinstance(q, dict):
                        continue
                    if not self._validate_m4_question(q_dict):
                        continue
                if modal_type == "m4_training":
                    if not isinstance(q, dict):
                        continue
                    if not self._validate_m4_training_question(q_dict):
                        continue

                # Generate unique query ID
                query_hash = hashlib.md5(query_text.encode()).hexdigest()[:8]
                query_id = f"{passage.passage_id}_q{idx}_{query_hash}"

                queries.append(GeneratedQuery(
                    query_id=query_id,
                    query_text=query_text,
                    query_type=query_type,
                    target_modality=modal_type,
                    passage_id=passage.passage_id,
                    difficulty=self._estimate_difficulty(query_text, query_type),
                    metadata={
                        "modalities_required": q_dict.get("modalities_required", [modal_type]),
                        "doc_ids": q_dict.get("doc_ids"),
                        "evidence_passage_ids": q_dict.get("evidence_passage_ids"),
                        "turns": q_dict.get("turns"),
                        "reasoning_steps": q_dict.get("reasoning_steps"),
                        "gold_answer": q_dict.get("gold_answer"),
                        "evidence": q_dict.get("evidence"),
                        "difficulty_label": q_dict.get("difficulty"),
                        "estimated_tokens": q_dict.get("estimated_tokens")
                    }
                ))

        except json.JSONDecodeError as e:
            print(f"Failed to parse LLM response: {e}")
            print(f"Response: {response[:200]}...")

        return queries

    def _validate_m4_question(self, question: Dict[str, Any]) -> bool:
        """Validate M4 question structure and coreference constraints."""
        turns = question.get("turns") or []
        if len(turns) < 2:
            return False
        reasoning_steps = question.get("reasoning_steps") or []
        if len(reasoning_steps) < 2:
            return False
        if not question.get("gold_answer"):
            return False
        if not question.get("doc_ids") or len(question.get("doc_ids")) < 2:
            return False
        modalities = question.get("modalities_required") or []
        if len(set(modalities)) < 2:
            return False
        # Coreference injection: require pronouns in follow-up turns.
        pronouns = {"it", "they", "that", "those", "these", "its", "their"}
        followups = " ".join(turns[1:]).lower()
        if not any(p in followups.split() for p in pronouns):
            return False
        return True

    def _validate_m4_training_question(self, question: Dict[str, Any]) -> bool:
        """Validate evidence-rich M4 training queries."""
        if not question.get("text"):
            return False
        evidence = question.get("evidence") or {}
        primary = evidence.get("primary_source") or {}
        if not primary.get("page") or not primary.get("section"):
            return False
        if not primary.get("bbox") or not primary.get("text_span"):
            return False
        if question.get("type") == "multi_hop":
            if not evidence.get("secondary_sources"):
                return False
        if question.get("type") == "multi_modal":
            if not evidence.get("required_images"):
                return False
        return True

    def _estimate_difficulty(self, query_text: str, query_type: str) -> float:
        """Estimate query difficulty (0.0 to 1.0)."""
        difficulty = 0.5

        # Query length contributes to difficulty
        word_count = len(query_text.split())
        if word_count > 20:
            difficulty += 0.1
        elif word_count < 8:
            difficulty -= 0.1

        # Query type affects difficulty
        type_difficulty = {
            "factual": 0.0,
            "descriptive": 0.1,
            "comparative": 0.2,
            "computational": 0.3,
            "trend": 0.2,
            "identification": 0.2,
            "interpretive": 0.3,
            "semantic": 0.2,
            "variable": 0.1,
            "application": 0.3,
            "derivation": 0.4,
            "structural": 0.2,
            "process": 0.3,
            "relationship": 0.3,
            "inferential": 0.3,
            "cross_modal_reasoning": 0.4,
            "multi_hop_reasoning": 0.5,
            "multi_modal_integration": 0.5,
            "multi_doc_synthesis": 0.5,
            "multi_turn_interaction": 0.5
        }
        difficulty += type_difficulty.get(query_type, 0.0)

        # Keywords suggesting complexity
        complex_keywords = ["why", "how", "explain", "compare", "analyze", "relationship"]
        for kw in complex_keywords:
            if kw in query_text.lower():
                difficulty += 0.05

        return min(1.0, max(0.0, difficulty))

    def generate(
        self,
        passage: Any,
        num_queries: int = 3
    ) -> List[GeneratedQuery]:
        """
        Generate queries for a single passage.

        Args:
            passage: Passage object with content and modality
            num_queries: Number of queries to generate

        Returns:
            List of GeneratedQuery objects
        """
        prompt = self._build_prompt(passage, num_queries)
        response = self._call_llm(prompt)

        if response:
            return self._parse_response(response, passage)

        return []

    def generate_batch(
        self,
        passages: List[Any],
        num_queries: int = 3,
        max_workers: int = 4
    ) -> Dict[str, List[GeneratedQuery]]:
        """
        Generate queries for multiple passages.

        Args:
            passages: List of Passage objects
            num_queries: Queries per passage
            max_workers: Parallel workers (limited by rate limit)

        Returns:
            Dict mapping passage_id to list of queries
        """
        results = {}

        # Use ThreadPoolExecutor for I/O-bound API calls
        # But limit concurrency due to rate limits
        effective_workers = min(max_workers, max(1, self.rate_limit // 10))

        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = {
                executor.submit(self.generate, passage, num_queries): passage
                for passage in passages
            }

            for future in futures:
                passage = futures[future]
                try:
                    queries = future.result()
                    results[passage.passage_id] = queries
                except Exception as e:
                    print(f"Query generation failed for {passage.passage_id}: {e}")
                    results[passage.passage_id] = []

        return results

    def generate_cross_modal_queries(
        self,
        passages: List[Any],
        num_queries: int = 2
    ) -> List[GeneratedQuery]:
        """
        Generate cross-modal queries that require multiple modalities.

        Args:
            passages: List of passages from same document
            num_queries: Number of cross-modal queries

        Returns:
            List of cross-modal queries
        """
        if len(passages) < 2:
            return []

        # Group passages by modality
        by_modality = {}
        for p in passages:
            modal = p.modal_type.value if hasattr(p.modal_type, 'value') else p.modal_type
            if modal not in by_modality:
                by_modality[modal] = []
            by_modality[modal].append(p)

        # Need at least 2 different modalities
        if len(by_modality) < 2:
            return []

        # Build combined content
        content_parts = []
        modalities_used = []
        for modal_type, modal_passages in list(by_modality.items())[:3]:
            if modal_passages:
                p = modal_passages[0]
                content_parts.append(f"[{modal_type.upper()}]\n{p.content[:500]}")
                modalities_used.append(modal_type)

        combined_content = "\n\n".join(content_parts)

        # Create pseudo-passage for template
        class PseudoPassage:
            def __init__(self):
                self.modal_type = "cross_modal"
                self.content = combined_content
                self.context = None
                self.passage_id = f"cross_{passages[0].doc_id}"

        pseudo = PseudoPassage()
        return self.generate(pseudo, num_queries)

    def generate_m4_queries(
        self,
        passages: List[Any],
        num_queries: int = 4,
        max_docs: int = 3,
        max_passages_per_doc: int = 4
    ) -> List[GeneratedQuery]:
        """
        Generate M4 queries requiring multi-hop, multi-modal, multi-document,
        and multi-turn reasoning.

        Args:
            passages: List of passages from multiple documents
            num_queries: Number of M4 queries
            max_docs: Max documents to include in the prompt
            max_passages_per_doc: Max passages per document in the prompt

        Returns:
            List of GeneratedQuery objects
        """
        if not passages:
            return []

        prompt, selected_doc_ids = self._build_m4_prompt(
            passages,
            num_queries=num_queries,
            max_docs=max_docs,
            max_passages_per_doc=max_passages_per_doc
        )
        response = self._call_llm(prompt)

        class PseudoPassage:
            def __init__(self, doc_ids):
                self.modal_type = "m4"
                self.content = ""
                self.context = None
                self.passage_id = f"m4_{'_'.join(doc_ids)}"

        pseudo = PseudoPassage(selected_doc_ids)
        if response:
            return self._parse_response(response, pseudo)
        return []

    def generate_m4_training_queries(
        self,
        arxiv_id: str,
        text_content: str,
        structure_content: str,
        assets_dir: str
    ) -> List[GeneratedQuery]:
        """
        Generate evidence-rich training queries from parsed paper assets.

        Args:
            arxiv_id: Paper arXiv ID
            text_content: Parsed markdown content
            structure_content: JSONL structure content
            assets_dir: Assets directory path string

        Returns:
            List of GeneratedQuery objects
        """
        prompt = self._build_m4_training_prompt(
            arxiv_id=arxiv_id,
            text_content=text_content,
            structure_content=structure_content,
            assets_dir=assets_dir
        )
        response = self._call_llm(prompt)

        class PseudoPassage:
            def __init__(self, doc_id):
                self.modal_type = "m4_training"
                self.content = ""
                self.context = None
                self.passage_id = f"m4_training_{doc_id}"

        pseudo = PseudoPassage(arxiv_id)
        if response:
            return self._parse_response(response, pseudo)
        return []
