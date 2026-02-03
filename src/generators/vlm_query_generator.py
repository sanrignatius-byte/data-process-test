"""
VLM Query Generator for Multimodal Contrastive Learning

Generates diverse queries using Vision-Language Models (VLM/MLLM) like Qwen3-VL.
Enables true multimodal understanding by processing actual images alongside text.

Supported Models:
- Qwen3-VL (via vLLM or transformers)
- GPT-4V/GPT-4o (OpenAI)
- Claude 3.5 Sonnet (Anthropic)
- LLaVA (local)
"""

import json
import time
import base64
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
import re

try:
    from PIL import Image
    import io
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    import torch
    QWEN_LOCAL_AVAILABLE = True
except ImportError:
    QWEN_LOCAL_AVAILABLE = False

try:
    from openai import OpenAI as VLLMClient
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


class QueryModalityType(Enum):
    """Types of queries based on modality requirements."""
    UNIMODAL_TEXT = "unimodal_text"           # Only text needed
    UNIMODAL_VISUAL = "unimodal_visual"       # Only image needed
    MULTIMODAL_GROUNDED = "multimodal_grounded"  # Image + text reference
    CROSS_MODAL_REASONING = "cross_modal_reasoning"  # Multiple modalities


@dataclass
class VLMGeneratedQuery:
    """Represents a VLM-generated query with modality information."""
    query_id: str
    query_text: str
    query_type: str
    query_modality: QueryModalityType
    target_modality: str
    passage_id: str
    difficulty: float = 0.5
    requires_image: bool = False
    visual_grounding: Optional[str] = None  # Region/element in image
    metadata: Dict[str, Any] = field(default_factory=dict)


class VLMPromptTemplates:
    """
    Multimodal prompt templates designed for Vision-Language Models.

    Each template is designed to leverage actual image understanding,
    not just text descriptions.
    """

    # ============ TABLE Prompts ============
    TABLE_VISUAL = """You are analyzing a table image from a scientific paper.

Look at this table image carefully and generate {num_queries} diverse questions.

Requirements:
1. Questions MUST require looking at the actual table to answer
2. Include these types:
   - **Visual Reading**: "What value is in row X, column Y?"
   - **Comparison**: "Which method achieves the highest score in column Z?"
   - **Calculation**: "What is the difference between A and B?"
   - **Pattern**: "Which column shows the most variation?"
3. Use specific cell references visible in the image
4. Vary difficulty from simple lookup to multi-step reasoning

{context_section}

Generate exactly {num_queries} questions in JSON format:
{{"questions": [
    {{"text": "question", "type": "visual_reading|comparison|calculation|pattern", "visual_grounding": "specific cell/region", "requires_image": true}},
    ...
]}}

Output only valid JSON."""

    TABLE_HYBRID = """You are analyzing a table from a scientific paper.

Table (markdown format):
```
{content}
```

Table image is also provided for visual verification.

{context_section}

Generate {num_queries} questions that test understanding of this data:
1. Some questions should require the IMAGE (visual layout, formatting)
2. Some questions should use TEXT (exact values, calculations)
3. Include cross-reference questions

{{"questions": [
    {{"text": "question", "type": "type", "requires_image": true/false, "visual_grounding": "region or null"}},
    ...
]}}

Output only valid JSON."""

    # ============ FIGURE Prompts ============
    FIGURE_VISUAL = """You are analyzing a scientific figure/chart.

Look at this figure carefully and generate {num_queries} diverse questions.

Requirements:
1. Questions MUST require visual understanding to answer
2. Include these types:
   - **Trend Analysis**: "What trend does the blue line show?"
   - **Value Reading**: "What is the approximate value at x=10?"
   - **Comparison**: "Which curve has the steepest slope?"
   - **Identification**: "What do the different colors represent?"
   - **Interpretation**: "What does this figure demonstrate about X?"
3. Reference specific visual elements (colors, axes, legends, markers)
4. Questions should NOT be answerable from caption alone

{context_section}

Generate exactly {num_queries} questions in JSON format:
{{"questions": [
    {{"text": "question", "type": "trend|reading|comparison|identification|interpretation", "visual_grounding": "element description", "requires_image": true}},
    ...
]}}

Output only valid JSON."""

    FIGURE_WITH_CAPTION = """You are analyzing a scientific figure.

Figure Caption: {content}

The figure image is provided above. Generate {num_queries} questions:

1. **Image-only questions**: Can ONLY be answered by looking at the figure
2. **Caption-grounded questions**: Connect caption claims to visual evidence
3. **Inference questions**: What can be concluded from both?

{context_section}

{{"questions": [
    {{"text": "question", "type": "type", "requires_image": true/false, "reasoning": "why this type"}},
    ...
]}}

Output only valid JSON."""

    # ============ FORMULA Prompts ============
    FORMULA_VISUAL = """You are analyzing a mathematical formula/equation image.

Look at this formula carefully and generate {num_queries} diverse questions.

Requirements:
1. Questions should test formula understanding:
   - **Symbol Recognition**: "What Greek letter appears in the denominator?"
   - **Structure**: "How many terms are in the summation?"
   - **Semantic**: "What physical quantity does this formula calculate?"
   - **Variable Meaning**: "What does the subscript i represent?"
   - **Application**: "How would you compute X using this formula?"
2. Reference specific visual elements (fractions, subscripts, operators)
3. Include both visual parsing and conceptual understanding

{context_section}

Generate exactly {num_queries} questions in JSON format:
{{"questions": [
    {{"text": "question", "type": "symbol|structure|semantic|variable|application", "visual_grounding": "formula part", "requires_image": true/false}},
    ...
]}}

Output only valid JSON."""

    FORMULA_LATEX = """You are analyzing a mathematical formula.

LaTeX representation:
```latex
{content}
```

Formula image is also provided.

{context_section}

Generate {num_queries} questions testing formula understanding:
1. **Parsing questions**: Require reading the rendered formula image
2. **Semantic questions**: Can use either representation
3. **Application questions**: How to use/modify the formula

{{"questions": [
    {{"text": "question", "type": "type", "requires_image": true/false}},
    ...
]}}

Output only valid JSON."""

    # ============ INFOGRAPHIC Prompts ============
    INFOGRAPHIC_VISUAL = """You are analyzing an infographic/diagram from a scientific paper.

This could be: architecture diagram, flowchart, pipeline, system overview, etc.

Look carefully and generate {num_queries} diverse questions.

Requirements:
1. Questions MUST require understanding the visual structure:
   - **Component**: "What are the main components/modules?"
   - **Flow**: "What is the sequence of operations?"
   - **Connection**: "How does component A connect to B?"
   - **Role**: "What is the purpose of the highlighted block?"
   - **Reasoning**: "Why is X positioned after Y in the pipeline?"
2. Reference specific visual elements (boxes, arrows, labels)
3. Test understanding of the overall system/process

{context_section}

Generate exactly {num_queries} questions in JSON format:
{{"questions": [
    {{"text": "question", "type": "component|flow|connection|role|reasoning", "visual_grounding": "specific element", "requires_image": true}},
    ...
]}}

Output only valid JSON."""

    # ============ CROSS-MODAL Prompts ============
    CROSS_MODAL = """You are analyzing multimodal scientific content.

You have access to:
{modality_list}

Generate {num_queries} questions that REQUIRE integrating information from MULTIPLE modalities.

Requirements:
1. Questions MUST need both text AND visual information
2. Example: "According to Table 2, which method achieves the result shown in Figure 3?"
3. Questions should test cross-referencing ability
4. Specify which modalities are required for each question

{{"questions": [
    {{"text": "question", "type": "cross_modal_reasoning", "modalities_required": ["table", "figure"], "requires_image": true}},
    ...
]}}

Output only valid JSON."""


class VLMQueryGenerator(ABC):
    """Abstract base class for VLM-based query generators."""

    @abstractmethod
    def generate(
        self,
        passage: Any,
        num_queries: int = 3,
        query_modes: List[QueryModalityType] = None
    ) -> List[VLMGeneratedQuery]:
        """Generate queries for a passage using VLM."""
        pass

    @abstractmethod
    def generate_with_image(
        self,
        image_path: str,
        text_content: str,
        modal_type: str,
        num_queries: int = 3
    ) -> List[VLMGeneratedQuery]:
        """Generate queries with explicit image input."""
        pass


class MultimodalQueryGenerator(VLMQueryGenerator):
    """
    Production VLM Query Generator supporting multiple backends.

    Backends:
    - qwen_local: Local Qwen3-VL via transformers
    - qwen_vllm: Qwen3-VL via vLLM server
    - openai: GPT-4V/GPT-4o
    - anthropic: Claude 3.5 Sonnet
    """

    def __init__(
        self,
        backend: str = "qwen_vllm",
        model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        device: str = "cuda:0",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        rate_limit: int = 30,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        max_image_size: int = 1024,  # Max dimension for image resize
    ):
        """
        Initialize VLM Query Generator.

        Args:
            backend: VLM backend ("qwen_local", "qwen_vllm", "openai", "anthropic")
            model: Model name/path
            api_base: API base URL (for vLLM or custom endpoints)
            api_key: API key
            device: GPU device for local models
            temperature: Generation temperature
            max_tokens: Max tokens in response
            rate_limit: Requests per minute
            max_retries: Max retry attempts
            retry_delay: Base delay between retries
            max_image_size: Max image dimension (for resizing)
        """
        self.backend = backend
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.device = device
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.rate_limit = rate_limit
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_image_size = max_image_size

        # Initialize client based on backend
        self.client = None
        self.processor = None
        self.local_model = None
        self._init_backend()

        # Rate limiting
        self._last_request_time = 0
        self._min_interval = 60.0 / rate_limit

        # Templates
        self.templates = VLMPromptTemplates()

    def _init_backend(self):
        """Initialize the VLM backend."""
        if self.backend == "qwen_local":
            if not QWEN_LOCAL_AVAILABLE:
                raise ImportError(
                    "transformers and torch required for local Qwen. "
                    "Install: pip install transformers torch"
                )
            print(f"Loading local Qwen model: {self.model}")
            self.local_model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model,
                torch_dtype=torch.bfloat16,
                device_map=self.device
            )
            self.processor = AutoProcessor.from_pretrained(self.model)

        elif self.backend == "qwen_vllm":
            # Use OpenAI-compatible client for vLLM
            if not VLLM_AVAILABLE:
                raise ImportError("openai package required for vLLM client")
            api_base = self.api_base or "http://localhost:8000/v1"
            self.client = VLLMClient(
                base_url=api_base,
                api_key=self.api_key or "dummy"  # vLLM doesn't require real key
            )

        elif self.backend == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("openai package not installed")
            self.client = OpenAI(api_key=self.api_key) if self.api_key else OpenAI()

        elif self.backend == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("anthropic package not installed")
            self.client = anthropic.Anthropic(
                api_key=self.api_key
            ) if self.api_key else anthropic.Anthropic()

        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _encode_image(self, image_path: str) -> Tuple[str, str]:
        """
        Encode image to base64 with optional resizing.

        Returns:
            Tuple of (base64_string, media_type)
        """
        if not PIL_AVAILABLE:
            raise ImportError("PIL required for image processing")

        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with Image.open(path) as img:
            # Convert to RGB if needed
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')

            # Resize if too large
            if max(img.size) > self.max_image_size:
                ratio = self.max_image_size / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.LANCZOS)

            # Encode to base64
            buffer = io.BytesIO()
            img_format = 'JPEG'
            img.save(buffer, format=img_format, quality=85)
            base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')

            return base64_str, f"image/{img_format.lower()}"

    def _get_template(self, modal_type: str, has_image: bool = True) -> str:
        """Get appropriate template based on modality and image availability."""
        templates = {
            ("table", True): self.templates.TABLE_VISUAL,
            ("table", False): self.templates.TABLE_HYBRID,
            ("figure", True): self.templates.FIGURE_VISUAL,
            ("figure", False): self.templates.FIGURE_WITH_CAPTION,
            ("formula", True): self.templates.FORMULA_VISUAL,
            ("formula", False): self.templates.FORMULA_LATEX,
            ("infographic", True): self.templates.INFOGRAPHIC_VISUAL,
            ("infographic", False): self.templates.INFOGRAPHIC_VISUAL,
            ("cross_modal", True): self.templates.CROSS_MODAL,
            ("cross_modal", False): self.templates.CROSS_MODAL,
        }

        # Default to figure template for unknown types
        return templates.get((modal_type.lower(), has_image), self.templates.FIGURE_VISUAL)

    def _build_prompt(
        self,
        passage: Any,
        num_queries: int = 3,
        has_image: bool = True
    ) -> str:
        """Build prompt for a passage."""
        modal_type = passage.modal_type.value if hasattr(passage.modal_type, 'value') else passage.modal_type
        template = self._get_template(modal_type, has_image)

        # Build context section
        context_section = ""
        if hasattr(passage, 'context') and passage.context:
            context_section = f"Paper Context:\n{passage.context[:500]}"

        # Truncate very long content
        content = passage.content if hasattr(passage, 'content') else ""
        if len(content) > 1500:
            content = content[:1500] + "...[truncated]"

        return template.format(
            content=content,
            context_section=context_section,
            num_queries=num_queries,
            modality_list=f"- {modal_type}: {content[:200]}..."
        )

    def _rate_limit_wait(self) -> None:
        """Wait if needed to respect rate limit."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()

    def _call_vlm_local(
        self,
        prompt: str,
        image_path: Optional[str] = None
    ) -> Optional[str]:
        """Call local Qwen model."""
        messages = []
        content = []

        if image_path and Path(image_path).exists():
            content.append({
                "type": "image",
                "image": image_path
            })

        content.append({
            "type": "text",
            "text": prompt
        })

        messages.append({
            "role": "user",
            "content": content
        })

        # Process inputs
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if image_path and Path(image_path).exists():
            image = Image.open(image_path)
            inputs = self.processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt"
            ).to(self.device)
        else:
            inputs = self.processor(
                text=[text],
                padding=True,
                return_tensors="pt"
            ).to(self.device)

        # Generate
        with torch.no_grad():
            output_ids = self.local_model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True
            )

        # Decode
        response = self.processor.batch_decode(
            output_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0]

        return response

    def _call_vlm_vllm(
        self,
        prompt: str,
        image_path: Optional[str] = None
    ) -> Optional[str]:
        """Call Qwen via vLLM server."""
        messages = [{"role": "user", "content": []}]

        # Add image if available
        if image_path and Path(image_path).exists():
            base64_img, media_type = self._encode_image(image_path)
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_type};base64,{base64_img}"
                }
            })

        # Add text prompt
        messages[0]["content"].append({
            "type": "text",
            "text": prompt
        })

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        return response.choices[0].message.content

    def _call_vlm_openai(
        self,
        prompt: str,
        image_path: Optional[str] = None
    ) -> Optional[str]:
        """Call OpenAI GPT-4V/4o."""
        messages = [{
            "role": "system",
            "content": "You are an expert at analyzing scientific documents and generating high-quality questions for AI training."
        }]

        user_content = []

        # Add image if available
        if image_path and Path(image_path).exists():
            base64_img, media_type = self._encode_image(image_path)
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_type};base64,{base64_img}",
                    "detail": "high"
                }
            })

        user_content.append({
            "type": "text",
            "text": prompt
        })

        messages.append({"role": "user", "content": user_content})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"}
        )

        return response.choices[0].message.content

    def _call_vlm_anthropic(
        self,
        prompt: str,
        image_path: Optional[str] = None
    ) -> Optional[str]:
        """Call Anthropic Claude."""
        content = []

        # Add image if available
        if image_path and Path(image_path).exists():
            base64_img, media_type = self._encode_image(image_path)
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64_img
                }
            })

        content.append({
            "type": "text",
            "text": prompt
        })

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": content}]
        )

        return response.content[0].text

    def _call_vlm(
        self,
        prompt: str,
        image_path: Optional[str] = None
    ) -> Optional[str]:
        """Call VLM API with retry logic."""
        for attempt in range(self.max_retries):
            try:
                self._rate_limit_wait()

                if self.backend == "qwen_local":
                    return self._call_vlm_local(prompt, image_path)
                elif self.backend == "qwen_vllm":
                    return self._call_vlm_vllm(prompt, image_path)
                elif self.backend == "openai":
                    return self._call_vlm_openai(prompt, image_path)
                elif self.backend == "anthropic":
                    return self._call_vlm_anthropic(prompt, image_path)

            except Exception as e:
                print(f"VLM call attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))

        return None

    def _parse_response(
        self,
        response: str,
        passage: Any,
        has_image: bool
    ) -> List[VLMGeneratedQuery]:
        """Parse VLM response into VLMGeneratedQuery objects."""
        queries = []

        try:
            response = response.strip()

            # Handle markdown code blocks
            if response.startswith("```"):
                match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response)
                if match:
                    response = match.group(1).strip()

            data = json.loads(response)
            questions = data.get("questions", [])

            modal_type = passage.modal_type.value if hasattr(passage.modal_type, 'value') else passage.modal_type
            passage_id = passage.passage_id if hasattr(passage, 'passage_id') else "unknown"

            for idx, q in enumerate(questions):
                if isinstance(q, dict):
                    query_text = q.get("text", q.get("question", ""))
                    query_type = q.get("type", "factual")
                    requires_image = q.get("requires_image", has_image)
                    visual_grounding = q.get("visual_grounding")
                elif isinstance(q, str):
                    query_text = q
                    query_type = "factual"
                    requires_image = has_image
                    visual_grounding = None
                else:
                    continue

                if not query_text:
                    continue

                # Determine query modality type
                if requires_image and visual_grounding:
                    query_modality = QueryModalityType.MULTIMODAL_GROUNDED
                elif requires_image:
                    query_modality = QueryModalityType.UNIMODAL_VISUAL
                else:
                    query_modality = QueryModalityType.UNIMODAL_TEXT

                # Generate unique query ID
                query_hash = hashlib.md5(query_text.encode()).hexdigest()[:8]
                query_id = f"{passage_id}_vlm_q{idx}_{query_hash}"

                queries.append(VLMGeneratedQuery(
                    query_id=query_id,
                    query_text=query_text,
                    query_type=query_type,
                    query_modality=query_modality,
                    target_modality=modal_type,
                    passage_id=passage_id,
                    difficulty=self._estimate_difficulty(query_text, query_type, requires_image),
                    requires_image=requires_image,
                    visual_grounding=visual_grounding,
                    metadata={
                        "backend": self.backend,
                        "model": self.model,
                        "has_source_image": has_image
                    }
                ))

        except json.JSONDecodeError as e:
            print(f"Failed to parse VLM response: {e}")
            print(f"Response: {response[:300]}...")

        return queries

    def _estimate_difficulty(
        self,
        query_text: str,
        query_type: str,
        requires_image: bool
    ) -> float:
        """Estimate query difficulty (0.0 to 1.0)."""
        difficulty = 0.5

        # Word count
        word_count = len(query_text.split())
        if word_count > 20:
            difficulty += 0.1
        elif word_count < 8:
            difficulty -= 0.1

        # Query type difficulty
        type_difficulty = {
            "visual_reading": 0.1,
            "trend": 0.2,
            "comparison": 0.25,
            "calculation": 0.3,
            "pattern": 0.3,
            "identification": 0.2,
            "interpretation": 0.35,
            "semantic": 0.25,
            "structure": 0.2,
            "application": 0.35,
            "reasoning": 0.4,
            "cross_modal_reasoning": 0.45
        }
        difficulty += type_difficulty.get(query_type.lower(), 0.15)

        # Multimodal queries are generally harder
        if requires_image:
            difficulty += 0.1

        # Complex keywords
        complex_keywords = ["why", "how", "explain", "compare", "analyze", "relationship", "difference"]
        for kw in complex_keywords:
            if kw in query_text.lower():
                difficulty += 0.03

        return min(1.0, max(0.0, difficulty))

    def generate(
        self,
        passage: Any,
        num_queries: int = 3,
        query_modes: List[QueryModalityType] = None
    ) -> List[VLMGeneratedQuery]:
        """
        Generate queries for a passage using VLM.

        Args:
            passage: Passage object with content, modality, and optional image_path
            num_queries: Number of queries to generate
            query_modes: Specific query modality types to generate (optional)

        Returns:
            List of VLMGeneratedQuery objects
        """
        # Check if image is available
        has_image = (
            hasattr(passage, 'image_path') and
            passage.image_path and
            Path(passage.image_path).exists()
        )
        image_path = passage.image_path if has_image else None

        # Build prompt
        prompt = self._build_prompt(passage, num_queries, has_image)

        # Call VLM
        response = self._call_vlm(prompt, image_path)

        if response:
            queries = self._parse_response(response, passage, has_image)

            # Filter by query modes if specified
            if query_modes:
                queries = [q for q in queries if q.query_modality in query_modes]

            return queries

        return []

    def generate_with_image(
        self,
        image_path: str,
        text_content: str,
        modal_type: str,
        num_queries: int = 3,
        passage_id: str = "custom"
    ) -> List[VLMGeneratedQuery]:
        """
        Generate queries with explicit image and text input.

        Args:
            image_path: Path to the image file
            text_content: Text description/content
            modal_type: Type of modality ("table", "figure", etc.)
            num_queries: Number of queries to generate
            passage_id: ID for the passage

        Returns:
            List of VLMGeneratedQuery objects
        """
        # Create pseudo-passage object
        class PseudoPassage:
            def __init__(self):
                self.modal_type = modal_type
                self.content = text_content
                self.context = None
                self.passage_id = passage_id
                self.image_path = image_path

        return self.generate(PseudoPassage(), num_queries)

    def generate_batch(
        self,
        passages: List[Any],
        num_queries: int = 3,
        max_workers: int = 4
    ) -> Dict[str, List[VLMGeneratedQuery]]:
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

        # Limit concurrency for rate limiting
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
                    print(f"VLM query generation failed for {passage.passage_id}: {e}")
                    results[passage.passage_id] = []

        return results

    def generate_cross_modal_queries(
        self,
        passages: List[Any],
        num_queries: int = 2
    ) -> List[VLMGeneratedQuery]:
        """
        Generate cross-modal queries requiring multiple modalities.

        Args:
            passages: List of passages from same document
            num_queries: Number of cross-modal queries

        Returns:
            List of cross-modal queries
        """
        if len(passages) < 2:
            return []

        # Group by modality
        by_modality = {}
        for p in passages:
            modal = p.modal_type.value if hasattr(p.modal_type, 'value') else p.modal_type
            if modal not in by_modality:
                by_modality[modal] = []
            by_modality[modal].append(p)

        if len(by_modality) < 2:
            return []

        # Build multimodal content description
        modality_list = []
        image_path = None
        for modal_type, modal_passages in list(by_modality.items())[:3]:
            if modal_passages:
                p = modal_passages[0]
                content_preview = p.content[:300] if p.content else "No content"
                modality_list.append(f"- {modal_type.upper()}: {content_preview}")
                # Use first available image
                if not image_path and hasattr(p, 'image_path') and p.image_path:
                    if Path(p.image_path).exists():
                        image_path = p.image_path

        # Create pseudo-passage for cross-modal
        class CrossModalPassage:
            def __init__(self):
                self.modal_type = "cross_modal"
                self.content = "\n".join(modality_list)
                self.context = None
                self.passage_id = f"cross_{passages[0].doc_id}"
                self.image_path = image_path

        queries = self.generate(CrossModalPassage(), num_queries)

        # Mark all as cross-modal
        for q in queries:
            q.query_modality = QueryModalityType.CROSS_MODAL_REASONING

        return queries


def create_vlm_generator(config: Dict[str, Any]) -> MultimodalQueryGenerator:
    """
    Factory function to create VLM generator from config.

    Args:
        config: Configuration dictionary with vlm_query_generation section

    Returns:
        Configured MultimodalQueryGenerator instance
    """
    vlm_config = config.get("vlm_query_generation", {})

    return MultimodalQueryGenerator(
        backend=vlm_config.get("backend", "qwen_vllm"),
        model=vlm_config.get("model", "Qwen/Qwen2.5-VL-7B-Instruct"),
        api_base=vlm_config.get("api_base"),
        api_key=vlm_config.get("api_key"),
        device=vlm_config.get("device", "cuda:0"),
        temperature=vlm_config.get("temperature", 0.7),
        max_tokens=vlm_config.get("max_tokens", 1024),
        rate_limit=vlm_config.get("rate_limit", 30),
        max_retries=vlm_config.get("max_retries", 3),
        retry_delay=vlm_config.get("retry_delay", 2.0),
        max_image_size=vlm_config.get("max_image_size", 1024)
    )
