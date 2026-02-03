# Multimodal Query Design for Contrastive Learning

## Overview

This document describes the design of multimodal queries for training contrastive learning models. The goal is to create `<query, positive_passage, negative_passage>` triplets that enable models to learn fine-grained multimodal understanding.

## Query Taxonomy

### 1. By Modality Requirement

```
QueryModalityType
├── UNIMODAL_TEXT        # Query answerable from text only
├── UNIMODAL_VISUAL      # Query answerable from image only
├── MULTIMODAL_GROUNDED  # Query needs image + text reference
└── CROSS_MODAL_REASONING # Query needs multiple modality fusion
```

### 2. By Target Content Type

```
Target Modality
├── TABLE       # Structured data tables
├── FIGURE      # Charts, plots, graphs
├── FORMULA     # Mathematical equations
├── INFOGRAPHIC # Diagrams, flowcharts, architectures
└── TEXT        # Prose paragraphs
```

---

## Query Design by Modality

### TABLE Queries

Tables contain structured data. Queries should test ability to:
- **Read specific cells** (visual parsing)
- **Compare across rows/columns** (relational reasoning)
- **Calculate aggregates** (numerical reasoning)
- **Identify patterns** (trend analysis)

| Query Type | Example | Requires Image |
|------------|---------|----------------|
| Visual Reading | "What value is in row 3, column 'F1-Score'?" | Yes |
| Comparison | "Which model achieves the highest BLEU score?" | Depends |
| Calculation | "What is the average accuracy across all methods?" | No |
| Pattern | "Which metric shows the most variation across models?" | Yes |

**Prompt Strategy:**
```
TABLE_VISUAL: Use when image available - test visual cell reading
TABLE_HYBRID: Use when both text+image - mix visual and textual queries
```

### FIGURE Queries

Figures include charts, plots, and visualizations. Queries should test:
- **Trend reading** (line charts, time series)
- **Value estimation** (bar charts, scatter plots)
- **Element identification** (legends, labels, colors)
- **Interpretation** (what the figure demonstrates)

| Query Type | Example | Requires Image |
|------------|---------|----------------|
| Trend | "What trend does the blue line show between x=0 and x=10?" | Yes |
| Reading | "What is the approximate y-value when x=5?" | Yes |
| Identification | "What do the different colored bars represent?" | Yes |
| Interpretation | "What conclusion can be drawn about model A vs B?" | Depends |

**Key Principle:** Figure queries should NOT be answerable from caption alone.

### FORMULA Queries

Mathematical formulas require both visual parsing and semantic understanding:
- **Symbol recognition** (Greek letters, operators)
- **Structure understanding** (summations, integrals, fractions)
- **Semantic meaning** (what the formula computes)
- **Application** (how to use the formula)

| Query Type | Example | Requires Image |
|------------|---------|----------------|
| Symbol | "What Greek letter appears in the denominator?" | Yes |
| Structure | "How many terms are in the summation?" | Yes |
| Semantic | "What physical quantity does this formula calculate?" | No |
| Variable | "What does the subscript i represent in context?" | Depends |
| Application | "How would you compute the loss for a single sample?" | No |

### INFOGRAPHIC Queries

Infographics (diagrams, flowcharts, architectures) test structural understanding:
- **Component identification** (what modules exist)
- **Flow/sequence** (order of operations)
- **Connections** (how parts relate)
- **Role understanding** (purpose of each part)

| Query Type | Example | Requires Image |
|------------|---------|----------------|
| Component | "What are the main modules in this architecture?" | Yes |
| Flow | "What is the sequence of operations from input to output?" | Yes |
| Connection | "How does the encoder connect to the decoder?" | Yes |
| Role | "What is the purpose of the attention mechanism?" | Depends |

---

## Contrastive Learning Triplet Design

### Triplet Structure

```python
ContrastiveTriplet = {
    "query": str,                    # The question
    "query_type": str,               # Type of query
    "query_modality": str,           # unimodal_visual / multimodal_grounded / etc.
    "requires_image": bool,          # Does query need image to answer?

    "positive": {
        "content": str,              # Text content (markdown table, caption, etc.)
        "image_path": str,           # Path to image (if available)
        "modal_type": str,           # table / figure / formula / infographic
        "metadata": dict             # Additional info (rows, cols, etc.)
    },

    "negatives": [
        {
            "content": str,
            "image_path": str,
            "modal_type": str,
            "negative_type": str,    # hard_same_modal / cross_modal / random
            "similarity_score": float # Embedding similarity (for hard negatives)
        }
    ],

    "difficulty_score": float        # 0.0 - 1.0
}
```

### Negative Sampling Strategies

#### 1. Hard Same-Modal Negatives (60%)
Select passages of the same modality with similar but different content:
- Same table topic, different data
- Similar chart type, different trends
- Related formula, different computation

**Why:** Forces model to learn fine-grained distinctions within modality.

#### 2. Cross-Modal Negatives (30%)
Select passages of different modality that might seem related:
- Table mentioned in text but query asks about figure
- Formula referenced in diagram but query asks about table

**Why:** Forces model to correctly match query to appropriate modality.

#### 3. Random Negatives (10%)
Randomly selected passages from other documents.

**Why:** Provides easy negatives to stabilize training.

---

## Query Generation Pipeline

### With VLM (Recommended)

```
Passage ─────┬─── image_path ───▶ ┌─────────────┐
             │                    │             │
             └─── content ──────▶ │  VLM Model  │───▶ Queries
                                  │ (Qwen3-VL)  │
             └─── context ──────▶ │             │
                                  └─────────────┘
```

**Advantages:**
- Can see actual images for visual queries
- Generates image-grounded questions
- Better query quality for visual content

### Without VLM (Fallback)

```
Passage ─────┬─── content ──────▶ ┌─────────────┐
             │                    │             │
             └─── context ──────▶ │  Text LLM   │───▶ Queries
                                  │ (GPT-4o-mini)│
                                  └─────────────┘
```

**Limitations:**
- Cannot generate true visual queries
- Questions based on text descriptions only
- Lower quality for figure/infographic queries

---

## Implementation Usage

### Basic Usage

```python
from src.generators.vlm_query_generator import MultimodalQueryGenerator

# Initialize with Qwen3-VL via vLLM
generator = MultimodalQueryGenerator(
    backend="qwen_vllm",
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    api_base="http://localhost:8000/v1"
)

# Generate queries for a passage
queries = generator.generate(passage, num_queries=4)

# Generate with explicit image
queries = generator.generate_with_image(
    image_path="/path/to/table.png",
    text_content="Table showing model performance...",
    modal_type="table",
    num_queries=4
)
```

### Batch Processing

```python
# Generate for multiple passages
results = generator.generate_batch(passages, num_queries=3, max_workers=4)

# Generate cross-modal queries
cross_queries = generator.generate_cross_modal_queries(doc_passages, num_queries=2)
```

---

## Query Quality Guidelines

### Good Queries

1. **Specific:** Reference concrete elements (row 3, blue line, variable x)
2. **Grounded:** Cannot be answered without the passage
3. **Diverse:** Mix of difficulty levels and types
4. **Unambiguous:** Single clear answer

### Bad Queries

1. **Generic:** "What does this show?" (too vague)
2. **Answerable from common knowledge:** "What is machine learning?"
3. **Unanswerable:** Asking about information not in the passage
4. **Ambiguous:** Multiple valid interpretations

---

## Recommended Configuration

```yaml
vlm_query_generation:
  enabled: true
  backend: "qwen_vllm"  # Best balance of speed and quality

  queries_per_modality:
    table: 4      # Rich structured data
    figure: 4     # Complex visual content
    formula: 3    # Moderate complexity
    infographic: 4  # Complex diagrams
    text: 2       # Simpler content

  query_type_ratios:
    unimodal_visual: 0.3      # 30% pure visual
    unimodal_text: 0.2        # 20% pure text
    multimodal_grounded: 0.35 # 35% grounded
    cross_modal_reasoning: 0.15 # 15% cross-modal
```

---

## Evaluation Metrics

### Query Quality Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Answerability | Can query be answered from passage? | >95% |
| Groundedness | Does query require passage (not common knowledge)? | >90% |
| Type Diversity | Distribution across query types | Entropy >0.8 |
| Difficulty Spread | Range of difficulty scores | 0.3-0.9 |

### Triplet Quality Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Hard Negative Similarity | Cosine similarity of hard negatives | 0.6-0.85 |
| Positive Relevance | Query-positive embedding similarity | >0.7 |
| Margin | pos_sim - max(neg_sim) | >0.15 |
