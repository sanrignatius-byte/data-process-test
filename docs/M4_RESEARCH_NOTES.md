# M4 Multi-hop Query Generation: Research Notes and Implementation Guide

## Overview

This document summarizes the research findings and implementation approach for M4 (Multi-hop, Multi-modal, Multi-document, Multi-turn) query generation for training data.

## Academic Background

### Key Benchmarks and Datasets

#### 1. M4DocBench (Dong et al., 2025)
- **Purpose**: First benchmark for "deep research" in multimodal document collections
- **Scale**: 158 expert-annotated questions across 304 multimodal documents
- **Domains**: Research papers, insurance, education, finance
- **Key Features**:
  - Multi-hop reasoning requiring cross-document evidence linking
  - Cross-modal integration (text, tables, figures, equations)
  - Rich evidence annotation (document, page, fine-grained regions)
- **Results**: Doc-Researcher achieves 50.6% accuracy, 3.4x better than baselines

#### 2. CoQA (Conversational Question Answering)
- **Scale**: 127,000 questions across 8,000 conversations
- **Coreference Statistics**:
  - ~50% explicit coreference (pronouns: "him", "it", "her")
  - ~20% implicit coreference (requiring inference)
- **Key Insight**: Natural multi-turn dialogues heavily rely on coreference

#### 3. MT-Bench++ (Multi-turn Benchmark)
- **Design**: 8-turn evaluation with 80 sessions, 640 utterances
- **Focus**: Ellipsis and anaphora in follow-up questions
- **Key Insight**: Testing multi-turn context understanding requires systematic coreference

### Key Methods

#### TRACE: Knowledge Triple-Grounded Reasoning Chains (EMNLP 2024)
- Uses KG triples to construct condensed reasoning chains
- Finding: More condensed chains improve Exact Match over full document retrieval
- **Implication**: Evidence chains should be explicit and structured, not implicit

#### RT-RAG: Tree-Structured Decomposition
- Decomposes complex questions into consensus-validated tree structure
- Bottom-up traversal with query refinement
- Hierarchical information integration
- **Implication**: Multi-hop questions benefit from explicit decomposition

#### HiRAG: Hierarchical RAG
- 5-module multi-hop QA system
- Document/chunk-level hierarchical retrieval
- Single-candidate, rethinking-based filtering
- **Implication**: Hierarchical evidence organization improves retrieval

#### Cross-Document Coreference
- SciCo-Radar: Dynamic definitions via context retrieval
- Graph-based inference for entity relationships
- **Implication**: Cross-document linking requires explicit entity resolution

## M4 Dimensions Explained

### 1. Multi-hop Reasoning

**Definition**: Questions requiring 2+ distinct evidence points connected by reasoning steps.

**Key Principles (Bridge Logic)**:
- Each hop must be connected by a shared entity/concept
- Reasoning steps must be explicit: `Evidence A → Bridge Entity → Evidence B → Answer`
- If reasoning path cannot be written explicitly, the question is invalid

**Implementation**:
```python
# From cross_document_linker.py
def build_evidence_chain(passages, bridge_entities):
    # Ensure each consecutive node shares at least one entity
    for i in range(len(nodes) - 1):
        shared = set(node_a.entities) & set(node_b.entities)
        if not shared:
            # Invalid chain - no bridge
            return None
```

**Validation Criteria**:
- `len(reasoning_steps) >= 2`
- Each step connects to the next via shared entity
- Answer cannot be derived from any single evidence source

### 2. Multi-modal Integration

**Definition**: Questions requiring evidence from 2+ modalities (text, table, figure, formula).

**Key Principles (Dependency Check)**:
- Reject questions answerable from text-only or visual-only
- Each modality must contribute unique information
- Cross-modal reasoning should be explicit

**Implementation**:
```python
# Validation
modalities = question.get("modalities_required") or []
if len(set(modalities)) < 2:
    return False  # Reject single-modality questions
```

**Modality Types**:
| Type | Examples | Typical Information |
|------|----------|-------------------|
| Text | Paragraphs, captions | Methods, analysis, context |
| Table | Data tables | Numerical results, comparisons |
| Figure | Charts, diagrams | Trends, architectures, flows |
| Formula | Equations | Mathematical relationships |

### 3. Multi-document Synthesis

**Definition**: Questions requiring evidence from 2+ documents.

**Key Principles**:
- Documents must have semantic relationship (shared entities/concepts)
- Cross-document reasoning requires entity linking
- Cannot be simply "compare paper A and paper B" (too vague)

**Implementation**:
```python
# From cross_document_linker.py
def find_cross_document_links(doc_ids):
    # Compare entities across document pairs
    for ent_a in entities_a:
        for ent_b in entities_b:
            similarity = compute_entity_similarity(ent_a, ent_b)
            if similarity >= threshold:
                links.append(CrossDocumentLink(...))
```

**Entity Types for Cross-Document Linking**:
- Methods/Models: BERT, Transformer, GPT
- Datasets: SQuAD, GLUE, ImageNet
- Metrics: F1, accuracy, BLEU
- Tasks: QA, NER, classification
- Concepts: attention, embedding, fine-tuning

### 4. Multi-turn Interaction

**Definition**: Questions phrased as 2-3 turn dialogues with natural coreference.

**Key Principles (Coreference Injection)**:
- Follow-up turns must use pronouns instead of repeating entities
- Each turn should feel natural in conversation
- Full answer only obtainable after all turns

**Implementation**:
```python
# Coreference validation
pronouns = {"it", "they", "that", "those", "these", "its", "their"}
followups = " ".join(turns[1:]).lower()
if not any(p in followups.split() for p in pronouns):
    return False  # No coreference in follow-ups
```

**Good Example**:
```
Turn 1: "What architecture does BERT use for encoding?"
Turn 2: "How does it differ from the original Transformer?"
Turn 3: "What advantages does this give for downstream tasks?"
```

**Bad Example** (no coreference):
```
Turn 1: "What architecture does BERT use?"
Turn 2: "What architecture does GPT use?"  # No connection
```

## Implementation Architecture

### Module Structure

```
src/
├── linkers/
│   ├── __init__.py
│   └── cross_document_linker.py    # Entity extraction & linking
├── generators/
│   ├── __init__.py
│   ├── query_generator.py          # Original query generation
│   └── m4_query_generator.py       # Enhanced M4 generation
```

### Data Flow

```
Passages (from MinerU)
    ↓
[1] Entity Extraction
    - Pattern matching for scientific entities
    - Method/Dataset/Metric/Task/Concept extraction
    ↓
[2] Cross-Document Linking
    - Entity similarity computation
    - Build entity co-occurrence graph
    ↓
[3] Evidence Chain Construction
    - Find linkable passage groups
    - Build chains with bridge entities
    - Generate reasoning steps
    ↓
[4] M4 Query Generation
    - Bridging question generation
    - Multi-turn conversion with coreference
    - Full M4 synthesis
    ↓
[5] Evidence Verification
    - Validate answer against sources
    - Check reasoning step validity
    ↓
[6] Output: Validated M4 Queries
```

### Key Classes

#### CrossDocumentLinker
- `extract_entities_from_passage()`: Pattern-based entity extraction
- `build_document_entities()`: Group mentions into entities
- `find_cross_document_links()`: Find entity matches across documents
- `build_evidence_chain()`: Construct reasoning chains
- `find_linkable_passage_groups()`: Select passages for query generation

#### M4QueryGenerator
- `generate_bridging_question()`: Create multi-hop foundation
- `convert_to_multi_turn()`: Add coreference for dialogue
- `generate_full_m4_query()`: Synthesize all dimensions
- `verify_evidence()`: Validate query against sources

## Prompt Engineering

### Bridging Question Prompt
Key elements:
1. Explicit evidence sources with metadata
2. Shared concept identification
3. Requirement that BOTH sources are needed

### Multi-turn Conversion Prompt
Key elements:
1. Original question + answer + entities
2. Explicit coreference requirements
3. Example of good coreference usage

### Full M4 Generation Prompt
Key elements:
1. Evidence chain with node details
2. All four M4 requirements stated
3. Validation criteria for rejection
4. Structured output with evidence mapping

## Quality Assurance

### Validation Checks

1. **Structural Validation**:
   - `len(turns) >= 2`
   - `len(reasoning_steps) >= 2`
   - `len(doc_ids) >= 2`
   - `len(modalities) >= 2`

2. **Semantic Validation**:
   - Coreference in follow-up turns
   - Answer supported by evidence
   - Reasoning steps logically connect

3. **Evidence Verification**:
   - LLM-based validation of answer derivability
   - Check each claimed evidence source

### Common Failure Modes

| Issue | Detection | Mitigation |
|-------|-----------|------------|
| Single-source answerable | LLM verification | Require multi-source dependency |
| Missing coreference | Pronoun pattern check | Reject and regenerate |
| Hallucinated evidence | Source text matching | Verify spans exist |
| Trivial multi-hop | Reasoning step analysis | Require substantive bridges |

## Usage Examples

### Basic Usage

```python
from src.linkers import CrossDocumentLinker
from src.generators import create_m4_generator

# Initialize
linker = CrossDocumentLinker()
generator = create_m4_generator(provider="anthropic")
generator.linker = linker

# Generate queries
queries = generator.generate_queries_for_passages(
    passages=all_passages,
    num_queries=10,
    require_full_m4=True
)

# Check statistics
stats = generator.get_generation_statistics()
print(f"Entities extracted: {stats['total_entities']}")
print(f"Cross-doc links: {stats['total_cross_doc_links']}")
```

### Custom Entity Patterns

```python
# Add domain-specific patterns
linker.ENTITY_PATTERNS[EntityType.METHOD].append(
    r'\b(RAG|Retrieval-Augmented Generation)\b'
)
```

## References

### Academic Papers

1. **M4DocBench**: Dong et al. (2025). "Multimodal Document Research Benchmark"
2. **TRACE**: "Constructing Knowledge-Grounded Reasoning Chains" (EMNLP 2024)
3. **CoQA**: Reddy et al. "Conversational Question Answering"
4. **MT-Bench++**: Multi-turn evaluation benchmark
5. **HiRAG**: Zhang et al. (2024). Hierarchical RAG for multi-hop QA
6. **SciCo-Radar**: Cross-document coreference with dynamic definitions

### Web Resources

- [M4DocBench](https://www.emergentmind.com/topics/m4docbench)
- [MMMU Benchmark](https://mmmu-benchmark.github.io/)
- [Multi-hop QA Survey](https://arxiv.org/pdf/2204.09140)

## Future Improvements

1. **Embedding-based Entity Linking**: Use sentence transformers for semantic similarity
2. **Graph Neural Networks**: Model document relationships as graphs
3. **Iterative Refinement**: Multi-round generation with self-critique
4. **Human-in-the-loop**: Expert validation for high-quality subset
5. **Domain Adaptation**: Fine-tune patterns for specific research domains
