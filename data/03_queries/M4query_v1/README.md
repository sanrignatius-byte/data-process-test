# M4query v1 — Multi-hop Cross-modal QA Dataset

Multi-hop, cross-modal question-answer dataset built from academic papers,
designed for contrastive learning and embedding model training.

## Dataset Structure

```
M4query_v1/
├── queries.jsonl              # 473 QC-passed queries
├── corpus.jsonl               # 2809 passage chunks (MinerU elements)
├── train_triplets.jsonl       # 473 contrastive triplets (query→pos/neg)
├── qrels.jsonl                # 946 query-passage relevance labels
├── stats.json                 # Dataset statistics
├── documents/                 # Source documents (57 papers)
│   └── {doc_id}/
│       ├── mineru/            # MinerU parsed: structure.json, images, formulas
│       └── latex/             # LaTeX source (when available)
├── graphs/                    # Document knowledge graphs
│   ├── pruned_graph.json
│   ├── hub_scores.json
│   └── multimodal_elements.json
└── candidates/                # Enriched pair candidates
    ├── hub_candidates_intra_doc.json
    └── m2_diverse_candidates_intra_doc.json
```

## Training Data Format

### `train_triplets.jsonl` (for contrastive learning / embedding training)

Each line:
```json
{
  "query_id": "l3_de_1511.00830_0000",
  "query": "...",
  "doc_id": "1511.00830",
  "hop_distance": 3,
  "positive": [
    {"passage_id": "...", "text": "...", "type": "paragraph"}
  ],
  "negative": [
    {"passage_id": "...", "text": "...", "type": "table", "neg_source": "intra_doc"},
    {"passage_id": "...", "text": "...", "type": "paragraph", "neg_source": "cross_doc"}
  ]
}
```

### `corpus.jsonl` (passage pool for retrieval)

Each line:
```json
{
  "passage_id": "1511.00830_elem_042",
  "doc_id": "1511.00830",
  "text": "...",
  "type": "paragraph|table|figure|equation",
  "section": "3.2 Method",
  "page": 5
}
```

### `qrels.jsonl` (relevance labels)

Each line:
```json
{
  "query_id": "l3_de_1511.00830_0000",
  "passage_id": "1511.00830_elem_042",
  "relevance": 1
}
```

## Statistics

- Queries: 473
- Corpus passages: 2809
- Training triplets: 473
- Unique documents: 57
- Hop distribution: {2: 189, 3: 275, 4: 2, 5: 7}
- Style: {'academic': 354, 'real_user': 119}
- MinerU coverage: 57/57
- LaTeX coverage: 56/57

Generated: 2026-04-16 14:13
