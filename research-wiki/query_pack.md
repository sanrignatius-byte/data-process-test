# Query Pack

Project direction:
Use multimodal document graphs to improve evidence localization, support QA, and synthesize high-quality SFT data. Current priority is delivery and validated retrieval gains, not graph complexity for its own sake.

Top gaps:
- G1: Dense retrieval still misses early evidence positions on multimodal documents.
- G2: Cross-document graph connectivity is weak; intra-doc virtual edges alone are not enough.
- G3: Delivery data needs positives, negatives, and lighter QC.
- G4: QA-side evidence for graph value is missing.
- G5: Repo structure is crowded and raises maintenance overhead.

Top ideas:
- idea:001 explicit bridge-edge rerank with hub-aware static prior. This is the dominant contribution and the current best-supported method.
- idea:002 cross-doc section-summary similarity edges with citation boost. This is the next validation gate, not the main story yet.
- idea:003 Method C long-chain query synthesis. Useful for data generation, but currently a supporting branch rather than the main paper story.

Failed or downgraded ideas:
- Treating intra-doc virtual edges as the default graph enhancement is currently a bad default for precision-oriented rerank.
- Using pass rate or multihop coverage alone as the proof of project value is not acceptable.
- Letting graph experiments silently run on the wrong layer is no longer allowed; `--graph-sources` must be explicit.

Top claims:
- C1 supported: explicit-only plus static prior improves rerank over dense baseline on the corrected rebuilt set.
- C2 supported: adding all intra-doc virtual edges dilutes precision metrics relative to explicit-only.
- C3 pending: cross-doc summary edges with citation boost may improve corrected explicit-only rerank.
- C4 framing: graph value must be proven on retrieval, QA, and data synthesis.

Active chain:
Wrong graph default -> old graph conclusions become provisional -> corrected explicit-only rerank becomes the new baseline -> cross-doc summary edges are tested only against that baseline.

Open unknowns:
- Will cross-doc summary edges improve R@1 and MRR without reintroducing noise?
- How should graph signals be injected into QA evaluation?
- What is the minimal QC needed for reliable delivery data?

