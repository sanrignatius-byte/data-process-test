"""Graph construction, hub detection, and candidate enumeration.

Modules:
  - ``algorithms``  — PageRank, label propagation
  - ``hubs``        — bridge hub / adjacent backbone bridge detection
  - ``candidates``  — multi-hop candidate path enumeration

The heavy topology graph construction and analysis logic from
``analyze_latex_graph_topology.py`` is re-exported from here after
scripts are updated to import from ``src.graph``.
"""

__all__: list = []
