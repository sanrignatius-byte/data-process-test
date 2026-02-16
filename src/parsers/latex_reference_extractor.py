"""
LaTeX Reference Extractor

Parses .tex / .bbl files to extract structured cross-references and build
a per-document reference DAG.  Works on arXiv e-print sources downloaded by
scripts/download_latex_sources.py.

Key outputs per document:
  - labels:  {label_key → LabelInfo}   (anchors defined by \\label{})
  - refs:    [RefInstance, ...]          (all \\ref / \\eqref / \\cite usages)
  - edges:   [source_label → target_label, ...]   (directed reference edges)
  - bib:     {cite_key → BibEntry}      (from .bbl)

Designed to be merged with the MinerU-based DAG in
src/linkers/multimodal_relationship_builder.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class LabelType(Enum):
    """Semantic type inferred from label prefix or enclosing environment."""
    FIGURE = "figure"
    TABLE = "table"
    EQUATION = "equation"
    SECTION = "section"
    ALGORITHM = "algorithm"
    APPENDIX = "appendix"
    THEOREM = "theorem"       # theorem, lemma, proposition, corollary, conjecture, claim
    DEFINITION = "definition" # definition, assumption, hypothesis, axiom, notation
    PROOF = "proof"
    EXAMPLE = "example"       # example, remark, note, observation, exercise
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LabelInfo:
    """A \\label{} anchor in the LaTeX source."""
    key: str                  # e.g. "fig:model"
    label_type: LabelType
    line_no: int              # 1-based line number (original file)
    environment: Optional[str]  # enclosing env: figure, table, equation …
    caption: Optional[str]    # nearest \\caption{} text
    file_path: Optional[str]  # .tex file where defined
    _merged_idx: int = -1     # 0-based line index in merged content (internal use)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "label_type": self.label_type.value,
            "line_no": self.line_no,
            "environment": self.environment,
            "caption": self.caption,
            "file_path": self.file_path,
        }


@dataclass
class RefInstance:
    """A single \\ref{} / \\eqref{} / \\cite{} usage."""
    target_key: str           # the label / cite key being referenced
    ref_type: str             # "ref" | "eqref" | "pageref" | "cite"
    line_no: int              # 1-based line number (original file)
    context: str              # surrounding text (±80 chars)
    source_env: Optional[str]  # enclosing env at the reference site
    file_path: Optional[str]
    _merged_idx: int = -1     # 0-based line index in merged content (internal use)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_key": self.target_key,
            "ref_type": self.ref_type,
            "line_no": self.line_no,
            "context": self.context[:300],
            "source_env": self.source_env,
            "file_path": self.file_path,
        }


@dataclass
class BibEntry:
    """A bibliography entry parsed from .bbl."""
    cite_key: str
    title: Optional[str] = None
    authors: Optional[str] = None
    year: Optional[str] = None
    raw: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cite_key": self.cite_key,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
        }


@dataclass
class LatexRefEdge:
    """Directed reference edge: source_label → target_label."""
    source_label: str
    target_label: str
    source_type: str          # LabelType.value of source
    target_type: str          # LabelType.value of target
    ref_text: str             # original \\ref{...} text
    context: str              # surrounding text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_label": self.source_label,
            "target_label": self.target_label,
            "source_type": self.source_type,
            "target_type": self.target_type,
            "ref_text": self.ref_text,
            "context": self.context[:300],
        }


@dataclass
class LatexDocumentGraph:
    """Full reference graph for one LaTeX document."""
    doc_id: str
    labels: Dict[str, LabelInfo] = field(default_factory=dict)
    refs: List[RefInstance] = field(default_factory=list)
    edges: List[LatexRefEdge] = field(default_factory=list)
    bib: Dict[str, BibEntry] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "num_labels": len(self.labels),
            "num_refs": len(self.refs),
            "num_edges": len(self.edges),
            "num_bib_entries": len(self.bib),
            "labels": {k: v.to_dict() for k, v in self.labels.items()},
            "refs": [r.to_dict() for r in self.refs],
            "edges": [e.to_dict() for e in self.edges],
            "bib": {k: v.to_dict() for k, v in self.bib.items()},
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# \label{key}
RE_LABEL = re.compile(r'\\label\{([^}]+)\}')

# \ref{key}, \eqref{key}, \pageref{key}, \autoref{key}, \cref{key}
RE_REF = re.compile(
    r'\\(?P<cmd>ref|eqref|pageref|autoref|cref|Cref)\{(?P<key>[^}]+)\}'
)

# \cite{key1, key2}, \citep{...}, \citet{...}, \citealp{...}
RE_CITE = re.compile(
    r'\\cite(?:p|t|alp|alt|author|year|num)?\*?\{(?P<keys>[^}]+)\}'
)

# \begin{env} / \end{env}
RE_BEGIN_ENV = re.compile(r'\\begin\{(\w+)\}')
RE_END_ENV = re.compile(r'\\end\{(\w+)\}')

# \caption{...}  (may span lines, we capture the first line only here)
RE_CAPTION = re.compile(r'\\caption(?:\[[^\]]*\])?\{(.+)')

# \input{file} / \include{file}
RE_INPUT = re.compile(r'\\(?:input|include)\{([^}]+)\}')

# \includegraphics[...]{file}
RE_INCLUDEGRAPHICS = re.compile(
    r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}'
)

# \section{}, \subsection{}, etc.
RE_SECTION = re.compile(
    r'\\(section|subsection|subsubsection|paragraph|chapter)\*?\{([^}]+)\}'
)

# Label prefix → type mapping
_PREFIX_MAP = {
    "fig": LabelType.FIGURE,
    "figure": LabelType.FIGURE,
    "tab": LabelType.TABLE,
    "table": LabelType.TABLE,
    "tbl": LabelType.TABLE,
    "eq": LabelType.EQUATION,
    "eqn": LabelType.EQUATION,
    "equation": LabelType.EQUATION,
    "sec": LabelType.SECTION,
    "section": LabelType.SECTION,
    "subsec": LabelType.SECTION,
    "chap": LabelType.SECTION,
    "alg": LabelType.ALGORITHM,
    "algo": LabelType.ALGORITHM,
    "app": LabelType.APPENDIX,
    "appendix": LabelType.APPENDIX,
    # Theorem-like
    "thm": LabelType.THEOREM,
    "theorem": LabelType.THEOREM,
    "lem": LabelType.THEOREM,
    "lemma": LabelType.THEOREM,
    "prop": LabelType.THEOREM,
    "proposition": LabelType.THEOREM,
    "cor": LabelType.THEOREM,
    "corollary": LabelType.THEOREM,
    "conj": LabelType.THEOREM,
    "conjecture": LabelType.THEOREM,
    "claim": LabelType.THEOREM,
    "fact": LabelType.THEOREM,
    # Definition-like
    "def": LabelType.DEFINITION,
    "defn": LabelType.DEFINITION,
    "definition": LabelType.DEFINITION,
    "assumption": LabelType.DEFINITION,
    "hyp": LabelType.DEFINITION,
    "hypothesis": LabelType.DEFINITION,
    "axiom": LabelType.DEFINITION,
    "notation": LabelType.DEFINITION,
    "condition": LabelType.DEFINITION,
    # Proof
    "proof": LabelType.PROOF,
    "pf": LabelType.PROOF,
    # Example/remark
    "ex": LabelType.EXAMPLE,
    "example": LabelType.EXAMPLE,
    "rem": LabelType.EXAMPLE,
    "remark": LabelType.EXAMPLE,
    "note": LabelType.EXAMPLE,
    "obs": LabelType.EXAMPLE,
    "observation": LabelType.EXAMPLE,
}

# Environment → type mapping
_ENV_MAP = {
    "figure": LabelType.FIGURE,
    "figure*": LabelType.FIGURE,
    "subfigure": LabelType.FIGURE,
    "wrapfigure": LabelType.FIGURE,
    "table": LabelType.TABLE,
    "table*": LabelType.TABLE,
    "tabular": LabelType.TABLE,
    "tabular*": LabelType.TABLE,
    "longtable": LabelType.TABLE,
    "equation": LabelType.EQUATION,
    "equation*": LabelType.EQUATION,
    "align": LabelType.EQUATION,
    "align*": LabelType.EQUATION,
    "alignat": LabelType.EQUATION,
    "alignat*": LabelType.EQUATION,
    "gather": LabelType.EQUATION,
    "gather*": LabelType.EQUATION,
    "multline": LabelType.EQUATION,
    "multline*": LabelType.EQUATION,
    "eqnarray": LabelType.EQUATION,
    "eqnarray*": LabelType.EQUATION,
    "flalign": LabelType.EQUATION,
    "flalign*": LabelType.EQUATION,
    "math": LabelType.EQUATION,
    "displaymath": LabelType.EQUATION,
    "algorithm": LabelType.ALGORITHM,
    "algorithm*": LabelType.ALGORITHM,
    "algorithmic": LabelType.ALGORITHM,
    "algorithm2e": LabelType.ALGORITHM,
    # Theorem-like
    "theorem": LabelType.THEOREM,
    "lemma": LabelType.THEOREM,
    "proposition": LabelType.THEOREM,
    "corollary": LabelType.THEOREM,
    "conjecture": LabelType.THEOREM,
    "claim": LabelType.THEOREM,
    "fact": LabelType.THEOREM,
    # Definition-like
    "definition": LabelType.DEFINITION,
    "assumption": LabelType.DEFINITION,
    "hypothesis": LabelType.DEFINITION,
    "axiom": LabelType.DEFINITION,
    "notation": LabelType.DEFINITION,
    "condition": LabelType.DEFINITION,
    # Proof
    "proof": LabelType.PROOF,
    "proof*": LabelType.PROOF,
    # Example/remark
    "example": LabelType.EXAMPLE,
    "example*": LabelType.EXAMPLE,
    "remark": LabelType.EXAMPLE,
    "remark*": LabelType.EXAMPLE,
    "note": LabelType.EXAMPLE,
    "observation": LabelType.EXAMPLE,
    "exercise": LabelType.EXAMPLE,
}


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------

class LaTeXReferenceExtractor:
    """
    Extract labels, refs, cites and build a per-document reference graph
    from LaTeX source files.
    """

    def __init__(self, context_chars: int = 120):
        self.context_chars = context_chars

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def extract(
        self,
        doc_id: str,
        extract_dir: Path,
        main_tex: Optional[Path] = None,
        bbl_path: Optional[Path] = None,
    ) -> LatexDocumentGraph:
        """
        Extract full reference graph for one document.

        Args:
            doc_id: arXiv ID or document identifier
            extract_dir: root dir of the extracted LaTeX source
            main_tex: path to main .tex (auto-detected if None)
            bbl_path: path to .bbl file (auto-detected if None)
        """
        graph = LatexDocumentGraph(doc_id=doc_id)

        # Find main .tex
        if main_tex is None:
            main_tex = self._find_main_tex(extract_dir)
        if main_tex is None:
            graph.metadata["error"] = "no_tex_found"
            return graph

        # Resolve all \input{} / \include{} → merged content
        full_content, file_lines = self._resolve_inputs(main_tex, extract_dir)
        if not full_content:
            graph.metadata["error"] = "empty_tex"
            return graph

        lines = full_content.split("\n")
        graph.metadata["total_lines"] = len(lines)
        graph.metadata["main_tex"] = str(main_tex.relative_to(extract_dir))

        # 1) Extract labels
        env_stack = self._build_env_map(lines)
        graph.labels = self._extract_labels(lines, env_stack, file_lines)

        # 2) Extract refs
        graph.refs = self._extract_refs(lines, env_stack, file_lines)

        # 3) Build edges: reference edges (enclosing element + section fallback)
        graph.edges = self._build_edges(graph.labels, graph.refs, lines, env_stack)

        # 4) Add containment edges: section → contained elements
        containment = self._build_containment_edges(graph.labels, lines)
        graph.edges.extend(containment)

        # 5) Parse .bbl
        if bbl_path is None:
            bbl_path = self._find_bbl(extract_dir)
        if bbl_path is not None:
            graph.bib = self._parse_bbl(bbl_path)

        # Stats
        graph.metadata["num_input_files"] = len(set(
            fl[1] for fl in file_lines.values() if fl[1]
        )) if file_lines else 1

        return graph

    # -----------------------------------------------------------------------
    # File resolution
    # -----------------------------------------------------------------------

    @staticmethod
    def _find_main_tex(extract_dir: Path) -> Optional[Path]:
        """Same heuristic as download_latex_sources.py."""
        tex_files = list(extract_dir.rglob("*.tex"))
        if not tex_files:
            return None
        for f in tex_files:
            try:
                text = f.read_text(errors="replace")
                if "\\documentclass" in text:
                    return f
            except Exception:
                continue
        for name in ("main.tex", "paper.tex", "ms.tex", "article.tex"):
            for f in tex_files:
                if f.name.lower() == name:
                    return f
        return max(tex_files, key=lambda f: f.stat().st_size)

    @staticmethod
    def _find_bbl(extract_dir: Path) -> Optional[Path]:
        bbl_files = list(extract_dir.rglob("*.bbl"))
        return bbl_files[0] if bbl_files else None

    def _resolve_inputs(
        self, main_tex: Path, extract_dir: Path, _visited: Optional[Set[str]] = None
    ) -> Tuple[str, Dict[int, Tuple[int, Optional[str]]]]:
        """
        Recursively resolve \\input{} / \\include{}, return merged text and
        a mapping: merged_line_no → (original_line_no, source_file).
        """
        if _visited is None:
            _visited = set()

        canonical = str(main_tex.resolve())
        if canonical in _visited:
            return "", {}
        _visited.add(canonical)

        try:
            content = main_tex.read_text(errors="replace")
        except Exception:
            return "", {}

        result_lines: List[str] = []
        file_lines: Dict[int, Tuple[int, Optional[str]]] = {}
        rel_path = str(main_tex.relative_to(extract_dir)) if extract_dir in main_tex.parents or main_tex.parent == extract_dir else main_tex.name

        for orig_line_no, line in enumerate(content.split("\n"), 1):
            # Strip comments (but not \%)
            stripped = self._strip_comment(line)

            m = RE_INPUT.search(stripped)
            if m:
                input_name = m.group(1)
                if not input_name.endswith(".tex"):
                    input_name += ".tex"
                # Try multiple locations
                candidates = [
                    main_tex.parent / input_name,
                    extract_dir / input_name,
                ]
                resolved = None
                for c in candidates:
                    if c.exists():
                        resolved = c
                        break

                if resolved:
                    sub_content, sub_file_lines = self._resolve_inputs(
                        resolved, extract_dir, _visited
                    )
                    sub_lines = sub_content.split("\n") if sub_content else []
                    for sl in sub_lines:
                        merged_idx = len(result_lines)
                        result_lines.append(sl)
                        # Map to the sub-file (sub_file_lines already set)
                        if merged_idx in sub_file_lines:
                            file_lines[merged_idx] = sub_file_lines[merged_idx]
                        else:
                            file_lines[merged_idx] = (0, str(resolved.relative_to(extract_dir)) if extract_dir in resolved.parents or resolved.parent == extract_dir else resolved.name)
                    continue

            merged_idx = len(result_lines)
            result_lines.append(stripped)
            file_lines[merged_idx] = (orig_line_no, rel_path)

        return "\n".join(result_lines), file_lines

    @staticmethod
    def _strip_comment(line: str) -> str:
        """Strip LaTeX comment (% not preceded by \\)."""
        i = 0
        while i < len(line):
            if line[i] == '%' and (i == 0 or line[i - 1] != '\\'):
                return line[:i]
            i += 1
        return line

    # -----------------------------------------------------------------------
    # Environment tracking
    # -----------------------------------------------------------------------

    def _build_env_map(
        self, lines: List[str]
    ) -> Dict[int, List[str]]:
        """
        For each line, compute the stack of enclosing environments.
        Returns: {line_idx → [env_name, ...]}  (innermost last)
        """
        env_map: Dict[int, List[str]] = {}
        stack: List[str] = []

        for idx, line in enumerate(lines):
            # Process all \begin and \end on this line (in order)
            for m in RE_BEGIN_ENV.finditer(line):
                stack.append(m.group(1))
            env_map[idx] = list(stack)
            for m in RE_END_ENV.finditer(line):
                env_name = m.group(1)
                # Pop matching env (tolerate unbalanced)
                for j in range(len(stack) - 1, -1, -1):
                    if stack[j] == env_name:
                        stack.pop(j)
                        break

        return env_map

    def _innermost_env(self, env_stack: Dict[int, List[str]], line_idx: int) -> Optional[str]:
        """Get the innermost enclosing environment at a given line."""
        envs = env_stack.get(line_idx, [])
        return envs[-1] if envs else None

    # -----------------------------------------------------------------------
    # Label extraction
    # -----------------------------------------------------------------------

    def _extract_labels(
        self,
        lines: List[str],
        env_stack: Dict[int, List[str]],
        file_lines: Dict[int, Tuple[int, Optional[str]]],
    ) -> Dict[str, LabelInfo]:
        labels: Dict[str, LabelInfo] = {}

        for idx, line in enumerate(lines):
            for m in RE_LABEL.finditer(line):
                key = m.group(1).strip()
                env = self._innermost_env(env_stack, idx)
                label_type = self._infer_label_type(key, env)
                caption = self._find_nearest_caption(lines, idx)
                orig_line, src_file = file_lines.get(idx, (idx + 1, None))

                labels[key] = LabelInfo(
                    key=key,
                    label_type=label_type,
                    line_no=orig_line,
                    environment=env,
                    caption=caption,
                    file_path=src_file,
                    _merged_idx=idx,
                )
        return labels

    @staticmethod
    def _infer_label_type(key: str, env: Optional[str]) -> LabelType:
        """Infer type from label prefix (fig:, tab:, eq:) or enclosing env."""
        # 1) Try prefix
        if ":" in key:
            prefix = key.split(":")[0].lower()
            if prefix in _PREFIX_MAP:
                return _PREFIX_MAP[prefix]

        # 2) Try enclosing environment
        if env:
            env_lower = env.lower()
            if env_lower in _ENV_MAP:
                return _ENV_MAP[env_lower]

        return LabelType.UNKNOWN

    @staticmethod
    def _find_nearest_caption(lines: List[str], label_idx: int, window: int = 10) -> Optional[str]:
        """Look ±window lines around \\label for a \\caption{}."""
        start = max(0, label_idx - window)
        end = min(len(lines), label_idx + window + 1)
        for i in range(start, end):
            m = RE_CAPTION.search(lines[i])
            if m:
                caption_text = m.group(1)
                # Try to close the brace
                depth = 1
                result = []
                for ch in caption_text:
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            break
                    result.append(ch)
                return "".join(result).strip()
        return None

    # -----------------------------------------------------------------------
    # Ref extraction
    # -----------------------------------------------------------------------

    def _extract_refs(
        self,
        lines: List[str],
        env_stack: Dict[int, List[str]],
        file_lines: Dict[int, Tuple[int, Optional[str]]],
    ) -> List[RefInstance]:
        refs: List[RefInstance] = []

        full_text = "\n".join(lines)

        for idx, line in enumerate(lines):
            # \\ref, \\eqref, etc.
            for m in RE_REF.finditer(line):
                cmd = m.group("cmd")
                key = m.group("key").strip()
                env = self._innermost_env(env_stack, idx)
                orig_line, src_file = file_lines.get(idx, (idx + 1, None))
                ctx = self._get_line_context(lines, idx)
                refs.append(RefInstance(
                    target_key=key,
                    ref_type=cmd,
                    line_no=orig_line,
                    context=ctx,
                    source_env=env,
                    file_path=src_file,
                    _merged_idx=idx,
                ))

            # \\cite, \\citep, etc. (can have comma-separated keys)
            for m in RE_CITE.finditer(line):
                keys_str = m.group("keys")
                env = self._innermost_env(env_stack, idx)
                orig_line, src_file = file_lines.get(idx, (idx + 1, None))
                ctx = self._get_line_context(lines, idx)
                for key in keys_str.split(","):
                    key = key.strip()
                    if key:
                        refs.append(RefInstance(
                            target_key=key,
                            ref_type="cite",
                            line_no=orig_line,
                            context=ctx,
                            source_env=env,
                            file_path=src_file,
                            _merged_idx=idx,
                        ))

        return refs

    def _get_line_context(self, lines: List[str], idx: int) -> str:
        """Get ±1 line around the reference for context."""
        start = max(0, idx - 1)
        end = min(len(lines), idx + 2)
        return " ".join(lines[start:end]).strip()

    # -----------------------------------------------------------------------
    # Edge construction
    # -----------------------------------------------------------------------

    def _build_edges(
        self,
        labels: Dict[str, LabelInfo],
        refs: List[RefInstance],
        lines: List[str],
        env_stack: Dict[int, List[str]],
    ) -> List[LatexRefEdge]:
        """
        Build directed edges with two strategies:
          1. Enclosing labeled environment → target  (high precision)
          2. Nearest preceding section → target       (high recall fallback)

        Uses _merged_idx (0-based merged content index) for all positional
        comparisons, fixing the original line_no coordinate mismatch.
        """
        edges: List[LatexRefEdge] = []
        seen: Set[Tuple[str, str]] = set()

        # label → line range in merged indices
        label_ranges = self._compute_label_ranges(labels, lines, env_stack)

        # sorted section labels for fallback attribution
        section_list = self._build_section_list(labels)

        for ref in refs:
            if ref.ref_type == "cite":
                continue  # citations don't create intra-doc edges

            target_key = ref.target_key
            if target_key not in labels:
                continue  # dangling ref (undefined label)

            ref_idx = ref._merged_idx
            if ref_idx < 0:
                continue

            # Strategy 1: enclosing labeled environment (high precision)
            source_label_key = self._find_enclosing_label(
                ref_idx, label_ranges, labels
            )

            # Strategy 2: nearest preceding section (high recall)
            if source_label_key is None:
                source_label_key = self._find_section_for_line(
                    ref_idx, section_list
                )

            if source_label_key is None or source_label_key == target_key:
                continue  # self-ref or truly free text

            edge_pair = (source_label_key, target_key)
            if edge_pair in seen:
                continue
            seen.add(edge_pair)

            source_info = labels[source_label_key]
            target_info = labels[target_key]

            edges.append(LatexRefEdge(
                source_label=source_label_key,
                target_label=target_key,
                source_type=source_info.label_type.value,
                target_type=target_info.label_type.value,
                ref_text=f"\\{ref.ref_type}{{{target_key}}}",
                context=ref.context,
            ))

        return edges

    def _compute_label_ranges(
        self,
        labels: Dict[str, LabelInfo],
        lines: List[str],
        env_stack: Dict[int, List[str]],
    ) -> Dict[str, Tuple[int, int]]:
        """
        For each label, estimate the line range of its enclosing environment.
        Returns {label_key → (start_merged_idx, end_merged_idx)}.
        Uses _merged_idx from LabelInfo (no rescanning needed).
        """
        ranges: Dict[str, Tuple[int, int]] = {}

        for key, info in labels.items():
            idx = info._merged_idx
            if idx < 0:
                continue

            env = info.environment
            if env is None:
                # No enclosing env → narrow range (±5 lines)
                ranges[key] = (max(0, idx - 5), min(len(lines), idx + 5))
                continue

            # Walk backward to find \begin{env}
            start = idx
            for i in range(idx, max(0, idx - 200) - 1, -1):
                if f"\\begin{{{env}}}" in lines[i]:
                    start = i
                    break

            # Walk forward to find \end{env}
            end = idx
            for i in range(idx, min(len(lines), idx + 200)):
                if f"\\end{{{env}}}" in lines[i]:
                    end = i
                    break

            ranges[key] = (start, end)

        return ranges

    @staticmethod
    def _find_enclosing_label(
        ref_line_no: int,
        label_ranges: Dict[str, Tuple[int, int]],
        labels: Dict[str, LabelInfo],
    ) -> Optional[str]:
        """
        Find which labeled element encloses the given line.
        If multiple match, pick the narrowest range.
        """
        best_key: Optional[str] = None
        best_span = float("inf")

        for key, (start, end) in label_ranges.items():
            if start <= ref_line_no <= end:
                span = end - start
                if span < best_span:
                    best_span = span
                    best_key = key

        return best_key

    # -----------------------------------------------------------------------
    # Section attribution helpers (P0 fix)
    # -----------------------------------------------------------------------

    @staticmethod
    def _build_section_list(
        labels: Dict[str, LabelInfo],
    ) -> List[Tuple[int, str]]:
        """
        Get sorted list of (merged_idx, label_key) for section-type labels.
        Used for the fallback section-attribution strategy in _build_edges.
        """
        sections: List[Tuple[int, str]] = []
        for key, info in labels.items():
            if info.label_type in (LabelType.SECTION, LabelType.APPENDIX):
                if info._merged_idx >= 0:
                    sections.append((info._merged_idx, key))
        sections.sort()
        return sections

    @staticmethod
    def _find_section_for_line(
        line_idx: int,
        section_list: List[Tuple[int, str]],
    ) -> Optional[str]:
        """Find nearest section label at or before the given line (binary search)."""
        if not section_list:
            return None
        best: Optional[str] = None
        for sec_idx, sec_key in section_list:
            if sec_idx <= line_idx:
                best = sec_key
            else:
                break
        return best

    # -----------------------------------------------------------------------
    # Containment edges (P2: section → element)
    # -----------------------------------------------------------------------

    def _build_containment_edges(
        self,
        labels: Dict[str, LabelInfo],
        lines: List[str],
    ) -> List[LatexRefEdge]:
        """
        Build section → element containment edges.
        Each non-section element gets an edge from its narrowest containing
        section, based on merged line positions.
        """
        edges: List[LatexRefEdge] = []

        # Build section ranges: (start_idx, end_idx, key)
        section_list = self._build_section_list(labels)
        if not section_list:
            return edges

        section_ranges: List[Tuple[int, int, str]] = []
        for i, (idx, key) in enumerate(section_list):
            end = (
                section_list[i + 1][0] - 1
                if i + 1 < len(section_list)
                else len(lines) - 1
            )
            section_ranges.append((idx, end, key))

        seen: Set[Tuple[str, str]] = set()

        for elem_key, elem_info in labels.items():
            # Skip section/appendix labels themselves
            if elem_info.label_type in (LabelType.SECTION, LabelType.APPENDIX):
                continue

            elem_idx = elem_info._merged_idx
            if elem_idx < 0:
                continue

            # Find narrowest containing section
            best_key: Optional[str] = None
            best_span = float("inf")

            for start, end, sec_key in section_ranges:
                if start <= elem_idx <= end:
                    span = end - start
                    if span < best_span:
                        best_span = span
                        best_key = sec_key

            if best_key and (best_key, elem_key) not in seen:
                seen.add((best_key, elem_key))
                sec_info = labels[best_key]
                edges.append(LatexRefEdge(
                    source_label=best_key,
                    target_label=elem_key,
                    source_type=sec_info.label_type.value,
                    target_type=elem_info.label_type.value,
                    ref_text="[containment]",
                    context=f"{elem_key} is within {best_key}",
                ))

        return edges

    # -----------------------------------------------------------------------
    # Bibliography (.bbl) parsing
    # -----------------------------------------------------------------------

    @staticmethod
    def _parse_bbl(bbl_path: Path) -> Dict[str, BibEntry]:
        """Parse a .bbl file to extract cite keys and basic metadata."""
        entries: Dict[str, BibEntry] = {}

        try:
            content = bbl_path.read_text(errors="replace")
        except Exception:
            return entries

        # Pattern: \bibitem{key} or \bibitem[...]{key}
        bibitem_re = re.compile(
            r'\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}'
        )

        # Split by \bibitem
        parts = bibitem_re.split(content)
        # parts[0] = preamble, then alternating: key, text, key, text, ...

        for i in range(1, len(parts) - 1, 2):
            cite_key = parts[i].strip()
            raw_text = parts[i + 1].strip() if i + 1 < len(parts) else ""

            # Try to extract title (usually after author line, in some formats)
            # This is best-effort; .bbl formats vary wildly
            title = None
            # Common pattern: \newblock {\em Title}
            title_m = re.search(
                r'\\newblock\s+(?:\\(?:em|it|textit)\s*\{)?([^}]+)\}?',
                raw_text
            )
            if title_m:
                title = title_m.group(1).strip().rstrip(".")

            # Year: look for (YYYY) or , YYYY, or YYYY.
            year_m = re.search(r'(?:^|\D)((?:19|20)\d{2})(?:\D|$)', raw_text)
            year = year_m.group(1) if year_m else None

            entries[cite_key] = BibEntry(
                cite_key=cite_key,
                title=title,
                year=year,
                raw=raw_text[:500],
            )

        return entries

    # -----------------------------------------------------------------------
    # Multi-hop path finding
    # -----------------------------------------------------------------------

    @staticmethod
    def find_multihop_paths(
        graph: LatexDocumentGraph,
        max_hops: int = 3,
    ) -> List[List[str]]:
        """
        Find all simple paths of length 2..max_hops between labels of
        *different* types.  Returns list of [label_key, ...] paths.
        """
        # Build adjacency (both directions for undirected search)
        adj: Dict[str, Set[str]] = {}
        for edge in graph.edges:
            adj.setdefault(edge.source_label, set()).add(edge.target_label)
            adj.setdefault(edge.target_label, set()).add(edge.source_label)

        paths: List[List[str]] = []

        def _dfs(current: str, path: List[str]) -> None:
            if len(path) > max_hops + 1:
                return
            if len(path) >= 3:  # at least 2-hop
                start_type = graph.labels[path[0]].label_type
                end_type = graph.labels[path[-1]].label_type
                if start_type != end_type:
                    paths.append(list(path))
            for neighbor in adj.get(current, set()):
                if neighbor not in path and neighbor in graph.labels:
                    path.append(neighbor)
                    _dfs(neighbor, path)
                    path.pop()

        for start in graph.labels:
            _dfs(start, [start])

        return paths
