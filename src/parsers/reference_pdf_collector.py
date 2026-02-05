"""Collect and download PDFs for papers referenced by an arXiv paper."""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from src.utils.file_utils import ensure_dir, safe_json_dump


@dataclass
class ReferenceRecord:
    """Metadata + download status for one referenced paper."""

    reference_paper_id: str
    title: str
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    citation_count: int = 0
    source_urls: List[str] = field(default_factory=list)
    selected_url: Optional[str] = None
    local_path: Optional[str] = None
    download_success: bool = False
    failure_reason: Optional[str] = None


class ReferencePDFCollector:
    """Download all obtainable reference PDFs for a given arXiv ID."""

    SEMANTIC_SCHOLAR_GRAPH = "https://api.semanticscholar.org/graph/v1"

    def __init__(
        self,
        output_dir: Path,
        api_key: Optional[str] = None,
        semantic_scholar_delay: float = 1.0,
    ):
        self.output_dir = output_dir
        ensure_dir(self.output_dir)
        self.semantic_scholar_delay = semantic_scholar_delay
        self._last_s2_call = 0.0

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "m4-reference-collector/1.2"})
        if api_key:
            self.session.headers["x-api-key"] = api_key

    def _rate_limit_s2(self) -> None:
        elapsed = time.time() - self._last_s2_call
        if elapsed < self.semantic_scholar_delay:
            time.sleep(self.semantic_scholar_delay - elapsed)
        self._last_s2_call = time.time()

    def _s2_get(self, url: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        self._rate_limit_s2()
        try:
            response = self.session.get(url, params=params, timeout=40)
            if response.status_code == 200:
                return response.json()
            print(f"[SemanticScholar] {response.status_code}: {response.text[:200]}")
            return None
        except requests.RequestException as exc:
            print(f"[SemanticScholar] request failed: {exc}")
            return None

    def _resolve_seed_paper_id(self, arxiv_id: str) -> Optional[str]:
        clean_id = arxiv_id.replace("arXiv:", "").strip()
        url = f"{self.SEMANTIC_SCHOLAR_GRAPH}/paper/arXiv:{clean_id}"
        data = self._s2_get(url, params={"fields": "paperId,title"})
        if not data:
            return None
        return data.get("paperId")

    def _fetch_references(self, paper_id: str, page_size: int = 500) -> List[Dict[str, Any]]:
        nested_fields = (
            "citedPaper.paperId,citedPaper.title,citedPaper.authors,citedPaper.year,"
            "citedPaper.venue,citedPaper.externalIds,citedPaper.citationCount,citedPaper.openAccessPdf"
        )
        fallback_fields = "paperId,title,authors,year,venue,externalIds,citationCount,openAccessPdf"

        url = f"{self.SEMANTIC_SCHOLAR_GRAPH}/paper/{paper_id}/references"
        offset = 0
        refs: List[Dict[str, Any]] = []

        while True:
            payload = self._s2_get(
                url,
                params={"fields": nested_fields, "limit": page_size, "offset": offset},
            )
            if not payload:
                break

            # Compatibility fallback for deployments that don't return expected nested payload.
            if "data" not in payload:
                payload = self._s2_get(
                    url,
                    params={"fields": fallback_fields, "limit": page_size, "offset": offset},
                ) or {}

            page = payload.get("data", [])
            if not page:
                break

            refs.extend(page)
            if len(page) < page_size:
                break

            offset += page_size

        return refs

    @staticmethod
    def _normalize_arxiv_id(raw: Optional[str]) -> Optional[str]:
        if not raw:
            return None
        cleaned = raw.replace("arXiv:", "").strip()
        cleaned = re.sub(r"v\d+$", "", cleaned)
        return cleaned

    @staticmethod
    def _extract_cited_paper(ref_item: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize API variations to a single cited-paper dict."""
        cited = ref_item.get("citedPaper")
        if isinstance(cited, dict) and cited:
            return cited

        # Some responses expose fields directly at top-level reference item.
        top_level_keys = {
            "paperId", "title", "authors", "year", "venue", "externalIds", "citationCount", "openAccessPdf"
        }
        if any(key in ref_item for key in top_level_keys):
            return ref_item

        return {}

    def _candidate_urls(self, cited: Dict[str, Any]) -> List[str]:
        urls: List[str] = []
        external_ids = cited.get("externalIds") or {}

        arxiv_id = self._normalize_arxiv_id(external_ids.get("ArXiv"))
        if arxiv_id:
            urls.append(f"https://arxiv.org/pdf/{arxiv_id}.pdf")

        oa = cited.get("openAccessPdf") or {}
        oa_url = oa.get("url")
        if oa_url:
            urls.append(oa_url)

        doi = external_ids.get("DOI")
        if doi:
            urls.append(f"https://doi.org/{doi}")

        deduped: List[str] = []
        for item in urls:
            if item not in deduped:
                deduped.append(item)
        return deduped

    def _download_from_candidates(self, record: ReferenceRecord, timeout: int = 60) -> Tuple[bool, Optional[str]]:
        safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", record.reference_paper_id or record.title)[:120]
        output_path = self.output_dir / f"{safe_name}.pdf"
        if output_path.exists():
            record.local_path = str(output_path)
            record.download_success = True
            return True, None

        for url in record.source_urls:
            try:
                response = self.session.get(url, stream=True, timeout=timeout, allow_redirects=True)
                if response.status_code != 200:
                    continue

                content_type = (response.headers.get("content-type") or "").lower()
                chunk_iter = response.iter_content(chunk_size=8192)
                first_chunk = next(chunk_iter, b"")
                if not first_chunk:
                    continue

                is_pdf = b"%PDF" in first_chunk[:1024] or "pdf" in content_type
                if not is_pdf:
                    continue

                tmp_path = output_path.with_suffix(".tmp")
                with open(tmp_path, "wb") as handle:
                    handle.write(first_chunk)
                    for chunk in chunk_iter:
                        if chunk:
                            handle.write(chunk)
                tmp_path.replace(output_path)

                record.selected_url = url
                record.local_path = str(output_path)
                record.download_success = True
                return True, None
            except requests.RequestException:
                continue

        return False, "No downloadable PDF found in candidate URLs"

    def collect_from_arxiv(
        self,
        arxiv_id: str,
        max_references: Optional[int] = None,
        min_citations: int = 0,
    ) -> List[ReferenceRecord]:
        """Collect and download all available references for an arXiv paper."""

        seed_paper_id = self._resolve_seed_paper_id(arxiv_id)
        if not seed_paper_id:
            raise ValueError(f"Could not resolve Semantic Scholar paperId for arXiv:{arxiv_id}")

        raw_refs = self._fetch_references(seed_paper_id)
        print(f"Found {len(raw_refs)} total references from Semantic Scholar")

        records: List[ReferenceRecord] = []
        for item in raw_refs:
            cited = self._extract_cited_paper(item)
            if not cited:
                continue

            citation_count = cited.get("citationCount", 0) or 0
            if citation_count < min_citations:
                continue

            external_ids = cited.get("externalIds") or {}
            arxiv_ref_id = self._normalize_arxiv_id(external_ids.get("ArXiv"))
            doi = external_ids.get("DOI")

            reference_paper_id = cited.get("paperId") or arxiv_ref_id or doi or cited.get("title", "unknown")
            record = ReferenceRecord(
                reference_paper_id=str(reference_paper_id),
                title=cited.get("title") or "Unknown",
                authors=[a.get("name", "") for a in cited.get("authors", [])],
                year=cited.get("year"),
                venue=cited.get("venue"),
                doi=doi,
                arxiv_id=arxiv_ref_id,
                citation_count=citation_count,
            )
            record.source_urls = self._candidate_urls(cited)
            records.append(record)

        records.sort(key=lambda x: x.citation_count, reverse=True)
        if max_references is not None:
            records = records[:max_references]

        for record in records:
            success, reason = self._download_from_candidates(record)
            if not success:
                record.failure_reason = reason

        return records

    def save_report(
        self,
        report_path: Path,
        source_arxiv_id: str,
        records: List[ReferenceRecord],
    ) -> None:
        summary = {
            "source_arxiv_id": source_arxiv_id,
            "total_references": len(records),
            "downloaded": sum(1 for r in records if r.download_success),
            "failed": sum(1 for r in records if not r.download_success),
            "records": [asdict(r) for r in records],
        }
        safe_json_dump(summary, report_path)
