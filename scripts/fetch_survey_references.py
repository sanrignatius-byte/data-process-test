#!/usr/bin/env python3
"""
Survey Reference Fetcher

从Survey论文中提取引用的论文，并下载它们的PDF。

支持的来源：
1. Semantic Scholar API - 获取论文引用列表
2. arXiv API - 下载PDF

Usage:
    # 通过Semantic Scholar ID获取引用
    python scripts/fetch_survey_references.py \
        --survey-id "204e3073870fae3d05bcbc2f6a8e263d9b72e776" \
        --output data/raw_pdfs \
        --max-papers 50

    # 通过arXiv ID获取引用
    python scripts/fetch_survey_references.py \
        --arxiv-id "2401.12345" \
        --output data/raw_pdfs \
        --max-papers 50

    # 从本地BibTeX文件提取
    python scripts/fetch_survey_references.py \
        --bibtex references.bib \
        --output data/raw_pdfs
"""

import argparse
import json
import re
import time
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.file_utils import ensure_dir, safe_json_dump


@dataclass
class PaperMetadata:
    """论文元数据"""
    paper_id: str  # Semantic Scholar ID or arXiv ID
    title: str
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None
    venue: Optional[str] = None
    abstract: Optional[str] = None
    citation_count: int = 0
    local_path: Optional[str] = None
    download_success: bool = False


class SemanticScholarClient:
    """Semantic Scholar API客户端"""

    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    RATE_LIMIT_DELAY = 1.0  # 秒

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.session = requests.Session()
        if api_key:
            self.session.headers["x-api-key"] = api_key
        self._last_request_time = 0

    def _rate_limit(self):
        """速率限制"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.RATE_LIMIT_DELAY:
            time.sleep(self.RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = time.time()

    def get_paper_by_arxiv(self, arxiv_id: str) -> Optional[Dict]:
        """通过arXiv ID获取论文信息"""
        self._rate_limit()
        url = f"{self.BASE_URL}/paper/arXiv:{arxiv_id}"
        params = {"fields": "paperId,title,authors,year,externalIds,venue,abstract,citationCount"}

        try:
            response = self.session.get(url, params=params, timeout=30)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                print(f"Paper not found: arXiv:{arxiv_id}")
                return None
            else:
                print(f"API error {response.status_code}: {response.text[:200]}")
                return None
        except Exception as e:
            print(f"Request failed: {e}")
            return None

    def get_paper_references(
        self,
        paper_id: str,
        limit: int = 100,
        fields: str = "paperId,title,authors,year,externalIds,venue,abstract,citationCount"
    ) -> List[Dict]:
        """获取论文的引用列表"""
        self._rate_limit()
        url = f"{self.BASE_URL}/paper/{paper_id}/references"
        params = {"fields": fields, "limit": limit}

        all_references = []
        offset = 0

        while True:
            params["offset"] = offset
            try:
                response = self.session.get(url, params=params, timeout=30)
                if response.status_code != 200:
                    print(f"API error {response.status_code}")
                    break

                data = response.json()
                references = data.get("data", [])

                if not references:
                    break

                all_references.extend(references)

                if len(references) < limit:
                    break

                offset += limit
                self._rate_limit()

            except Exception as e:
                print(f"Request failed: {e}")
                break

        return all_references

    def search_paper(self, query: str, limit: int = 10) -> List[Dict]:
        """搜索论文"""
        self._rate_limit()
        url = f"{self.BASE_URL}/paper/search"
        params = {
            "query": query,
            "limit": limit,
            "fields": "paperId,title,authors,year,externalIds,venue,abstract,citationCount"
        }

        try:
            response = self.session.get(url, params=params, timeout=30)
            if response.status_code == 200:
                return response.json().get("data", [])
            return []
        except Exception as e:
            print(f"Search failed: {e}")
            return []


class ArxivDownloader:
    """arXiv PDF下载器"""

    PDF_URL_TEMPLATE = "https://arxiv.org/pdf/{arxiv_id}.pdf"
    RATE_LIMIT_DELAY = 3.0  # arXiv要求3秒延迟

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        ensure_dir(output_dir)
        self.session = requests.Session()
        self._last_request_time = 0

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self.RATE_LIMIT_DELAY:
            time.sleep(self.RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = time.time()

    def download(self, arxiv_id: str, max_retries: int = 3) -> Optional[Path]:
        """下载arXiv论文PDF"""
        # 标准化arXiv ID
        arxiv_id = arxiv_id.replace("arXiv:", "").strip()

        # 检查是否已下载
        output_path = self.output_dir / f"{arxiv_id.replace('/', '_')}.pdf"
        if output_path.exists():
            print(f"Already downloaded: {arxiv_id}")
            return output_path

        url = self.PDF_URL_TEMPLATE.format(arxiv_id=arxiv_id)

        for attempt in range(max_retries):
            self._rate_limit()
            try:
                response = self.session.get(url, timeout=60, stream=True)
                if response.status_code == 200:
                    with open(output_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print(f"Downloaded: {arxiv_id}")
                    return output_path
                elif response.status_code == 404:
                    print(f"PDF not found: {arxiv_id}")
                    return None
                else:
                    print(f"Download failed ({response.status_code}): {arxiv_id}")
            except Exception as e:
                print(f"Download error (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(5 * (attempt + 1))

        return None


def parse_bibtex(bibtex_path: Path) -> List[Dict[str, str]]:
    """解析BibTeX文件提取arXiv IDs"""
    entries = []

    with open(bibtex_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 简单的BibTeX解析
    entry_pattern = r'@\w+\{([^,]+),([^@]+)\}'
    field_pattern = r'(\w+)\s*=\s*[{"]([^}"]+)[}"]'

    for match in re.finditer(entry_pattern, content, re.DOTALL):
        entry_id = match.group(1).strip()
        entry_content = match.group(2)

        entry = {"id": entry_id}
        for field_match in re.finditer(field_pattern, entry_content):
            field_name = field_match.group(1).lower()
            field_value = field_match.group(2).strip()
            entry[field_name] = field_value

        # 尝试提取arXiv ID
        arxiv_id = None
        if "eprint" in entry:
            arxiv_id = entry["eprint"]
        elif "arxiv" in entry:
            arxiv_id = entry["arxiv"]
        elif "url" in entry and "arxiv.org" in entry["url"]:
            match = re.search(r'arxiv.org/abs/(\d+\.\d+)', entry["url"])
            if match:
                arxiv_id = match.group(1)

        if arxiv_id:
            entry["arxiv_id"] = arxiv_id
            entries.append(entry)

    return entries


def fetch_survey_references(
    survey_id: Optional[str] = None,
    arxiv_id: Optional[str] = None,
    bibtex_path: Optional[Path] = None,
    output_dir: Path = Path("data/raw_pdfs"),
    max_papers: int = 100,
    min_citations: int = 0,
    api_key: Optional[str] = None
) -> List[PaperMetadata]:
    """
    获取Survey的引用论文并下载

    Args:
        survey_id: Semantic Scholar paper ID
        arxiv_id: arXiv ID
        bibtex_path: BibTeX文件路径
        output_dir: 输出目录
        max_papers: 最大下载论文数
        min_citations: 最小引用数过滤
        api_key: Semantic Scholar API key

    Returns:
        下载的论文元数据列表
    """
    client = SemanticScholarClient(api_key=api_key)
    downloader = ArxivDownloader(output_dir)
    papers: List[PaperMetadata] = []

    # 收集引用论文
    references = []

    if bibtex_path:
        # 从BibTeX文件提取
        print(f"Parsing BibTeX file: {bibtex_path}")
        entries = parse_bibtex(bibtex_path)
        for entry in entries:
            if "arxiv_id" in entry:
                references.append({
                    "citedPaper": {
                        "title": entry.get("title", "Unknown"),
                        "externalIds": {"ArXiv": entry["arxiv_id"]},
                        "authors": [{"name": a.strip()} for a in entry.get("author", "").split(" and ")],
                        "year": int(entry.get("year", 0)) if entry.get("year", "").isdigit() else None
                    }
                })
        print(f"Found {len(references)} papers with arXiv IDs in BibTeX")

    else:
        # 从Semantic Scholar获取引用
        paper_id = survey_id

        if arxiv_id and not survey_id:
            print(f"Looking up paper by arXiv ID: {arxiv_id}")
            paper_info = client.get_paper_by_arxiv(arxiv_id)
            if paper_info:
                paper_id = paper_info["paperId"]
                print(f"Found Semantic Scholar ID: {paper_id}")
                print(f"Paper title: {paper_info.get('title', 'Unknown')}")
            else:
                print("Could not find paper in Semantic Scholar")
                return []

        if not paper_id:
            print("No paper ID provided")
            return []

        print(f"Fetching references for paper: {paper_id}")
        references = client.get_paper_references(paper_id, limit=500)
        print(f"Found {len(references)} references")

    # 过滤有arXiv ID的论文
    arxiv_papers = []
    for ref in references:
        cited = ref.get("citedPaper", {})
        if not cited:
            continue

        external_ids = cited.get("externalIds", {})
        arxiv = external_ids.get("ArXiv")

        if arxiv:
            citation_count = cited.get("citationCount", 0) or 0
            if citation_count >= min_citations:
                arxiv_papers.append({
                    "arxiv_id": arxiv,
                    "title": cited.get("title", "Unknown"),
                    "authors": [a.get("name", "") for a in cited.get("authors", [])],
                    "year": cited.get("year"),
                    "citation_count": citation_count,
                    "venue": cited.get("venue"),
                    "abstract": cited.get("abstract")
                })

    # 按引用数排序，取top N
    arxiv_papers.sort(key=lambda x: x["citation_count"], reverse=True)
    arxiv_papers = arxiv_papers[:max_papers]

    print(f"\nFound {len(arxiv_papers)} papers with arXiv IDs (filtered by citations >= {min_citations})")

    # 下载PDF
    print(f"\nDownloading PDFs to {output_dir}...")

    for paper_info in tqdm(arxiv_papers, desc="Downloading"):
        arxiv = paper_info["arxiv_id"]

        paper = PaperMetadata(
            paper_id=arxiv,
            title=paper_info["title"],
            authors=paper_info["authors"],
            year=paper_info["year"],
            arxiv_id=arxiv,
            citation_count=paper_info["citation_count"],
            venue=paper_info["venue"],
            abstract=paper_info["abstract"]
        )

        # 下载PDF
        pdf_path = downloader.download(arxiv)
        if pdf_path:
            paper.local_path = str(pdf_path)
            paper.download_success = True

        papers.append(paper)

    # 保存元数据
    metadata_path = output_dir / "survey_references_metadata.json"
    metadata = {
        "source": {
            "survey_id": survey_id,
            "arxiv_id": arxiv_id,
            "bibtex_path": str(bibtex_path) if bibtex_path else None
        },
        "total_references": len(references),
        "papers_with_arxiv": len(arxiv_papers),
        "downloaded": sum(1 for p in papers if p.download_success),
        "papers": [asdict(p) for p in papers]
    }
    safe_json_dump(metadata, metadata_path)
    print(f"\nMetadata saved to {metadata_path}")

    # 统计
    successful = sum(1 for p in papers if p.download_success)
    print(f"\n{'='*50}")
    print(f"Download complete: {successful}/{len(papers)} papers")
    print(f"Output directory: {output_dir}")

    return papers


def main():
    parser = argparse.ArgumentParser(
        description="Fetch and download papers referenced by a survey"
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--survey-id",
        help="Semantic Scholar paper ID of the survey"
    )
    source_group.add_argument(
        "--arxiv-id",
        help="arXiv ID of the survey (e.g., 2401.12345)"
    )
    source_group.add_argument(
        "--bibtex",
        type=Path,
        help="Path to BibTeX file with references"
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/raw_pdfs"),
        help="Output directory for downloaded PDFs"
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=100,
        help="Maximum number of papers to download"
    )
    parser.add_argument(
        "--min-citations",
        type=int,
        default=0,
        help="Minimum citation count filter"
    )
    parser.add_argument(
        "--api-key",
        help="Semantic Scholar API key (optional, increases rate limit)"
    )

    args = parser.parse_args()

    fetch_survey_references(
        survey_id=args.survey_id,
        arxiv_id=args.arxiv_id,
        bibtex_path=args.bibtex,
        output_dir=args.output,
        max_papers=args.max_papers,
        min_citations=args.min_citations,
        api_key=args.api_key
    )


if __name__ == "__main__":
    main()
