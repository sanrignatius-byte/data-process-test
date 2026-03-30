#!/usr/bin/env python3
"""
Survey Reference Fetcher

从Survey论文中提取引用的论文，并下载它们的PDF。

支持的来源：
1. Semantic Scholar API - 获取论文引用列表
2. arXiv API - 搜索相关论文并下载PDF
3. 本地BibTeX文件

Usage:
    # 通过Semantic Scholar ID获取引用
    python scripts/fetch_survey_references.py \
        --survey-id "204e3073870fae3d05bcbc2f6a8e263d9b72e776" \
        --output data/raw_pdfs \
        --max-papers 50

    # 通过arXiv ID获取引用（如果Semantic Scholar没有索引，会用关键词搜索）
    python scripts/fetch_survey_references.py \
        --arxiv-id "2501.09959" \
        --output data/raw_pdfs \
        --max-papers 50

    # 直接用关键词搜索arXiv论文
    python scripts/fetch_survey_references.py \
        --search "multi-turn dialogue LLM" \
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


class ArxivSearcher:
    """arXiv API搜索器 - 当Semantic Scholar没有索引时的备选方案"""

    BASE_URL = "http://export.arxiv.org/api/query"
    RATE_LIMIT_DELAY = 3.0

    def __init__(self):
        self.session = requests.Session()
        self._last_request_time = 0

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self.RATE_LIMIT_DELAY:
            time.sleep(self.RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = time.time()

    def search(
        self,
        query: str,
        max_results: int = 50,
        categories: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        搜索arXiv论文

        Args:
            query: 搜索关键词
            max_results: 最大结果数
            categories: 限制的arXiv类别，如 ["cs.CL", "cs.AI"]
        """
        self._rate_limit()

        # 构建搜索查询
        search_query = f"all:{query}"
        if categories:
            cat_query = " OR ".join(f"cat:{cat}" for cat in categories)
            search_query = f"({search_query}) AND ({cat_query})"

        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }

        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=60)
            if response.status_code != 200:
                print(f"arXiv API error: {response.status_code}")
                return []

            # 解析Atom feed
            return self._parse_atom_feed(response.text)

        except Exception as e:
            print(f"arXiv search failed: {e}")
            return []

    def _parse_atom_feed(self, xml_text: str) -> List[Dict[str, Any]]:
        """解析arXiv Atom feed"""
        import xml.etree.ElementTree as ET

        papers = []
        ns = {
            'atom': 'http://www.w3.org/2005/Atom',
            'arxiv': 'http://arxiv.org/schemas/atom'
        }

        try:
            root = ET.fromstring(xml_text)

            for entry in root.findall('atom:entry', ns):
                # 提取arXiv ID
                id_elem = entry.find('atom:id', ns)
                if id_elem is None:
                    continue

                arxiv_url = id_elem.text
                arxiv_id = arxiv_url.split('/abs/')[-1]
                # 去掉版本号
                arxiv_id = re.sub(r'v\d+$', '', arxiv_id)

                # 提取标题
                title_elem = entry.find('atom:title', ns)
                title = title_elem.text.strip().replace('\n', ' ') if title_elem is not None else "Unknown"

                # 提取作者
                authors = []
                for author in entry.findall('atom:author', ns):
                    name_elem = author.find('atom:name', ns)
                    if name_elem is not None:
                        authors.append(name_elem.text)

                # 提取摘要
                summary_elem = entry.find('atom:summary', ns)
                abstract = summary_elem.text.strip() if summary_elem is not None else ""

                # 提取发布日期
                published_elem = entry.find('atom:published', ns)
                year = None
                if published_elem is not None:
                    year = int(published_elem.text[:4])

                # 提取类别
                categories = []
                for cat in entry.findall('arxiv:primary_category', ns):
                    categories.append(cat.get('term'))
                for cat in entry.findall('atom:category', ns):
                    categories.append(cat.get('term'))

                papers.append({
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "authors": authors,
                    "abstract": abstract,
                    "year": year,
                    "categories": list(set(categories)),
                    "citation_count": 0  # arXiv API不提供引用数
                })

        except ET.ParseError as e:
            print(f"XML parse error: {e}")

        return papers

    def get_paper_title(self, arxiv_id: str) -> Optional[str]:
        """获取论文标题（用于从arXiv ID推断搜索关键词）"""
        self._rate_limit()
        params = {
            "id_list": arxiv_id,
            "max_results": 1
        }

        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=30)
            if response.status_code == 200:
                papers = self._parse_atom_feed(response.text)
                if papers:
                    return papers[0].get("title")
        except Exception as e:
            print(f"Failed to get paper title: {e}")

        return None


def extract_search_keywords(title: str) -> str:
    """
    从论文标题提取搜索关键词

    过滤掉常见的停用词，保留有意义的术语
    """
    # 常见的学术论文停用词
    stopwords = {
        'a', 'an', 'the', 'of', 'for', 'and', 'or', 'in', 'on', 'to', 'with',
        'by', 'from', 'as', 'at', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'can', 'that', 'this', 'these',
        'those', 'it', 'its', 'their', 'our', 'your', 'what', 'which', 'who',
        'how', 'when', 'where', 'why', 'via', 'using', 'based', 'towards',
        'through', 'into', 'over', 'under', 'between', 'about', 'survey',
        'comprehensive', 'review', 'study', 'analysis', 'approach', 'method',
        'methods', 'new', 'novel', 'improved', 'towards', 'recent', 'advances'
    }

    # 清理标题
    title = title.lower()
    title = re.sub(r'[^\w\s-]', ' ', title)

    # 分词并过滤
    words = title.split()
    keywords = [w for w in words if w not in stopwords and len(w) > 2]

    # 保留最重要的关键词（前8个）
    keywords = keywords[:8]

    return ' '.join(keywords)


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
    search_query: Optional[str] = None,
    output_dir: Path = Path("data/raw_pdfs"),
    max_papers: int = 100,
    min_citations: int = 0,
    api_key: Optional[str] = None,
    categories: List[str] = None
) -> List[PaperMetadata]:
    """
    获取Survey的引用论文并下载

    Args:
        survey_id: Semantic Scholar paper ID
        arxiv_id: arXiv ID
        bibtex_path: BibTeX文件路径
        search_query: 直接用关键词搜索arXiv
        output_dir: 输出目录
        max_papers: 最大下载论文数
        min_citations: 最小引用数过滤
        api_key: Semantic Scholar API key
        categories: arXiv类别过滤，如 ["cs.CL", "cs.AI"]

    Returns:
        下载的论文元数据列表
    """
    client = SemanticScholarClient(api_key=api_key)
    arxiv_searcher = ArxivSearcher()
    downloader = ArxivDownloader(output_dir)
    papers: List[PaperMetadata] = []

    # 收集引用论文
    references = []
    arxiv_papers = []
    use_arxiv_search = False

    if search_query:
        # 直接用关键词搜索arXiv
        print(f"Searching arXiv with query: {search_query}")
        use_arxiv_search = True
        arxiv_papers = arxiv_searcher.search(
            query=search_query,
            max_results=max_papers * 2,  # 搜索更多以便过滤
            categories=categories
        )
        print(f"Found {len(arxiv_papers)} papers from arXiv search")

    elif bibtex_path:
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
        paper_title = None

        if arxiv_id and not survey_id:
            print(f"Looking up paper by arXiv ID: {arxiv_id}")
            paper_info = client.get_paper_by_arxiv(arxiv_id)
            if paper_info:
                paper_id = paper_info["paperId"]
                paper_title = paper_info.get("title", "")
                print(f"Found Semantic Scholar ID: {paper_id}")
                print(f"Paper title: {paper_title}")
            else:
                # Semantic Scholar没有索引这篇论文，使用arXiv备选方案
                print("Paper not found in Semantic Scholar, trying arXiv fallback...")

                # 从arXiv获取论文标题
                paper_info_arxiv = arxiv_searcher.search(
                    query=arxiv_id,
                    max_results=1
                )

                if paper_info_arxiv:
                    paper_title = paper_info_arxiv[0].get("title", "")
                    print(f"Found paper on arXiv: {paper_title}")

                    # 从标题提取关键词进行搜索
                    keywords = extract_search_keywords(paper_title)
                    print(f"Searching for related papers with keywords: {keywords}")

                    use_arxiv_search = True
                    arxiv_papers = arxiv_searcher.search(
                        query=keywords,
                        max_results=max_papers * 2,
                        categories=categories or ["cs.CL", "cs.AI", "cs.LG"]
                    )
                    print(f"Found {len(arxiv_papers)} related papers from arXiv")
                else:
                    print(f"Could not find paper {arxiv_id} on arXiv either")
                    return []

        if not use_arxiv_search:
            if not paper_id:
                print("No paper ID provided")
                return []

            print(f"Fetching references for paper: {paper_id}")
            references = client.get_paper_references(paper_id, limit=500)
            print(f"Found {len(references)} references")

    # 处理引用列表（仅当不是直接arXiv搜索时）
    if not use_arxiv_search:
        # 过滤有arXiv ID的论文
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

        # 按引用数排序
        arxiv_papers.sort(key=lambda x: x.get("citation_count", 0), reverse=True)

    # 取top N
    arxiv_papers = arxiv_papers[:max_papers]

    source_type = "arXiv search" if use_arxiv_search else "references"
    print(f"\nFound {len(arxiv_papers)} papers from {source_type}")

    # 下载PDF
    print(f"\nDownloading PDFs to {output_dir}...")

    for paper_info in tqdm(arxiv_papers, desc="Downloading"):
        arxiv = paper_info["arxiv_id"]

        paper = PaperMetadata(
            paper_id=arxiv,
            title=paper_info.get("title", "Unknown"),
            authors=paper_info.get("authors", []),
            year=paper_info.get("year"),
            arxiv_id=arxiv,
            citation_count=paper_info.get("citation_count", 0),
            venue=paper_info.get("venue"),
            abstract=paper_info.get("abstract")
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
            "bibtex_path": str(bibtex_path) if bibtex_path else None,
            "search_query": search_query,
            "used_arxiv_fallback": use_arxiv_search
        },
        "total_references": len(references) if not use_arxiv_search else 0,
        "total_found": len(arxiv_papers),
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
    source_group.add_argument(
        "--search",
        help="Search arXiv directly with keywords (e.g., 'multi-turn dialogue LLM')"
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
        "--categories",
        nargs="+",
        default=None,
        help="arXiv categories to filter (e.g., cs.CL cs.AI cs.LG)"
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
        search_query=args.search,
        output_dir=args.output,
        max_papers=args.max_papers,
        min_citations=args.min_citations,
        api_key=args.api_key,
        categories=args.categories
    )


if __name__ == "__main__":
    main()
