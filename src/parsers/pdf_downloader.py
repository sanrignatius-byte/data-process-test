"""PDF downloader from various sources (arXiv, Semantic Scholar, etc.)."""

import os
import time
import asyncio
import aiohttp
import requests
from pathlib import Path
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
from datetime import datetime
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
import hashlib
import math


@dataclass
class PaperMetadata:
    """Metadata for a downloaded paper."""
    paper_id: str
    title: str
    authors: List[str]
    abstract: str
    categories: List[str]
    published_date: str
    source: str
    pdf_url: str
    local_path: Optional[str] = None


class PDFDownloader:
    """
    Multi-source PDF downloader with rate limiting and retry logic.

    Supports:
    - arXiv (primary source for scientific papers)
    - Semantic Scholar API
    - Direct URLs
    """

    ARXIV_API_URL = "http://export.arxiv.org/api/query"
    ARXIV_PDF_BASE = "https://arxiv.org/pdf/"
    SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/v1/paper/"

    def __init__(
        self,
        output_dir: str,
        max_concurrent: int = 5,
        rate_limit_delay: float = 3.0,  # arXiv requires 3s between requests
        max_retries: int = 3
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_concurrent = max_concurrent
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self._downloaded_ids: Set[str] = set()
        self._load_existing()

    def _load_existing(self) -> None:
        """Load already downloaded paper IDs."""
        for pdf_file in self.output_dir.glob("*.pdf"):
            self._downloaded_ids.add(pdf_file.stem)

    def search_arxiv(
        self,
        categories: List[str],
        max_results: int = 50,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        search_query: Optional[str] = None
    ) -> List[PaperMetadata]:
        """
        Search arXiv for papers in given categories.

        Args:
            categories: List of arXiv categories (e.g., ["cs.CL", "cs.CV"])
            max_results: Maximum number of results
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD)
            search_query: Additional search query

        Returns:
            List of paper metadata
        """
        papers = []

        for category in categories:
            # Build query
            query_parts = [f"cat:{category}"]
            if search_query:
                query_parts.append(search_query)

            query = " AND ".join(query_parts)

            per_category_limit = max(1, math.ceil(max_results / max(1, len(categories))))
            params = {
                "search_query": query,
                "start": 0,
                "max_results": per_category_limit,
                "sortBy": "submittedDate",
                "sortOrder": "descending"
            }

            try:
                response = requests.get(
                    self.ARXIV_API_URL,
                    params=params,
                    timeout=30
                )
                response.raise_for_status()

                # Parse XML response
                root = ET.fromstring(response.content)
                ns = {"atom": "http://www.w3.org/2005/Atom"}

                for entry in root.findall("atom:entry", ns):
                    paper_id = entry.find("atom:id", ns).text.split("/abs/")[-1]
                    # Clean version from ID
                    paper_id = paper_id.split("v")[0] if "v" in paper_id else paper_id

                    # Skip if already downloaded
                    if paper_id.replace(".", "_") in self._downloaded_ids:
                        continue

                    title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
                    abstract = entry.find("atom:summary", ns).text.strip().replace("\n", " ")

                    authors = [
                        author.find("atom:name", ns).text
                        for author in entry.findall("atom:author", ns)
                    ]

                    categories_list = [
                        cat.get("term")
                        for cat in entry.findall("atom:category", ns)
                    ]

                    published = entry.find("atom:published", ns).text[:10]

                    # Filter by date if specified
                    if date_from and published < date_from:
                        continue
                    if date_to and published > date_to:
                        continue

                    pdf_url = f"{self.ARXIV_PDF_BASE}{paper_id}.pdf"

                    papers.append(PaperMetadata(
                        paper_id=paper_id,
                        title=title,
                        authors=authors,
                        abstract=abstract,
                        categories=categories_list,
                        published_date=published,
                        source="arxiv",
                        pdf_url=pdf_url
                    ))

                # Rate limiting
                time.sleep(self.rate_limit_delay)

            except Exception as e:
                print(f"Error searching arXiv category {category}: {e}")
                continue

        return papers[:max_results]

    def download_arxiv_papers(
        self,
        categories: List[str],
        target_count: int,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        search_query: Optional[str] = None,
        initial_buffer: int = 50,
        buffer_step: int = 50,
        max_rounds: int = 6,
        max_results_cap: int = 2000,
        use_async: bool = True
    ) -> List[PaperMetadata]:
        """
        Search arXiv and download PDFs until the target count is reached (best effort).

        Args:
            categories: List of arXiv categories
            target_count: Target number of PDFs to download
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD)
            search_query: Additional search query
            initial_buffer: Extra papers to fetch beyond the target
            buffer_step: How many more results to add per search expansion
            max_rounds: Maximum number of search expansions
            max_results_cap: Maximum total results to request from arXiv
            use_async: Use async downloads for better performance

        Returns:
            List of successfully downloaded paper metadata
        """
        if target_count <= 0:
            return []

        seen_ids: Set[str] = set(self._downloaded_ids)
        successful: List[PaperMetadata] = []
        candidate_queue: List[PaperMetadata] = []

        max_results = min(max_results_cap, target_count + initial_buffer)
        rounds = 0
        exhausted = False

        while len(successful) < target_count:
            needed = target_count - len(successful)

            while len(candidate_queue) < needed + buffer_step and not exhausted:
                papers = self.search_arxiv(
                    categories=categories,
                    max_results=max_results,
                    date_from=date_from,
                    date_to=date_to,
                    search_query=search_query
                )

                added = 0
                for paper in papers:
                    if paper.paper_id in seen_ids:
                        continue
                    seen_ids.add(paper.paper_id)
                    candidate_queue.append(paper)
                    added += 1

                if added == 0:
                    exhausted = True
                    break

                if max_results >= max_results_cap:
                    exhausted = True
                    break

                max_results = min(max_results_cap, max_results + buffer_step)
                rounds += 1
                if rounds >= max_rounds:
                    exhausted = True
                    break

            if not candidate_queue:
                break

            batch_size = min(len(candidate_queue), needed + buffer_step)
            batch = candidate_queue[:batch_size]
            candidate_queue = candidate_queue[batch_size:]

            downloaded = self.download_batch(batch, use_async=use_async)
            successful.extend(downloaded)

        return successful[:target_count]

    def download_pdf(
        self,
        paper: PaperMetadata,
        timeout: int = 60
    ) -> Optional[Path]:
        """
        Download a single PDF.

        Args:
            paper: Paper metadata
            timeout: Download timeout in seconds

        Returns:
            Path to downloaded file or None if failed
        """
        # Sanitize filename
        safe_id = paper.paper_id.replace("/", "_").replace(".", "_")
        output_path = self.output_dir / f"{safe_id}.pdf"

        if output_path.exists():
            paper.local_path = str(output_path)
            return output_path

        for attempt in range(self.max_retries):
            try:
                response = requests.get(
                    paper.pdf_url,
                    timeout=timeout,
                    stream=True,
                    headers={"User-Agent": "Mozilla/5.0 (Research Bot)"}
                )
                response.raise_for_status()

                # Verify it's actually a PDF
                content_type = response.headers.get("content-type", "")
                if "pdf" not in content_type.lower() and not response.content[:4] == b"%PDF":
                    print(f"Warning: {paper.paper_id} may not be a valid PDF")

                # Write to temp file first, then rename (atomic write)
                temp_path = output_path.with_suffix(".tmp")
                with open(temp_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                temp_path.rename(output_path)
                paper.local_path = str(output_path)
                self._downloaded_ids.add(safe_id)

                return output_path

            except Exception as e:
                print(f"Attempt {attempt + 1} failed for {paper.paper_id}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff

        return None

    async def _async_download(
        self,
        session: aiohttp.ClientSession,
        paper: PaperMetadata,
        semaphore: asyncio.Semaphore
    ) -> Optional[Path]:
        """Async download helper."""
        safe_id = paper.paper_id.replace("/", "_").replace(".", "_")
        output_path = self.output_dir / f"{safe_id}.pdf"

        if output_path.exists():
            paper.local_path = str(output_path)
            return output_path

        async with semaphore:
            for attempt in range(self.max_retries):
                try:
                    async with session.get(
                        paper.pdf_url,
                        timeout=aiohttp.ClientTimeout(total=60),
                        headers={"User-Agent": "Mozilla/5.0 (Research Bot)"}
                    ) as response:
                        response.raise_for_status()
                        content = await response.read()

                        temp_path = output_path.with_suffix(".tmp")
                        with open(temp_path, "wb") as f:
                            f.write(content)

                        temp_path.rename(output_path)
                        paper.local_path = str(output_path)
                        self._downloaded_ids.add(safe_id)

                        return output_path

                except Exception as e:
                    print(f"Async attempt {attempt + 1} failed for {paper.paper_id}: {e}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(2 ** attempt)

            return None

    async def download_batch_async(
        self,
        papers: List[PaperMetadata]
    ) -> List[PaperMetadata]:
        """
        Download multiple PDFs concurrently.

        Args:
            papers: List of paper metadata

        Returns:
            List of successfully downloaded papers
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)
        connector = aiohttp.TCPConnector(limit=self.max_concurrent)

        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                self._async_download(session, paper, semaphore)
                for paper in papers
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        successful = []
        for paper, result in zip(papers, results):
            if isinstance(result, Path):
                successful.append(paper)
            elif isinstance(result, Exception):
                print(f"Download failed for {paper.paper_id}: {result}")

        return successful

    def download_batch(
        self,
        papers: List[PaperMetadata],
        use_async: bool = True
    ) -> List[PaperMetadata]:
        """
        Download multiple PDFs.

        Args:
            papers: List of paper metadata
            use_async: Use async downloads for better performance

        Returns:
            List of successfully downloaded papers
        """
        if use_async:
            return asyncio.run(self.download_batch_async(papers))

        # Synchronous fallback
        successful = []
        for paper in papers:
            result = self.download_pdf(paper)
            if result:
                successful.append(paper)
            time.sleep(self.rate_limit_delay)

        return successful

    def save_metadata(self, papers: List[PaperMetadata], filepath: str) -> None:
        """Save paper metadata to JSON file."""
        import json

        data = []
        for paper in papers:
            data.append({
                "paper_id": paper.paper_id,
                "title": paper.title,
                "authors": paper.authors,
                "abstract": paper.abstract,
                "categories": paper.categories,
                "published_date": paper.published_date,
                "source": paper.source,
                "pdf_url": paper.pdf_url,
                "local_path": paper.local_path
            })

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_download_stats(self) -> Dict:
        """Get download statistics."""
        pdf_files = list(self.output_dir.glob("*.pdf"))
        total_size = sum(f.stat().st_size for f in pdf_files)

        return {
            "total_downloaded": len(pdf_files),
            "total_size_mb": total_size / (1024 * 1024),
            "output_dir": str(self.output_dir)
        }


def download_papers_for_training(
    output_dir: str,
    target_count: int = 200,
    categories: List[str] = None
) -> List[PaperMetadata]:
    """
    Convenience function to download papers for training.

    Args:
        output_dir: Output directory for PDFs
        target_count: Target number of papers
        categories: arXiv categories

    Returns:
        List of downloaded paper metadata
    """
    if categories is None:
        categories = ["cs.CL", "cs.CV", "cs.LG", "cs.AI"]

    downloader = PDFDownloader(output_dir)

    print(f"Searching arXiv for papers in categories: {categories}")
    successful = downloader.download_arxiv_papers(
        categories=categories,
        target_count=target_count,
        date_from="2024-01-01"
    )

    print(f"Successfully downloaded {len(successful)} papers")
    if len(successful) < target_count:
        print(
            f"Warning: Only downloaded {len(successful)} / {target_count} papers. "
            "You may need to widen date range or categories."
        )

    # Save metadata
    metadata_path = Path(output_dir) / "paper_metadata.json"
    downloader.save_metadata(successful, str(metadata_path))

    return successful
