"""
Main Pipeline for Multimodal Contrastive Learning Data Generation

Orchestrates the entire workflow:
1. PDF Download → 2. MinerU Parsing → 3. Modal Extraction →
4. Query Generation → 5. Negative Sampling → 6. Dataset Output
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm

from .utils.config import Config
from .utils.logging_utils import PipelineLogger
from .utils.file_utils import (
    ensure_dir, safe_json_dump, safe_json_load,
    write_jsonl, read_jsonl, get_pdf_files
)
from .parsers.pdf_downloader import PDFDownloader, download_papers_for_training
from .parsers.mineru_parser import MinerUParser, ParsedDocument
from .parsers.modal_extractor import ModalExtractor, Passage, ModalityType
from .generators.query_generator import MultimodalQueryGenerator, GeneratedQuery
from .samplers.negative_sampler import HardNegativeSampler, ContrastiveTriplet


@dataclass
class PipelineStats:
    """Statistics for pipeline execution."""
    start_time: datetime
    end_time: Optional[datetime] = None
    total_pdfs: int = 0
    parsed_pdfs: int = 0
    failed_pdfs: int = 0
    total_elements: int = 0
    total_passages: int = 0
    total_queries: int = 0
    total_triplets: int = 0
    modal_distribution: Dict[str, int] = None
    query_type_distribution: Dict[str, int] = None
    query_modality_distribution: Dict[str, int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": (self.end_time - self.start_time).total_seconds() if self.end_time else None,
            "total_pdfs": self.total_pdfs,
            "parsed_pdfs": self.parsed_pdfs,
            "failed_pdfs": self.failed_pdfs,
            "parse_success_rate": self.parsed_pdfs / max(1, self.total_pdfs),
            "total_elements": self.total_elements,
            "total_passages": self.total_passages,
            "total_queries": self.total_queries,
            "total_triplets": self.total_triplets,
            "modal_distribution": self.modal_distribution or {},
            "query_type_distribution": self.query_type_distribution or {},
            "query_modality_distribution": self.query_modality_distribution or {}
        }


class ContrastiveDataPipeline:
    """
    Main pipeline for generating multimodal contrastive learning data.

    Designed for deployment on NTU EEE cluster with 4x A2000 GPUs.
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize pipeline with configuration.

        Args:
            config_path: Path to configuration file
        """
        self.config = Config(config_path)

        # Setup logging
        self.logger = PipelineLogger(
            name="contrastive_pipeline",
            log_dir=self.config.paths.get("log_dir", "./logs")
        )

        # Initialize components
        self._init_components()

        # Create directories
        self._setup_directories()

        # Checkpoint tracking
        self.checkpoint_path = Path(self.config.paths["checkpoint_dir"]) / "checkpoint.json"

    def _init_components(self) -> None:
        """Initialize pipeline components."""
        # MinerU Parser with GPU parallelization
        self.parser = MinerUParser(
            output_dir=self.config.paths["mineru_output"],
            backend=self.config.mineru.backend,
            devices=self.config.mineru.devices,
            num_workers=self.config.mineru.num_workers,
            language=self.config.mineru.language,
            timeout=self.config.mineru.timeout
        )

        # Modal Extractor
        self.extractor = ModalExtractor(self.config.modal_types)

        # Query Generator (lazy initialization - requires API key)
        self._query_gen = None

        # Negative Sampler
        neg_config = self.config.negative_sampling
        self.sampler = HardNegativeSampler(
            num_negatives=neg_config.num_negatives,
            strategy=neg_config.strategy,
            distribution=neg_config.distribution,
            use_embeddings=neg_config.use_embeddings,
            embedding_model=neg_config.embedding_model
        )

    @property
    def query_generator(self) -> MultimodalQueryGenerator:
        """Lazy-load query generator."""
        if self._query_gen is None:
            qg_config = self.config.query_generation
            self._query_gen = MultimodalQueryGenerator(
                provider=qg_config.provider,
                model=qg_config.model,
                temperature=qg_config.temperature,
                max_tokens=qg_config.max_tokens,
                rate_limit=qg_config.rate_limit,
                max_retries=qg_config.max_retries,
                retry_delay=qg_config.retry_delay
            )
        return self._query_gen

    def _setup_directories(self) -> None:
        """Create necessary directories."""
        for path_key, path_value in self.config.paths.items():
            ensure_dir(path_value)

    def _save_checkpoint(self, state: Dict[str, Any]) -> None:
        """Save checkpoint for resume capability."""
        state["timestamp"] = datetime.now().isoformat()
        safe_json_dump(state, self.checkpoint_path)

    def _load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load checkpoint if exists."""
        return safe_json_load(self.checkpoint_path)

    # =========================================================================
    # Stage 1: PDF Download
    # =========================================================================

    def download_pdfs(
        self,
        target_count: int = None,
        categories: List[str] = None
    ) -> List[Path]:
        """
        Download PDFs from configured sources.

        Args:
            target_count: Number of PDFs to download
            categories: arXiv categories to search

        Returns:
            List of downloaded PDF paths
        """
        self.logger.info("Starting PDF download stage")

        target_count = target_count or self.config.constraints["target_docs"]
        download_config = self.config.download

        categories = categories or download_config.get("arxiv_categories", ["cs.CL", "cs.CV"])

        downloader = PDFDownloader(
            output_dir=self.config.paths["pdf_input"],
            max_concurrent=download_config.get("concurrent_downloads", 5),
            max_retries=download_config.get("max_retries", 3)
        )

        successful = downloader.download_arxiv_papers(
            categories=categories,
            target_count=target_count,
            date_from=download_config.get("date_from"),
            date_to=download_config.get("date_to"),
            search_query=download_config.get("search_query")
        )

        # Save metadata
        metadata_path = Path(self.config.paths["pdf_input"]) / "metadata.json"
        downloader.save_metadata(successful, str(metadata_path))

        self.logger.info(f"Downloaded {len(successful)} PDFs (target {target_count})")
        if len(successful) < target_count:
            self.logger.warning(
                "PDF download target not reached. Consider widening date range or categories."
            )
        self.logger.update_metric("docs_processed", len(successful), increment=False)

        return [Path(p.local_path) for p in successful if p.local_path]

    # =========================================================================
    # Stage 2: Document Parsing
    # =========================================================================

    def parse_documents(
        self,
        pdf_paths: List[Path] = None,
        resume: bool = True
    ) -> List[ParsedDocument]:
        """
        Parse PDF documents using MinerU.

        Args:
            pdf_paths: List of PDF paths (auto-detect if None)
            resume: Whether to skip already parsed documents

        Returns:
            List of ParsedDocument objects
        """
        self.logger.info("Starting document parsing stage")

        if pdf_paths is None:
            pdf_paths = get_pdf_files(self.config.paths["pdf_input"])

        # Check for already parsed documents
        if resume:
            parsed_ids = set()
            output_dir = Path(self.config.paths["mineru_output"])
            for d in output_dir.iterdir():
                if d.is_dir():
                    parsed_ids.add(d.name)

            original_count = len(pdf_paths)
            pdf_paths = [p for p in pdf_paths if p.stem not in parsed_ids]
            skipped = original_count - len(pdf_paths)
            if skipped > 0:
                self.logger.info(f"Skipping {skipped} already parsed documents")

        if not pdf_paths:
            self.logger.info("No documents to parse")
            return []

        self.logger.info(f"Parsing {len(pdf_paths)} documents with {self.config.mineru.num_workers} workers")

        # Progress callback
        def progress_callback(completed: int, total: int, result: ParsedDocument):
            if result.success:
                self.logger.update_metric("docs_processed")
            else:
                self.logger.update_metric("docs_failed")
                self.logger.error(f"Parse failed for {result.doc_id}: {result.error_message}")

        # Parse with GPU parallelization
        results = self.parser.parse_batch(
            [str(p) for p in pdf_paths],
            progress_callback=progress_callback
        )

        saved = 0
        for result in results:
            if result.success:
                if self.parser.save_structure(result):
                    saved += 1

        successful = [r for r in results if r.success]
        self.logger.info(f"Successfully parsed {len(successful)}/{len(results)} documents")
        if saved:
            self.logger.info(f"Saved {saved} structure.json files for resume")

        return results

    # =========================================================================
    # Stage 3: Modal Extraction
    # =========================================================================

    def extract_passages(
        self,
        parsed_docs: List[ParsedDocument]
    ) -> Tuple[List[Passage], Dict[str, List[Passage]]]:
        """
        Extract passages from parsed documents.

        Args:
            parsed_docs: List of ParsedDocument objects

        Returns:
            Tuple of (all_passages, passages_by_doc)
        """
        self.logger.info("Starting modal extraction stage")

        all_passages = []
        passages_by_doc = {}

        for doc in tqdm(parsed_docs, desc="Extracting passages"):
            if not doc.success:
                continue

            passages = self.extractor.extract_passages(
                elements=doc.elements,
                doc_id=doc.doc_id,
                include_context=True
            )

            # Filter by quality
            passages = self.extractor.filter_by_quality(passages, min_score=0.4)

            # Limit per document
            max_per_doc = self.config.constraints.get("max_elements_per_doc", 50)
            passages = passages[:max_per_doc]

            if len(passages) >= self.config.constraints.get("min_elements_per_doc", 5):
                all_passages.extend(passages)
                passages_by_doc[doc.doc_id] = passages
                self.logger.update_metric("elements_extracted", len(passages))

        # Log distribution
        distribution = self.extractor.get_modal_distribution(all_passages)
        self.logger.info(f"Modal distribution: {distribution}")

        return all_passages, passages_by_doc

    # =========================================================================
    # Stage 4: Query Generation
    # =========================================================================

    def generate_queries(
        self,
        passages: List[Passage],
        passages_by_doc: Dict[str, List[Passage]]
    ) -> Dict[str, List[GeneratedQuery]]:
        """
        Generate queries for passages.

        Args:
            passages: List of all passages
            passages_by_doc: Passages grouped by document

        Returns:
            Dict mapping passage_id to list of queries
        """
        self.logger.info("Starting query generation stage")

        num_queries = self.config.query_generation.queries_per_element

        # Generate queries in batches
        all_query_data = {}

        batch_size = self.config.query_generation.batch_size

        for i in tqdm(range(0, len(passages), batch_size), desc="Generating queries"):
            batch = passages[i:i + batch_size]

            try:
                batch_results = self.query_generator.generate_batch(
                    batch,
                    num_queries=num_queries
                )
                all_query_data.update(batch_results)

                total_queries = sum(len(qs) for qs in batch_results.values())
                self.logger.update_metric("queries_generated", total_queries)

                # Checkpoint every N batches
                if (i // batch_size) % 10 == 0:
                    self._save_checkpoint({
                        "stage": "query_generation",
                        "processed_passages": i + len(batch),
                        "total_queries": sum(len(qs) for qs in all_query_data.values())
                    })

            except Exception as e:
                self.logger.error(f"Query generation batch failed: {e}")
                continue

        # Generate cross-modal queries for each document
        self.logger.info("Generating cross-modal queries")
        for doc_id, doc_passages in passages_by_doc.items():
            if len(doc_passages) >= 2:
                try:
                    cross_queries = self.query_generator.generate_cross_modal_queries(
                        doc_passages, num_queries=2
                    )
                    if cross_queries:
                        # Assign to first multi-modal passage
                        key = doc_passages[0].passage_id
                        if key in all_query_data:
                            all_query_data[key].extend(cross_queries)
                        else:
                            all_query_data[key] = cross_queries
                except Exception as e:
                    self.logger.error(f"Cross-modal query generation failed for {doc_id}: {e}")

        type_distribution, modality_distribution = self._summarize_queries(all_query_data)
        if type_distribution:
            self.logger.info(f"Query type distribution: {type_distribution}")
        if modality_distribution:
            self.logger.info(f"Query modality distribution: {modality_distribution}")

        self.logger.info(f"Generated {sum(len(qs) for qs in all_query_data.values())} queries")

        return all_query_data

    def _summarize_queries(
        self,
        query_data: Dict[str, List[GeneratedQuery]]
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Summarize query counts by type and target modality."""
        type_distribution: Dict[str, int] = {}
        modality_distribution: Dict[str, int] = {}

        for queries in query_data.values():
            for query in queries:
                query_type = getattr(query, "query_type", None)
                if query_type is None and isinstance(query, dict):
                    query_type = query.get("query_type", "unknown")
                query_type = query_type or "unknown"
                type_distribution[query_type] = type_distribution.get(query_type, 0) + 1

                target_modality = getattr(query, "target_modality", None)
                if target_modality is None and isinstance(query, dict):
                    target_modality = query.get("target_modality", "unknown")
                target_modality = target_modality or "unknown"
                modality_distribution[target_modality] = (
                    modality_distribution.get(target_modality, 0) + 1
                )

        return type_distribution, modality_distribution

    # =========================================================================
    # Stage 5: Triplet Construction
    # =========================================================================

    def construct_triplets(
        self,
        query_data: Dict[str, List[GeneratedQuery]],
        all_passages: List[Passage]
    ) -> List[ContrastiveTriplet]:
        """
        Construct contrastive learning triplets.

        Args:
            query_data: Dict mapping passage_id to queries
            all_passages: All passages for negative sampling

        Returns:
            List of ContrastiveTriplet objects
        """
        self.logger.info("Starting triplet construction stage")

        # Build passage lookup
        passages_by_id = {p.passage_id: p for p in all_passages}

        # Construct triplets
        triplets = self.sampler.construct_triplets(
            query_data=query_data,
            passage_pool=all_passages,
            passages_by_id=passages_by_id
        )

        self.logger.update_metric("triplets_created", len(triplets))
        self.logger.info(f"Constructed {len(triplets)} triplets")

        return triplets

    # =========================================================================
    # Stage 6: Output
    # =========================================================================

    def save_dataset(
        self,
        triplets: List[ContrastiveTriplet],
        output_name: str = "contrastive_dataset"
    ) -> Dict[str, str]:
        """
        Save dataset in configured format.

        Args:
            triplets: List of triplets
            output_name: Output file name (without extension)

        Returns:
            Dict with output file paths
        """
        self.logger.info("Saving dataset")

        output_dir = Path(self.config.paths["dataset_output"])
        output_config = self.config.output

        # Split train/validation
        train_ratio = output_config.get("train_ratio", 0.9)
        split_idx = int(len(triplets) * train_ratio)

        train_triplets = triplets[:split_idx]
        val_triplets = triplets[split_idx:]

        output_paths = {}

        output_format = output_config.get("format", "jsonl")

        if output_format == "jsonl":
            # JSONL format (one triplet per line)
            train_path = output_dir / f"{output_name}_train.jsonl"
            val_path = output_dir / f"{output_name}_val.jsonl"

            write_jsonl(
                [t.to_training_format() for t in train_triplets],
                train_path
            )
            write_jsonl(
                [t.to_training_format() for t in val_triplets],
                val_path
            )

            output_paths["train"] = str(train_path)
            output_paths["validation"] = str(val_path)

        elif output_format == "json":
            # Single JSON file
            output_path = output_dir / f"{output_name}.json"
            data = {
                "train": [t.to_training_format() for t in train_triplets],
                "validation": [t.to_training_format() for t in val_triplets],
                "metadata": {
                    "total_triplets": len(triplets),
                    "train_count": len(train_triplets),
                    "val_count": len(val_triplets),
                    "created_at": datetime.now().isoformat()
                }
            }
            safe_json_dump(data, output_path)
            output_paths["dataset"] = str(output_path)

        self.logger.info(f"Dataset saved to {output_paths}")

        return output_paths

    # =========================================================================
    # Full Pipeline Execution
    # =========================================================================

    def run(
        self,
        target_docs: int = None,
        skip_download: bool = False,
        skip_parse: bool = False,
        resume: bool = True
    ) -> PipelineStats:
        """
        Run the full pipeline.

        Args:
            target_docs: Target number of documents
            skip_download: Skip download stage (use existing PDFs)
            skip_parse: Skip parse stage (use existing parsed data)
            resume: Resume from checkpoint if available

        Returns:
            Pipeline execution statistics
        """
        stats = PipelineStats(start_time=datetime.now())
        target_docs = target_docs or self.config.constraints["target_docs"]

        self.logger.info("=" * 60)
        self.logger.info("Starting Multimodal Contrastive Data Pipeline")
        self.logger.info(f"Target documents: {target_docs}")
        self.logger.info("=" * 60)

        try:
            # Stage 1: Download
            if not skip_download:
                pdf_paths = self.download_pdfs(target_count=target_docs)
            else:
                pdf_paths = get_pdf_files(self.config.paths["pdf_input"])
                self.logger.info(f"Using {len(pdf_paths)} existing PDFs")

            stats.total_pdfs = len(pdf_paths)

            # Stage 2: Parse
            if not skip_parse:
                parsed_docs = self.parse_documents(pdf_paths, resume=resume)
            else:
                # Load from existing parsed data
                parsed_docs = self._load_parsed_documents()

            stats.parsed_pdfs = sum(1 for d in parsed_docs if d.success)
            stats.failed_pdfs = sum(1 for d in parsed_docs if not d.success)

            # Stage 3: Extract
            all_passages, passages_by_doc = self.extract_passages(
                [d for d in parsed_docs if d.success]
            )

            stats.total_elements = sum(len(d.elements) for d in parsed_docs if d.success)
            stats.total_passages = len(all_passages)
            stats.modal_distribution = self.extractor.get_modal_distribution(all_passages)

            if not all_passages:
                self.logger.error("No passages extracted, aborting")
                return stats

            # Stage 4: Generate Queries
            query_data = self.generate_queries(all_passages, passages_by_doc)

            stats.total_queries = sum(len(qs) for qs in query_data.values())
            (
                stats.query_type_distribution,
                stats.query_modality_distribution
            ) = self._summarize_queries(query_data)

            # Stage 5: Construct Triplets
            triplets = self.construct_triplets(query_data, all_passages)

            stats.total_triplets = len(triplets)

            # Stage 6: Save
            output_paths = self.save_dataset(triplets)

            # Save statistics
            stats.end_time = datetime.now()
            stats_path = Path(self.config.paths["dataset_output"]) / "pipeline_stats.json"
            safe_json_dump(stats.to_dict(), stats_path)

            # Log summary
            self.logger.log_summary()

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc=e)
            stats.end_time = datetime.now()
            raise

        return stats

    def _load_parsed_documents(self) -> List[ParsedDocument]:
        """Load previously parsed documents."""
        parsed_docs = []
        output_dir = Path(self.config.paths["mineru_output"])

        for doc_dir in output_dir.iterdir():
            if not doc_dir.is_dir():
                continue

            structure_file = doc_dir / "structure.json"
            if structure_file.exists():
                try:
                    data = safe_json_load(structure_file)
                    if data:
                        # Reconstruct ParsedDocument
                        from .parsers.mineru_parser import ParsedElement
                        elements = []
                        for elem_data in data.get("elements", []):
                            elements.append(ParsedElement(
                                element_id=elem_data.get("element_id", ""),
                                doc_id=data.get("doc_id", doc_dir.name),
                                page_idx=elem_data.get("page_idx", 0),
                                element_type=elem_data.get("type", "text"),
                                content=elem_data.get("content", ""),
                                bbox=elem_data.get("bbox"),
                                image_path=elem_data.get("image_path"),
                                metadata=elem_data.get("metadata", {})
                            ))

                        parsed_docs.append(ParsedDocument(
                            doc_id=data.get("doc_id", doc_dir.name),
                            pdf_path=data.get("pdf_path", ""),
                            output_path=str(doc_dir),
                            total_pages=data.get("total_pages", 0),
                            elements=elements,
                            parse_time=0,
                            success=True
                        ))
                except Exception as e:
                    self.logger.error(f"Failed to load {doc_dir}: {e}")

        self.logger.info(f"Loaded {len(parsed_docs)} parsed documents")
        return parsed_docs


def run_pipeline(
    config_path: str = "configs/config.yaml",
    target_docs: int = 200,
    skip_download: bool = False,
    skip_parse: bool = False
) -> PipelineStats:
    """
    Convenience function to run the pipeline.

    Args:
        config_path: Path to config file
        target_docs: Target number of documents
        skip_download: Skip download stage
        skip_parse: Skip parse stage

    Returns:
        Pipeline statistics
    """
    pipeline = ContrastiveDataPipeline(config_path)
    return pipeline.run(
        target_docs=target_docs,
        skip_download=skip_download,
        skip_parse=skip_parse
    )
