"""Logging utilities for the pipeline."""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str = "pipeline",
    log_file: Optional[str] = None,
    level: str = "INFO",
    format_str: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.

    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
        format_str: Log format string
        console: Whether to log to console

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Default format
    if format_str is None:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(format_str)

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class PipelineLogger:
    """Enhanced logger for pipeline with metrics tracking."""

    def __init__(self, name: str = "pipeline", log_dir: str = "./logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"{name}_{timestamp}.log"

        self.logger = setup_logger(name, str(log_file))

        # Metrics tracking
        self.metrics = {
            "docs_processed": 0,
            "docs_failed": 0,
            "elements_extracted": 0,
            "queries_generated": 0,
            "triplets_created": 0,
            "errors": []
        }

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def warning(self, msg: str) -> None:
        self.logger.warning(msg)

    def error(self, msg: str, exc: Optional[Exception] = None) -> None:
        self.logger.error(msg)
        self.metrics["errors"].append({
            "message": msg,
            "exception": str(exc) if exc else None,
            "timestamp": datetime.now().isoformat()
        })

    def debug(self, msg: str) -> None:
        self.logger.debug(msg)

    def update_metric(self, key: str, value: int = 1, increment: bool = True) -> None:
        """Update a metric value."""
        if key in self.metrics:
            if increment:
                self.metrics[key] += value
            else:
                self.metrics[key] = value

    def get_summary(self) -> dict:
        """Get metrics summary."""
        return {
            **self.metrics,
            "success_rate": (
                self.metrics["docs_processed"] /
                (self.metrics["docs_processed"] + self.metrics["docs_failed"])
                if (self.metrics["docs_processed"] + self.metrics["docs_failed"]) > 0
                else 0
            )
        }

    def log_summary(self) -> None:
        """Log final metrics summary."""
        summary = self.get_summary()
        self.info("=" * 60)
        self.info("Pipeline Execution Summary")
        self.info("=" * 60)
        self.info(f"Documents processed: {summary['docs_processed']}")
        self.info(f"Documents failed: {summary['docs_failed']}")
        self.info(f"Success rate: {summary['success_rate']:.2%}")
        self.info(f"Elements extracted: {summary['elements_extracted']}")
        self.info(f"Queries generated: {summary['queries_generated']}")
        self.info(f"Triplets created: {summary['triplets_created']}")
        self.info(f"Total errors: {len(summary['errors'])}")
        self.info("=" * 60)
