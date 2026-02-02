"""Configuration management module."""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class MinerUConfig:
    """MinerU parser configuration."""
    backend: str = "auto"
    devices: list = field(default_factory=lambda: ["cuda:0"])
    num_workers: int = 4
    output_format: str = "md"
    language: str = "en"
    timeout: int = 300


@dataclass
class QueryGenConfig:
    """Query generation configuration."""
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    local_model_path: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 200
    queries_per_element: int = 3
    batch_size: int = 10
    rate_limit: int = 60
    max_retries: int = 3
    retry_delay: int = 2


@dataclass
class NegativeSamplingConfig:
    """Negative sampling configuration."""
    strategy: str = "modal_mixed"
    num_negatives: int = 3
    distribution: Dict[str, float] = field(default_factory=lambda: {
        "hard_same_modal": 0.6,
        "cross_modal": 0.3,
        "random": 0.1
    })
    use_embeddings: bool = True
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"


class Config:
    """Main configuration class."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = Path(config_path)
        self._raw_config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._raw_config = yaml.safe_load(f) or {}
        else:
            print(f"Warning: Config file {self.config_path} not found, using defaults")
            self._raw_config = {}

    def reload(self) -> None:
        """Reload configuration from file."""
        self._load_config()

    @property
    def mineru(self) -> MinerUConfig:
        """Get MinerU configuration."""
        cfg = self._raw_config.get("mineru", {})
        return MinerUConfig(
            backend=cfg.get("backend", "auto"),
            devices=cfg.get("devices", ["cuda:0"]),
            num_workers=cfg.get("num_workers", 4),
            output_format=cfg.get("output_format", "md"),
            language=cfg.get("language", "en"),
            timeout=cfg.get("timeout", 300)
        )

    @property
    def query_generation(self) -> QueryGenConfig:
        """Get query generation configuration."""
        cfg = self._raw_config.get("query_generation", {})
        return QueryGenConfig(
            provider=cfg.get("provider", "openai"),
            model=cfg.get("model", "gpt-4o-mini"),
            local_model_path=cfg.get("local_model_path"),
            temperature=cfg.get("temperature", 0.7),
            max_tokens=cfg.get("max_tokens", 200),
            queries_per_element=cfg.get("queries_per_element", 3),
            batch_size=cfg.get("batch_size", 10),
            rate_limit=cfg.get("rate_limit", 60),
            max_retries=cfg.get("max_retries", 3),
            retry_delay=cfg.get("retry_delay", 2)
        )

    @property
    def negative_sampling(self) -> NegativeSamplingConfig:
        """Get negative sampling configuration."""
        cfg = self._raw_config.get("negative_sampling", {})
        return NegativeSamplingConfig(
            strategy=cfg.get("strategy", "modal_mixed"),
            num_negatives=cfg.get("num_negatives", 3),
            distribution=cfg.get("distribution", {
                "hard_same_modal": 0.6,
                "cross_modal": 0.3,
                "random": 0.1
            }),
            use_embeddings=cfg.get("use_embeddings", True),
            embedding_model=cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        )

    @property
    def paths(self) -> Dict[str, str]:
        """Get path configuration."""
        defaults = {
            "pdf_input": "./data/raw_pdfs",
            "mineru_output": "./data/mineru_output",
            "dataset_output": "./data/contrastive_data",
            "checkpoint_dir": "./data/checkpoints",
            "log_dir": "./logs"
        }
        return {**defaults, **self._raw_config.get("paths", {})}

    @property
    def download(self) -> Dict[str, Any]:
        """Get download configuration."""
        return self._raw_config.get("download", {})

    @property
    def constraints(self) -> Dict[str, Any]:
        """Get processing constraints."""
        defaults = {
            "target_docs": 200,
            "max_elements_per_doc": 50,
            "min_elements_per_doc": 5,
            "skip_on_error": True,
            "checkpoint_interval": 10
        }
        return {**defaults, **self._raw_config.get("constraints", {})}

    @property
    def modal_types(self) -> Dict[str, Any]:
        """Get modal type configurations."""
        return self._raw_config.get("modal_types", {})

    @property
    def output(self) -> Dict[str, Any]:
        """Get output configuration."""
        defaults = {
            "format": "jsonl",
            "train_ratio": 0.9,
            "include_images": True,
            "compress": False
        }
        return {**defaults, **self._raw_config.get("output", {})}

    def get(self, key: str, default: Any = None) -> Any:
        """Get raw configuration value."""
        return self._raw_config.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access to configuration."""
        return self._raw_config[key]
