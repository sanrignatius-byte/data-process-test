"""Utilities for recording LLM token usage in append-only JSONL logs."""

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional


class TokenUsageLogger:
    """Thread-safe JSONL logger for LLM token usage auditing."""

    def __init__(self, log_file: str):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]

    def log(
        self,
        *,
        provider: str,
        model: str,
        operation: str,
        prompt: str,
        input_tokens: Optional[int],
        output_tokens: Optional[int],
        response_chars: Optional[int] = None,

        success: bool = True,
        error: Optional[str] = None,

        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "provider": provider,
            "model": model,
            "operation": operation,
            "prompt_hash": self._hash_text(prompt),
            "prompt_chars": len(prompt or ""),
            "response_chars": response_chars,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": (
                (input_tokens or 0) + (output_tokens or 0)
                if (input_tokens is not None or output_tokens is not None)
                else None
            ),

            "success": success,
            "error": error,
            "metadata": metadata or {},
        }
        line = json.dumps(record, ensure_ascii=False, default=str)

        with self._lock:
            with self.log_file.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
