"""Persistent token usage logger for tracking Anthropic API costs across runs.

Usage in a script:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.utils.token_logger import log_run

    log_run(
        script="my_script",
        model="claude-sonnet-4-5-20250929",
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        extra={"queries_generated": 42, "qc_pass": 30},
    )
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Pricing per million tokens (USD) — update when Anthropic changes rates
_PRICING: Dict[str, Dict[str, float]] = {
    "claude-opus-4-6":            {"input": 15.0,  "output": 75.0},
    "claude-opus-4":              {"input": 15.0,  "output": 75.0},
    "claude-sonnet-4-5":          {"input":  3.0,  "output": 15.0},
    "claude-sonnet-4":            {"input":  3.0,  "output": 15.0},
    "claude-haiku-4-5":           {"input":  0.80, "output":  4.0},
    "claude-haiku-4":             {"input":  0.80, "output":  4.0},
    "claude-3-5-sonnet":          {"input":  3.0,  "output": 15.0},
    "claude-3-5-haiku":           {"input":  0.80, "output":  4.0},
    "claude-3-opus":              {"input": 15.0,  "output": 75.0},
}
_DEFAULT_PRICING = {"input": 3.0, "output": 15.0}

# Default log file (relative to project root, scripts are run from there)
DEFAULT_LOG_FILE = Path("logs/token_usage.jsonl")


def _get_price(model: str) -> Dict[str, float]:
    """Return per-million-token pricing for a model string."""
    for prefix, price in _PRICING.items():
        if prefix in model:
            return price
    return _DEFAULT_PRICING


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost for an API call."""
    p = _get_price(model)
    return input_tokens * p["input"] / 1_000_000 + output_tokens * p["output"] / 1_000_000


def log_run(
    script: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    extra: Optional[Dict[str, Any]] = None,
    log_file: Optional[Path] = None,
) -> None:
    """Append one run record to the persistent token usage log.

    Args:
        script:        Short name of the calling script (e.g. "generate_l2_queries").
        model:         Anthropic model ID used for this run.
        input_tokens:  Total input tokens consumed across the entire run.
        output_tokens: Total output tokens produced across the entire run.
        extra:         Optional dict of additional metadata (qc_pass, queries_generated…).
        log_file:      Override path; defaults to logs/token_usage.jsonl.
    """
    if input_tokens == 0 and output_tokens == 0:
        return  # dry-run or empty batch — nothing to record

    target = Path(log_file) if log_file else DEFAULT_LOG_FILE
    target.parent.mkdir(parents=True, exist_ok=True)

    cost = estimate_cost(model, input_tokens, output_tokens)
    record: Dict[str, Any] = {
        "ts":            datetime.now().isoformat(timespec="seconds"),
        "script":        script,
        "model":         model,
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
        "cost_usd":      round(cost, 6),
    }
    if extra:
        record.update(extra)

    with open(target, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        f"[token_log] {target} ← "
        f"${cost:.4f}  ({input_tokens:,} in + {output_tokens:,} out)"
    )


def summarize(log_file: Optional[Path] = None) -> None:
    """Print cumulative token usage stats from the persistent log."""
    target = Path(log_file) if log_file else DEFAULT_LOG_FILE

    if not target.exists():
        print(f"[token_log] No log file at {target}")
        return

    records = []
    with open(target, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not records:
        print(f"[token_log] {target} is empty.")
        return

    total_in   = sum(r.get("input_tokens",  0) for r in records)
    total_out  = sum(r.get("output_tokens", 0) for r in records)
    total_cost = sum(r.get("cost_usd",      0) for r in records)

    by_script: Dict[str, Dict[str, Any]] = {}
    for r in records:
        s = r.get("script", "unknown")
        if s not in by_script:
            by_script[s] = {"runs": 0, "input": 0, "output": 0, "cost": 0.0}
        by_script[s]["runs"]   += 1
        by_script[s]["input"]  += r.get("input_tokens",  0)
        by_script[s]["output"] += r.get("output_tokens", 0)
        by_script[s]["cost"]   += r.get("cost_usd",      0)

    print(f"\n{'='*62}")
    print(f"Cumulative Token Usage  [{target}]")
    print(f"{'='*62}")
    print(f"  Runs:          {len(records)}")
    print(f"  Input tokens:  {total_in:>15,}")
    print(f"  Output tokens: {total_out:>15,}")
    print(f"  Total cost:    ${total_cost:>10.4f}")
    print(f"\n  Breakdown by script:")
    for s, d in sorted(by_script.items()):
        print(
            f"    {s:<40s} {d['runs']:3d} run(s)  "
            f"${d['cost']:.4f}  "
            f"({d['input']:,} in + {d['output']:,} out)"
        )
    print(f"{'='*62}")


if __name__ == "__main__":
    # Allow: python -m src.utils.token_logger  OR  python src/utils/token_logger.py
    summarize()
