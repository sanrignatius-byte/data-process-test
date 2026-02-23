"""Persistent token usage logger — SQLite backend, management-ready reports.

Every query-generation run is recorded in logs/token_usage.db with full
per-run details: timestamp, script, model, purpose, token counts, cost,
and task-level metadata (qc_pass, queries_written, …).

Usage in a script
-----------------
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.utils.token_logger import log_run

    log_run(
        script="generate_l2_queries",
        model="claude-sonnet-4-5-20250929",
        purpose="L2 cross-document query generation (citation-based candidates)",
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        extra={"pairs_processed": 43, "qc_pass": 19, "qc_fail": 23},
    )

CLI usage
---------
    # Print last 30-day summary  (default)
    python src/utils/token_logger.py

    # Custom window
    python src/utils/token_logger.py --days 7

    # All-time
    python src/utils/token_logger.py --all

    # Export CSV (e.g. for spreadsheet hand-off to manager)
    python src/utils/token_logger.py --csv > token_report.csv

    # Migrate old JSONL log into SQLite
    python src/utils/token_logger.py --migrate logs/token_usage.jsonl
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Pricing table (USD per million tokens) — update when Anthropic changes rates
# ---------------------------------------------------------------------------
_PRICING: Dict[str, Dict[str, float]] = {
    "claude-opus-4-6":   {"input": 15.0,  "output": 75.0},
    "claude-opus-4":     {"input": 15.0,  "output": 75.0},
    "claude-sonnet-4-5": {"input":  3.0,  "output": 15.0},
    "claude-sonnet-4":   {"input":  3.0,  "output": 15.0},
    "claude-haiku-4-5":  {"input":  0.80, "output":  4.0},
    "claude-haiku-4":    {"input":  0.80, "output":  4.0},
    "claude-3-5-sonnet": {"input":  3.0,  "output": 15.0},
    "claude-3-5-haiku":  {"input":  0.80, "output":  4.0},
    "claude-3-opus":     {"input": 15.0,  "output": 75.0},
}
_DEFAULT_PRICING = {"input": 3.0, "output": 15.0}

DEFAULT_DB = Path("logs/token_usage.db")

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_price(model: str) -> Dict[str, float]:
    for prefix, price in _PRICING.items():
        if prefix in model:
            return price
    return _DEFAULT_PRICING


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost from token counts."""
    p = _get_price(model)
    return input_tokens * p["input"] / 1_000_000 + output_tokens * p["output"] / 1_000_000


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS token_usage (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ts              TEXT    NOT NULL,
            script          TEXT    NOT NULL,
            model           TEXT    NOT NULL,
            purpose         TEXT    NOT NULL DEFAULT '',
            input_tokens    INTEGER NOT NULL,
            output_tokens   INTEGER NOT NULL,
            cost_usd        REAL    NOT NULL,
            pairs_processed INTEGER,
            queries_written INTEGER,
            qc_pass         INTEGER,
            qc_fail         INTEGER,
            parse_failures  INTEGER,
            output_file     TEXT,
            extra_json      TEXT
        )
    """)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def log_run(
    script: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    purpose: str = "",
    extra: Optional[Dict[str, Any]] = None,
    db_path: Optional[Path] = None,
) -> None:
    """Record one run to the audit database.

    Args:
        script:        Calling script name (e.g. "generate_l2_queries").
        model:         Anthropic model ID.
        purpose:       Human-readable description of what this run produced.
                       Example: "L1 dual-evidence v4.2 — 118 pairs (figure+table)"
        input_tokens:  Total input tokens for the entire run.
        output_tokens: Total output tokens for the entire run.
        extra:         Optional metadata dict.  Known keys (pairs_processed,
                       queries_written, qc_pass, qc_fail, parse_failures,
                       output) are stored in dedicated columns; anything else
                       goes into extra_json.
        db_path:       Override DB path; defaults to logs/token_usage.db.
    """
    if input_tokens == 0 and output_tokens == 0:
        return  # dry-run / empty batch — nothing billable to record

    ex = extra or {}
    cost = estimate_cost(model, input_tokens, output_tokens)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Split known fields from catch-all
    known = {"pairs_processed", "queries_written", "qc_pass", "qc_fail",
             "parse_failures", "output"}
    rest = {k: v for k, v in ex.items() if k not in known}

    conn = _connect(Path(db_path) if db_path else DEFAULT_DB)
    conn.execute(
        """INSERT INTO token_usage
           (ts, script, model, purpose, input_tokens, output_tokens, cost_usd,
            pairs_processed, queries_written, qc_pass, qc_fail, parse_failures,
            output_file, extra_json)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            ts, script, model, purpose,
            input_tokens, output_tokens, round(cost, 6),
            ex.get("pairs_processed"), ex.get("queries_written"),
            ex.get("qc_pass"),         ex.get("qc_fail"),
            ex.get("parse_failures"),  ex.get("output"),
            json.dumps(rest, ensure_ascii=False) if rest else None,
        ),
    )
    conn.commit()
    conn.close()

    print(
        f"[token_log] Recorded → ${cost:.4f}  "
        f"({input_tokens:,} in + {output_tokens:,} out)  "
        f"[{script}]"
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fetch(db_path: Path, since: Optional[datetime] = None) -> List[sqlite3.Row]:
    if not db_path.exists():
        return []
    conn = _connect(db_path)
    if since:
        rows = conn.execute(
            "SELECT * FROM token_usage WHERE ts >= ? ORDER BY ts",
            (since.isoformat(timespec="seconds"),),
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM token_usage ORDER BY ts").fetchall()
    conn.close()
    return rows


def report(
    days: Optional[int] = 30,
    all_time: bool = False,
    db_path: Optional[Path] = None,
) -> None:
    """Print a management-friendly token usage report to stdout."""
    target = Path(db_path) if db_path else DEFAULT_DB
    since = None if all_time else datetime.now(timezone.utc) - timedelta(days=days or 30)
    rows = _fetch(target, since)

    window = "all time" if all_time else f"last {days} day(s)"

    if not rows:
        print(f"[token_log] No records in {target} ({window}).")
        return

    total_in   = sum(r["input_tokens"]  for r in rows)
    total_out  = sum(r["output_tokens"] for r in rows)
    total_cost = sum(r["cost_usd"]      for r in rows)

    # Group by script
    by_script: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        s = r["script"]
        if s not in by_script:
            by_script[s] = {"runs": 0, "input": 0, "output": 0, "cost": 0.0}
        by_script[s]["runs"]   += 1
        by_script[s]["input"]  += r["input_tokens"]
        by_script[s]["output"] += r["output_tokens"]
        by_script[s]["cost"]   += r["cost_usd"]

    first_ts = rows[0]["ts"]
    last_ts  = rows[-1]["ts"]

    W = 68
    print(f"\n{'='*W}")
    print(f"  Token Usage Audit Report  ({window})")
    print(f"{'='*W}")
    print(f"  DB:            {target}")
    print(f"  Period:        {first_ts}  →  {last_ts}")
    print(f"  Total runs:    {len(rows)}")
    print(f"  Input tokens:  {total_in:>15,}")
    print(f"  Output tokens: {total_out:>15,}")
    print(f"  Total cost:    ${total_cost:>10.4f}")
    print(f"\n  By script:")
    print(f"  {'Script':<42} {'Runs':>4}  {'Cost (USD)':>10}  {'Input tok':>12}  {'Output tok':>11}")
    print(f"  {'-'*42} {'-'*4}  {'-'*10}  {'-'*12}  {'-'*11}")
    for s, d in sorted(by_script.items(), key=lambda x: -x[1]["cost"]):
        print(f"  {s:<42} {d['runs']:>4}  ${d['cost']:>9.4f}  {d['input']:>12,}  {d['output']:>11,}")

    print(f"\n  Run detail (most recent first):")
    print(f"  {'Timestamp':<22} {'Script':<32} {'$':>7}  {'In':>8}  {'Out':>7}  {'QC✓':>5}  Purpose")
    print(f"  {'-'*22} {'-'*32} {'-'*7}  {'-'*8}  {'-'*7}  {'-'*5}  {'-'*30}")
    for r in reversed(rows):
        qc = str(r["qc_pass"]) if r["qc_pass"] is not None else "-"
        purpose = (r["purpose"] or "")[:50]
        print(
            f"  {r['ts']:<22} {r['script']:<32} "
            f"${r['cost_usd']:>6.4f}  {r['input_tokens']:>8,}  "
            f"{r['output_tokens']:>7,}  {qc:>5}  {purpose}"
        )
    print(f"{'='*W}\n")


def report_csv(days: Optional[int] = None, all_time: bool = True,
               db_path: Optional[Path] = None) -> str:
    """Return a CSV string of all run records."""
    target = Path(db_path) if db_path else DEFAULT_DB
    since = None if all_time else datetime.now(timezone.utc) - timedelta(days=days or 30)
    rows = _fetch(target, since)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "id", "ts", "script", "model", "purpose",
        "input_tokens", "output_tokens", "cost_usd",
        "pairs_processed", "queries_written", "qc_pass", "qc_fail",
        "parse_failures", "output_file",
    ])
    for r in rows:
        writer.writerow([
            r["id"], r["ts"], r["script"], r["model"], r["purpose"],
            r["input_tokens"], r["output_tokens"], r["cost_usd"],
            r["pairs_processed"], r["queries_written"], r["qc_pass"],
            r["qc_fail"], r["parse_failures"], r["output_file"],
        ])
    return buf.getvalue()


# ---------------------------------------------------------------------------
# JSONL migration helper
# ---------------------------------------------------------------------------

def migrate_jsonl(jsonl_path: Path, db_path: Optional[Path] = None) -> int:
    """Import records from a legacy JSONL log into SQLite.

    Returns the number of records imported.
    """
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        print(f"[token_log] JSONL not found: {jsonl_path}")
        return 0

    imported = 0
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            log_run(
                script=r.get("script", "unknown"),
                model=r.get("model", "unknown"),
                purpose=r.get("purpose", ""),
                input_tokens=r.get("input_tokens", 0),
                output_tokens=r.get("output_tokens", 0),
                extra={k: v for k, v in r.items()
                       if k not in {"ts", "script", "model", "purpose",
                                    "input_tokens", "output_tokens", "cost_usd"}},
                db_path=db_path,
            )
            imported += 1
    print(f"[token_log] Migrated {imported} record(s) from {jsonl_path}")
    return imported


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli() -> None:
    ap = argparse.ArgumentParser(
        description="Token usage audit log — view or export records.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python src/utils/token_logger.py                      # last 30 days
  python src/utils/token_logger.py --days 7             # last 7 days
  python src/utils/token_logger.py --all                # all time
  python src/utils/token_logger.py --csv > report.csv   # export for manager
  python src/utils/token_logger.py --migrate logs/token_usage.jsonl
""",
    )
    ap.add_argument("--days",    type=int,  default=30,  help="Window in days (default: 30)")
    ap.add_argument("--all",     action="store_true",    help="Show all-time records")
    ap.add_argument("--csv",     action="store_true",    help="Output CSV to stdout")
    ap.add_argument("--db",      default=str(DEFAULT_DB), help="SQLite DB path")
    ap.add_argument("--migrate", metavar="JSONL",        help="Import a legacy JSONL file")
    args = ap.parse_args()

    db = Path(args.db)

    if args.migrate:
        migrate_jsonl(Path(args.migrate), db_path=db)
        return

    if args.csv:
        print(report_csv(days=None, all_time=True, db_path=db), end="")
        return

    report(days=args.days, all_time=args.all, db_path=db)


if __name__ == "__main__":
    _cli()
