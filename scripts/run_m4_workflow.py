#!/usr/bin/env python3
"""
M4 Complete Workflow - 从Survey到M4 Query的完整流程

完整工作流：
1. 获取Survey引用的论文 (fetch_survey_references)
2. 使用MinerU解析PDF (parse_only)
3. 生成M4跨文档Query (generate_m4_queries)

Usage:
    # 从arXiv Survey开始
    python scripts/run_m4_workflow.py \
        --arxiv-id 2401.12345 \
        --output-dir data/m4_output \
        --max-papers 30 \
        --num-queries 50

    # 从本地PDF目录开始（跳过下载）
    python scripts/run_m4_workflow.py \
        --skip-download \
        --pdf-dir data/raw_pdfs \
        --output-dir data/m4_output

    # 从已解析的MinerU输出开始（跳过下载和解析）
    python scripts/run_m4_workflow.py \
        --skip-download \
        --skip-parse \
        --mineru-dir data/mineru_output \
        --output-dir data/m4_output
"""

import argparse
import json
import sys
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.file_utils import ensure_dir, safe_json_dump


def run_command(cmd: List[str], description: str) -> bool:
    """执行命令并打印输出"""
    print(f"\n{'='*60}")
    print(f"Step: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60 + '\n')

    try:
        result = subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed with return code {e.returncode}")
        return False


def _make_json_serializable(value: Any) -> Any:
    """Recursively convert argparse/path objects to JSON-safe values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _make_json_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_make_json_serializable(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(
        description="M4 Complete Workflow - Survey to M4 Query Generation"
    )

    # 输入选项
    input_group = parser.add_argument_group("Input Options")
    input_group.add_argument(
        "--arxiv-id",
        help="arXiv ID of the survey paper"
    )
    input_group.add_argument(
        "--survey-id",
        help="Semantic Scholar ID of the survey"
    )
    input_group.add_argument(
        "--bibtex",
        type=Path,
        help="BibTeX file with references"
    )
    input_group.add_argument(
        "--pdf-dir",
        type=Path,
        help="Directory with existing PDFs (use with --skip-download)"
    )
    input_group.add_argument(
        "--mineru-dir",
        type=Path,
        help="Directory with MinerU output (use with --skip-parse)"
    )

    # 输出选项
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("data/m4_output"),
        help="Base output directory"
    )
    output_group.add_argument(
        "--domain",
        default="research",
        help="Domain label for generated queries"
    )

    # 控制选项
    control_group = parser.add_argument_group("Control Options")
    control_group.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip paper download step"
    )
    control_group.add_argument(
        "--skip-parse",
        action="store_true",
        help="Skip MinerU parsing step"
    )
    control_group.add_argument(
        "--max-papers",
        type=int,
        default=50,
        help="Maximum papers to download"
    )
    control_group.add_argument(
        "--num-queries",
        type=int,
        default=30,
        help="Number of M4 queries to generate"
    )
    control_group.add_argument(
        "--min-citations",
        type=int,
        default=5,
        help="Minimum citation count for papers"
    )
    control_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be done"
    )
    control_group.add_argument(
        "--relaxed",
        action="store_true",
        help="Relax M4 requirements during query generation"
    )

    # 解析选项
    parse_group = parser.add_argument_group("Parse Options")
    parse_group.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of MinerU workers"
    )
    parse_group.add_argument(
        "--devices",
        nargs="+",
        default=["cuda:0"],
        help="CUDA devices for MinerU"
    )

    # LLM选项
    llm_group = parser.add_argument_group("LLM Options")
    llm_group.add_argument(
        "--provider",
        choices=["anthropic", "openai"],
        default="anthropic",
        help="LLM provider"
    )
    llm_group.add_argument(
        "--model",
        default="claude-sonnet-4-20250514",
        help="LLM model"
    )

    args = parser.parse_args()

    # 设置目录
    output_dir = args.output_dir
    pdf_dir = args.pdf_dir or output_dir / "raw_pdfs"
    mineru_dir = args.mineru_dir or output_dir / "mineru_output"
    queries_dir = output_dir / "m4_queries"

    ensure_dir(output_dir)
    ensure_dir(pdf_dir)
    ensure_dir(mineru_dir)
    ensure_dir(queries_dir)

    # 记录配置
    config = {
        "start_time": datetime.now().isoformat(),
        "args": _make_json_serializable(vars(args)),
        "directories": {
            "output": str(output_dir),
            "pdfs": str(pdf_dir),
            "mineru": str(mineru_dir),
            "queries": str(queries_dir)
        }
    }

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           M4 Complete Workflow                               ║
║                                                              ║
║  Survey → Papers → Parse → M4 Queries                        ║
╚══════════════════════════════════════════════════════════════╝

Configuration:
  Output Directory: {output_dir}
  Max Papers: {args.max_papers}
  Num Queries: {args.num_queries}
  Skip Download: {args.skip_download}
  Skip Parse: {args.skip_parse}
  Dry Run: {args.dry_run}
""")

    if args.dry_run:
        print("[DRY RUN MODE - No actual commands will be executed]\n")
    if args.relaxed:
        print("[RELAXED MODE - M4 constraints will be loosened in query generation]\n")

    steps_completed = []

    # Step 1: Download papers
    if not args.skip_download:
        if not args.arxiv_id and not args.survey_id and not args.bibtex:
            print("Error: Must provide --arxiv-id, --survey-id, or --bibtex")
            sys.exit(1)

        cmd = ["python", "scripts/fetch_survey_references.py"]

        if args.arxiv_id:
            cmd.extend(["--arxiv-id", args.arxiv_id])
        elif args.survey_id:
            cmd.extend(["--survey-id", args.survey_id])
        elif args.bibtex:
            cmd.extend(["--bibtex", str(args.bibtex)])

        cmd.extend([
            "--output", str(pdf_dir),
            "--max-papers", str(args.max_papers),
            "--min-citations", str(args.min_citations)
        ])

        if not args.dry_run:
            if run_command(cmd, "Fetching survey references"):
                steps_completed.append("download")
            else:
                print("Warning: Download step failed, continuing...")
        else:
            print(f"Would run: {' '.join(cmd)}")
            steps_completed.append("download (dry)")
    else:
        print(f"\nSkipping download, using PDFs from: {pdf_dir}")
        steps_completed.append("download (skipped)")

    # Step 2: Parse with MinerU
    if not args.skip_parse:
        cmd = [
            "python", "scripts/parse_only.py",
            "--input", str(pdf_dir),
            "--output", str(mineru_dir),
            "--workers", str(args.workers),
            "--devices", *args.devices
        ]

        if not args.dry_run:
            if run_command(cmd, "Parsing PDFs with MinerU"):
                steps_completed.append("parse")
            else:
                print("Warning: Parse step failed, continuing...")
        else:
            print(f"Would run: {' '.join(cmd)}")
            steps_completed.append("parse (dry)")
    else:
        print(f"\nSkipping parse, using MinerU output from: {mineru_dir}")
        steps_completed.append("parse (skipped)")

    # Step 3: Generate M4 Queries
    cmd = [
        "python", "scripts/generate_m4_queries.py",
        "--input", str(mineru_dir),
        "--output", str(queries_dir / "queries"),
        "--max-docs", str(min(args.max_papers, 20)),  # M4 works best with fewer docs
        "--num-queries", str(args.num_queries),
        "--provider", args.provider,
        "--model", args.model
    ]
    if args.relaxed:
        cmd.append("--relaxed")

    if not args.dry_run:
        if run_command(cmd, "Generating M4 Queries"):
            steps_completed.append("generate")
        else:
            print("Warning: Query generation failed")
    else:
        print(f"Would run: {' '.join(cmd)}")
        steps_completed.append("generate (dry)")

    # 保存工作流配置
    config["end_time"] = datetime.now().isoformat()
    config["steps_completed"] = steps_completed
    safe_json_dump(config, output_dir / "workflow_config.json")

    # 打印总结
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    Workflow Complete                         ║
╚══════════════════════════════════════════════════════════════╝

Steps completed: {', '.join(steps_completed)}

Output locations:
  - PDFs: {pdf_dir}
  - MinerU output: {mineru_dir}
  - M4 Queries: {queries_dir}
  - Workflow config: {output_dir / 'workflow_config.json'}

Next steps:
  1. Review generated queries in {queries_dir}
  2. Optionally run with --relaxed flag for more queries
  3. Use queries for training/evaluation
""")


if __name__ == "__main__":
    main()
