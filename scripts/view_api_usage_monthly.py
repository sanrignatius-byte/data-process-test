#!/usr/bin/env python3
"""
本地 API 日志月度用量统计脚本（不修改 local_api_logger/viewer.py）。

适配日志结构：
  {log_dir}/stats/{model}/{task_or_user}_{YYYY-MM}.jsonl
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, Optional


def _extract_name_from_stem(stem: str) -> str:
    """
    从文件名 stem 提取 task/user 名称。

    支持：
    - xxx_2026-03
    - xxx_202603
    """
    match = re.search(r"_(?:\d{4}-\d{2}|\d{6})$", stem)
    if match:
        return stem[: match.start()]
    return stem


def collect_stats(log_dir: str, model: Optional[str] = None, month: Optional[str] = None) -> Dict[str, Any]:
    stats = {
        "total_calls": 0,
        "successful_calls": 0,
        "failed_calls": 0,
        "zero_output_calls": 0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "by_model": defaultdict(
            lambda: {
                "calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "zero_output_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
        ),
        "by_name": defaultdict(
            lambda: {
                "calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "zero_output_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
        ),
    }

    stats_path = Path(log_dir) / "stats"
    if not stats_path.exists():
        return stats

    for model_dir in stats_path.iterdir():
        if not model_dir.is_dir():
            continue

        current_model = model_dir.name
        if model and current_model != model:
            continue

        for stats_file in model_dir.glob("*.jsonl"):
            if month and month not in stats_file.stem:
                continue

            name = _extract_name_from_stem(stats_file.stem)

            with open(stats_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                    except Exception:
                        continue

                    prompt_tokens = entry.get("prompt_tokens", 0)
                    completion_tokens = entry.get("completion_tokens", 0)
                    total_tokens = entry.get("total_tokens", 0)
                    success = entry.get("success", True)

                    stats["total_calls"] += 1
                    stats["by_model"][current_model]["calls"] += 1
                    stats["by_name"][name]["calls"] += 1

                    if success:
                        stats["successful_calls"] += 1
                        stats["by_model"][current_model]["successful_calls"] += 1
                        stats["by_name"][name]["successful_calls"] += 1
                    else:
                        stats["failed_calls"] += 1
                        stats["by_model"][current_model]["failed_calls"] += 1
                        stats["by_name"][name]["failed_calls"] += 1

                    if completion_tokens == 0:
                        stats["zero_output_calls"] += 1
                        stats["by_model"][current_model]["zero_output_calls"] += 1
                        stats["by_name"][name]["zero_output_calls"] += 1

                    stats["total_prompt_tokens"] += prompt_tokens
                    stats["total_completion_tokens"] += completion_tokens
                    stats["total_tokens"] += total_tokens

                    stats["by_model"][current_model]["prompt_tokens"] += prompt_tokens
                    stats["by_model"][current_model]["completion_tokens"] += completion_tokens
                    stats["by_model"][current_model]["total_tokens"] += total_tokens

                    stats["by_name"][name]["prompt_tokens"] += prompt_tokens
                    stats["by_name"][name]["completion_tokens"] += completion_tokens
                    stats["by_name"][name]["total_tokens"] += total_tokens

    stats["by_model"] = dict(stats["by_model"])
    stats["by_name"] = dict(stats["by_name"])
    return stats


def print_summary(stats: Dict[str, Any], model: Optional[str], month: Optional[str]) -> None:
    print("=" * 80)
    print("API 调用统计摘要（monthly script）")
    if model:
        print(f"模型: {model}")
    if month:
        print(f"月份: {month}")
    print("=" * 80)

    total_calls = stats["total_calls"]
    success_rate = (stats["successful_calls"] / total_calls * 100) if total_calls > 0 else 0

    print(f"\n总调用次数: {stats['total_calls']:,}")
    print(f"  ✓ 成功: {stats['successful_calls']:,}")
    print(f"  ✗ 失败: {stats['failed_calls']:,}")
    print(f"  ⚠ 输出token为0: {stats['zero_output_calls']:,}")
    print(f"  成功率: {success_rate:.1f}%")
    print(f"\n总输入 Tokens: {stats['total_prompt_tokens']:,}")
    print(f"总输出 Tokens: {stats['total_completion_tokens']:,}")
    print(f"总 Tokens: {stats['total_tokens']:,}")

    if stats["by_model"]:
        print("\n按模型统计:")
        print("-" * 80)
        for model_name, model_stats in stats["by_model"].items():
            model_calls = model_stats["calls"]
            model_success_rate = (model_stats["successful_calls"] / model_calls * 100) if model_calls > 0 else 0
            print(f"\n{model_name}:")
            print(
                f"  调用次数: {model_stats['calls']:,} "
                f"(成功: {model_stats['successful_calls']:,}, 失败: {model_stats['failed_calls']:,})"
            )
            print(f"  成功率: {model_success_rate:.1f}%")
            print(f"  输出token为0: {model_stats['zero_output_calls']:,}")
            print(f"  输入 Tokens: {model_stats['prompt_tokens']:,}")
            print(f"  输出 Tokens: {model_stats['completion_tokens']:,}")
            print(f"  总 Tokens: {model_stats['total_tokens']:,}")

    if stats["by_name"]:
        print("\n按任务/用户统计:")
        print("-" * 80)
        for name, name_stats in stats["by_name"].items():
            name_calls = name_stats["calls"]
            name_success_rate = (name_stats["successful_calls"] / name_calls * 100) if name_calls > 0 else 0
            print(f"\n{name}:")
            print(
                f"  调用次数: {name_stats['calls']:,} "
                f"(成功: {name_stats['successful_calls']:,}, 失败: {name_stats['failed_calls']:,})"
            )
            print(f"  成功率: {name_success_rate:.1f}%")
            print(f"  输出token为0: {name_stats['zero_output_calls']:,}")
            print(f"  输入 Tokens: {name_stats['prompt_tokens']:,}")
            print(f"  输出 Tokens: {name_stats['completion_tokens']:,}")
            print(f"  总 Tokens: {name_stats['total_tokens']:,}")

    print("\n" + "=" * 80)


def export_csv(log_dir: str, output_file: str, model: Optional[str], month: Optional[str]) -> None:
    stats_path = Path(log_dir) / "stats"
    if not stats_path.exists():
        print("没有找到统计数据")
        return

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ["时间戳", "模型", "任务或用户", "输入Tokens", "输出Tokens", "总Tokens", "耗时(ms)", "成功"]
        )

        for model_dir in stats_path.iterdir():
            if not model_dir.is_dir():
                continue

            current_model = model_dir.name
            if model and current_model != model:
                continue

            for stats_file in model_dir.glob("*.jsonl"):
                if month and month not in stats_file.stem:
                    continue

                name = _extract_name_from_stem(stats_file.stem)

                with open(stats_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            entry = json.loads(line.strip())
                        except Exception:
                            continue
                        writer.writerow(
                            [
                                entry.get("timestamp", ""),
                                current_model,
                                name,
                                entry.get("prompt_tokens", 0),
                                entry.get("completion_tokens", 0),
                                entry.get("total_tokens", 0),
                                entry.get("duration_ms", ""),
                                "是" if entry.get("success", True) else "否",
                            ]
                        )

    print(f"数据已导出到: {output_file}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="本地 API 日志月度用量统计")
    parser.add_argument("--log-dir", default="api_logs", help="日志根目录，默认 api_logs")
    parser.add_argument("--month", default=None, help="按月份过滤，例如 2026-03")
    parser.add_argument("--model", default=None, help="按模型过滤，例如 claude-sonnet-4-20250514")
    parser.add_argument("--export-csv", default=None, help="导出筛选结果到 CSV 文件路径")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stats = collect_stats(log_dir=args.log_dir, model=args.model, month=args.month)
    print_summary(stats=stats, model=args.model, month=args.month)
    if args.export_csv:
        export_csv(log_dir=args.log_dir, output_file=args.export_csv, model=args.model, month=args.month)


if __name__ == "__main__":
    main()

