#!/usr/bin/env python3
"""
MinerU Output Diagnostic Tool

检查MinerU解析输出的实际结构，帮助诊断解析问题。

Usage:
    python scripts/diagnose_mineru_output.py /path/to/mineru_output
    python scripts/diagnose_mineru_output.py /path/to/mineru_output --doc-id paper_001
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict


def scan_directory_structure(output_dir: Path, max_depth: int = 4) -> dict:
    """递归扫描目录结构"""
    structure = {
        "path": str(output_dir),
        "files": [],
        "dirs": [],
        "file_count": 0,
        "dir_count": 0
    }

    if not output_dir.exists():
        structure["error"] = "目录不存在"
        return structure

    for item in sorted(output_dir.iterdir()):
        if item.is_file():
            file_info = {
                "name": item.name,
                "size": item.stat().st_size,
                "ext": item.suffix
            }
            structure["files"].append(file_info)
            structure["file_count"] += 1
        elif item.is_dir() and max_depth > 0:
            sub_structure = scan_directory_structure(item, max_depth - 1)
            structure["dirs"].append(sub_structure)
            structure["dir_count"] += 1

    return structure


def find_all_files(output_dir: Path, pattern: str = "*") -> list:
    """查找所有匹配的文件"""
    return list(output_dir.rglob(pattern))


def analyze_mineru_output(output_dir: Path) -> dict:
    """分析MinerU输出目录"""
    analysis = {
        "output_dir": str(output_dir),
        "exists": output_dir.exists(),
        "documents": [],
        "summary": {
            "total_docs": 0,
            "successful_docs": 0,
            "failed_docs": 0,
            "file_types_found": defaultdict(int)
        }
    }

    if not output_dir.exists():
        print(f"❌ 错误: 目录不存在 - {output_dir}")
        return analysis

    # 查找所有可能的文档输出目录
    doc_dirs = []

    # 方式1: 直接子目录
    for item in output_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            doc_dirs.append(item)

    # 如果没有子目录，可能输出直接在当前目录
    if not doc_dirs:
        doc_dirs = [output_dir]

    print(f"\n📁 扫描目录: {output_dir}")
    print(f"   找到 {len(doc_dirs)} 个潜在文档目录\n")

    for doc_dir in doc_dirs:
        doc_analysis = analyze_single_document(doc_dir)
        analysis["documents"].append(doc_analysis)

        if doc_analysis["has_content"]:
            analysis["summary"]["successful_docs"] += 1
        else:
            analysis["summary"]["failed_docs"] += 1
        analysis["summary"]["total_docs"] += 1

        for ext, count in doc_analysis["file_types"].items():
            analysis["summary"]["file_types_found"][ext] += count

    return analysis


def analyze_single_document(doc_dir: Path) -> dict:
    """分析单个文档的输出"""
    doc_analysis = {
        "doc_dir": str(doc_dir),
        "doc_name": doc_dir.name,
        "has_content": False,
        "structure": {},
        "file_types": defaultdict(int),
        "important_files": {
            "markdown": [],
            "json": [],
            "images": [],
            "formula": []
        },
        "issues": []
    }

    # 检查 auto 子目录 (MinerU常见结构)
    auto_dir = doc_dir / "auto"
    content_dir = auto_dir if auto_dir.exists() else doc_dir

    # 查找所有文件
    all_files = list(doc_dir.rglob("*"))

    for f in all_files:
        if f.is_file():
            ext = f.suffix.lower()
            doc_analysis["file_types"][ext] += 1

            # 分类重要文件
            if ext == ".md":
                doc_analysis["important_files"]["markdown"].append(str(f.relative_to(doc_dir)))
            elif ext == ".json":
                doc_analysis["important_files"]["json"].append(str(f.relative_to(doc_dir)))
            elif ext in [".png", ".jpg", ".jpeg", ".gif"]:
                doc_analysis["important_files"]["images"].append(str(f.relative_to(doc_dir)))

    # 检查关键文件
    key_files = {
        # MinerU新版输出格式
        "structure.json": ["structure.json", "auto/structure.json"],
        "formula.md": ["formula.md", "auto/formula.md"],
        # MinerU旧版输出格式
        "content_list.json": ["content_list.json", "auto/content_list.json"],
        "middle.json": ["middle.json", "auto/middle.json"],
        # 通用
        "content.md": ["content.md", "auto/content.md", f"{doc_dir.name}.md", f"auto/{doc_dir.name}.md"],
    }

    doc_analysis["structure"]["key_files"] = {}
    for key_name, possible_paths in key_files.items():
        found = False
        for p in possible_paths:
            full_path = doc_dir / p
            if full_path.exists():
                doc_analysis["structure"]["key_files"][key_name] = str(p)
                found = True
                break
        if not found:
            doc_analysis["structure"]["key_files"][key_name] = None

    # 检查是否有有效内容
    has_md = len(doc_analysis["important_files"]["markdown"]) > 0
    has_json = len(doc_analysis["important_files"]["json"]) > 0
    has_images = len(doc_analysis["important_files"]["images"]) > 0

    doc_analysis["has_content"] = has_md or has_json

    # 记录问题
    if not has_md:
        doc_analysis["issues"].append("缺少Markdown文件")
    if not has_json:
        doc_analysis["issues"].append("缺少JSON结构文件")
    if not has_images:
        doc_analysis["issues"].append("没有提取到图片")

    # 检查是否使用新版格式
    if doc_analysis["structure"]["key_files"].get("structure.json"):
        doc_analysis["mineru_version"] = "新版 (structure.json)"
    elif doc_analysis["structure"]["key_files"].get("content_list.json"):
        doc_analysis["mineru_version"] = "旧版 (content_list.json)"
    else:
        doc_analysis["mineru_version"] = "未知"

    return doc_analysis


def inspect_json_file(json_path: Path) -> dict:
    """深入检查JSON文件结构"""
    inspection = {
        "path": str(json_path),
        "size": json_path.stat().st_size,
        "readable": False,
        "structure": None,
        "sample": None
    }

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        inspection["readable"] = True

        if isinstance(data, dict):
            inspection["structure"] = {k: type(v).__name__ for k in list(data.keys())[:20]}
            inspection["top_level_keys"] = list(data.keys())
        elif isinstance(data, list):
            inspection["structure"] = f"list with {len(data)} items"
            if data:
                if isinstance(data[0], dict):
                    inspection["item_keys"] = list(data[0].keys())[:10]
                inspection["sample"] = str(data[0])[:500]
    except Exception as e:
        inspection["error"] = str(e)

    return inspection


def print_analysis(analysis: dict):
    """打印分析结果"""
    print("\n" + "="*70)
    print("📊 MinerU 输出分析报告")
    print("="*70)

    summary = analysis["summary"]
    print(f"\n📁 输出目录: {analysis['output_dir']}")
    print(f"📄 总文档数: {summary['total_docs']}")
    print(f"✅ 成功解析: {summary['successful_docs']}")
    print(f"❌ 解析失败: {summary['failed_docs']}")

    print(f"\n📑 文件类型统计:")
    for ext, count in sorted(summary['file_types_found'].items(), key=lambda x: -x[1]):
        print(f"   {ext or '(无扩展名)'}: {count}")

    # 显示前几个文档的详细信息
    print(f"\n📋 文档详情 (显示前5个):")
    print("-"*70)

    for doc in analysis["documents"][:5]:
        status = "✅" if doc["has_content"] else "❌"
        print(f"\n{status} {doc['doc_name']}")
        print(f"   版本: {doc.get('mineru_version', '未知')}")
        print(f"   Markdown文件: {len(doc['important_files']['markdown'])}")
        print(f"   JSON文件: {len(doc['important_files']['json'])}")
        print(f"   图片: {len(doc['important_files']['images'])}")

        if doc["structure"]["key_files"]:
            print(f"   关键文件:")
            for key, path in doc["structure"]["key_files"].items():
                if path:
                    print(f"      ✓ {key}: {path}")
                else:
                    print(f"      ✗ {key}: 未找到")

        if doc["issues"]:
            print(f"   ⚠️  问题: {', '.join(doc['issues'])}")

    if len(analysis["documents"]) > 5:
        print(f"\n... 还有 {len(analysis['documents']) - 5} 个文档未显示")


def suggest_fixes(analysis: dict):
    """根据分析结果建议修复方案"""
    print("\n" + "="*70)
    print("💡 建议修复方案")
    print("="*70)

    issues = []

    # 检查是否有structure.json但代码不支持
    has_new_format = False
    has_old_format = False

    for doc in analysis["documents"]:
        if doc["structure"]["key_files"].get("structure.json"):
            has_new_format = True
        if doc["structure"]["key_files"].get("content_list.json"):
            has_old_format = True

    if has_new_format and not has_old_format:
        issues.append({
            "问题": "MinerU使用新版输出格式 (structure.json)，但代码只支持旧版",
            "建议": "需要更新 mineru_parser.py 来支持 structure.json 格式",
            "priority": "高"
        })

    if analysis["summary"]["failed_docs"] > 0:
        issues.append({
            "问题": f"有 {analysis['summary']['failed_docs']} 个文档解析失败",
            "建议": "检查MinerU日志，可能是PDF损坏或GPU内存不足",
            "priority": "高"
        })

    # 检查图片
    total_images = analysis["summary"]["file_types_found"].get(".png", 0) + \
                   analysis["summary"]["file_types_found"].get(".jpg", 0)
    if total_images == 0:
        issues.append({
            "问题": "没有提取到任何图片",
            "建议": "检查MinerU配置，确保启用了图片提取",
            "priority": "中"
        })

    if not issues:
        print("\n✅ 没有发现明显问题!")
    else:
        for i, issue in enumerate(issues, 1):
            print(f"\n{i}. [{issue['priority']}优先级] {issue['问题']}")
            print(f"   💡 {issue['建议']}")

    return issues


def main():
    parser = argparse.ArgumentParser(description="MinerU输出诊断工具")
    parser.add_argument("output_dir", help="MinerU输出目录路径")
    parser.add_argument("--doc-id", help="只检查特定文档")
    parser.add_argument("--inspect-json", help="深入检查特定JSON文件")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.inspect_json:
        json_path = Path(args.inspect_json)
        if not json_path.exists():
            json_path = output_dir / args.inspect_json

        if json_path.exists():
            inspection = inspect_json_file(json_path)
            print(f"\n📋 JSON文件检查: {inspection['path']}")
            print(f"   大小: {inspection['size']} bytes")
            print(f"   可读: {inspection['readable']}")
            if inspection.get("top_level_keys"):
                print(f"   顶层键: {inspection['top_level_keys']}")
            if inspection.get("structure"):
                print(f"   结构: {inspection['structure']}")
            if inspection.get("item_keys"):
                print(f"   项目键: {inspection['item_keys']}")
            if inspection.get("sample"):
                print(f"   示例: {inspection['sample'][:200]}...")
        else:
            print(f"❌ JSON文件不存在: {json_path}")
        return

    if args.doc_id:
        doc_path = output_dir / args.doc_id
        if not doc_path.exists():
            # 尝试查找
            matches = list(output_dir.glob(f"*{args.doc_id}*"))
            if matches:
                doc_path = matches[0]

        if doc_path.exists():
            doc_analysis = analyze_single_document(doc_path)
            print(f"\n📄 文档分析: {args.doc_id}")
            print(json.dumps(doc_analysis, indent=2, ensure_ascii=False, default=str))
        else:
            print(f"❌ 找不到文档: {args.doc_id}")
        return

    # 完整分析
    analysis = analyze_mineru_output(output_dir)
    print_analysis(analysis)
    suggest_fixes(analysis)

    # 保存分析结果
    report_path = output_dir / "diagnosis_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n📝 详细报告已保存到: {report_path}")


if __name__ == "__main__":
    main()
