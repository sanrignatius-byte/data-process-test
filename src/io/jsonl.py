"""JSONL 读写的统一入口 —— 带 UTF-8 + flush 语义，消除散落各处的副本。

Phase A.1 血泪教训：任何跑 >5 分钟的脚本，输出文件必须每条 flush，否则进程中断
= 全部白跑。本模块的 :func:`write_jsonl` / :func:`append_jsonl` 默认 flush 每条。

向后兼容：底层委托到 :mod:`src.utils.file_utils`，现有代码无需同步迁移。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Union

from src.utils.file_utils import ensure_dir, read_jsonl as _read_jsonl

PathLike = Union[str, Path]


def load_jsonl(filepath: PathLike) -> List[Dict[str, Any]]:
    """一次性读入 JSONL 文件为 list —— 小到中等规模文件用这个。

    空文件或不存在返回 ``[]``；解析失败的行会被静默跳过（与现有
    ``file_utils.read_jsonl`` 行为一致，便于旧脚本平滑切换）。
    """
    return _read_jsonl(filepath)


def iter_jsonl(filepath: PathLike) -> Iterator[Dict[str, Any]]:
    """流式逐行读 JSONL —— 大文件用这个省内存。

    解析失败的行会被静默跳过。UTF-8-sig 兼容以吃掉 Windows BOM。
    """
    p = Path(filepath)
    if not p.exists():
        return
    with open(p, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def write_jsonl(
    data: Iterable[Dict[str, Any]],
    filepath: PathLike,
    *,
    append: bool = False,
    flush_every: int = 1,
) -> int:
    """写 JSONL，默认覆盖写，每条 flush 一次（长作业续跑必须）。

    Args:
        data:         可迭代的 dict 序列。
        filepath:     目标路径；父目录不存在会自动创建。
        append:       ``True`` 时追加到现有文件（默认覆盖写）。
        flush_every:  每写多少条 flush 一次到磁盘。默认 1 = 每条都刷盘，
                      牺牲一点性能换崩溃安全。大批量可以调到 100。

    Returns:
        实际写入的行数。
    """
    p = Path(filepath)
    ensure_dir(p.parent)
    mode = "a" if append else "w"
    count = 0
    with open(p, mode, encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
            if flush_every > 0 and count % flush_every == 0:
                f.flush()
    return count


def append_jsonl(
    item_or_items: Union[Dict[str, Any], Iterable[Dict[str, Any]]],
    filepath: PathLike,
) -> int:
    """追加单条 dict 或一批 dict 到 JSONL —— 立即 flush。

    对于 resume / skip-done 场景，推荐每条处理完就 append_jsonl 一次，
    这样进程中断后 ``--resume`` 能准确跳过已完成的条目。
    """
    if isinstance(item_or_items, dict):
        items: Iterable[Dict[str, Any]] = [item_or_items]
    else:
        items = item_or_items
    return write_jsonl(items, filepath, append=True, flush_every=1)


__all__ = ["load_jsonl", "iter_jsonl", "write_jsonl", "append_jsonl"]
