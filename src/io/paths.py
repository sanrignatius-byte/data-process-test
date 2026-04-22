"""项目路径常量 + 解析器 —— 消除 40+ 处 ``Path(__file__).resolve().parent.parent``。

CLAUDE.md 铁律 4：代码中用 :data:`PROJECT_ROOT` 动态获取根目录，**禁止**硬编码
绝对路径（如 ``/projects/_hdd/myyyx1/...``）。所有数据路径都是相对于项目根的。

本模块用 ``Path(__file__).resolve().parents[2]`` —— ``src/io/paths.py`` → 回退
两级到项目根。如果文件移动到别的子目录，parents 的层数要同步更新。
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

# src/io/paths.py → src/io/ → src/ → project root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
CONFIG_DIR: Path = PROJECT_ROOT / "config"


def resolve_data_path(rel: Union[str, Path]) -> Path:
    """把相对于 ``data/`` 的路径字符串拼成绝对路径。

    Example::

        resolve_data_path("03_queries/l3.jsonl")
        # → <project_root>/data/03_queries/l3.jsonl
    """
    return DATA_DIR / Path(rel)


__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "LOGS_DIR",
    "CONFIG_DIR",
    "resolve_data_path",
]
