"""统一 I/O 基元层 —— 消除 24+ 处重复的 load_jsonl / PROJECT_ROOT 散乱代码。

这一层是"解耦计划 2026-04-22" Phase 1 的产物。所有被 ≥2 个脚本 import 的 I/O
工具都应该集中在这里，而不是各写各的。

模块划分：
  - :mod:`src.io.jsonl`  — JSONL 读写（load/write/iter/append），UTF-8 + flush 语义
  - :mod:`src.io.paths`  — PROJECT_ROOT / DATA_DIR / LOGS_DIR / CONFIG_DIR 常量 +
                           ``resolve_data_path()`` helper

向后兼容注意：现有的 ``src.utils.file_utils`` 仍然保留且可用，本模块的
``load_jsonl`` / ``write_jsonl`` 内部委托到它，这样旧脚本不需要立即迁移。
后续按阶段把调用方切过来就行，不要一次全改。

典型用法::

    from src.io import load_jsonl, write_jsonl, PROJECT_ROOT, resolve_data_path

    rows = load_jsonl(resolve_data_path("03_queries/l3.jsonl"))
    write_jsonl(rows, PROJECT_ROOT / "data" / "out.jsonl")
"""

from src.io.jsonl import load_jsonl, write_jsonl, iter_jsonl, append_jsonl
from src.io.paths import (
    PROJECT_ROOT,
    DATA_DIR,
    LOGS_DIR,
    CONFIG_DIR,
    resolve_data_path,
)

__all__ = [
    "load_jsonl",
    "write_jsonl",
    "iter_jsonl",
    "append_jsonl",
    "PROJECT_ROOT",
    "DATA_DIR",
    "LOGS_DIR",
    "CONFIG_DIR",
    "resolve_data_path",
]
