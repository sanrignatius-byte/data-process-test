"""图像处理工具 —— 路径解析 + base64 编码，跨环境通吃。

之前 3 个脚本各写各的 encode_image / load_image_b64 / resolve_image_path，
现在统一收到这里。

最值得说的是 resolve_image_path()——维护了一个 _KNOWN_PREFIXES 列表，
处理集群 / 本地 / CI 三种环境下图像路径不一致的问题。
四级策略：直接路径 → 已知前缀剥离 → /data/mineru_output/ 后缀提取 → 通用 /data/ 重定向。
"""

from __future__ import annotations

import base64
import mimetypes
import sys
from pathlib import Path
from typing import Optional, Tuple

# Maximum image size (bytes) we send to the LLM.
MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5 MB

# Project root for resolving relative paths.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Known absolute-path prefixes from various cluster / CI environments.
# When a stored path starts with one of these but doesn't exist locally,
# we strip the prefix and re-root under PROJECT_ROOT.
_KNOWN_PREFIXES = [
    "/projects/myyyx1/data-process-test/",
    "/cluster/scratch/myyyx1/data-process-test/",
    "/home/runner/work/data-process-test/data-process-test/",
]


# ── Path resolution ───────────────────────────────────────────────────────────

def resolve_image_path(raw_path: str) -> Optional[Path]:
    """多策略图像路径解析 —— 集群/本地/CI 环境随便切换不报错。

    试四种策略：
      1. 直接路径（绝对 or 相对于项目根）
      2. 已知前缀剥离（/projects/myyyx1/... → 本地 data/...）
      3. /data/mineru_output/ 后缀提取
      4. 通用 /data/ 重定向
    全试完还找不到就返回 None 并打个 stderr 提示。
    """
    if not raw_path:
        return None

    p = Path(raw_path)
    if not p.is_absolute():
        p = PROJECT_ROOT / raw_path
    if p.exists():
        return p

    normed = raw_path.replace("\\", "/")

    # Strategy 1: known prefix stripping
    for prefix in _KNOWN_PREFIXES:
        if normed.startswith(prefix):
            relative = normed[len(prefix):]
            candidate = PROJECT_ROOT / relative
            if candidate.exists():
                return candidate

    # Strategy 2: extract suffix after '/data/mineru_output/'
    parts = normed.split("/data/mineru_output/")
    if len(parts) == 2:
        candidate = PROJECT_ROOT / "data" / "mineru_output" / parts[1]
        if candidate.exists():
            return candidate

    # Strategy 3: generic – find '/data/' and re-root everything after it
    idx = normed.find("/data/")
    if idx >= 0:
        relative = normed[idx + 1:]  # keep 'data/...'
        candidate = PROJECT_ROOT / relative
        if candidate.exists():
            return candidate

    print(f"  [resolve_image_path] MISS: {raw_path!r}", file=sys.stderr)
    return None


def _fallback_image_path(raw_path: str) -> Optional[Path]:
    """Resolve cross-environment paths via known prefix stripping.

    Called as a fallback by :func:`encode_image` when the direct path fails.
    Tries ``/data/mineru_output/`` split as well.
    """
    if not raw_path:
        return None
    normed = raw_path.replace("\\", "/")
    for prefix in _KNOWN_PREFIXES:
        if normed.startswith(prefix):
            relative = normed[len(prefix):]
            candidate = PROJECT_ROOT / relative
            if candidate.exists():
                return candidate
    # /data/mineru_output/ split
    parts = normed.split("/data/mineru_output/")
    if len(parts) == 2:
        candidate = PROJECT_ROOT / "data" / "mineru_output" / parts[1]
        if candidate.exists():
            return candidate
    # generic /data/ re-root
    idx = normed.find("/data/")
    if idx >= 0:
        candidate = PROJECT_ROOT / normed[idx + 1:]
        if candidate.exists():
            return candidate
    return None


# ── Image encoding ────────────────────────────────────────────────────────────

def encode_image(path: Optional[str]) -> Optional[Tuple[str, str]]:
    """读图 → base64 编码，返回 (base64字符串, mime_type)。

    文件 <500B 跳过（大概率是坏图/占位符）。
    找不到路径就走 _fallback_image_path 兜底。
    """
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / path
    if not p.exists():
        fallback = _fallback_image_path(path)
        if fallback:
            p = fallback
    if not p.exists() or p.stat().st_size < 500:
        return None
    ext = p.suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(
        ext, "image/jpeg",
    )
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8"), mime


def load_image_b64(image_path: str) -> Optional[Tuple[str, str]]:
    """加载图片为 base64 —— 比 encode_image 多一层大小限制(5MB)和 mime 自动推断。"""
    resolved = resolve_image_path(image_path)
    if resolved is None:
        return None

    mime, _ = mimetypes.guess_type(str(resolved))
    if not mime:
        mime = "image/jpeg"

    try:
        with open(resolved, "rb") as f:
            data = f.read()
        if len(data) > MAX_IMAGE_BYTES:
            return None
        return base64.b64encode(data).decode("ascii"), mime
    except (IOError, OSError):
        return None
