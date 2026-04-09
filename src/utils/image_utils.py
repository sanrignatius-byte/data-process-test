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
    "/projects/_hdd/myyyx1/data-process-test/",
    "/projects/myyyx1/data-process-test/",
    "/cluster/scratch/myyyx1/data-process-test/",
    "/home/runner/work/data-process-test/data-process-test/",
]

_MINERU_PREFIX = "data/mineru_output/"
_MINERU_00RAW_PREFIX = "data/00_raw/mineru_output/"


# ── Internal helpers ──────────────────────────────────────────────────────────

def _try_00raw_variant(relative: str) -> Optional[Path]:
    """If *relative* contains ``data/mineru_output/``, try ``data/00_raw/mineru_output/``."""
    if _MINERU_PREFIX not in relative:
        return None
    alt = relative.replace(_MINERU_PREFIX, _MINERU_00RAW_PREFIX, 1)
    candidate = PROJECT_ROOT / alt
    return candidate if candidate.exists() else None


def _resolve_core(normed: str) -> Optional[Path]:
    """Shared resolution logic used by both public and fallback resolvers.

    Strategies (in order):
      0. Relative ``data/mineru_output/...`` → ``data/00_raw/mineru_output/...``
      1. Known prefix stripping  (+00_raw variant)
      2. ``/data/mineru_output/`` or ``/data/00_raw/mineru_output/`` suffix extraction
      3. Generic ``/data/`` re-root  (+00_raw variant)
    """
    # Strategy 0: relative path starting with data/mineru_output/
    if normed.startswith(_MINERU_PREFIX):
        suffix = normed[len(_MINERU_PREFIX):]
        candidate = PROJECT_ROOT / _MINERU_00RAW_PREFIX / suffix
        if candidate.exists():
            return candidate

    # Strategy 1: known prefix stripping
    for prefix in _KNOWN_PREFIXES:
        if normed.startswith(prefix):
            relative = normed[len(prefix):]
            candidate = PROJECT_ROOT / relative
            if candidate.exists():
                return candidate
            alt = _try_00raw_variant(relative)
            if alt:
                return alt

    # Strategy 2: extract suffix after /data/mineru_output/ (or /data/00_raw/...)
    for marker in ("/data/00_raw/mineru_output/", "/data/mineru_output/"):
        parts = normed.split(marker)
        if len(parts) == 2:
            for base in (_MINERU_00RAW_PREFIX, _MINERU_PREFIX):
                candidate = PROJECT_ROOT / base / parts[1]
                if candidate.exists():
                    return candidate

    # Strategy 3: generic – find '/data/' and re-root
    idx = normed.find("/data/")
    if idx >= 0:
        relative = normed[idx + 1:]  # keep 'data/...'
        candidate = PROJECT_ROOT / relative
        if candidate.exists():
            return candidate
        alt = _try_00raw_variant(relative)
        if alt:
            return alt

    return None


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
    result = _resolve_core(normed)
    if result:
        return result

    print(f"  [resolve_image_path] MISS: {raw_path!r}", file=sys.stderr)
    return None


def _fallback_image_path(raw_path: str) -> Optional[Path]:
    """Resolve cross-environment paths via known prefix stripping.

    Called as a fallback by :func:`encode_image` when the direct path fails.
    Delegates to the same core resolver as :func:`resolve_image_path`.
    """
    if not raw_path:
        return None
    normed = raw_path.replace("\\", "/")
    return _resolve_core(normed)


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
