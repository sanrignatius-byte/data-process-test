"""File operation utilities."""

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import shutil


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if not."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_json_dump(data: Any, filepath: Union[str, Path], indent: int = 2) -> bool:
    """
    Safely dump JSON to file with atomic write.

    Args:
        data: Data to serialize
        filepath: Target file path
        indent: JSON indentation

    Returns:
        True if successful, False otherwise
    """
    filepath = Path(filepath)
    temp_path = filepath.with_suffix('.tmp')

    try:
        ensure_dir(filepath.parent)
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        temp_path.rename(filepath)
        return True
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise e


def safe_json_load(filepath: Union[str, Path], default: Any = None) -> Any:
    """
    Safely load JSON from file.

    Args:
        filepath: Source file path
        default: Default value if file doesn't exist or is invalid

    Returns:
        Loaded data or default value
    """
    filepath = Path(filepath)

    if not filepath.exists():
        return default

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return default


def write_jsonl(data: List[Dict], filepath: Union[str, Path], append: bool = False) -> int:
    """
    Write data to JSONL file.

    Args:
        data: List of dictionaries to write
        filepath: Target file path
        append: Whether to append to existing file

    Returns:
        Number of lines written
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)

    mode = 'a' if append else 'w'
    count = 0

    with open(filepath, mode, encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            count += 1

    return count


def read_jsonl(filepath: Union[str, Path]) -> List[Dict]:
    """
    Read JSONL file.

    Args:
        filepath: Source file path

    Returns:
        List of dictionaries
    """
    filepath = Path(filepath)
    data = []

    if not filepath.exists():
        return data

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    return data


def compute_file_hash(filepath: Union[str, Path], algorithm: str = 'md5') -> str:
    """Compute hash of a file."""
    filepath = Path(filepath)
    hash_func = hashlib.new(algorithm)

    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hash_func.update(chunk)

    return hash_func.hexdigest()


def compute_content_hash(content: str, length: int = 8) -> str:
    """Compute short hash of content string."""
    return hashlib.md5(content.encode('utf-8')).hexdigest()[:length]


def copy_with_structure(src: Path, dst: Path, relative_to: Path) -> Path:
    """Copy file maintaining directory structure."""
    relative_path = src.relative_to(relative_to)
    target_path = dst / relative_path
    ensure_dir(target_path.parent)
    shutil.copy2(src, target_path)
    return target_path


def get_pdf_files(directory: Union[str, Path], recursive: bool = True) -> List[Path]:
    """Get all PDF files in directory."""
    directory = Path(directory)
    pattern = "**/*.pdf" if recursive else "*.pdf"
    return sorted(directory.glob(pattern))


def cleanup_temp_files(directory: Union[str, Path], patterns: List[str] = None) -> int:
    """Clean up temporary files."""
    directory = Path(directory)
    if patterns is None:
        patterns = ["*.tmp", "*.temp", "*.partial"]

    count = 0
    for pattern in patterns:
        for file in directory.rglob(pattern):
            try:
                file.unlink()
                count += 1
            except Exception:
                pass

    return count
