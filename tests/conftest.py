"""Shared pytest fixtures for the test suite."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is importable without pip install -e .
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
