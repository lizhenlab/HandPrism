"""Resolve repository-local CLI imports without relying on shell PYTHONPATH."""

from pathlib import Path
import sys


def use_workspace() -> None:
    """Prefer this script's checkout; never infer an import root from the cwd."""
    root = Path(__file__).resolve().parents[1]
    paths = [str(root / "src"), str(root)]
    sys.path[:] = paths + [value for value in sys.path if value not in paths]
