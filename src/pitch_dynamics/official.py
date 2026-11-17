"""The only access to the organisers' evaluation kit in official/."""

import importlib.util
import subprocess
import sys
from functools import cache
from pathlib import Path
from types import ModuleType

from pitch_dynamics.paths import OFFICIAL


@cache
def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metrics() -> ModuleType:
    """The official possession metrics module."""
    return _load("possession_metrics", OFFICIAL / "evaluation" / "possession_metrics.py")


def games() -> ModuleType:
    """The official per-game constants module."""
    return _load("games", OFFICIAL / "games.py")


def validate(csv_path: Path) -> None:
    """Run the official submission validator; raises CalledProcessError if the file is invalid."""
    script = OFFICIAL / "evaluation" / "validate_submission.py"
    subprocess.run([sys.executable, str(script), str(csv_path)], check=True)
