"""Filesystem locations used across the package."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = ROOT / "official"
DATA = Path(os.environ.get("MTAH_DATA_DIR") or ROOT / "data")
CACHE = Path(os.environ.get("PITCH_DYNAMICS_CACHE") or ROOT / ".cache")
ARTIFACTS = ROOT / "artifacts"
