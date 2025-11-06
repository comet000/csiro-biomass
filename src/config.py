# src/config.py
# =============================================================
# Minimal, portable config for local (macOS) and Kaggle
# - No recursion or project discovery games
# - Works when notebooks live in ./notebooks and package in ./src
# - Kaggle data dir is auto-detected; local uses ./input_local by default
# =============================================================
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Dict

# -------------------------------------------------------------
# Environment detection
# -------------------------------------------------------------
def _is_kaggle() -> bool:
    if os.path.isdir("/kaggle"):
        return True
    # Some Kaggle environments set kernel-related env vars
    return any(k for k in os.environ if k.startswith("KAGGLE"))

# -------------------------------------------------------------
# Roots and dirs
# -------------------------------------------------------------
# This file is in <project_root>/src/config.py → project_root = parents[1]
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# Allow override via env var (absolute or relative)
_env_data = os.getenv("CSIRO_DATA_DIR", "").strip()
DEFAULT_LOCAL_DATA_DIR = PROJECT_ROOT / "input_local"

def _find_kaggle_data_dir() -> Optional[Path]:
    """
    Search /kaggle/input/* for a folder that has train.csv and test.csv.
    This avoids hard-coding the competition folder name.
    """
    base = Path("/kaggle/input")
    if not base.exists():
        return None
    candidates = []
    for p in base.iterdir():
        if not p.is_dir():
            continue
        train_csv = p / "train.csv"
        test_csv = p / "test.csv"
        if train_csv.exists() and test_csv.exists():
            candidates.append(p)
    # Prefer exact match folder names if multiple are present
    # Fallback: pick the first deterministically (sorted)
    return sorted(candidates)[0] if candidates else None

def _resolve_data_dir() -> Path:
    if _env_data:
        p = Path(_env_data).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"CSIRO_DATA_DIR points to missing path: {p}")
        return p
    if _is_kaggle():
        kd = _find_kaggle_data_dir()
        if kd is None:
            # Still allow mounting a custom dataset as /kaggle/input_local
            alt = Path("/kaggle/input_local")
            if alt.exists():
                return alt
            raise FileNotFoundError(
                "Could not auto-detect the Kaggle dataset folder in /kaggle/input. "
                "Mount the competition data or set $CSIRO_DATA_DIR."
            )
        return kd
    # Local default
    return DEFAULT_LOCAL_DATA_DIR

DATA_DIR: Path = _resolve_data_dir()

# Image folders (train/ and test/ under data dir)
TRAIN_IMG_DIR: Path = DATA_DIR / "train"
TEST_IMG_DIR: Path  = DATA_DIR / "test"

# CSVs
TRAIN_CSV: Path = DATA_DIR / "train.csv"
TEST_CSV: Path  = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_CSV: Path = (DATA_DIR / "sample_submission.csv")

# Output dir (local → ./output; Kaggle → /kaggle/working/output)
OUTPUT_ROOT: Path = (Path("/kaggle/working/output") if _is_kaggle()
                     else PROJECT_ROOT / "output")
OUTPUT_DIRS: Dict[str, Path] = {
    "submissions":  OUTPUT_ROOT / "submissions",
    "checkpoints":  OUTPUT_ROOT / "checkpoints",
    "logs":         OUTPUT_ROOT / "logs",
}

def ensure_output_dirs() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for p in OUTPUT_DIRS.values():
        p.mkdir(parents=True, exist_ok=True)

def verify_layout(strict: bool = True) -> None:
    """
    Validate that required files/dirs exist. Raises if strict.
    """
    missing = []
    for path in [TRAIN_CSV, TEST_CSV, TRAIN_IMG_DIR, TEST_IMG_DIR]:
        if not path.exists():
            missing.append(str(path))
    if missing and strict:
        raise FileNotFoundError(
            "Required data artifacts are missing:\n- " + "\n- ".join(missing) +
            "\nSet $CSIRO_DATA_DIR or fix the dataset mount."
        )

__all__ = [
    "PROJECT_ROOT", "DATA_DIR",
    "TRAIN_IMG_DIR", "TEST_IMG_DIR",
    "TRAIN_CSV", "TEST_CSV", "SAMPLE_SUBMISSION_CSV",
    "OUTPUT_ROOT", "OUTPUT_DIRS",
    "ensure_output_dirs", "verify_layout",
]