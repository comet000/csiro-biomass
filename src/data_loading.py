# =============================================================
# 📦 Data Loading Utilities — Production-Safe Version
# =============================================================
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import pandas as pd
from PIL import Image, UnidentifiedImageError
import numpy as np

from .config import (
    TRAIN_CSV, TEST_CSV,
    TRAIN_IMG_DIR, TEST_IMG_DIR,
)

# -------------------------------------------------------------
# Column inference
# -------------------------------------------------------------
_IMG_HINTS = ("image", "img", "filename", "file_name", "filepath", "path")
_TGT_HINTS = ("biomass", "target", "label", "y")

def _infer_col(df: pd.DataFrame, hints: Tuple[str, ...]) -> Optional[str]:
    for c in df.columns:
        if c in hints:
            return c
    lower = {c.lower(): c for c in df.columns}
    for key, orig in lower.items():
        if any(h in key for h in hints):
            return orig
    return None

def infer_image_col(df: pd.DataFrame) -> str:
    col = _infer_col(df, _IMG_HINTS)
    if col is None:
        raise KeyError(f"Could not infer image filename column from {list(df.columns)}")
    return col

def infer_target_col(df: pd.DataFrame) -> Optional[str]:
    return _infer_col(df, _TGT_HINTS)

# -------------------------------------------------------------
# CSV readers
# -------------------------------------------------------------
def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path, low_memory=False)

# -------------------------------------------------------------
# Helper: normalize relative paths
# -------------------------------------------------------------
def _normalize_relpath(p: str) -> str:
    """Remove redundant train/ or test/ prefixes from CSV paths."""
    p = str(p).strip()
    if p.startswith("train/"):
        return p[len("train/"):]
    if p.startswith("test/"):
        return p[len("test/"):]
    return p

# -------------------------------------------------------------
# Train/Test loaders
# -------------------------------------------------------------
def load_train_data() -> Tuple[pd.DataFrame, str, str]:
    df = read_csv(TRAIN_CSV)
    img_col = infer_image_col(df)
    tgt_col = infer_target_col(df)
    if tgt_col is None:
        raise KeyError("Training CSV must contain a target column "
                       f"(one of {', '.join(_TGT_HINTS)} or similar).")

    # Normalize and join paths safely
    df["image_path"] = df[img_col].apply(_normalize_relpath)
    df["image_path"] = df["image_path"].apply(lambda x: str(TRAIN_IMG_DIR / x))
    return df, img_col, tgt_col

def load_test_data() -> Tuple[pd.DataFrame, str]:
    df = read_csv(TEST_CSV)
    img_col = infer_image_col(df)

    df["image_path"] = df[img_col].apply(_normalize_relpath)
    df["image_path"] = df["image_path"].apply(lambda x: str(TEST_IMG_DIR / x))
    return df, img_col

# -------------------------------------------------------------
# Image validation utilities
# -------------------------------------------------------------
def sample_image_stats(
    df: pd.DataFrame,
    image_path_col: str = "image_path",
    max_samples: int = 200,
) -> Dict[str, object]:
    if image_path_col not in df.columns:
        raise KeyError(f"Column '{image_path_col}' not in DataFrame")

    paths = df[image_path_col].tolist()
    rng = np.random.default_rng(42)
    idx = rng.choice(len(paths), size=min(max_samples, len(paths)), replace=False)
    sizes, missing, bad = [], 0, 0

    for i in idx:
        p = Path(paths[i])
        if not p.exists():
            missing += 1
            continue
        try:
            with Image.open(p) as im:
                sizes.append(im.size)
        except (UnidentifiedImageError, OSError):
            bad += 1

    sizes_arr = np.array(sizes) if sizes else np.zeros((0, 2))
    return {
        "n_examined": int(len(idx)),
        "missing": int(missing),
        "bad": int(bad),
        "sizes_count": int(sizes_arr.shape[0]),
        "width_mean": float(sizes_arr[:, 0].mean()) if sizes_arr.size else None,
        "height_mean": float(sizes_arr[:, 1].mean()) if sizes_arr.size else None,
        "width_min": int(sizes_arr[:, 0].min()) if sizes_arr.size else None,
        "height_min": int(sizes_arr[:, 1].min()) if sizes_arr.size else None,
        "width_max": int(sizes_arr[:, 0].max()) if sizes_arr.size else None,
        "height_max": int(sizes_arr[:, 1].max()) if sizes_arr.size else None,
    }
