# =============================================================
# 🧩 CSIRO Data Pipeline — Image Dataset + Transform Builders
# =============================================================
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np

from src.config import DATA_DIR, TRAIN_IMG_DIR, TEST_IMG_DIR
from src.data_loading import load_train_data, load_test_data


# -------------------------------------------------------------
# 1️⃣  Build train / validation transforms
# -------------------------------------------------------------
def get_transforms(img_size: int = 224, is_train: bool = True):
    """Return torchvision transform pipeline for train/test."""
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


# -------------------------------------------------------------
# 2️⃣  Core Dataset Class
# -------------------------------------------------------------
class CSIRODataset(Dataset):
    """
    Image (and optional metadata) dataset for CSIRO Biomass Prediction.

    Returns:
        (image_tensor, target_tensor) for training
        (image_tensor,) for inference
    """
    def __init__(self, df: pd.DataFrame, img_dir: Path,
                 transform=None, include_metadata: bool = False,
                 metadata_cols=None, target_col: str = "target"):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.include_metadata = include_metadata
        self.metadata_cols = metadata_cols or []
        self.target_col = target_col

        # Preload metadata subset if requested
        if self.include_metadata and self.metadata_cols:
            self.meta_df = self.df[self.metadata_cols].copy()
        else:
            self.meta_df = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = Path(row["image_path"])
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        # --- Load and transform image ---
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # --- Target (for training) ---
        target = torch.tensor(row[self.target_col], dtype=torch.float32) if self.target_col in row else None

        # --- Metadata (optional future fusion) ---
        if self.include_metadata and self.meta_df is not None:
            meta_feats = self.meta_df.iloc[idx].values.astype(np.float32)
            if target is not None:
                return image, torch.tensor(meta_feats), target
            else:
                return image, torch.tensor(meta_feats)
        else:
            if target is not None:
                return image, target
            else:
                return (image,)


# -------------------------------------------------------------
# 3️⃣  DataLoader Builders
# -------------------------------------------------------------
def create_dataloaders(batch_size=16, img_size=224, num_workers=2,
                       include_metadata=False, metadata_cols=None):
    """
    Create train and test dataloaders using CSIRODataset.
    """
    # Load CSVs
    train_df, _, _ = load_train_data()
    test_df, _ = load_test_data()

    # Transforms
    train_tfms = get_transforms(img_size=img_size, is_train=True)
    test_tfms = get_transforms(img_size=img_size, is_train=False)

    # Datasets
    train_dataset = CSIRODataset(
        train_df, TRAIN_IMG_DIR,
        transform=train_tfms,
        include_metadata=include_metadata,
        metadata_cols=metadata_cols,
    )
    test_dataset = CSIRODataset(
        test_df, TEST_IMG_DIR,
        transform=test_tfms,
        include_metadata=include_metadata,
        metadata_cols=metadata_cols,
        target_col=None,
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=True
    )

    print(f"✅ Train dataset size: {len(train_dataset)}  |  Batches: {len(train_loader)}")
    print(f"✅ Test  dataset size:  {len(test_dataset)}   |  Batches: {len(test_loader)}")

    return train_loader, test_loader


# -------------------------------------------------------------
# 4️⃣  Robust batch unpacking helper
# -------------------------------------------------------------
def unpack_batch(batch):
    """
    Unpacks a batch into (imgs, metas, targets)
    Supports:
        - [imgs, targets]
        - [(imgs, metas), targets]
        - fallback to batch[0], batch[1]
    """
    if len(batch) == 2 and isinstance(batch[0], torch.Tensor):
        imgs, targets = batch
        metas = None
    elif len(batch) == 2 and isinstance(batch[0], (list, tuple)):
        (imgs, metas), targets = batch
        if metas is not None and metas.numel() == 0:
            metas = None
    else:
        try:
            imgs, targets = batch[0], batch[1]
            metas = None
        except Exception:
            raise ValueError(f"Unexpected batch format: {batch}")
    return imgs, metas, targets


# -------------------------------------------------------------
# 5️⃣  Standalone sanity check
# -------------------------------------------------------------
if __name__ == "__main__":
    print("🔍 Running CSIRODataset sanity check...")
    train_loader, test_loader = create_dataloaders(batch_size=4, img_size=128)
    one_batch = next(iter(train_loader))

    # Handle both formats
    if isinstance(one_batch[0], torch.Tensor):
        images, targets = one_batch
        metas = None
    else:
        (images, metas), targets = one_batch

    images = images.to("cpu", dtype=torch.float32)
    if targets is not None:
        targets = targets.to("cpu", dtype=torch.float32)
    if metas is not None:
        metas = metas.to("cpu", dtype=torch.float32)

    print(f"Batch images: {images.shape} | Batch targets: {targets.shape if targets is not None else 'None'}")
    print("✅ Sanity check passed.")
