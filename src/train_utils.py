# =============================================================
# 🧠 TRAINING UTILITIES — Cross-Environment Compatible
# =============================================================
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda import amp
from tqdm.auto import tqdm
import numpy as np
import time
import math
import os

# -------------------------------------------------------------
# 🧩 Environment setup
# -------------------------------------------------------------
IS_KAGGLE = Path("/kaggle").exists()
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if getattr(torch.backends, "mps", None)
    and torch.backends.mps.is_available()
    else "cpu"
)
IS_GPU = DEVICE in ("cuda", "mps")

# -------------------------------------------------------------
# ⚙️ Utility: compute RMSE, R²
# -------------------------------------------------------------
def rmse(y_true, y_pred):
    return float(torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item())

def r2_score(y_true, y_pred):
    y_true_mean = torch.mean(y_true)
    ss_tot = torch.sum((y_true - y_true_mean) ** 2)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    return float(1 - ss_res / ss_tot)

# -------------------------------------------------------------
# 🧱 Training + Validation Epochs
# -------------------------------------------------------------
def train_one_epoch(model, dataloader, optimizer, scaler, criterion, device, epoch, total_epochs):
    model.train()
    running_loss = 0.0
    n_batches = len(dataloader)
    pbar = tqdm(dataloader, desc=f"🧩 Train [{epoch+1}/{total_epochs}]", leave=False)

    for batch in pbar:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device).float()

        optimizer.zero_grad(set_to_none=True)
        with amp.autocast(enabled=(device == "cuda")):
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    avg_loss = running_loss / n_batches
    return avg_loss


@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    y_true, y_pred = [], []

    pbar = tqdm(dataloader, desc="🧪 Validation", leave=False)
    for batch in pbar:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device).float()

        outputs = model(inputs).squeeze()
        loss = criterion(outputs, targets)

        running_loss += loss.item()
        y_true.append(targets.detach().cpu())
        y_pred.append(outputs.detach().cpu())

    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)
    val_loss = running_loss / len(dataloader)
    val_rmse = rmse(y_true, y_pred)
    val_r2 = r2_score(y_true, y_pred)

    return val_loss, val_rmse, val_r2


# -------------------------------------------------------------
# 💾 Checkpoint handling
# -------------------------------------------------------------
def save_checkpoint(model, optimizer, epoch, output_dir, name="model_ckpt.pt"):
    ckpt_path = Path(output_dir) / name
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    torch.save(ckpt, ckpt_path)
    print(f"💾 Saved checkpoint to: {ckpt_path}")


def load_checkpoint(model, optimizer, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    print(f"📦 Loaded checkpoint from: {ckpt_path}")
    return ckpt.get("epoch", 0)


# -------------------------------------------------------------
# 🧠 Full training routine
# -------------------------------------------------------------
def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    device=DEVICE,
    epochs=10,
    output_dir="checkpoints",
    ckpt_name="model_ckpt.pt",
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    scaler = amp.GradScaler(enabled=(device == "cuda"))
    best_rmse = math.inf
    best_epoch = 0

    print(f"🚀 Starting training for {epochs} epochs on {device.upper()}")
    for epoch in range(epochs):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, criterion, device, epoch, epochs)
        val_loss, val_rmse, val_r2 = validate_one_epoch(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step(val_loss)

        elapsed = time.time() - t0
        print(f"[Epoch {epoch+1:02d}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | RMSE: {val_rmse:.4f} | R²: {val_r2:.4f} | ⏱ {elapsed:.1f}s")

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_epoch = epoch + 1
            save_checkpoint(model, optimizer, best_epoch, output_dir, ckpt_name)

    print(f"🏁 Training complete. Best RMSE={best_rmse:.4f} at epoch {best_epoch}.")
    return best_rmse, best_epoch
