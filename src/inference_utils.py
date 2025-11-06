# src/inference_utils.py
import numpy as np
import torch
from src.data_pipeline import unpack_batch

def predict_batch(model, dataloader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in dataloader:
            imgs, metas, targets = unpack_batch(batch)
            imgs = imgs.to(device, dtype=torch.float32)
            if metas is not None:
                metas = metas.to(device, dtype=torch.float32)
            if targets is not None:
                y_true.append(targets.cpu().numpy())
            preds = model(imgs, metas)
            y_pred.append(preds.detach().cpu().numpy())
    if y_true:
        y_true = np.concatenate(y_true)
    else:
        y_true = None
    y_pred = np.concatenate(y_pred)
    return y_true, y_pred

def evaluate_predictions(y_true, y_pred):
    rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = float(np.sum((y_true - y_pred)**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true))**2))
    r2 = float(1 - ss_res / (ss_tot + 1e-12))
    return {"rmse": rmse, "mae": mae, "r2": r2}
