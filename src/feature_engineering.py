# src/feature_engineering.py
import torch
import torch.nn as nn

def process_metadata(meta_tensor, meta_hidden=(32, 16)):
    """
    Optional: normalize/encode metadata and return tensor suitable for fusion
    """
    meta_net = nn.Sequential(
        nn.Linear(meta_tensor.shape[1], meta_hidden[0]), nn.ReLU(inplace=True),
        nn.Linear(meta_hidden[0], meta_hidden[1]), nn.ReLU(inplace=True),
    )
    return meta_net(meta_tensor)
