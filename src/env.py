# =============================================================
# 🌍 Environment Utilities — Cursor / macOS / Kaggle Compatible
# =============================================================
import platform
import torch

def get_env():
    """Return (device, num_workers, pin_memory) for this runtime."""
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # macOS MPS spawn prefers single worker; others can parallelize
    num_workers = 0 if platform.system() == "Darwin" else 2
    pin_memory = torch.cuda.is_available()  # ignored safely on MPS

    print(f"🧭 Device: {device} | num_workers={num_workers} | pin_memory={pin_memory}")
    return device, num_workers, pin_memory
