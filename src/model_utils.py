# =============================================================
# 🧠 Model Utilities — Cross-Environment Safe
# =============================================================
from pathlib import Path
import torch
import torch.nn as nn
import timm

# -------------------------------------------------------------
# Environment flags
# -------------------------------------------------------------
IS_KAGGLE = Path("/kaggle").exists()
DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
)
IS_GPU = DEVICE == "cuda" or DEVICE == "mps"

# -------------------------------------------------------------
# 🔧 Base model builder
# -------------------------------------------------------------
def build_effnet_b0(pretrained: bool = True, num_outputs: int = 1) -> nn.Module:
    """EfficientNet-B0 backbone + regression head"""
    model = timm.create_model("efficientnet_b0", pretrained=pretrained, in_chans=3)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_outputs)
    return model


def build_swin_v2s(pretrained: bool = True, num_outputs: int = 1) -> nn.Module:
    """Swin Transformer V2 Small backbone + regression head"""
    model = timm.create_model("swinv2_small_window8_256", pretrained=pretrained, in_chans=3)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_outputs)
    return model


def build_dinov2_base(pretrained: bool = True, num_outputs: int = 1) -> nn.Module:
    """DINOv2 Base ViT backbone + regression head"""
    model = timm.create_model("vit_base_patch16_dinov2", pretrained=pretrained, in_chans=3)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_outputs)
    return model


# -------------------------------------------------------------
# 🧠 FusionRegressor for metadata-safe training
# -------------------------------------------------------------
class FusionRegressor(nn.Module):
    def __init__(self, backbone_name="efficientnet_b0", meta_in=2, meta_hidden=(32,16)):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=False, num_classes=0, global_pool="avg"
        )
        feat_dim = getattr(self.backbone, "num_features", 1280)
        self.meta = nn.Sequential(
            nn.Linear(meta_in, meta_hidden[0]), nn.ReLU(inplace=True),
            nn.Linear(meta_hidden[0], meta_hidden[1]), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(feat_dim + meta_hidden[1], 256), nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, imgs, metas=None):
        f_img = self.backbone(imgs)
        if metas is not None:
            f_meta = self.meta(metas)
        else:
            # Safe zero metadata if not present
            f_meta = torch.zeros(
                (f_img.shape[0], self.meta[-2].out_features),
                device=f_img.device,
                dtype=f_img.dtype
            )
        fused = torch.cat([f_img, f_meta], dim=1)
        return self.head(fused).squeeze(1)


# -------------------------------------------------------------
# 📦 Helper to create model dynamically
# -------------------------------------------------------------
def build_model(name: str = "effnet_b0", pretrained: bool = True, num_outputs: int = 1) -> nn.Module:
    name = name.lower()
    if name in ["effnet_b0", "efficientnet_b0", "effnet"]:
        return build_effnet_b0(pretrained, num_outputs)
    elif name == "effnet_b0_fusion":
        return FusionRegressor()
    elif name in ["swin_v2s", "swinv2_small"]:
        return build_swin_v2s(pretrained, num_outputs)
    elif name in ["dinov2_base", "vit_dinov2"]:
        return build_dinov2_base(pretrained, num_outputs)
    else:
        raise ValueError(f"Unknown model name: {name}")


# -------------------------------------------------------------
# 🧪 Quick self-test
# -------------------------------------------------------------
if __name__ == "__main__":
    m = build_model("effnet_b0")
    x = torch.randn(2, 3, 224, 224)
    out = m(x)
    print(f"✅ Built model OK — Output shape: {out.shape} on device={DEVICE}")
