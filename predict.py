#!/usr/bin/env python3
"""
Rice Disease Classification - Inference Script
Supports MambaCNN, MambaCNN Lite (Edge), and YOLO (Baseline) models.

Usage:
    python predict.py --image <path_to_image> --model <mamba|mamba_lite|yolo>

Examples:
    python predict.py --image test_leaf.jpg --model mamba_lite  # Edge model (<1MB)
    python predict.py --image test_leaf.jpg --model mamba       # Original Mamba
    python predict.py --image test_leaf.jpg --model yolo        # Baseline
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Disease class names
CLASS_NAMES = [
    "Bacterial_Leaf_Blight",
    "Brown_Spot",
    "Healthy_Rice_Leaf",
    "Leaf_Blast",
    "Leaf_scald",
    "Sheath_Blight"
]

# Normalization constants (ImageNet)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# ============================================
# MambaCNN Model Definition
# ============================================

class DepthwiseSeparable(nn.Module):
    """Depthwise separable convolution for MambaCNN Lite."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch), nn.GELU())
        self.pw = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU())
    def forward(self, x): return self.pw(self.dw(x))


class MambaBlock(nn.Module):
    """Mamba S6 block with selective scan mechanism."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.dt_rank = max(1, self.d_inner // 16)

        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A_init = torch.log(
            torch.arange(1, d_state + 1, dtype=torch.float32)
            .unsqueeze(0).repeat(self.d_inner, 1)
        )
        self.A_log = nn.Parameter(A_init)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def _selective_scan(self, x):
        B, L, _ = x.shape
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        xBC = self.x_proj(x)
        dt_raw, B_p, C = xBC.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt_raw))
        dA = torch.exp(torch.einsum('bld,ds->blds', dt, A))
        dB = torch.einsum('bld,bls->blds', dt, B_p)
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for i in range(L):
            h = dA[:, i] * h + dB[:, i] * x[:, i, :, None]
            ys.append(torch.einsum('bds,bs->bd', h, C[:, i]))
        y = torch.stack(ys, dim=1)
        return y + x * D.to(x.dtype)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        xz = self.in_proj(x)
        x_h, z = xz.chunk(2, dim=-1)
        x_h = self.conv1d(x_h.transpose(1, 2))[:, :, :x_h.shape[1]].transpose(1, 2)
        x_h = F.silu(x_h)
        y = self._selective_scan(x_h)
        y = y * F.silu(z)
        return self.out_proj(y) + residual


class MambaCNN(nn.Module):
    """
    MambaCNN for rice disease classification.
    Architecture: CNN Stem -> Patchify -> Mamba S6 Blocks -> Classification
    """

    def __init__(self, num_classes: int = 6, d_model: int = 64, n_mamba: int = 2):
        super().__init__()

        def cnn_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
                nn.MaxPool2d(2)
            )

        self.stem = nn.Sequential(
            cnn_block(3, 32),
            cnn_block(32, 64),
            cnn_block(64, d_model),
            cnn_block(d_model, d_model)
        )
        self.mamba = nn.Sequential(*[MambaBlock(d_model) for _ in range(n_mamba)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(d_model, num_classes))

    def forward(self, x):
        x = self.stem(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.mamba(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)


class MambaBlockLite(nn.Module):
    """Lite Mamba block with smaller state size."""
    def __init__(self, d_model: int, d_state: int = 8, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.dt_rank = max(1, self.d_inner // 16)
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                 padding=d_conv - 1, groups=self.d_inner, bias=True)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        A_init = torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(self.d_inner, 1))
        self.A_log = nn.Parameter(A_init)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def _selective_scan(self, x):
        B, L, _ = x.shape
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        xBC = self.x_proj(x)
        dt_raw, B_p, C = xBC.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt_raw))
        dA = torch.exp(torch.einsum('bld,ds->blds', dt, A))
        dB = torch.einsum('bld,bls->blds', dt, B_p)
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for i in range(L):
            h = dA[:, i] * h + dB[:, i] * x[:, i, :, None]
            ys.append(torch.einsum('bds,bs->bd', h, C[:, i]))
        return torch.stack(ys, dim=1) + x * D.to(x.dtype)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        xz = self.in_proj(x)
        x_h, z = xz.chunk(2, dim=-1)
        x_h = self.conv1d(x_h.transpose(1, 2))[:, :, :x_h.shape[1]].transpose(1, 2)
        x_h = F.silu(x_h)
        y = self._selective_scan(x_h)
        return self.out_proj(y * F.silu(z)) + residual


class MambaCNNLite(nn.Module):
    """
    MambaCNN Lite for edge deployment.
    Uses depthwise separable convolutions and smaller state size.
    Input: 96x96x3
    """
    def __init__(self, num_classes: int = 6, d_model: int = 64, n_mamba: int = 2, d_state: int = 8):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.GELU(), nn.MaxPool2d(2),
            DepthwiseSeparable(16, 32), nn.MaxPool2d(2),
            DepthwiseSeparable(32, d_model), nn.MaxPool2d(2),
            DepthwiseSeparable(d_model, d_model), nn.MaxPool2d(2))
        self.mamba = nn.Sequential(*[MambaBlockLite(d_model, d_state=d_state) for _ in range(n_mamba)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(d_model, num_classes))

    def forward(self, x):
        x = self.stem(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.mamba(x)
        x = self.norm(x)
        return self.head(x.mean(dim=1))


# ============================================
# Inference Functions
# ============================================

def load_mamba_model(checkpoint_path: str, device: torch.device, lite: bool = False):
    """Load MambaCNN or MambaCNN Lite model from checkpoint."""
    if lite:
        model = MambaCNNLite(num_classes=len(CLASS_NAMES), d_model=64, n_mamba=2, d_state=8)
    else:
        model = MambaCNN(num_classes=len(CLASS_NAMES), d_model=64, n_mamba=2)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_yolo_model(checkpoint_path: str):
    """Load YOLO classification model."""
    from ultralytics import YOLO
    model = YOLO(checkpoint_path)
    return model


def preprocess_image_mamba(image_path: str, image_size: int = 128) -> torch.Tensor:
    """Preprocess image for MambaCNN model."""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)


def predict_mamba(model, image_path: str, device: torch.device, image_size: int = 128) -> dict:
    """Run inference with MambaCNN or MambaCNN Lite model."""
    image_tensor = preprocess_image_mamba(image_path, image_size).to(device)

    # Warm-up run
    with torch.no_grad():
        _ = model(image_tensor)

    # Timed inference
    start_time = time.perf_counter()
    with torch.no_grad():
        logits = model(image_tensor)
        probs = F.softmax(logits, dim=1)
    end_time = time.perf_counter()

    inference_time_ms = (end_time - start_time) * 1000

    pred_idx = probs.argmax(dim=1).item()
    confidence = probs[0, pred_idx].item()

    return {
        "predicted_class": CLASS_NAMES[pred_idx],
        "confidence": confidence * 100,
        "inference_time_ms": inference_time_ms,
        "all_probabilities": {
            CLASS_NAMES[i]: probs[0, i].item() * 100
            for i in range(len(CLASS_NAMES))
        }
    }


def predict_yolo(model, image_path: str) -> dict:
    """Run inference with YOLO model."""
    # Warm-up run
    _ = model.predict(image_path, verbose=False)

    # Timed inference
    start_time = time.perf_counter()
    results = model.predict(image_path, verbose=False)
    end_time = time.perf_counter()

    inference_time_ms = (end_time - start_time) * 1000

    result = results[0]
    probs = result.probs
    pred_idx = probs.top1
    confidence = probs.top1conf.item()

    return {
        "predicted_class": result.names[pred_idx],
        "confidence": confidence * 100,
        "inference_time_ms": inference_time_ms,
        "all_probabilities": {
            result.names[i]: probs.data[i].item() * 100
            for i in range(len(result.names))
        }
    }


def find_model_path(model_type: str) -> str:
    """Find the best model checkpoint path."""
    base_dir = Path(__file__).parent

    if model_type == "mamba_lite":
        # MambaCNN Lite - Edge model (<1MB), best seed is 1
        paths = [
            base_dir / "train_mamba_lite_results" / "outputs" / "runs" / "seed_1" / "best_mamba_cnn_lite_seed1.pth",
            base_dir / "models" / "mamba_lite.pth",
        ]
    elif model_type == "mamba":
        # Original MambaCNN
        paths = [
            base_dir / "train_mamba_results" / "outputs" / "runs" / "seed_123" / "best_mamba_cnn_seed123.pth",
            base_dir / "models" / "mamba.pth",
        ]
    else:  # yolo
        paths = [
            base_dir / "train_yolo_results" / "outputs" / "best_model" / "best.pt",
            base_dir / "train_yolo_results" / "runs" / "classify" / "outputs" / "runs" / "seed_789" / "weights" / "best.pt",
            base_dir / "models" / "yolo_baseline.pt",
        ]

    for path in paths:
        if path.exists():
            return str(path)

    raise FileNotFoundError(
        f"Could not find {model_type} model checkpoint. "
        f"Searched paths: {[str(p) for p in paths]}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Rice Disease Classification Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python predict.py --image leaf.jpg --model mamba_lite   # Edge model (<1MB)
    python predict.py --image leaf.jpg --model mamba        # Original MambaCNN
    python predict.py --image leaf.jpg --model yolo         # YOLO baseline
    python predict.py --image leaf.jpg --model mamba_lite --checkpoint path/to/model.pth
        """
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        choices=["mamba", "mamba_lite", "yolo"],
        default="mamba_lite",
        help="Model type: 'mamba_lite' (Edge, <1MB), 'mamba' (original), or 'yolo' (Baseline, ~3MB)"
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        default=None,
        help="Path to model checkpoint (auto-detected if not specified)"
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="auto",
        help="Device: 'cpu', 'cuda', or 'auto'"
    )

    args = parser.parse_args()

    # Validate image path
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        return 1

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Find checkpoint
    checkpoint = args.checkpoint or find_model_path(args.model)

    print(f"\n{'='*60}")
    print(f"Rice Disease Classification")
    print(f"{'='*60}")
    print(f"Model:      {args.model.upper()}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Device:     {device}")
    print(f"Image:      {args.image}")
    print(f"{'='*60}\n")

    # Load model and run inference
    if args.model in ["mamba", "mamba_lite"]:
        is_lite = args.model == "mamba_lite"
        image_size = 96 if is_lite else 128
        model = load_mamba_model(checkpoint, device, lite=is_lite)
        result = predict_mamba(model, args.image, device, image_size=image_size)
    else:  # yolo
        model = load_yolo_model(checkpoint)
        result = predict_yolo(model, args.image)

    # Display results
    print(f"Prediction:     {result['predicted_class']}")
    print(f"Confidence:     {result['confidence']:.2f}%")
    print(f"Inference Time: {result['inference_time_ms']:.2f} ms")
    print(f"\nAll Class Probabilities:")
    print(f"{'-'*40}")
    for class_name, prob in sorted(
        result['all_probabilities'].items(),
        key=lambda x: x[1],
        reverse=True
    ):
        bar = '#' * int(prob / 5)
        print(f"  {class_name:<25} {prob:6.2f}% {bar}")

    print(f"\n{'='*60}\n")

    return 0


if __name__ == "__main__":
    exit(main())
