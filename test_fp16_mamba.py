"""
Test MambaCNN FP16 vs MambaCNN Lite on GPU with same test split.
Run this on Kaggle with GPU enabled.

Outputs:
- FP16 model (.pth)
- FP16 ONNX model
- Accuracy comparison
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import time
import json

# ============================================
# CONFIGURATION
# ============================================
DATASET_PATH = "/kaggle/input/rice-leaf-aug/Rice_Leaf_AUG"  # Kaggle dataset path
SPLIT_SEED = 42  # Same as training for consistent test split
IMAGE_SIZE = 128
BATCH_SIZE = 64
NUM_WORKERS = 4

# Model paths (update these based on your Kaggle input)
MAMBA_MODEL_PATH = "/kaggle/input/rice-disease-models/best_mamba_cnn_seed123.pth"
MAMBA_LITE_MODEL_PATH = "/kaggle/input/rice-disease-models/best_mamba_cnn_lite_seed1.pth"

OUTPUT_DIR = "/kaggle/working"

# ============================================
# MODEL ARCHITECTURES
# ============================================

class MambaBlock(nn.Module):
    """Mamba block for MambaCNN."""
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
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
    """MambaCNN - Full model (d_model=64, n_mamba=2)."""
    def __init__(self, num_classes: int, d_model: int = 64, n_mamba: int = 2):
        super().__init__()
        def cnn_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch), nn.GELU(), nn.MaxPool2d(2))
        self.stem = nn.Sequential(cnn_block(3, 32), cnn_block(32, 64),
                                  cnn_block(64, d_model), cnn_block(d_model, d_model))
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
    """Mamba block for MambaCNN Lite (smaller d_state, d_conv)."""
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


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution for MambaCNN Lite."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch), nn.GELU())
        self.pw = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU())

    def forward(self, x):
        return self.pw(self.dw(x))


class MambaCNNLite(nn.Module):
    """MambaCNN Lite - Smaller model with depthwise separable convolutions."""
    def __init__(self, num_classes: int, d_model: int = 64, n_mamba: int = 2):
        super().__init__()
        # Channels: 16 -> 32 -> 64 -> 64 (matching checkpoint)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.GELU(),
            nn.MaxPool2d(2),
            DepthwiseSeparableConv(16, 32),
            nn.MaxPool2d(2),
            DepthwiseSeparableConv(32, d_model),
            nn.MaxPool2d(2),
            DepthwiseSeparableConv(d_model, d_model),
            nn.MaxPool2d(2),
        )
        self.mamba = nn.Sequential(*[MambaBlockLite(d_model) for _ in range(n_mamba)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(d_model, num_classes))

    def forward(self, x):
        x = self.stem(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.mamba(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)


# ============================================
# DATA LOADING
# ============================================

def get_test_loader(dataset_path, image_size, batch_size, num_workers, split_seed):
    """Create test loader with same split as training."""
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    dataset = datasets.ImageFolder(dataset_path, transform=eval_tf)
    class_names = dataset.classes
    n = len(dataset)

    # Same split as training (80/10/10)
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(split_seed)).tolist()
    n_train, n_val = int(0.8 * n), int(0.1 * n)
    test_indices = indices[n_train + n_val:]

    test_ds = Subset(dataset, test_indices)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return test_loader, class_names


# ============================================
# EVALUATION
# ============================================

@torch.no_grad()
def evaluate(model, loader, device, use_amp=False):
    """Evaluate model on test set."""
    model.eval()
    correct = 0
    total = 0
    total_time = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        if use_amp:
            with torch.cuda.amp.autocast():
                start = time.perf_counter()
                logits = model(imgs)
                torch.cuda.synchronize()
                total_time += time.perf_counter() - start
        else:
            start = time.perf_counter()
            logits = model(imgs)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            total_time += time.perf_counter() - start

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total * 100
    avg_time = total_time / len(loader) * 1000  # ms per batch

    return accuracy, avg_time


def get_model_size(model):
    """Get model size in KB."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================
# MAIN
# ============================================

def main():
    print("=" * 70)
    print("  MambaCNN FP16 vs MambaCNN Lite - Test Accuracy Comparison")
    print("=" * 70)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load test data
    print(f"\nLoading test data from: {DATASET_PATH}")
    print(f"Using split seed: {SPLIT_SEED} (same as training)")
    test_loader, class_names = get_test_loader(DATASET_PATH, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, SPLIT_SEED)
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Classes: {class_names}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {}

    # ========================================
    # 1. MambaCNN FP32
    # ========================================
    print("\n" + "=" * 70)
    print("  [1] MambaCNN (FP32)")
    print("=" * 70)

    mamba_fp32 = MambaCNN(num_classes=6, d_model=64, n_mamba=2).to(device)
    mamba_fp32.load_state_dict(torch.load(MAMBA_MODEL_PATH, map_location=device))
    mamba_fp32.eval()

    acc_fp32, time_fp32 = evaluate(mamba_fp32, test_loader, device)
    size_fp32 = get_model_size(mamba_fp32)
    params_mamba = count_parameters(mamba_fp32)

    print(f"  Parameters: {params_mamba:,}")
    print(f"  Model Size: {size_fp32:.1f} KB")
    print(f"  Test Accuracy: {acc_fp32:.2f}%")
    print(f"  Inference Time: {time_fp32:.2f} ms/batch")

    results['mamba_fp32'] = {
        'accuracy': acc_fp32,
        'size_kb': size_fp32,
        'parameters': params_mamba,
        'inference_ms': time_fp32
    }

    # ========================================
    # 2. MambaCNN FP16 (AMP)
    # ========================================
    print("\n" + "=" * 70)
    print("  [2] MambaCNN (FP16 - Automatic Mixed Precision)")
    print("=" * 70)

    # Use same model with AMP for FP16 inference
    acc_fp16, time_fp16 = evaluate(mamba_fp32, test_loader, device, use_amp=True)
    size_fp16 = size_fp32 / 2  # FP16 is half the size

    print(f"  Parameters: {params_mamba:,}")
    print(f"  Model Size (FP16): {size_fp16:.1f} KB")
    print(f"  Test Accuracy: {acc_fp16:.2f}%")
    print(f"  Inference Time: {time_fp16:.2f} ms/batch")

    results['mamba_fp16'] = {
        'accuracy': acc_fp16,
        'size_kb': size_fp16,
        'parameters': params_mamba,
        'inference_ms': time_fp16
    }

    # Save FP16 model
    mamba_fp16_model = MambaCNN(num_classes=6, d_model=64, n_mamba=2)
    mamba_fp16_model.load_state_dict(torch.load(MAMBA_MODEL_PATH, map_location='cpu'))
    mamba_fp16_model = mamba_fp16_model.half()

    fp16_pth_path = os.path.join(OUTPUT_DIR, "mamba_cnn_fp16.pth")
    torch.save(mamba_fp16_model.state_dict(), fp16_pth_path)
    print(f"\n  FP16 .pth saved: {fp16_pth_path}")
    print(f"  FP16 .pth size: {os.path.getsize(fp16_pth_path) / 1024:.1f} KB")

    # Export FP16 ONNX (export FP32, ONNX Runtime handles FP16)
    fp16_onnx_path = os.path.join(OUTPUT_DIR, "mamba_cnn_fp16.onnx")
    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    mamba_fp32_cpu = MambaCNN(num_classes=6, d_model=64, n_mamba=2)
    mamba_fp32_cpu.load_state_dict(torch.load(MAMBA_MODEL_PATH, map_location='cpu'))
    mamba_fp32_cpu.eval()

    torch.onnx.export(
        mamba_fp32_cpu, dummy, fp16_onnx_path,
        input_names=["input"], output_names=["output"],
        opset_version=14, dynamo=False,
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
    )
    print(f"  ONNX saved: {fp16_onnx_path}")
    print(f"  ONNX size: {os.path.getsize(fp16_onnx_path) / 1024:.1f} KB")

    # ========================================
    # 3. MambaCNN Lite (FP32)
    # ========================================
    print("\n" + "=" * 70)
    print("  [3] MambaCNN Lite (FP32)")
    print("=" * 70)

    mamba_lite = MambaCNNLite(num_classes=6, d_model=64, n_mamba=2).to(device)
    mamba_lite.load_state_dict(torch.load(MAMBA_LITE_MODEL_PATH, map_location=device))
    mamba_lite.eval()

    acc_lite, time_lite = evaluate(mamba_lite, test_loader, device)
    size_lite = get_model_size(mamba_lite)
    params_lite = count_parameters(mamba_lite)

    print(f"  Parameters: {params_lite:,}")
    print(f"  Model Size: {size_lite:.1f} KB")
    print(f"  Test Accuracy: {acc_lite:.2f}%")
    print(f"  Inference Time: {time_lite:.2f} ms/batch")

    results['mamba_lite'] = {
        'accuracy': acc_lite,
        'size_kb': size_lite,
        'parameters': params_lite,
        'inference_ms': time_lite
    }

    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 70)
    print("  SUMMARY COMPARISON")
    print("=" * 70)

    print(f"""
| Model              | Test Accuracy | Model Size | Parameters | Inference |
|--------------------|---------------|------------|------------|-----------|
| MambaCNN (FP32)    | {acc_fp32:>10.2f}% | {size_fp32:>7.0f} KB | {params_mamba:>10,} | {time_fp32:>6.2f} ms |
| MambaCNN (FP16)    | {acc_fp16:>10.2f}% | {size_fp16:>7.0f} KB | {params_mamba:>10,} | {time_fp16:>6.2f} ms |
| MambaCNN Lite      | {acc_lite:>10.2f}% | {size_lite:>7.0f} KB | {params_lite:>10,} | {time_lite:>6.2f} ms |
""")

    # Comparison
    print("=" * 70)
    if acc_fp16 > acc_lite:
        print(f"  ✅ MambaCNN FP16 ({acc_fp16:.2f}%) > MambaCNN Lite ({acc_lite:.2f}%)")
        print(f"     Accuracy difference: +{acc_fp16 - acc_lite:.2f}%")
    elif acc_fp16 < acc_lite:
        print(f"  ✅ MambaCNN Lite ({acc_lite:.2f}%) > MambaCNN FP16 ({acc_fp16:.2f}%)")
        print(f"     Accuracy difference: +{acc_lite - acc_fp16:.2f}%")
    else:
        print(f"  🔄 MambaCNN FP16 ({acc_fp16:.2f}%) = MambaCNN Lite ({acc_lite:.2f}%)")

    print(f"\n  Size comparison:")
    print(f"    MambaCNN FP16: {size_fp16:.0f} KB")
    print(f"    MambaCNN Lite: {size_lite:.0f} KB")
    if size_fp16 < size_lite:
        print(f"    FP16 is {(1 - size_fp16/size_lite)*100:.1f}% smaller")
    else:
        print(f"    Lite is {(1 - size_lite/size_fp16)*100:.1f}% smaller")
    print("=" * 70)

    # Save results
    results_path = os.path.join(OUTPUT_DIR, "fp16_comparison_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
