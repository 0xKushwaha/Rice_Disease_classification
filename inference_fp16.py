"""
FP16 MambaCNN Inference Script
Runs inference using FP16 model weights on CPU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import time
import argparse
import os


# MambaCNN Architecture (same as training)
class MambaBlock(nn.Module):
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


# Class names
CLASS_NAMES = [
    'Bacterial_Leaf_Blight',
    'Brown_Spot',
    'Healthy_Rice_Leaf',
    'Leaf_Blast',
    'Leaf_scald',
    'Sheath_Blight'
]

# Preprocessing
def get_transform():
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def load_model(checkpoint_path: str, device: str = 'cpu'):
    """Load FP16 model and convert to FP32 for CPU inference."""
    model = MambaCNN(num_classes=6, d_model=64, n_mamba=2)

    # Load FP16 weights
    state_dict = torch.load(checkpoint_path, map_location=device)

    # Convert FP16 weights to FP32 for CPU inference
    state_dict_fp32 = {k: v.float() if v.dtype == torch.float16 else v for k, v in state_dict.items()}
    model.load_state_dict(state_dict_fp32)

    model.to(device)
    model.eval()
    return model


def predict(model, image_path: str, device: str = 'cpu', warmup_runs: int = 3, benchmark_runs: int = 10):
    """Run inference and return prediction with timing."""
    transform = get_transform()

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)

    # Benchmark runs
    times = []
    with torch.no_grad():
        for _ in range(benchmark_runs):
            start = time.perf_counter()
            output = model(input_tensor)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

    avg_time = sum(times) / len(times)

    # Get prediction
    probs = F.softmax(output, dim=1)
    confidence, pred_idx = torch.max(probs, dim=1)

    return {
        'class': CLASS_NAMES[pred_idx.item()],
        'confidence': confidence.item() * 100,
        'inference_time_ms': avg_time,
        'all_probs': {CLASS_NAMES[i]: probs[0, i].item() * 100 for i in range(len(CLASS_NAMES))}
    }


def main():
    parser = argparse.ArgumentParser(description='MambaCNN FP16 Inference')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--checkpoint', type=str, default='mamba_fp16/mamba_cnn_fp16.pth',
                        help='Path to FP16 checkpoint')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'],
                        help='Device for inference')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark without image')
    args = parser.parse_args()

    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        print("Make sure you have the FP16 model in mamba_fp16/ folder")
        return

    # Load model
    print(f"Loading FP16 model from: {args.checkpoint}")
    model = load_model(args.checkpoint, args.device)
    print(f"Model loaded on: {args.device}")

    # Get model size
    model_size = os.path.getsize(args.checkpoint) / 1024
    print(f"Model size: {model_size:.1f} KB")

    if args.benchmark or args.image is None:
        # Benchmark with dummy input
        print("\nRunning benchmark with dummy input...")
        dummy_input = torch.randn(1, 3, 128, 128).to(args.device)

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(100):
                start = time.perf_counter()
                _ = model(dummy_input)
                end = time.perf_counter()
                times.append((end - start) * 1000)

        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        print(f"\nBenchmark Results (100 runs):")
        print(f"  Average: {avg_time:.2f} ms")
        print(f"  Min: {min_time:.2f} ms")
        print(f"  Max: {max_time:.2f} ms")
        return

    # Run inference on image
    if not os.path.exists(args.image):
        print(f"Error: Image not found at {args.image}")
        return

    print(f"\nRunning inference on: {args.image}")
    result = predict(model, args.image, args.device)

    print("\n" + "=" * 60)
    print("Rice Disease Classification (FP16 Model)")
    print("=" * 60)
    print(f"Prediction:     {result['class']}")
    print(f"Confidence:     {result['confidence']:.2f}%")
    print(f"Inference Time: {result['inference_time_ms']:.2f} ms")
    print("\nAll Class Probabilities:")
    print("-" * 40)
    sorted_probs = sorted(result['all_probs'].items(), key=lambda x: x[1], reverse=True)
    for cls, prob in sorted_probs:
        bar = '#' * int(prob / 5)
        print(f"  {cls:25s} {prob:5.2f}% {bar}")
    print("=" * 60)


if __name__ == "__main__":
    main()
