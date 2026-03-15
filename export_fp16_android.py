"""
Export MambaCNN FP16 model to ONNX for Android.
Converts FP16 weights to FP32 for ONNX Runtime Android compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def main():
    # Load FP16 model
    model_path = "mamba_fp16/mamba_cnn_fp16.pth"
    print(f"Loading FP16 model from: {model_path}")

    model = MambaCNN(num_classes=6, d_model=64, n_mamba=2)
    state_dict = torch.load(model_path, map_location='cpu')

    # Convert FP16 weights to FP32
    state_dict_fp32 = {k: v.float() if v.dtype == torch.float16 else v for k, v in state_dict.items()}
    model.load_state_dict(state_dict_fp32)
    model.eval()

    print("Model loaded and converted to FP32 for ONNX export")

    # Export to ONNX with opset 14 (IR version 7, compatible with Android)
    dummy_input = torch.randn(1, 3, 128, 128)
    output_path = "RiceDiseaseClassifier/app/src/main/assets/mamba_fp16.onnx"

    print(f"Exporting to ONNX with opset 14...")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=14,
        do_constant_folding=True,
        dynamo=False,
    )

    print(f"Model exported to: {output_path}")

    # Verify
    import onnx
    onnx_model = onnx.load(output_path)
    print(f"ONNX IR version: {onnx_model.ir_version}")
    print(f"ONNX opset version: {onnx_model.opset_import[0].version}")

    # Test inference
    import onnxruntime as ort
    session = ort.InferenceSession(output_path)
    test_input = dummy_input.numpy()
    output = session.run(None, {"input": test_input})
    print(f"Test inference output shape: {output[0].shape}")
    print("Export successful!")

    # Show file size
    import os
    size_kb = os.path.getsize(output_path) / 1024
    print(f"ONNX file size: {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
