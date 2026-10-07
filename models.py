import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


def haar_dwt(x: torch.Tensor) -> torch.Tensor:
    """Orthogonal, parameter-free 2D Haar Discrete Wavelet Transform.
    Converts [B, C, H, W] into 4 sub-bands [B, 4*C, H/2, W/2] (LL, LH, HL, HH).
    """
    x0 = x[:, :, 0::2, 0::2]  # top-left
    x1 = x[:, :, 1::2, 0::2]  # bottom-left
    x2 = x[:, :, 0::2, 1::2]  # top-right
    x3 = x[:, :, 1::2, 1::2]  # bottom-right

    a = x0 + x2
    b = x1 + x3
    c = x0 - x2
    d = x1 - x3

    ll = 0.5 * (a + b)
    lh = 0.5 * (a - b)
    hl = 0.5 * (c + d)
    hh = 0.5 * (c - d)
    return torch.cat([ll, lh, hl, hh], dim=1)


def haar_iwt(x: torch.Tensor) -> torch.Tensor:
    """Orthogonal, parameter-free 2D Inverse Haar Discrete Wavelet Transform.
    Reconstructs [B, C, H, W] from sub-bands [B, 4*C, H/2, W/2].
    """
    c = x.shape[1] // 4
    ll = x[:, 0 * c : 1 * c, :, :]
    lh = x[:, 1 * c : 2 * c, :, :]
    hl = x[:, 2 * c : 3 * c, :, :]
    hh = x[:, 3 * c : 4 * c, :, :]

    p = ll + hl
    q = lh + hh
    r = ll - hl
    s = lh - hh

    x0 = 0.5 * (p + q)
    x1 = 0.5 * (p - q)
    x2 = 0.5 * (r + s)
    x3 = 0.5 * (r - s)

    b, _, h2, w2 = ll.shape
    out = torch.empty(b, c, h2 * 2, w2 * 2, dtype=x.dtype, device=x.device)
    out[:, :, 0::2, 0::2] = x0
    out[:, :, 1::2, 0::2] = x1
    out[:, :, 0::2, 1::2] = x2
    out[:, :, 1::2, 1::2] = x3
    return out


class RepConv2d(nn.Module):
    """Re-parameterizable 3x3 Conv2d layer without BatchNorm.
    During training: multi-branch (3x3 conv + 1x1 conv + identity if in==out).
    During deploy: fused analytically into a single Conv2d(in_ch, out_ch, 3, padding=1).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.is_deployed = False

        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=True)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=True)

    def get_equivalent_kernel_bias(self):
        pad_size = self.kernel_size // 2
        w_1x1 = F.pad(self.conv1x1.weight, (pad_size, pad_size, pad_size, pad_size))
        k_rep = self.conv3x3.weight + w_1x1
        b_rep = self.conv3x3.bias + self.conv1x1.bias

        if self.in_channels == self.out_channels:
            w_id = torch.zeros_like(self.conv3x3.weight)
            for i in range(self.in_channels):
                w_id[i, i, pad_size, pad_size] = 1.0
            k_rep = k_rep + w_id

        return k_rep, b_rep

    def switch_to_deploy(self):
        if self.is_deployed:
            return
        k_rep, b_rep = self.get_equivalent_kernel_bias()
        self.conv_deploy = nn.Conv2d(
            self.in_channels, self.out_channels, self.kernel_size,
            padding=self.padding, bias=True, device=k_rep.device, dtype=k_rep.dtype
        )
        self.conv_deploy.weight.data.copy_(k_rep)
        self.conv_deploy.bias.data.copy_(b_rep)

        del self.conv3x3
        del self.conv1x1
        self.is_deployed = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_deployed:
            return self.conv_deploy(x)
        out = self.conv3x3(x) + self.conv1x1(x)
        if self.in_channels == self.out_channels:
            out = out + x
        return out


class DepthwiseECB(nn.Module):
    """Depthwise Edge-oriented Convolution Block (ECB) with analytical fusion."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2
        self.is_deployed = False

        self.conv_normal = nn.Conv2d(
            channels, channels, kernel_size, padding=self.pad, groups=channels, bias=True
        )
        self.conv_expand = nn.Conv2d(channels, channels, 1, padding=0, groups=channels, bias=True)
        self.conv_squeeze = nn.Conv2d(
            channels, channels, kernel_size, padding=self.pad, groups=channels, bias=True
        )

        dx = torch.zeros(1, 1, kernel_size, kernel_size)
        dy = torch.zeros(1, 1, kernel_size, kernel_size)
        lap = torch.zeros(1, 1, kernel_size, kernel_size)

        if kernel_size == 3:
            dx[0, 0] = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
            dy[0, 0] = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
            lap[0, 0] = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
        elif kernel_size == 5:
            dx[0, 0, 0, 0] = -1.0
            dx[0, 0, 0, 4] = 1.0
            dx[0, 0, 2, 0] = -2.0
            dx[0, 0, 2, 4] = 2.0
            dx[0, 0, 4, 0] = -1.0
            dx[0, 0, 4, 4] = 1.0

            dy[0, 0, 0, 0] = -1.0
            dy[0, 0, 0, 2] = -2.0
            dy[0, 0, 0, 4] = -1.0
            dy[0, 0, 4, 0] = 1.0
            dy[0, 0, 4, 2] = 2.0
            dy[0, 0, 4, 4] = 1.0

            lap[0, 0, 0, 2] = 1.0
            lap[0, 0, 2, 0] = 1.0
            lap[0, 0, 2, 2] = -4.0
            lap[0, 0, 2, 4] = 1.0
            lap[0, 0, 4, 2] = 1.0
        else:
            raise ValueError(f"Unsupported kernel_size: {kernel_size}")

        self.register_buffer("mask_dx", dx.repeat(channels, 1, 1, 1))
        self.register_buffer("mask_dy", dy.repeat(channels, 1, 1, 1))
        self.register_buffer("mask_lap", lap.repeat(channels, 1, 1, 1))

        self.scale_dx = nn.Parameter(torch.randn(channels, 1, 1, 1) * 1e-3)
        self.bias_dx = nn.Parameter(torch.zeros(channels))

        self.scale_dy = nn.Parameter(torch.randn(channels, 1, 1, 1) * 1e-3)
        self.bias_dy = nn.Parameter(torch.zeros(channels))

        self.scale_lap = nn.Parameter(torch.randn(channels, 1, 1, 1) * 1e-3)
        self.bias_lap = nn.Parameter(torch.zeros(channels))

    def get_equivalent_kernel_bias(self):
        k_normal = self.conv_normal.weight
        b_normal = self.conv_normal.bias

        k_seq = self.conv_squeeze.weight * self.conv_expand.weight
        b_seq = self.conv_squeeze.bias + self.conv_squeeze.weight.sum(dim=(1, 2, 3)) * self.conv_expand.bias

        k_dx = self.scale_dx * self.mask_dx
        b_dx = self.bias_dx

        k_dy = self.scale_dy * self.mask_dy
        b_dy = self.bias_dy

        k_lap = self.scale_lap * self.mask_lap
        b_lap = self.bias_lap

        k_rep = k_normal + k_seq + k_dx + k_dy + k_lap
        b_rep = b_normal + b_seq + b_dx + b_dy + b_lap
        return k_rep, b_rep

    def switch_to_deploy(self):
        if self.is_deployed:
            return
        k_rep, b_rep = self.get_equivalent_kernel_bias()
        self.conv_deploy = nn.Conv2d(
            self.channels, self.channels, self.kernel_size,
            padding=self.pad, groups=self.channels, bias=True,
            device=k_rep.device, dtype=k_rep.dtype
        )
        self.conv_deploy.weight.data.copy_(k_rep)
        self.conv_deploy.bias.data.copy_(b_rep)

        del self.conv_normal, self.conv_expand, self.conv_squeeze
        del self.scale_dx, self.bias_dx, self.mask_dx
        del self.scale_dy, self.bias_dy, self.mask_dy
        del self.scale_lap, self.bias_lap, self.mask_lap
        self.is_deployed = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_deployed:
            return self.conv_deploy(x)
        k_rep, b_rep = self.get_equivalent_kernel_bias()
        return F.conv2d(x, k_rep, b_rep, padding=self.pad, groups=self.channels)


class RepAFB(nn.Module):
    """Re-parameterized Asymmetric Frequency Block."""

    def __init__(self, c: int = 40):
        super().__init__()
        self.c = c
        self.c_lf = 10
        self.c_hf = 30

        self.lf_ecb = DepthwiseECB(self.c_lf, kernel_size=5)
        self.hf_ecb = DepthwiseECB(self.c_hf, kernel_size=3)
        self.proj_mod = nn.Conv2d(self.c_lf, self.c_lf, kernel_size=1, bias=True)

        self.expand_conv = nn.Conv2d(c, 80, kernel_size=1, bias=True)
        self.sca_conv = nn.Conv2d(40, 40, kernel_size=1, bias=True)
        self.proj_out = nn.Conv2d(40, 40, kernel_size=1, bias=True)

    def switch_to_deploy(self):
        self.lf_ecb.switch_to_deploy()
        self.hf_ecb.switch_to_deploy()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        f_lf = x[:, : self.c_lf, :, :]
        f_hf = x[:, self.c_lf :, :, :]

        f_lf = self.lf_ecb(f_lf)
        f_hf = self.hf_ecb(f_hf)

        m_lf = torch.tanh(self.proj_mod(f_lf))
        b, _, h, w = f_hf.shape
        f_hf_mod = (f_hf.reshape(b, 3, self.c_lf, h, w) * m_lf.unsqueeze(1)).flatten(1, 2)

        f_cat = torch.cat([f_lf, f_hf_mod], dim=1)

        x_gate = self.expand_conv(f_cat)
        x1, x2 = x_gate.chunk(2, dim=1)
        x_gate = x1 * x2

        attn = x_gate.mean(dim=(2, 3), keepdim=True)
        attn = torch.sigmoid(self.sca_conv(attn))
        x_sca = x_gate * attn

        out = self.proj_out(x_sca)
        return shortcut + out


class RepAFDenoiseNet(nn.Module):
    """RepAF Denoising Network with wavelet transforms and re-parameterized blocks."""

    def __init__(self, c: int = 40):
        super().__init__()
        self.c = c
        self.stem = RepConv2d(12, c, kernel_size=3, padding=1)

        self.stage1 = nn.Sequential(*[RepAFB(c) for _ in range(2)])
        self.stage2 = nn.Sequential(*[RepAFB(c) for _ in range(3)])
        self.stage3 = nn.Sequential(*[RepAFB(c) for _ in range(2)])

        self.head = RepConv2d(c, 12, kernel_size=3, padding=1)
        self.is_deployed = False

    def switch_to_deploy(self):
        if self.is_deployed:
            return
        self.stem.switch_to_deploy()
        for stage in (self.stage1, self.stage2, self.stage3):
            for block in stage:
                block.switch_to_deploy()
        self.head.switch_to_deploy()
        self.is_deployed = True

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        _, _, h, w = y.shape
        is_static = isinstance(h, int) and isinstance(w, int)
        pad_h = (2 - h % 2) % 2 if is_static else 0
        pad_w = (2 - w % 2) % 2 if is_static else 0

        if is_static and (pad_h > 0 or pad_w > 0):
            y_in = F.pad(y, (0, pad_w, 0, pad_h), mode="reflect")
        else:
            y_in = y

        feat = self.stem(haar_dwt(y_in))
        feat = self.stage3(self.stage2(self.stage1(feat)))
        n_hat = haar_iwt(self.head(feat))

        if is_static and (pad_h > 0 or pad_w > 0):
            n_hat = n_hat[:, :, :h, :w]

        return y - n_hat


def test_reparameterization_equivalence(device: str = "cpu", tol: float = 1e-5) -> float:
    """Unit test checking that the numerical difference between training-mode output
    and fused inference-mode output is negligible (max absolute error < 1e-5).
    """
    model = RepAFDenoiseNet().to(device)
    x = torch.randn(2, 3, 64, 64, device=device)

    with torch.no_grad():
        model.train()
        out_train = model(x)
        model.eval()
        model.switch_to_deploy()
        out_deploy = model(x)
        max_diff = (out_train - out_deploy).abs().max().item()

    assert max_diff < tol, f"Equivalence test failed: max diff = {max_diff:.8e} >= {tol}"
    return max_diff


def load_pretrained_weights(model: nn.Module, checkpoint_path: str, device: Optional[torch.device] = None) -> dict:
    ckpt = torch.load(checkpoint_path, map_location=device or "cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    if isinstance(ckpt, dict) and ckpt.get("quantized_format") == "symmetric_int8_per_channel":
        dequant = {}
        for k, v in state.items():
            if k.endswith("_scale"):
                continue
            dequant[k] = v.float() * state[f"{k}_scale"] if f"{k}_scale" in state else v
        state = dequant

    if any(".conv_deploy." in k for k in state.keys()):
        train_state = model.state_dict()
        for k, v in state.items():
            if "stem.conv_deploy" in k or "head.conv_deploy" in k:
                prefix, attr = k.rsplit(".conv_deploy.", 1)
                if attr == "bias":
                    train_state[f"{prefix}.conv3x3.bias"] = v.clone()
                    train_state[f"{prefix}.conv1x1.bias"] = torch.zeros_like(train_state[f"{prefix}.conv1x1.bias"])
                elif attr == "weight":
                    w = v.clone()
                    in_ch, out_ch = w.shape[1], w.shape[0]
                    if in_ch == out_ch:
                        pad_size = 3 // 2
                        w_id = torch.zeros_like(w)
                        for i in range(in_ch):
                            w_id[i, i, pad_size, pad_size] = 1.0
                        w = w - w_id
                    train_state[f"{prefix}.conv3x3.weight"] = w
                    train_state[f"{prefix}.conv1x1.weight"] = torch.zeros_like(train_state[f"{prefix}.conv1x1.weight"])
            elif ".conv_deploy." in k:
                prefix, attr = k.rsplit(".conv_deploy.", 1)
                if attr == "bias":
                    train_state[f"{prefix}.conv_normal.bias"] = v.clone()
                    train_state[f"{prefix}.conv_expand.bias"] = torch.zeros_like(train_state[f"{prefix}.conv_expand.bias"])
                    train_state[f"{prefix}.conv_squeeze.bias"] = torch.zeros_like(train_state[f"{prefix}.conv_squeeze.bias"])
                    train_state[f"{prefix}.bias_dx"] = torch.zeros_like(train_state[f"{prefix}.bias_dx"])
                    train_state[f"{prefix}.bias_dy"] = torch.zeros_like(train_state[f"{prefix}.bias_dy"])
                    train_state[f"{prefix}.bias_lap"] = torch.zeros_like(train_state[f"{prefix}.bias_lap"])
                elif attr == "weight":
                    train_state[f"{prefix}.conv_normal.weight"] = v.clone()
                    train_state[f"{prefix}.conv_expand.weight"] = torch.zeros_like(train_state[f"{prefix}.conv_expand.weight"])
                    train_state[f"{prefix}.conv_squeeze.weight"] = torch.zeros_like(train_state[f"{prefix}.conv_squeeze.weight"])
                    train_state[f"{prefix}.scale_dx"] = torch.zeros_like(train_state[f"{prefix}.scale_dx"])
                    train_state[f"{prefix}.scale_dy"] = torch.zeros_like(train_state[f"{prefix}.scale_dy"])
                    train_state[f"{prefix}.scale_lap"] = torch.zeros_like(train_state[f"{prefix}.scale_lap"])
            elif k in train_state:
                train_state[k] = v.clone()
        state = train_state

    model.load_state_dict(state, strict=False)
    return ckpt if isinstance(ckpt, dict) else {"state_dict": ckpt}


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running RepAF-Denoise Net reparameterization unit test on {dev}...")
    err = test_reparameterization_equivalence(device=dev)
    print(f"PASSED! Max absolute error: {err:.8e} (threshold: 1e-5)")
