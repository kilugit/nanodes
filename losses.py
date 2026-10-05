import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps2 = eps**2

    def forward(self, pred: torch.Tensor, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        diff = pred - target if target is not None else pred
        return torch.mean(torch.sqrt(diff.square() + self.eps2))


class SpatialGradientLoss(nn.Module):
    def __init__(self, lambda_grad: float = 0.05, channels: int = 3):
        super().__init__()
        self.lambda_grad = lambda_grad
        self.channels = channels

        sx = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        sy = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        sobel = torch.cat([torch.stack([sx, sy], dim=1).view(2, 1, 3, 3)] * channels, dim=0)
        self.register_buffer("sobel", sobel)

    def forward(self, pred: torch.Tensor, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        diff = pred - target if target is not None else pred
        g = F.conv2d(diff, self.sobel, padding=1, groups=self.channels)
        return (self.lambda_grad * 2.0) * g.abs().mean()


class Stage1Loss(nn.Module):
    def __init__(self, eps: float = 1e-3, lambda_grad: float = 0.05, channels: int = 3):
        super().__init__()
        self.charbonnier = CharbonnierLoss(eps=eps)
        self.gradient = SpatialGradientLoss(lambda_grad=lambda_grad, channels=channels)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return self.charbonnier(diff, None) + self.gradient(diff, None)


class PSNRLoss(nn.Module):
    def __init__(self, data_range: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.scale = data_range**2
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        mse = torch.mean(diff.square())
        return -10.0 * torch.log10(self.scale / (mse + self.eps))
