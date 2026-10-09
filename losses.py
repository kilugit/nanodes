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


class LaplacianLoss(nn.Module):
    def __init__(self, lambda_lap: float = 0.02, channels: int = 3):
        super().__init__()
        self.lambda_lap = lambda_lap
        self.channels = channels
        lap = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("lap", lap.repeat(channels, 1, 1, 1))

    def forward(self, pred: torch.Tensor, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        diff = pred - target if target is not None else pred
        curv = F.conv2d(diff, self.lap, padding=1, groups=self.channels)
        return self.lambda_lap * curv.abs().mean()


class FFTLoss(nn.Module):
    def __init__(self, lambda_fft: float = 0.05):
        super().__init__()
        self.lambda_fft = lambda_fft

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_fft = torch.fft.rfft2(pred.float(), norm="ortho")
        target_fft = torch.fft.rfft2(target.float(), norm="ortho")
        loss = F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))
        return self.lambda_fft * loss.to(pred.dtype)


class SSIMLoss(nn.Module):
    def __init__(self, lambda_ssim: float = 0.05):
        super().__init__()
        self.lambda_ssim = lambda_ssim

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        from metrics import calculate_ssim
        return self.lambda_ssim * (1.0 - calculate_ssim(pred, target, as_tensor=True))


class Stage1Loss(nn.Module):
    def __init__(
        self,
        eps: float = 1e-3,
        lambda_grad: float = 0.05,
        lambda_lap: float = 0.02,
        lambda_fft: float = 0.0,
        lambda_ssim: float = 0.0,
        channels: int = 3,
    ):
        super().__init__()
        self.charbonnier = CharbonnierLoss(eps=eps)
        self.gradient = SpatialGradientLoss(lambda_grad=lambda_grad, channels=channels)
        self.laplacian = LaplacianLoss(lambda_lap=lambda_lap, channels=channels)
        self.fft = FFTLoss(lambda_fft=lambda_fft) if lambda_fft > 0 else None
        self.ssim = SSIMLoss(lambda_ssim=lambda_ssim) if lambda_ssim > 0 else None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = self.charbonnier(diff) + self.gradient(diff) + self.laplacian(diff)
        if self.fft is not None:
            loss = loss + self.fft(pred, target)
        if self.ssim is not None:
            loss = loss + self.ssim(pred, target)
        return loss


class PSNRLoss(nn.Module):
    def __init__(self, data_range: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.scale = data_range**2
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        mse = torch.mean(diff.square())
        return -10.0 * torch.log10(self.scale / (mse + self.eps))


class Stage2SharpLoss(nn.Module):
    def __init__(
        self,
        data_range: float = 1.0,
        eps: float = 1e-6,
        lambda_grad: float = 0.05,
        lambda_lap: float = 0.02,
        lambda_fft: float = 0.0,
        lambda_ssim: float = 0.0,
        channels: int = 3,
    ):
        super().__init__()
        self.psnr = PSNRLoss(data_range=data_range, eps=eps)
        self.gradient = SpatialGradientLoss(lambda_grad=lambda_grad, channels=channels)
        self.laplacian = LaplacianLoss(lambda_lap=lambda_lap, channels=channels)
        self.fft = FFTLoss(lambda_fft=lambda_fft) if lambda_fft > 0 else None
        self.ssim = SSIMLoss(lambda_ssim=lambda_ssim) if lambda_ssim > 0 else None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = self.psnr(pred, target) + self.gradient(diff) + self.laplacian(diff)
        if self.fft is not None:
            loss = loss + self.fft(pred, target)
        if self.ssim is not None:
            loss = loss + self.ssim(pred, target)
        return loss
