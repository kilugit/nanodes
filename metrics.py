import torch
import torch.nn.functional as F

_WINDOW_CACHE = {}


def clear_metrics_cache():
    """Clears cached Gaussian windows to free memory."""
    _WINDOW_CACHE.clear()


def _get_gaussian_window(
    window_size: int, sigma: float, channels: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    key = (window_size, sigma, channels, str(device), dtype)
    window = _WINDOW_CACHE.get(key)
    if window is None:
        coords = torch.arange(window_size, dtype=torch.float32) - (window_size - 1) / 2.0
        gauss = torch.exp(-(coords**2) / (2 * sigma**2))
        kernel_1d = (gauss / gauss.sum()).unsqueeze(1)
        kernel_2d = kernel_1d.mm(kernel_1d.t()).unsqueeze(0).unsqueeze(0)
        window = kernel_2d.repeat(channels, 1, 1, 1).to(device=device, dtype=dtype)
        _WINDOW_CACHE[key] = window
    return window


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor, data_range: float = 1.0) -> float:
    diff = img1 - img2
    mse = torch.mean(diff.square())
    if mse == 0:
        return float("inf")
    return (10.0 * torch.log10((data_range**2) / mse)).item()


def calculate_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> float:
    channels = img1.size(1)
    window = _get_gaussian_window(window_size, sigma, channels, img1.device, img1.dtype)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channels)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channels) - mu1_mu2

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean().item()
