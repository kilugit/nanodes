import argparse
import os
import time
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from dataset import DenoisingDataset
from export import export_int8_quantization, export_onnx
from losses import PSNRLoss, Stage1Loss
from metrics import calculate_psnr, calculate_ssim
from models import RepAFDenoiseNet, test_reparameterization_equivalence


def get_device() -> torch.device:
    try:
        import torch_xla.core.xla_model as xm
        return xm.xla_device()
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            _ = torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except Exception as e:
            print(f"CUDA initialization failed ({e}), falling back to CPU.")
    return torch.device("cpu")


def get_progressive_patch_size(epoch: int, total_epochs: int, min_size: int = 128, max_size: int = 256) -> int:
    if total_epochs <= 1:
        return max_size
    alpha = min(1.0, max(0.0, epoch / (total_epochs - 1)))
    raw_size = min_size + alpha * (max_size - min_size)
    return int(round(raw_size / 32.0) * 32)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    bf16: bool = False,
) -> float:
    model.train()
    running_loss = 0.0
    valid_count = 0
    is_xla = (device.type == "xla")

    device_loader = loader
    if is_xla:
        try:
            import torch_xla.distributed.parallel_loader as pl
            device_loader = pl.MpDeviceLoader(loader, device)
        except Exception:
            device_loader = loader

    for noisy, clean in device_loader:
        if not is_xla or noisy.device != device:
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
            pred = model(noisy)
            loss = criterion(pred, clean)
        if not torch.isfinite(loss) or loss.abs().item() > 100.0:
            continue
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if torch.isfinite(grad_norm):
            if is_xla:
                import torch_xla.core.xla_model as xm
                xm.optimizer_step(optimizer)
            else:
                optimizer.step()
            running_loss += loss.item() * noisy.size(0)
            valid_count += noisy.size(0)
        else:
            optimizer.zero_grad(set_to_none=True)

    return running_loss / valid_count if valid_count > 0 else float("nan")


@torch.inference_mode()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, bf16: bool = False
) -> Tuple[float, float]:
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0
    is_xla = (device.type == "xla")

    eval_loader = loader
    if is_xla:
        try:
            import torch_xla.distributed.parallel_loader as pl
            eval_loader = pl.MpDeviceLoader(loader, device)
        except Exception:
            eval_loader = loader

    for noisy, clean in eval_loader:
        if not is_xla or noisy.device != device:
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
            pred = torch.clamp(model(noisy), 0.0, 1.0)
        if not torch.isfinite(pred).all():
            continue
        if is_xla:
            import torch_xla.core.xla_model as xm
            xm.mark_step()

        total_psnr += calculate_psnr(pred, clean) * noisy.size(0)
        total_ssim += calculate_ssim(pred, clean) * noisy.size(0)
        count += noisy.size(0)

    mean_psnr = total_psnr / count if count > 0 else 0.0
    mean_ssim = total_ssim / count if count > 0 else 0.0
    return mean_psnr, mean_ssim


def save_checkpoint(
    state: Dict, is_best: bool, save_dir: str = "checkpoints", filename: str = "checkpoint.pth"
):
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)

    # Detach and convert tensors to CPU for universal cross-platform checkpoint compatibility
    cpu_state = {}
    for k, v in state.items():
        if k == "state_dict" and isinstance(v, dict):
            cpu_state[k] = {pk: pv.cpu() if torch.is_tensor(pv) else pv for pk, pv in v.items()}
        elif torch.is_tensor(v):
            cpu_state[k] = v.cpu()
        else:
            cpu_state[k] = v

    try:
        import torch_xla.core.xla_model as xm
        xm.save(cpu_state, filepath)
        if is_best:
            xm.save(cpu_state, os.path.join(save_dir, "best_model.pth"))
    except Exception:
        torch.save(cpu_state, filepath)
        if is_best:
            torch.save(cpu_state, os.path.join(save_dir, "best_model.pth"))


def run_pipeline(args):
    device = get_device()
    if device.type == "cuda" and getattr(torch.version, "hip", None) is None:
        torch.backends.cudnn.benchmark = True
    dev_name = "Cloud TPU v6e-1 (Trillium)" if device.type == "xla" else (torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU")
    precision_tag = " [BF16]" if args.bf16 else ""
    print(f"RepAF-Denoise Net Training Pipeline on {device} ({dev_name}){precision_tag}")

    model = RepAFDenoiseNet(c=40).to(device)

    train_dataset = DenoisingDataset(
        noisy_dir=args.train_noisy_dir,
        clean_dir=args.train_clean_dir,
        patch_size=128,
        is_train=True,
        num_synthetic_samples=args.synthetic_samples,
        cache=True,
    )
    val_dataset = DenoisingDataset(
        noisy_dir=args.val_noisy_dir,
        clean_dir=args.val_clean_dir,
        patch_size=256,
        is_train=False,
        num_synthetic_samples=max(16, args.synthetic_samples // 4),
        cache=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size_val,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    best_psnr = -float("inf")

    # Stage 1: Coarse Training with Progressive Patch Sizes
    print(f"\nStage 1: {args.stage1_epochs} epochs | Batch Size: {args.batch_size_stage1}")
    optimizer_s1 = AdamW(model.parameters(), lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler_s1 = CosineAnnealingLR(optimizer_s1, T_max=args.stage1_epochs, eta_min=1e-6)
    criterion_s1 = Stage1Loss(eps=1e-3, lambda_grad=0.05).to(device)

    train_loader_s1 = DataLoader(
        train_dataset,
        batch_size=args.batch_size_stage1,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=(len(train_dataset) >= args.batch_size_stage1),
    )

    for epoch in range(1, args.stage1_epochs + 1):
        t0 = time.time()
        patch_size = get_progressive_patch_size(epoch - 1, args.stage1_epochs, min_size=128, max_size=256)
        train_dataset.set_patch_size(patch_size)

        loss = train_one_epoch(model, train_loader_s1, criterion_s1, optimizer_s1, device, bf16=args.bf16)
        scheduler_s1.step()

        val_psnr, val_ssim = evaluate(model, val_loader, device, bf16=args.bf16)
        is_best = val_psnr > best_psnr
        if is_best:
            best_psnr = val_psnr

        save_checkpoint(
            {
                "epoch": epoch,
                "stage": 1,
                "state_dict": model.state_dict(),
                "best_psnr": best_psnr,
                "optimizer": optimizer_s1.state_dict(),
            },
            is_best=is_best,
            save_dir=args.checkpoint_dir,
            filename="stage1_last.pth",
        )

        elapsed = time.time() - t0
        lr_curr = optimizer_s1.param_groups[0]["lr"]
        print(
            f"Stage 1 [Epoch {epoch:03d}/{args.stage1_epochs:03d}] Patch: {patch_size:03d}x{patch_size:03d} | "
            f"LR: {lr_curr:.6f} | Loss: {loss:.4f} | Val PSNR: {val_psnr:.2f} dB | Val SSIM: {val_ssim:.4f} "
            f"({'*BEST*' if is_best else ''}) [{elapsed:.1f}s]"
        )

    del optimizer_s1, scheduler_s1, criterion_s1, train_loader_s1
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "xla":
        import torch_xla.core.xla_model as xm
        xm.mark_step()

    best_ckpt = os.path.join(args.checkpoint_dir, "best_model.pth")
    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        best_psnr = ckpt.get("best_psnr", best_psnr)

    # Stage 2: Fine-Tuning with 512x512 Patches
    print(f"\nStage 2: {args.stage2_epochs} epochs | Batch Size: {args.batch_size_stage2}")
    train_dataset.set_patch_size(512)
    train_loader_s2 = DataLoader(
        train_dataset,
        batch_size=args.batch_size_stage2,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=(len(train_dataset) >= args.batch_size_stage2),
    )

    optimizer_s2 = AdamW(model.parameters(), lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler_s2 = CosineAnnealingLR(optimizer_s2, T_max=args.stage2_epochs, eta_min=1e-6)
    criterion_s2 = PSNRLoss(data_range=1.0, eps=1e-6).to(device)

    for epoch in range(1, args.stage2_epochs + 1):
        t0 = time.time()
        loss = train_one_epoch(model, train_loader_s2, criterion_s2, optimizer_s2, device, bf16=args.bf16)
        scheduler_s2.step()

        val_psnr, val_ssim = evaluate(model, val_loader, device, bf16=args.bf16)
        is_best = val_psnr > best_psnr
        if is_best:
            best_psnr = val_psnr

        save_checkpoint(
            {
                "epoch": epoch,
                "stage": 2,
                "state_dict": model.state_dict(),
                "best_psnr": best_psnr,
                "optimizer": optimizer_s2.state_dict(),
            },
            is_best=is_best,
            save_dir=args.checkpoint_dir,
            filename="stage2_last.pth",
        )

        elapsed = time.time() - t0
        lr_curr = optimizer_s2.param_groups[0]["lr"]
        print(
            f"Stage 2 [Epoch {epoch:02d}/{args.stage2_epochs:02d}] Patch: 512x512 | "
            f"LR: {lr_curr:.6f} | PSNR Loss: {loss:.4f} | Val PSNR: {val_psnr:.2f} dB | Val SSIM: {val_ssim:.4f} "
            f"({'*BEST*' if is_best else ''}) [{elapsed:.1f}s]"
        )

    # Re-parameterization & Deployment Export
    print("\nVerifying re-parameterization numerical equivalence...")
    max_err = test_reparameterization_equivalence(device=device.type, tol=1e-5)
    print(f"Max numerical error: {max_err:.8e}")

    best_ckpt = os.path.join(args.checkpoint_dir, "best_model.pth")
    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        print(f"Loaded best checkpoint (PSNR: {ckpt.get('best_psnr', 0.0):.2f} dB)")

    model.switch_to_deploy()
    deployed_ckpt = os.path.join(args.checkpoint_dir, "repaf_denoise_net_deployed.pth")
    torch.save(model.state_dict(), deployed_ckpt)
    print(f"Deployed checkpoint saved: {deployed_ckpt}")

    onnx_path = os.path.join(args.export_dir, "repaf_denoise_net.onnx")
    export_onnx(model, save_path=onnx_path, input_shape=(1, 3, 256, 256))

    int8_path = os.path.join(args.export_dir, "repaf_denoise_net_int8.pth")
    export_int8_quantization(model, save_path=int8_path, calibration_loader=val_loader)

    from dataset import clear_image_cache
    clear_image_cache()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("Training and export complete.")


def main():
    parser = argparse.ArgumentParser(description="RepAF-Denoise Net Training and Export Pipeline")
    parser.add_argument("--train-noisy-dir", type=str, default=None, help="Training noisy images directory")
    parser.add_argument("--train-clean-dir", type=str, default=None, help="Training clean images directory")
    parser.add_argument("--val-noisy-dir", type=str, default=None, help="Validation noisy images directory")
    parser.add_argument("--val-clean-dir", type=str, default=None, help="Validation clean images directory")
    parser.add_argument("--stage1-epochs", type=int, default=100, help="Epochs for Stage 1")
    parser.add_argument("--stage2-epochs", type=int, default=30, help="Epochs for Stage 2")
    parser.add_argument("--batch-size-stage1", type=int, default=32, help="Batch size for Stage 1")
    parser.add_argument("--batch-size-stage2", type=int, default=8, help="Batch size for Stage 2")
    parser.add_argument("--batch-size-val", type=int, default=4, help="Batch size for validation")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader num_workers")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--export-dir", type=str, default="exports", help="Directory to export models")
    parser.add_argument("--synthetic-samples", type=int, default=128, help="Synthetic samples if no dataset provided")
    parser.add_argument("--bf16", action="store_true", help="Enable BF16 mixed precision training")
    parser.add_argument("--dry-run", action="store_true", help="Run 1-epoch dry run")

    args = parser.parse_args()

    if args.dry_run:
        args.stage1_epochs = 1
        args.stage2_epochs = 1
        args.batch_size_stage1 = 4
        args.batch_size_stage2 = 2
        args.synthetic_samples = 8

    run_pipeline(args)


if __name__ == "__main__":
    main()
