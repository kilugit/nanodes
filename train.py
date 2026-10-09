import argparse
import copy
import os
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from dataset import DenoisingDataset
from devices import clear_device_cache, create_grad_scaler, get_amp_context, get_device as resolve_device, get_device_name
from export import export_int8_quantization, export_onnx
from losses import PSNRLoss, Stage1Loss, Stage2SharpLoss
from metrics import calculate_psnr, calculate_ssim
from models import RepAFDenoiseNet, load_pretrained_weights, test_reparameterization_equivalence


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad = False
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module):
        for ep, mp in zip(self.module.parameters(), model.parameters()):
            ep.data.lerp_(mp.data, 1.0 - self.decay)

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state_dict):
        self.module.load_state_dict(state_dict)


def get_device(hint: str = "auto") -> torch.device:
    return resolve_device(hint)


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
    fp16: bool = False,
    scaler: Optional[torch.amp.GradScaler] = None,
    ema: Optional[ModelEMA] = None,
    accum_steps: int = 1,
    channels_last: bool = False,
) -> float:
    model.train()
    running_loss = 0.0
    valid_count = 0
    is_xla = (device.type == "xla")
    amp_dtype = torch.float16 if fp16 else (torch.bfloat16 if bf16 else torch.float32)
    amp_enabled = (fp16 or bf16) and device.type in ("cuda", "xpu", "cpu")

    device_loader = loader
    if is_xla:
        try:
            import torch_xla.distributed.parallel_loader as pl
            device_loader = pl.MpDeviceLoader(loader, device)
        except Exception:
            device_loader = loader

    total_steps = len(loader)
    optimizer.zero_grad(set_to_none=True)

    for step, (noisy, clean) in enumerate(device_loader):
        if not is_xla or noisy.device != device:
            if noisy.dtype == torch.uint8:
                target_dtype = amp_dtype if amp_enabled else torch.float32
                noisy = noisy.to(device, dtype=target_dtype, non_blocking=True).div_(255.0)
                clean = clean.to(device, dtype=target_dtype, non_blocking=True).div_(255.0)
            else:
                noisy = noisy.to(device, non_blocking=True)
                clean = clean.to(device, non_blocking=True)
        if channels_last:
            noisy = noisy.to(memory_format=torch.channels_last)

        with get_amp_context(device, enabled=amp_enabled, dtype=amp_dtype):
            pred = model(noisy)
            loss = criterion(pred, clean)

        if not torch.isfinite(loss) or loss.abs().item() > 100.0:
            continue

        raw_loss = loss.item()
        if accum_steps > 1:
            loss = loss / accum_steps

        is_update_step = ((step + 1) % accum_steps == 0) or ((step + 1) == total_steps)

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            if is_update_step:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if torch.isfinite(grad_norm):
                    scaler.step(optimizer)
                    scaler.update()
                    if ema is not None:
                        ema.update(model)
                else:
                    scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            loss.backward()
            if is_update_step:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if torch.isfinite(grad_norm):
                    if is_xla:
                        import torch_xla.core.xla_model as xm
                        xm.optimizer_step(optimizer)
                    else:
                        optimizer.step()
                    if ema is not None:
                        ema.update(model)
                optimizer.zero_grad(set_to_none=True)

        running_loss += raw_loss * noisy.size(0)
        valid_count += noisy.size(0)

    return running_loss / valid_count if valid_count > 0 else float("nan")


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    bf16: bool = False,
    fp16: bool = False,
    channels_last: bool = False,
) -> Tuple[float, float]:
    model.eval()
    total_psnr = torch.zeros(1, device=device)
    total_ssim = torch.zeros(1, device=device)
    count = 0
    is_xla = (device.type == "xla")
    amp_dtype = torch.float16 if fp16 else (torch.bfloat16 if bf16 else torch.float32)
    amp_enabled = (fp16 or bf16) and device.type in ("cuda", "xpu", "cpu")

    eval_loader = loader
    if is_xla:
        try:
            import torch_xla.distributed.parallel_loader as pl
            eval_loader = pl.MpDeviceLoader(loader, device)
        except Exception:
            eval_loader = loader

    for noisy, clean in eval_loader:
        if not is_xla or noisy.device != device:
            if noisy.dtype == torch.uint8:
                target_dtype = amp_dtype if amp_enabled else torch.float32
                noisy = noisy.to(device, dtype=target_dtype, non_blocking=True).div_(255.0)
                clean = clean.to(device, dtype=target_dtype, non_blocking=True).div_(255.0)
            else:
                noisy = noisy.to(device, non_blocking=True)
                clean = clean.to(device, non_blocking=True)
        if channels_last:
            noisy = noisy.to(memory_format=torch.channels_last)

        with get_amp_context(device, enabled=amp_enabled, dtype=amp_dtype):
            pred = torch.clamp(model(noisy), 0.0, 1.0)
        if not torch.isfinite(pred).all():
            continue
        if is_xla:
            import torch_xla.core.xla_model as xm
            xm.mark_step()

        bs = noisy.size(0)
        total_psnr += calculate_psnr(pred, clean, as_tensor=True) * bs
        total_ssim += calculate_ssim(pred, clean, as_tensor=True) * bs
        count += bs

    mean_psnr = (total_psnr / count).item() if count > 0 else 0.0
    mean_ssim = (total_ssim / count).item() if count > 0 else 0.0
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
    device = get_device(getattr(args, "device", "auto"))
    if hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    if device.type == "cuda" and getattr(torch.version, "hip", None) is None:
        torch.backends.cudnn.benchmark = True

    if not args.fp16 and not args.bf16:
        if device.type in ("cuda", "xpu") and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            args.bf16 = True
        elif device.type in ("cuda", "xpu"):
            args.fp16 = True

    num_workers = args.num_workers
    if num_workers == 0 and not args.dry_run and os.cpu_count():
        num_workers = min(8, max(2, os.cpu_count() // 2))

    channels_last = getattr(args, "channels_last", True) and (device.type in ("cuda", "xpu"))
    dev_name = get_device_name(device)
    precision_tag = " [FP16]" if args.fp16 else (" [BF16]" if args.bf16 else "")
    cl_tag = " [Channels-Last]" if channels_last else ""
    ema_tag = f" [EMA beta={args.ema_decay}]" if getattr(args, "ema", True) else ""
    print(f"RepAF-Denoise Net Training Pipeline on {device} ({dev_name}){precision_tag}{cl_tag}{ema_tag}")

    scaler = create_grad_scaler(device, enabled=(args.fp16 and device.type in ("cuda", "xpu")))
    model = RepAFDenoiseNet(c=40).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    ema = ModelEMA(model, decay=args.ema_decay) if getattr(args, "ema", True) else None
    if ema is not None and channels_last:
        ema.module.to(memory_format=torch.channels_last)

    adv_aug = getattr(args, "advanced_augment", True)
    train_dataset = DenoisingDataset(
        noisy_dir=args.train_noisy_dir,
        clean_dir=args.train_clean_dir,
        patch_size=128,
        is_train=True,
        num_synthetic_samples=args.synthetic_samples,
        cache=True,
        preload_to_ram=args.preload_ram,
        advanced_augment=adv_aug,
        to_float=False,
    )
    val_dataset = DenoisingDataset(
        noisy_dir=args.val_noisy_dir,
        clean_dir=args.val_clean_dir,
        patch_size=256,
        is_train=False,
        num_synthetic_samples=max(16, args.synthetic_samples // 4),
        cache=True,
        preload_to_ram=args.preload_ram,
        to_float=False,
    )

    if args.dry_run:
        train_dataset.paired_files = train_dataset.paired_files[:8]
        val_dataset.paired_files = val_dataset.paired_files[:4]

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size_val,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type in ("cuda", "xpu")),
    )

    best_psnr = -float("inf")
    best_ema_psnr = -float("inf")

    if args.pretrained:
        if os.path.exists(args.pretrained):
            ckpt_info = load_pretrained_weights(model, args.pretrained, device=device)
            best_psnr = ckpt_info.get("best_psnr", -float("inf"))
            if ema is not None:
                ema.module.load_state_dict(model.state_dict())
            print(f"Loaded pre-trained weights from {args.pretrained}")
        else:
            print(f"Warning: Pre-trained file {args.pretrained} not found, initializing from scratch.")

    if getattr(args, "resume", None) and os.path.exists(args.resume):
        res_ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(res_ckpt.get("state_dict", res_ckpt))
        if "state_dict_ema" in res_ckpt and ema is not None:
            ema.load_state_dict(res_ckpt["state_dict_ema"])
        best_psnr = res_ckpt.get("best_psnr", best_psnr)
        best_ema_psnr = res_ckpt.get("best_ema_psnr", best_ema_psnr)
        print(f"Resumed from {args.resume} (best base: {best_psnr:.2f} dB, best EMA: {best_ema_psnr:.2f} dB)")

    if args.freeze_stem:
        for p in model.stem.parameters():
            p.requires_grad = False
        print("Stem parameters frozen.")

    def build_scheduler(opt, epochs, warmup=0):
        w = min(warmup, max(0, epochs // 4))
        if w > 0:
            s1 = LinearLR(opt, start_factor=0.1, total_iters=w)
            s2 = CosineAnnealingLR(opt, T_max=epochs - w, eta_min=1e-6)
            return SequentialLR(opt, schedulers=[s1, s2], milestones=[w])
        return CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    if args.finetune:
        base_psnr, base_ssim = evaluate(model, val_loader, device, bf16=args.bf16, fp16=args.fp16, channels_last=channels_last)
        print(f"\nPre-trained Baseline -> Val PSNR: {base_psnr:.2f} dB | Val SSIM: {base_ssim:.4f}")
        if base_psnr > best_psnr:
            best_psnr = base_psnr
    else:
        # Stage 1: Coarse Training with Progressive Patch Sizes
        print(f"\nStage 1: {args.stage1_epochs} epochs | Batch Size: {args.batch_size_stage1} | Workers: {num_workers}")
        optimizer_s1 = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4, betas=(0.9, 0.999), weight_decay=1e-4)
        scheduler_s1 = build_scheduler(optimizer_s1, args.stage1_epochs, warmup=args.warmup_epochs)
        criterion_s1 = Stage1Loss(eps=1e-3, lambda_grad=0.05, lambda_fft=args.lambda_fft, lambda_ssim=args.lambda_ssim).to(device)

        train_loader_s1 = DataLoader(
            train_dataset,
            batch_size=args.batch_size_stage1,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device.type in ("cuda", "xpu")),
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
            drop_last=(len(train_dataset) >= args.batch_size_stage1),
        )

        for epoch in range(1, args.stage1_epochs + 1):
            t0 = time.time()
            patch_size = args.patch_size or get_progressive_patch_size(epoch - 1, args.stage1_epochs, min_size=128, max_size=256)
            train_dataset.set_patch_size(patch_size)

            loss = train_one_epoch(
                model, train_loader_s1, criterion_s1, optimizer_s1, device,
                bf16=args.bf16, fp16=args.fp16, scaler=scaler,
                ema=ema, accum_steps=args.accum_steps, channels_last=channels_last
            )
            scheduler_s1.step()

            should_eval = (epoch % args.eval_interval == 0) or (epoch == args.stage1_epochs)
            if should_eval:
                val_psnr, val_ssim = evaluate(model, val_loader, device, bf16=args.bf16, fp16=args.fp16, channels_last=channels_last)
                is_best = val_psnr > best_psnr
                if is_best:
                    best_psnr = val_psnr
                val_str = f"Val PSNR: {val_psnr:.2f} dB | SSIM: {val_ssim:.4f} {'*BEST*' if is_best else ''}"

                if ema is not None:
                    ema_psnr, ema_ssim = evaluate(ema.module, val_loader, device, bf16=args.bf16, fp16=args.fp16, channels_last=channels_last)
                    is_ema_best = ema_psnr > best_ema_psnr
                    if is_ema_best:
                        best_ema_psnr = ema_psnr
                        save_checkpoint(
                            {"state_dict": ema.state_dict(), "best_psnr": best_ema_psnr, "stage": 1, "is_ema": True},
                            is_best=True, save_dir=args.checkpoint_dir, filename="best_model_ema.pth"
                        )
                    val_str += f" | EMA: {ema_psnr:.2f} dB {'*BEST*' if is_ema_best else ''}"
            else:
                is_best = False
                val_str = "Val: (skipped)"

            save_checkpoint(
                {
                    "epoch": epoch,
                    "stage": 1,
                    "state_dict": model.state_dict(),
                    "state_dict_ema": ema.state_dict() if ema is not None else None,
                    "best_psnr": best_psnr,
                    "best_ema_psnr": best_ema_psnr,
                    "optimizer": optimizer_s1.state_dict(),
                    "scheduler": scheduler_s1.state_dict(),
                },
                is_best=is_best,
                save_dir=args.checkpoint_dir,
                filename="stage1_last.pth",
            )

            elapsed = time.time() - t0
            lr_curr = optimizer_s1.param_groups[0]["lr"]
            print(
                f"Stage 1 [Epoch {epoch:03d}/{args.stage1_epochs:03d}] Patch: {patch_size:03d}x{patch_size:03d} | "
                f"LR: {lr_curr:.6f} | Loss: {loss:.4f} | {val_str} [{elapsed:.1f}s]"
            )

        del optimizer_s1, scheduler_s1, criterion_s1, train_loader_s1
        clear_device_cache(device)

        best_ckpt = os.path.join(args.checkpoint_dir, "best_model.pth")
        if os.path.exists(best_ckpt):
            ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["state_dict"])
            best_psnr = ckpt.get("best_psnr", best_psnr)

    # Stage 2 / Fine-Tuning
    s2_epochs = args.finetune_epochs if args.finetune else args.stage2_epochs
    s2_lr = args.finetune_lr if args.finetune else 1e-4
    stage_label = "Fine-Tuning" if args.finetune else "Stage 2"
    last_ckpt_name = "finetuned_last.pth" if args.finetune else "stage2_last.pth"

    print(f"\n{stage_label}: {s2_epochs} epochs | Batch Size: {args.batch_size_stage2} | LR: {s2_lr} | Workers: {num_workers}")
    train_dataset.set_patch_size(256)
    train_loader_s2 = DataLoader(
        train_dataset,
        batch_size=args.batch_size_stage2,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type in ("cuda", "xpu")),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=(len(train_dataset) >= args.batch_size_stage2),
    )

    optimizer_s2 = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=s2_lr, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler_s2 = build_scheduler(optimizer_s2, s2_epochs, warmup=max(1, args.warmup_epochs // 2))
    criterion_s2 = Stage2SharpLoss(data_range=1.0, eps=1e-6, lambda_fft=args.lambda_fft, lambda_ssim=args.lambda_ssim).to(device)

    for epoch in range(1, s2_epochs + 1):
        t0 = time.time()
        loss = train_one_epoch(
            model, train_loader_s2, criterion_s2, optimizer_s2, device,
            bf16=args.bf16, fp16=args.fp16, scaler=scaler,
            ema=ema, accum_steps=args.accum_steps, channels_last=channels_last
        )
        scheduler_s2.step()

        should_eval = (epoch % args.eval_interval == 0) or (epoch == s2_epochs)
        if should_eval:
            val_psnr, val_ssim = evaluate(model, val_loader, device, bf16=args.bf16, fp16=args.fp16, channels_last=channels_last)
            is_best = val_psnr > best_psnr
            if is_best:
                best_psnr = val_psnr
            val_str = f"Val PSNR: {val_psnr:.2f} dB | SSIM: {val_ssim:.4f} {'*BEST*' if is_best else ''}"

            if ema is not None:
                ema_psnr, ema_ssim = evaluate(ema.module, val_loader, device, bf16=args.bf16, fp16=args.fp16, channels_last=channels_last)
                is_ema_best = ema_psnr > best_ema_psnr
                if is_ema_best:
                    best_ema_psnr = ema_psnr
                    save_checkpoint(
                        {"state_dict": ema.state_dict(), "best_psnr": best_ema_psnr, "stage": 2, "is_ema": True},
                        is_best=True, save_dir=args.checkpoint_dir, filename="best_model_ema.pth"
                    )
                val_str += f" | EMA: {ema_psnr:.2f} dB {'*BEST*' if is_ema_best else ''}"
        else:
            is_best = False
            val_str = "Val: (skipped)"

        save_checkpoint(
            {
                "epoch": epoch,
                "stage": 2 if not args.finetune else "finetune",
                "state_dict": model.state_dict(),
                "state_dict_ema": ema.state_dict() if ema is not None else None,
                "best_psnr": best_psnr,
                "best_ema_psnr": best_ema_psnr,
                "optimizer": optimizer_s2.state_dict(),
                "scheduler": scheduler_s2.state_dict(),
            },
            is_best=is_best,
            save_dir=args.checkpoint_dir,
            filename=last_ckpt_name,
        )

        elapsed = time.time() - t0
        lr_curr = optimizer_s2.param_groups[0]["lr"]
        print(
            f"{stage_label} [Epoch {epoch:02d}/{s2_epochs:02d}] Patch: 256x256 | "
            f"LR: {lr_curr:.6f} | Loss: {loss:.4f} | {val_str} [{elapsed:.1f}s]"
        )

    # Re-parameterization & Deployment Export
    print("\nVerifying re-parameterization numerical equivalence...")
    max_err = test_reparameterization_equivalence(device=device.type, tol=1e-5)
    print(f"Max numerical error: {max_err:.8e}")

    # Use best EMA weights if available and better than or equal to base model
    deploy_model = model
    if ema is not None and best_ema_psnr >= best_psnr:
        deploy_model = ema.module
        print(f"Selecting best EMA model for deployment (EMA PSNR: {best_ema_psnr:.2f} dB vs Base: {best_psnr:.2f} dB)")
    else:
        best_ckpt = os.path.join(args.checkpoint_dir, "best_model.pth")
        if os.path.exists(best_ckpt):
            ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
            deploy_model.load_state_dict(ckpt["state_dict"])
            print(f"Loaded best base checkpoint (PSNR: {ckpt.get('best_psnr', 0.0):.2f} dB)")

    deploy_model.switch_to_deploy()
    deployed_ckpt = os.path.join(args.checkpoint_dir, "repaf_denoise_net_deployed.pth")
    torch.save(deploy_model.state_dict(), deployed_ckpt)
    print(f"Deployed checkpoint saved: {deployed_ckpt}")

    onnx_path = os.path.join(args.export_dir, "repaf_denoise_net.onnx")
    export_onnx(deploy_model, save_path=onnx_path, input_shape=(1, 3, 256, 256))

    int8_path = os.path.join(args.export_dir, "repaf_denoise_net_int8.pth")
    export_int8_quantization(deploy_model, save_path=int8_path, calibration_loader=val_loader)

    from dataset import clear_image_cache
    clear_image_cache()
    clear_device_cache(device)
    print("Training and export complete.")


def main():
    parser = argparse.ArgumentParser(description="RepAF-Denoise Net Training and Export Pipeline")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "rocm", "xpu", "mps", "xla", "cpu"], help="Hardware accelerator device")
    parser.add_argument("--train-noisy-dir", type=str, default=None, help="Training noisy images directory")
    parser.add_argument("--train-clean-dir", type=str, default=None, help="Training clean images directory")
    parser.add_argument("--val-noisy-dir", type=str, default=None, help="Validation noisy images directory")
    parser.add_argument("--val-clean-dir", type=str, default=None, help="Validation clean images directory")
    parser.add_argument("--stage1-epochs", type=int, default=100, help="Epochs for Stage 1")
    parser.add_argument("--stage2-epochs", type=int, default=30, help="Epochs for Stage 2")
    parser.add_argument("--batch-size-stage1", type=int, default=32, help="Batch size for Stage 1")
    parser.add_argument("--batch-size-stage2", type=int, default=16, help="Batch size for Stage 2")
    parser.add_argument("--batch-size-val", type=int, default=16, help="Batch size for validation")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader num_workers (0 = auto-detect)")
    parser.add_argument("--eval-interval", type=int, default=5, help="Epoch interval for validation")
    parser.add_argument("--preload-ram", action="store_true", help="Preload dataset into RAM")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--export-dir", type=str, default="exports", help="Directory to export models")
    parser.add_argument("--synthetic-samples", type=int, default=128, help="Synthetic samples if no dataset provided")
    parser.add_argument("--bf16", action="store_true", help="Enable BF16 mixed precision training")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 mixed precision training")
    parser.add_argument("--patch-size", type=int, default=None, help="Fixed patch size for Stage 1")
    parser.add_argument("--pretrained", type=str, default=None, help="Path to pre-trained checkpoint to load/fine-tune")
    parser.add_argument("--finetune", action="store_true", help="Fine-tune pre-trained model")
    parser.add_argument("--finetune-epochs", type=int, default=30, help="Epochs for fine-tuning")
    parser.add_argument("--finetune-lr", type=float, default=5e-5, help="Learning rate for fine-tuning")
    parser.add_argument("--freeze-stem", action="store_true", help="Freeze stem layer during fine-tuning")
    parser.add_argument("--dry-run", action="store_true", help="Run 1-epoch dry run")

    # SOTA additions (with sensible defaults adhering to KISS/YAGNI)
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True, help="Enable Model EMA")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="Model EMA decay rate")
    parser.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=True, help="Channels-last memory format")
    parser.add_argument("--warmup-epochs", type=int, default=3, help="Warmup epochs for learning rate scheduler")
    parser.add_argument("--accum-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--lambda-fft", type=float, default=0.0, help="Weight for 2D FFT frequency loss")
    parser.add_argument("--lambda-ssim", type=float, default=0.0, help="Weight for differentiable SSIM loss")
    parser.add_argument("--advanced-augment", action=argparse.BooleanOptionalAction, default=True, help="Enable paired channel and exposure augmentations")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")

    args = parser.parse_args()

    if args.dry_run:
        args.stage1_epochs = 1
        args.stage2_epochs = 1
        args.finetune_epochs = 1
        args.batch_size_stage1 = 4
        args.batch_size_stage2 = 2
        args.synthetic_samples = 8
        args.warmup_epochs = 0

    run_pipeline(args)


if __name__ == "__main__":
    main()
