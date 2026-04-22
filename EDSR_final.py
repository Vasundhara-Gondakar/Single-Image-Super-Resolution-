#!/usr/bin/env python3
"""
edsr_dota_train_with_plots.py
Single-file, complete EDSR training script adapted for DOTA v1 (1024x1024, x4),
with training/validation graphs and sample SR output saving.

Features (all-in-one):
- EDSR model (MeanShift, ResBlocks, Upsampler)
- Random aligned patch extraction (LR patch_size e.g. 48 -> HR patch_size * scale)
- Augmentation: random horizontal flip, vertical flip, 90-degree rotate
- Two dataset modes: standard (one patch per image) and multi (patches_per_image)
- Mixed precision (AMP), GradScaler, gradient clipping, optional gradient accumulation
- Validation uses full images and PSNR on Y-channel (shaves border = scale)
- Compute dataset RGB mean (one-time) and optionally use it in MeanShift
- Save best model by Val PSNR
- Save plots: loss curve and PSNR curve per epoch, training_log.json
- Save example SR/HR side-by-side images each epoch in plots/samples/

Usage:
    python edsr_dota_train_with_plots.py --hr_dir PATH/TO/HR --lr_dir PATH/TO/LR_x4 \
        --val_hr PATH/TO/VAL/HR --val_lr PATH/TO/VAL/LR_x4
    To compute dataset mean (one-time):
    python edsr_dota_train_with_plots.py --hr_dir PATH/TO/HR --compute_mean --dataset_mean_path dataset_mean.json
"""

import os
import glob
import argparse
import random
import json
from PIL import Image
import numpy as np
from tqdm import tqdm
import math

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt

# -------------------------
# Configurable defaults
# -------------------------
DEFAULTS = {
    "hr_dir": r"E:\ML\Training\x4_final\HR",
    "lr_dir": r"E:\ML\Training\x4_final\LR_x4",
    "val_hr": r"E:\ML\Training\x4_final\val_HR",
    "val_lr": r"E:\ML\Training\x4_final\val_LR",
    "scale": 4,
    "patch_size": 48,              # LR patch size
    "batch_size": 16,
    "num_workers": 4,
    "epochs": 100,
    "lr": 1e-4,
    "n_resblocks": 16,
    "n_feats": 64,
    "res_scale": 0.1,
    "patches_per_image": 1,        # set >1 to use multi-patch-per-image mode
    "compute_mean": False,         # one-time compute dataset mean and exit
    "dataset_mean_path": "dataset_mean.json",
    "use_dataset_mean": False,     # if True and mean file exists, use it
    "accumulation_steps": 1,       # use >1 to emulate larger batch via grad accumulation
    "save_path": "best_edsr_dota.pth",
    "seed": 42,
    "plots_dir": "plots",
    "save_sample_every_epoch": True,
    "sample_image_name": "sample_epoch_{:03d}.png",
}

# -------------------------
# Utilities
# -------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_rgb_mean_std(img_dir, exts=('png','jpg','jpeg')):
    files = []
    for e in exts:
        files += glob.glob(os.path.join(img_dir, f"*.{e}"))
    assert files, f"No images found in {img_dir}"
    mean = np.zeros(3, dtype=np.float64)
    sq_mean = np.zeros(3, dtype=np.float64)
    n = 0
    for p in tqdm(files, desc="Computing dataset mean/std"):
        im = Image.open(p).convert('RGB')
        arr = np.asarray(im).astype(np.float32) / 255.0
        mean += arr.mean(axis=(0,1))
        sq_mean += (arr**2).mean(axis=(0,1))
        n += 1
    mean /= n
    var = sq_mean / n - mean**2
    std = np.sqrt(np.maximum(var, 1e-12))
    return mean.tolist(), std.tolist()

# -------------------------
# Model components
# -------------------------
class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range=1.0, rgb_mean=(0.4488,0.4371,0.4040), rgb_std=(1.0,1.0,1.0), sign=-1):
        super().__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3,3,1,1) / std.view(3,1,1,1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class ResBlock(nn.Module):
    def __init__(self, n_feats=64, kernel_size=3, bias=True, act=nn.ReLU(True), res_scale=0.1):
        super().__init__()
        self.res_scale = res_scale
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size//2, bias=bias),
            act,
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size//2, bias=bias)
        )
    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        return x + res

class Upsampler(nn.Sequential):
    def __init__(self, scale, n_feats=64, bias=True):
        m = []
        if scale in (2,4):
            m.append(nn.Conv2d(n_feats, n_feats * 4, 3, padding=1, bias=bias))
            m.append(nn.PixelShuffle(2))
            if scale == 4:
                m.append(nn.Conv2d(n_feats, n_feats * 4, 3, padding=1, bias=bias))
                m.append(nn.PixelShuffle(2))
        else:
            m.append(nn.Conv2d(n_feats, n_feats, 3, padding=1, bias=bias))
        super().__init__(*m)

class EDSR(nn.Module):
    def __init__(self, scale=2, n_resblocks=16, n_feats=64, res_scale=0.1, dataset_mean=None):
        super().__init__()
        mean = dataset_mean if dataset_mean is not None else (0.4488,0.4371,0.4040)
        self.sub_mean = MeanShift(rgb_range=1.0, rgb_mean=mean, sign=-1)
        self.add_mean = MeanShift(rgb_range=1.0, rgb_mean=mean, sign=1)
        self.head = nn.Conv2d(3, n_feats, 3, padding=1)
        body = []
        for _ in range(n_resblocks):
            body.append(ResBlock(n_feats=n_feats, res_scale=res_scale))
        body.append(nn.Conv2d(n_feats, n_feats, 3, padding=1))
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(
            Upsampler(scale, n_feats),
            nn.Conv2d(n_feats, 3, 3, padding=1)
        )
    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        x = self.add_mean(x)
        return x

# -------------------------
# Dataset implementations
# -------------------------
class DotaEDSRDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, scale=2, patch_size=48, train=True, augment=True, exts=('png','jpg','jpeg')):
        self.hr_files = sorted(sum([glob.glob(os.path.join(hr_dir, f"*.{e}")) for e in exts], []))
        self.lr_files = sorted(sum([glob.glob(os.path.join(lr_dir, f"*.{e}")) for e in exts], []))
        assert len(self.hr_files) == len(self.lr_files), f"HR/LR count mismatch: {len(self.hr_files)} vs {len(self.lr_files)}"
        self.scale = scale
        self.patch_size = patch_size
        self.train = train
        self.augment = augment
    def __len__(self):
        return len(self.hr_files)
    def __getitem__(self, idx):
        hr = Image.open(self.hr_files[idx]).convert('RGB')
        lr = Image.open(self.lr_files[idx]).convert('RGB')
        if not self.train:
            return TF.to_tensor(lr), TF.to_tensor(hr)
        lr_w, lr_h = lr.size
        ps = self.patch_size
        if lr_w < ps or lr_h < ps:
            raise ValueError(f"LR img {self.lr_files[idx]} smaller than patch_size {ps}")
        tx = random.randint(0, lr_w - ps)
        ty = random.randint(0, lr_h - ps)
        lr_patch = lr.crop((tx, ty, tx + ps, ty + ps))
        hr_ps = ps * self.scale
        hr_tx, hr_ty = tx * self.scale, ty * self.scale
        hr_patch = hr.crop((hr_tx, hr_ty, hr_tx + hr_ps, hr_ty + hr_ps))
        if self.augment:
            if random.random() < 0.5:
                lr_patch = TF.hflip(lr_patch); hr_patch = TF.hflip(hr_patch)
            if random.random() < 0.5:
                lr_patch = TF.vflip(lr_patch); hr_patch = TF.vflip(hr_patch)
            if random.random() < 0.5:
                lr_patch = lr_patch.rotate(90); hr_patch = hr_patch.rotate(90)
        return TF.to_tensor(lr_patch), TF.to_tensor(hr_patch)

class DotaEDSRDatasetMulti(Dataset):
    def __init__(self, hr_dir, lr_dir, scale=2, patch_size=48, train=True, augment=True, patches_per_image=10, exts=('png','jpg','jpeg')):
        self.hr_files = sorted(sum([glob.glob(os.path.join(hr_dir, f"*.{e}")) for e in exts], []))
        self.lr_files = sorted(sum([glob.glob(os.path.join(lr_dir, f"*.{e}")) for e in exts], []))
        assert len(self.hr_files) == len(self.lr_files)
        self.scale = scale
        self.patch_size = patch_size
        self.train = train
        self.augment = augment
        self.patches_per_image = int(patches_per_image)
        self.num_images = len(self.hr_files)
    def __len__(self):
        return self.num_images * self.patches_per_image
    def __getitem__(self, idx):
        img_idx = idx % self.num_images
        hr = Image.open(self.hr_files[img_idx]).convert('RGB')
        lr = Image.open(self.lr_files[img_idx]).convert('RGB')
        if not self.train:
            return TF.to_tensor(lr), TF.to_tensor(hr)
        lr_w, lr_h = lr.size
        ps = self.patch_size
        if lr_w < ps or lr_h < ps:
            raise ValueError(f"LR img {self.lr_files[img_idx]} smaller than patch_size {ps}")
        tx = random.randint(0, lr_w - ps)
        ty = random.randint(0, lr_h - ps)
        lr_patch = lr.crop((tx, ty, tx + ps, ty + ps))
        hr_ps = ps * self.scale
        hr_tx, hr_ty = tx * self.scale, ty * self.scale
        hr_patch = hr.crop((hr_tx, hr_ty, hr_tx + hr_ps, hr_ty + hr_ps))
        if self.augment:
            if random.random() < 0.5:
                lr_patch = TF.hflip(lr_patch); hr_patch = TF.hflip(hr_patch)
            if random.random() < 0.5:
                lr_patch = TF.vflip(lr_patch); hr_patch = TF.vflip(hr_patch)
            if random.random() < 0.5:
                lr_patch = lr_patch.rotate(90); hr_patch = hr_patch.rotate(90)
        return TF.to_tensor(lr_patch), TF.to_tensor(hr_patch)

class DotaValDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, exts=('png','jpg','jpeg')):
        self.hr_files = sorted(sum([glob.glob(os.path.join(hr_dir, f"*.{e}")) for e in exts], []))
        self.lr_files = sorted(sum([glob.glob(os.path.join(lr_dir, f"*.{e}")) for e in exts], []))
        assert len(self.hr_files) == len(self.lr_files)
    def __len__(self): return len(self.hr_files)
    def __getitem__(self, idx):
        hr = Image.open(self.hr_files[idx]).convert('RGB')
        lr = Image.open(self.lr_files[idx]).convert('RGB')
        return TF.to_tensor(lr), TF.to_tensor(hr)

# -------------------------
# Metrics (Y-channel PSNR)
# -------------------------
def rgb_to_y(tensor):
    if tensor.dim() == 3:
        r,g,b = tensor[0], tensor[1], tensor[2]
        y = 0.299*r + 0.587*g + 0.114*b
        return y.unsqueeze(0)
    elif tensor.dim() == 4:
        r = tensor[:,0:1,...]; g = tensor[:,1:2,...]; b = tensor[:,2:3,...]
        y = 0.299*r + 0.587*g + 0.114*b
        return y
    else:
        raise ValueError("Unsupported tensor shape for rgb_to_y")

def psnr_y(sr, hr, shave_border=2, use_y=True, max_val=1.0):
    assert sr.shape == hr.shape, f"Shapes mismatch {sr.shape} vs {hr.shape}"
    if use_y:
        sr = rgb_to_y(sr); hr = rgb_to_y(hr)
    if shave_border > 0:
        sr = sr[..., shave_border:-shave_border, shave_border:-shave_border]
        hr = hr[..., shave_border:-shave_border, shave_border:-shave_border]
    mse = torch.mean((sr - hr)**2, dim=[1,2,3])
    psnr_batch = 10.0 * torch.log10((max_val**2) / (mse + 1e-12))
    return psnr_batch.mean().item()

# -------------------------
# Training & Validation loops
# -------------------------
def train_one_epoch(model, loader, optimizer, criterion, device, scaler, accumulation_steps=1):
    model.train()
    running_loss = 0.0
    total_batches = 0
    optimizer.zero_grad()
    for i, (lr, hr) in enumerate(tqdm(loader, desc="Train", leave=False)):
        lr = lr.to(device); hr = hr.to(device)
        with autocast():
            sr = model(lr)
            loss = criterion(sr, hr) / accumulation_steps
        scaler.scale(loss).backward()
        if (i + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        running_loss += loss.item() * accumulation_steps
        total_batches += 1
    return running_loss / total_batches if total_batches > 0 else 0.0

@torch.no_grad()
def validate(model, loader, device, scale, use_y=True, shave_border=None, save_sample=False, sample_path=None):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    criterion = nn.L1Loss()
    count = 0
    shave = scale if shave_border is None else shave_border
    sample_saved = False
    for lr, hr in tqdm(loader, desc="Val", leave=False):
        lr = lr.to(device); hr = hr.to(device)
        sr = model(lr)
        total_loss += criterion(sr, hr).item()
        total_psnr += psnr_y(sr, hr, shave_border=shave, use_y=use_y)
        count += 1
        if save_sample and (not sample_saved):
            # Save first sample as side-by-side: LR (upsampled for visualization), SR, HR
            # Convert tensors to numpy images
            # sr, hr are (B, C, H, W) with B likely 1
            s = sr[0].clamp(0.0,1.0).cpu().numpy().transpose(1,2,0) * 255.0
            h = hr[0].clamp(0.0,1.0).cpu().numpy().transpose(1,2,0) * 255.0
            l = TF.to_pil_image(lr[0])  # LR low-res image as PIL (not upsampled)
            # Upsample LR using PIL bicubic for visualization
            up_l = l.resize((h.shape[1], h.shape[0]), Image.BICUBIC)
            s_img = Image.fromarray(np.asarray(s).astype(np.uint8))
            h_img = Image.fromarray(np.asarray(h).astype(np.uint8))
            # Create side-by-side canvas
            W = up_l.width + s_img.width + h_img.width
            H = max(up_l.height, s_img.height, h_img.height)
            canvas = Image.new('RGB', (W, H))
            canvas.paste(up_l, (0,0))
            canvas.paste(s_img, (up_l.width, 0))
            canvas.paste(h_img, (up_l.width + s_img.width, 0))
            canvas.save(sample_path)
            sample_saved = True
    avg_loss = total_loss / count if count else 0.0
    avg_psnr = total_psnr / count if count else 0.0
    return avg_loss, avg_psnr

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="EDSR training for DOTA-v1 with plots (single-file)")
    parser.add_argument("--hr_dir", type=str, default=DEFAULTS["hr_dir"])
    parser.add_argument("--lr_dir", type=str, default=DEFAULTS["lr_dir"])
    parser.add_argument("--val_hr", type=str, default=DEFAULTS["val_hr"])
    parser.add_argument("--val_lr", type=str, default=DEFAULTS["val_lr"])
    parser.add_argument("--scale", type=int, default=DEFAULTS["scale"])
    parser.add_argument("--patch_size", type=int, default=DEFAULTS["patch_size"])
    parser.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--num_workers", type=int, default=DEFAULTS["num_workers"])
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--n_resblocks", type=int, default=DEFAULTS["n_resblocks"])
    parser.add_argument("--n_feats", type=int, default=DEFAULTS["n_feats"])
    parser.add_argument("--res_scale", type=float, default=DEFAULTS["res_scale"])
    parser.add_argument("--patches_per_image", type=int, default=DEFAULTS["patches_per_image"])
    parser.add_argument("--compute_mean", action="store_true", help="Compute dataset mean and exit")
    parser.add_argument("--dataset_mean_path", type=str, default=DEFAULTS["dataset_mean_path"])
    parser.add_argument("--use_dataset_mean", action="store_true", help="Use dataset mean if file exists")
    parser.add_argument("--accumulation_steps", type=int, default=DEFAULTS["accumulation_steps"])
    parser.add_argument("--save_path", type=str, default=DEFAULTS["save_path"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--plots_dir", type=str, default=DEFAULTS["plots_dir"])
    parser.add_argument("--patches_per_image_flag", dest='patches_per_image_flag', action='store_true', help="Use patches_per_image mode (use --patches_per_image to set count)")

    args = parser.parse_args()

    set_seed(args.seed)

    # one-time mean computation
    if args.compute_mean:
        mean, std = compute_rgb_mean_std(args.hr_dir)
        print(f"Computed mean: {mean}, std: {std}")
        with open(args.dataset_mean_path, 'w') as f:
            json.dump({"mean": mean, "std": std}, f)
        print(f"Saved dataset mean to {args.dataset_mean_path}")
        return

    dataset_mean = None
    if args.use_dataset_mean and os.path.exists(args.dataset_mean_path):
        with open(args.dataset_mean_path, 'r') as f:
            dd = json.load(f)
            dataset_mean = dd.get("mean", None)
            if dataset_mean is None:
                print("Mean file found but missing 'mean' key; using default DIV2K mean.")
            else:
                print(f"Using dataset mean: {dataset_mean}")
    else:
        print("Using default DIV2K mean (0.4488,0.4371,0.4040) unless you compute and set --use_dataset_mean")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # datasets
    if args.patches_per_image_flag and args.patches_per_image > 1:
        train_ds = DotaEDSRDatasetMulti(args.hr_dir, args.lr_dir, scale=args.scale, patch_size=args.patch_size, train=True, augment=True, patches_per_image=args.patches_per_image)
        print(f"Using multi-patch mode: {len(train_ds)} samples per epoch ({args.patches_per_image} patches per image)")
    else:
        train_ds = DotaEDSRDataset(args.hr_dir, args.lr_dir, scale=args.scale, patch_size=args.patch_size, train=True, augment=True)
        print(f"Using standard dataset mode: {len(train_ds)} samples per epoch (1 patch per image)")

    val_ds = DotaValDataset(args.val_hr, args.val_lr)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=max(0,args.num_workers), pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=max(1,args.num_workers//2), pin_memory=True)

    # model & training objects
    model = EDSR(scale=args.scale, n_resblocks=args.n_resblocks, n_feats=args.n_feats, res_scale=args.res_scale, dataset_mean=dataset_mean).to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    scaler = GradScaler()
    best_psnr = -1e9

    # plotting / logging setup
    os.makedirs(args.plots_dir, exist_ok=True)
    samples_dir = os.path.join(args.plots_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    train_losses = []
    val_losses = []
    val_psnrs = []

    # training loop
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, accumulation_steps=args.accumulation_steps)
        save_sample_path = os.path.join(samples_dir, DEFAULTS["sample_image_name"].format(epoch))
        val_loss, val_psnr = validate(model, val_loader, device, scale=args.scale, use_y=True, shave_border=None,
                                      save_sample=DEFAULTS["save_sample_every_epoch"], sample_path=save_sample_path)
        print(f"Train L1: {train_loss:.6f} | Val L1: {val_loss:.6f} | Val PSNR (Y): {val_psnr:.4f}")

        # track
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_psnrs.append(val_psnr)

        # save training log JSON
        log = {"train_losses": train_losses, "val_losses": val_losses, "val_psnrs": val_psnrs}
        with open(os.path.join(args.plots_dir, "training_log.json"), "w") as f:
            json.dump(log, f, indent=2)

        # save plots
        try:
            plt.figure(figsize=(8,5))
            plt.plot(train_losses, label="Train L1 Loss")
            plt.plot(val_losses, label="Val L1 Loss")
            plt.xlabel("Epoch")
            plt.ylabel("L1 Loss")
            plt.title("Training and Validation L1 Loss")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(args.plots_dir, "loss_curve.png"))
            plt.close()

            plt.figure(figsize=(8,5))
            plt.plot(val_psnrs, label="Val PSNR (Y)")
            plt.xlabel("Epoch")
            plt.ylabel("PSNR (dB)")
            plt.title("Validation PSNR (Y) Over Epochs")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(args.plots_dir, "psnr_curve.png"))
            plt.close()
        except Exception as e:
            print(f"Warning: failed to plot/save graphs: {e}")

        # save model if improved
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "psnr": best_psnr
            }, args.save_path)
            print(f"Saved best model to {args.save_path} (PSNR {best_psnr:.4f})")

    print("Training complete. Plots and samples saved in:", args.plots_dir)

if __name__ == "__main__":
    main()
