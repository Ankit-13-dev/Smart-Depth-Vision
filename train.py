"""
train.py
─────────
Full training script for Smart Depth Vision classifier.

Features:
  - Mixed precision (AMP) for RTX 3050
  - Cosine LR scheduler with warmup
  - Early stopping
  - Saves best checkpoint automatically
  - TensorBoard logging

Run:
    python train.py
"""

import os
import yaml
import time
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

ROOT     = Path(__file__).resolve().parent
CFG_PATH = ROOT / "config.yaml"
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

TR_CFG   = CFG["training"]
DS_CFG   = CFG["dataset"]
PATHS    = CFG["paths"]

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHTS_DIR = ROOT / PATHS["weights"]
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Imports ───────────────────────────────────────────────────────────────────
from utils.dataset import make_loaders
from models.classifier import build_model


# ── Loss ─────────────────────────────────────────────────────────────────────
def compute_loss(bin_logits, cat_logits, bin_labels, cat_labels,
                 bin_w: float = 1.0, cat_w: float = 0.5,
                 bin_class_weights=None):
    ce_bin = nn.CrossEntropyLoss(weight=bin_class_weights)
    ce_cat = nn.CrossEntropyLoss()
    loss_bin = ce_bin(bin_logits, bin_labels)
    loss_cat = ce_cat(cat_logits, cat_labels)
    return bin_w * loss_bin + cat_w * loss_cat, loss_bin, loss_cat


# ── Accuracy ──────────────────────────────────────────────────────────────────
def accuracy(logits, labels):
    return (logits.argmax(dim=1) == labels).float().mean().item()


# ── Warmup + Cosine Scheduler ─────────────────────────────────────────────────
def get_scheduler(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps  = total_epochs  * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Train one epoch ───────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, scaler, bin_class_weights=None):
    model.train()
    total_loss = bin_acc = cat_acc = 0.0
    bin_w = TR_CFG["loss_binary_weight"]
    cat_w = TR_CFG["loss_category_weight"]

    for tensors, bin_lbl, cat_lbl in loader:
        tensors = tensors.to(DEVICE, non_blocking=True)
        bin_lbl = bin_lbl.to(DEVICE, non_blocking=True)
        cat_lbl = cat_lbl.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        with autocast():
            bin_out, cat_out = model(tensors)
            loss, _, _ = compute_loss(bin_out, cat_out,
                                      bin_lbl, cat_lbl, bin_w, cat_w,
                                      bin_class_weights)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        bin_acc    += accuracy(bin_out, bin_lbl)
        cat_acc    += accuracy(cat_out, cat_lbl)

    n = len(loader)
    return total_loss/n, bin_acc/n, cat_acc/n


# ── Validate ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, bin_class_weights=None):
    model.eval()
    total_loss = bin_acc = cat_acc = 0.0
    bin_w = TR_CFG["loss_binary_weight"]
    cat_w = TR_CFG["loss_category_weight"]

    for tensors, bin_lbl, cat_lbl in loader:
        tensors = tensors.to(DEVICE, non_blocking=True)
        bin_lbl = bin_lbl.to(DEVICE, non_blocking=True)
        cat_lbl = cat_lbl.to(DEVICE, non_blocking=True)

        with autocast():
            bin_out, cat_out = model(tensors)
            loss, _, _ = compute_loss(bin_out, cat_out,
                                      bin_lbl, cat_lbl, bin_w, cat_w,
                                      bin_class_weights)

        total_loss += loss.item()
        bin_acc    += accuracy(bin_out, bin_lbl)
        cat_acc    += accuracy(cat_out, cat_lbl)

    n = len(loader)
    return total_loss/n, bin_acc/n, cat_acc/n


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "═"*60)
    print("  Smart Depth Vision — Training")
    print("═"*60)
    print(f"  Device : {DEVICE}")
    if DEVICE == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print()

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, _ = make_loaders(
        batch_size  = TR_CFG["batch_size"],
        num_workers = TR_CFG["num_workers"]
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(
        num_classes = DS_CFG["num_classes"],
        dropout     = CFG["models"]["classifier"]["dropout"],
        device      = DEVICE
    )

    # ── Class weights for binary imbalance ────────────────────────────────────
    n_2d = (torch.tensor(train_loader.dataset.df["label_2d3d"].values) == 0).sum().float()
    n_3d = (torch.tensor(train_loader.dataset.df["label_2d3d"].values) == 1).sum().float()
    total = n_2d + n_3d
    bin_class_weights = torch.tensor([total/n_2d, total/n_3d]).to(DEVICE)
    print(f"  Class weights  : 2D={bin_class_weights[0]:.2f}  3D={bin_class_weights[1]:.2f}")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = TR_CFG["learning_rate"],
        weight_decay = TR_CFG["weight_decay"]
    )

    # ── Scheduler ─────────────────────────────────────────────────────────────
    scheduler = get_scheduler(
        optimizer,
        warmup_epochs  = TR_CFG["warmup_epochs"],
        total_epochs   = TR_CFG["epochs"],
        steps_per_epoch= len(train_loader)
    )

    scaler  = GradScaler()
    writer  = SummaryWriter(str(ROOT / "runs"))

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss  = float("inf")
    no_improve     = 0
    best_ckpt_path = WEIGHTS_DIR / "best_model.pt"

    print(f"\n  Epochs     : {TR_CFG['epochs']}")
    print(f"  Batch size : {TR_CFG['batch_size']}")
    print(f"  LR         : {TR_CFG['learning_rate']}")
    print(f"  Early stop : {TR_CFG['early_stop']} epochs\n")
    print(f"  {'Epoch':>5}  {'TrainLoss':>10}  {'BinAcc':>7}  "
          f"{'CatAcc':>7}  {'ValLoss':>8}  {'ValBin':>7}  {'ValCat':>7}  {'LR':>8}")
    print("  " + "─"*72)

    for epoch in range(1, TR_CFG["epochs"] + 1):
        t0 = time.time()

        tr_loss, tr_bin, tr_cat = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, bin_class_weights)
        vl_loss, vl_bin, vl_cat = validate(model, val_loader, bin_class_weights)

        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        print(f"  {epoch:>5}  {tr_loss:>10.4f}  {tr_bin:>6.1%}  "
              f"{tr_cat:>6.1%}  {vl_loss:>8.4f}  {vl_bin:>6.1%}  "
              f"{vl_cat:>6.1%}  {lr_now:>8.6f}  ({elapsed:.0f}s)")

        # TensorBoard
        writer.add_scalar("Loss/train",    tr_loss, epoch)
        writer.add_scalar("Loss/val",      vl_loss, epoch)
        writer.add_scalar("Acc/train_bin", tr_bin,  epoch)
        writer.add_scalar("Acc/val_bin",   vl_bin,  epoch)
        writer.add_scalar("Acc/train_cat", tr_cat,  epoch)
        writer.add_scalar("Acc/val_cat",   vl_cat,  epoch)

        # Save periodic checkpoint
        if epoch % TR_CFG["save_every"] == 0:
            ckpt = WEIGHTS_DIR / f"epoch_{epoch:03d}.pt"
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "val_loss": vl_loss}, str(ckpt))

        # Save best model
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            no_improve    = 0
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "val_loss": vl_loss,
                        "val_bin_acc": vl_bin,
                        "val_cat_acc": vl_cat}, str(best_ckpt_path))
            print(f"  ✓ Best model saved  (val_loss={vl_loss:.4f})")
        else:
            no_improve += 1
            if no_improve >= TR_CFG["early_stop"]:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(no improvement for {TR_CFG['early_stop']} epochs)")
                break

    writer.close()
    print("\n" + "═"*60)
    print(f"  Training complete!")
    print(f"  Best val loss : {best_val_loss:.4f}")
    print(f"  Best model    : {best_ckpt_path}")
    print(f"  Next step     : python evaluate.py")
    print("═"*60 + "\n")
if __name__ == "__main__":
    main()
