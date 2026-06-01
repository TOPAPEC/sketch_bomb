"""Train a sketch classifier on the balanced QuickDraw 30-class set, log to MLflow.

Backbone is a timm model (ViT / CLIP-init ViT). Two modes:
  - linear_probe : freeze backbone, train only the classifier head (fast).
  - finetune     : train the whole network (lower backbone LR).

All params/metrics/artifacts are logged to MLflow (file store in sketch_clf/mlruns).
The best-val checkpoint is evaluated on the held-out test split; confusion matrix,
per-class accuracy, classification report and worst misclassifications are saved.

Usage examples:
  python train.py --model vit_small_patch16_224.augreg_in21k --mode finetune --epochs 12
  python train.py --model vit_base_patch16_clip_224.openai   --mode finetune --epochs 10
  python train.py --smoke   # tiny fast run to catch bugs
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import timm
from timm.data import resolve_data_config
import torchvision.transforms as T

from common import setup_mlflow, load_meta, load_split, SketchDataset, ROOT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_aug():
    # Modest geometric jitter -> mimics natural variation in hand sketches.
    # No horizontal flip: several classes (face, arm, telephone) are orientation-sensitive.
    return T.RandomAffine(degrees=12, translate=(0.08, 0.08), scale=(0.85, 1.10),
                          fill=255)


@torch.no_grad()
def evaluate(model, loader, n_classes):
    model.eval()
    all_logits, all_y = [], []
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(x)
        all_logits.append(logits.float().cpu())
        all_y.append(y)
    logits = torch.cat(all_logits)
    y = torch.cat(all_y)
    probs = logits.softmax(-1)
    top1 = (probs.argmax(-1) == y).float().mean().item()
    top3 = (probs.topk(3, dim=-1).indices == y[:, None]).any(-1).float().mean().item()
    top5 = (probs.topk(5, dim=-1).indices == y[:, None]).any(-1).float().mean().item()
    # macro F1
    preds = probs.argmax(-1)
    f1s = []
    for c in range(n_classes):
        tp = ((preds == c) & (y == c)).sum().item()
        fp = ((preds == c) & (y != c)).sum().item()
        fn = ((preds != c) & (y == c)).sum().item()
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    macro_f1 = float(np.mean(f1s))
    return {"top1": top1, "top3": top3, "top5": top5, "macro_f1": macro_f1}, \
        preds.numpy(), y.numpy(), probs.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vit_small_patch16_224.augreg_in21k")
    ap.add_argument("--mode", choices=["linear_probe", "finetune"], default="finetune")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)            # head LR
    ap.add_argument("--backbone-lr", type=float, default=2e-5)  # backbone LR (finetune)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--experiment", default="sketch-classifier")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    meta = load_meta()
    classes = meta["classes"]
    n_classes = meta["n_classes"]

    Xtr, ytr = load_split("train")
    Xva, yva = load_split("val")
    Xte, yte = load_split("test")

    if args.smoke:
        args.epochs = 1
        Xtr, ytr = Xtr[:1500], ytr[:1500]
        Xva, yva = Xva[:600], yva[:600]
        Xte, yte = Xte[:600], yte[:600]

    # model
    model = timm.create_model(args.model, pretrained=True, num_classes=n_classes)
    cfg = resolve_data_config({}, model=model)
    mean, std = cfg["mean"], cfg["std"]
    model = model.to(DEVICE)

    if args.mode == "linear_probe":
        for p in model.parameters():
            p.requires_grad_(False)
        head = model.get_classifier()
        for p in head.parameters():
            p.requires_grad_(True)

    aug = build_aug()
    dtr = SketchDataset(Xtr, ytr, mean, std, train=True, aug=aug)
    dva = SketchDataset(Xva, yva, mean, std, train=False)
    dte = SketchDataset(Xte, yte, mean, std, train=False)
    ltr = DataLoader(dtr, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                     pin_memory=True, drop_last=True, persistent_workers=True)
    lva = DataLoader(dva, batch_size=args.bs, shuffle=False, num_workers=args.workers,
                     pin_memory=True, persistent_workers=True)
    lte = DataLoader(dte, batch_size=args.bs, shuffle=False, num_workers=args.workers,
                     pin_memory=True, persistent_workers=True)

    # optimizer: separate LR for head vs backbone
    head_params = list(model.get_classifier().parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids and p.requires_grad]
    groups = [{"params": head_params, "lr": args.lr}]
    if args.mode == "finetune" and backbone_params:
        groups.append({"params": backbone_params, "lr": args.backbone_lr})
    opt = torch.optim.AdamW(groups, weight_decay=args.wd)
    steps_per_epoch = max(1, len(ltr))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda")
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    n_params = sum(p.numel() for p in model.parameters())
    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    mlflow = setup_mlflow(args.experiment)
    run_name = args.run_name or f"{args.model.split('.')[0]}_{args.mode}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model": args.model, "mode": args.mode, "epochs": args.epochs,
            "batch_size": args.bs, "head_lr": args.lr, "backbone_lr": args.backbone_lr,
            "weight_decay": args.wd, "warmup_epochs": args.warmup,
            "label_smoothing": args.label_smoothing, "n_classes": n_classes,
            "n_train": len(ytr), "n_val": len(yva), "n_test": len(yte),
            "img_sz": meta["img_sz"], "augment": "affine(deg12,trans.08,scale.85-1.1)",
            "total_params": n_params, "trainable_params": n_train_params,
            "init": "pretrained", "smoke": args.smoke,
        })

        best_val = -1.0
        best_state = None
        ckpt_dir = ROOT / "artifacts" / run_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        gstep = 0
        for epoch in range(args.epochs):
            model.train()
            running = 0.0
            te0 = time.time()
            for x, y in ltr:
                x = x.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(x)
                    loss = crit(logits, y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                running += loss.item()
                gstep += 1
            train_loss = running / steps_per_epoch
            val_metrics, *_ = evaluate(model, lva, n_classes)
            dt = time.time() - te0
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("lr", opt.param_groups[0]["lr"], step=epoch)
            for k, v in val_metrics.items():
                mlflow.log_metric(f"val_{k}", v, step=epoch)
            vram = torch.cuda.max_memory_allocated() / 1e9
            print(f"epoch {epoch+1}/{args.epochs} loss={train_loss:.3f} "
                  f"val_top1={val_metrics['top1']:.4f} val_top3={val_metrics['top3']:.4f} "
                  f"f1={val_metrics['macro_f1']:.4f} {dt:.1f}s vram={vram:.1f}G", flush=True)
            if val_metrics["top1"] > best_val:
                best_val = val_metrics["top1"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        train_time = time.time() - t0
        mlflow.log_metric("train_time_sec", train_time)
        mlflow.log_metric("max_vram_gb", torch.cuda.max_memory_allocated() / 1e9)
        mlflow.log_metric("best_val_top1", best_val)

        # ---- test eval with best checkpoint ----
        model.load_state_dict(best_state)
        test_metrics, preds, ytrue, probs = evaluate(model, lte, n_classes)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)
        print(f"TEST: {test_metrics}", flush=True)

        # save checkpoint + artifacts
        torch.save({"state_dict": best_state, "model": args.model,
                    "classes": classes, "mean": mean, "std": std},
                   ckpt_dir / "best.pt")
        # per-class accuracy
        per_class = {}
        for c in range(n_classes):
            m = ytrue == c
            per_class[classes[c]] = float((preds[m] == c).mean()) if m.sum() else 0.0
        (ckpt_dir / "per_class_acc.json").write_text(json.dumps(per_class, indent=2))
        # classification report + confusion
        from sklearn.metrics import classification_report, confusion_matrix
        rep = classification_report(ytrue, preds, target_names=classes,
                                    output_dict=True, zero_division=0)
        (ckpt_dir / "classification_report.json").write_text(json.dumps(rep, indent=2))
        cm = confusion_matrix(ytrue, preds, labels=list(range(n_classes)))
        np.save(ckpt_dir / "confusion_matrix.npy", cm)
        # top confusions
        confusions = []
        for i in range(n_classes):
            for j in range(n_classes):
                if i != j and cm[i, j] > 0:
                    confusions.append((classes[i], classes[j], int(cm[i, j])))
        confusions.sort(key=lambda x: -x[2])
        (ckpt_dir / "top_confusions.json").write_text(json.dumps(confusions[:40], indent=2))

        mlflow.log_artifacts(str(ckpt_dir), artifact_path="model")
        summary = {"test": test_metrics, "best_val_top1": best_val,
                   "train_time_sec": train_time,
                   "worst_classes": sorted(per_class.items(), key=lambda x: x[1])[:8],
                   "top_confusions": confusions[:12]}
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
