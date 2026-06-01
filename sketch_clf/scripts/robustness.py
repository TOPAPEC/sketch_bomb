"""Robustness analysis of a trained checkpoint vs the BEiT baseline.

Applies controlled perturbations to the test sketches (rotation, scale, stroke
thickening/thinning, translation, partial-stroke dropout / occlusion) and measures
top-1 degradation. Helps see whether the model relies on fragile cues. Results logged
to MLflow under a dedicated run and saved as JSON + a bar chart.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import cv2
from PIL import Image
import timm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import setup_mlflow, load_meta, load_split, ROOT

DEVICE = "cuda"


def perturb(img, kind):
    """img: (224,224) uint8 grayscale (black strokes on white). Return perturbed uint8."""
    h, w = img.shape
    if kind == "clean":
        return img
    if kind == "rotate15":
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 15, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderValue=255)
    if kind == "rotate30":
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 30, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderValue=255)
    if kind == "scale0.7":
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, 0.7)
        return cv2.warpAffine(img, M, (w, h), borderValue=255)
    if kind == "translate":
        M = np.float32([[1, 0, 25], [0, 1, 25]])
        return cv2.warpAffine(img, M, (w, h), borderValue=255)
    if kind == "thicken":
        inv = 255 - img
        inv = cv2.dilate(inv, np.ones((5, 5), np.uint8), iterations=1)
        return 255 - inv
    if kind == "thin":
        inv = 255 - img
        inv = cv2.erode(inv, np.ones((3, 3), np.uint8), iterations=1)
        return 255 - inv
    if kind == "occlude":  # blank out a 80px square (random-ish fixed location)
        out = img.copy()
        out[40:120, 40:120] = 255
        return out
    if kind == "noise":  # salt speckle
        out = img.copy()
        rng = np.random.RandomState(0)
        mask = rng.rand(h, w) < 0.02
        out[mask] = 0
        return out
    raise ValueError(kind)


@torch.no_grad()
def eval_model(model, X, y, mean, std, bs=256):
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    correct = 0
    for s in range(0, len(X), bs):
        batch = X[s:s + bs]
        t = torch.from_numpy(batch.astype(np.float32) / 255.0)[:, None].repeat(1, 3, 1, 1)
        t = (t - mean_t) / std_t
        t = t.to(DEVICE)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(t)
        pred = logits.argmax(-1).cpu().numpy()
        correct += (pred == y[s:s + bs]).sum()
    return correct / len(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=6000)  # test subset for speed
    args = ap.parse_args()

    meta = load_meta()
    n_classes = meta["n_classes"]
    Xte, yte = load_split("test")
    idx = np.random.RandomState(0).choice(len(yte), size=min(args.n, len(yte)), replace=False)
    Xte, yte = Xte[idx], yte[idx]

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = timm.create_model(ck["model"], pretrained=False, num_classes=n_classes)
    model.load_state_dict(ck["state_dict"])
    model = model.to(DEVICE).eval()
    mean, std = ck["mean"], ck["std"]

    kinds = ["clean", "rotate15", "rotate30", "scale0.7", "translate",
             "thicken", "thin", "occlude", "noise"]
    results = {}
    for kind in kinds:
        Xp = np.stack([perturb(im, kind) for im in Xte]).astype(np.uint8)
        acc = float(eval_model(model, Xp, yte, mean, std))
        results[kind] = acc
        print(f"  {kind:10s} top1={acc:.4f}", flush=True)

    out = ROOT / "artifacts" / "robustness"
    out.mkdir(parents=True, exist_ok=True)
    (out / "robustness.json").write_text(json.dumps(results, indent=2))

    # bar chart
    clean = results["clean"]
    fig, axx = plt.subplots(figsize=(9, 4.5))
    names = kinds
    vals = [results[k] for k in names]
    colors = ["#2a7" if k == "clean" else "#37a" for k in names]
    axx.bar(names, vals, color=colors)
    axx.axhline(clean, ls="--", color="gray", lw=1, label=f"clean={clean:.3f}")
    axx.set_ylabel("top-1 accuracy")
    axx.set_title(f"Robustness under perturbations ({ck['model'].split('.')[0]}, n={len(yte)})")
    axx.set_ylim(0, 1)
    for i, v in enumerate(vals):
        axx.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(out / "robustness.png", dpi=110)

    mlflow = setup_mlflow("sketch-classifier")
    with mlflow.start_run(run_name=f"robustness_{Path(args.ckpt).parent.name}"):
        mlflow.log_param("ckpt", args.ckpt)
        mlflow.log_param("model", ck["model"])
        mlflow.log_param("n_eval", len(yte))
        for k, v in results.items():
            mlflow.log_metric(f"robust_{k}", v)
        mlflow.log_metric("robust_drop_rotate30", clean - results["rotate30"])
        mlflow.log_metric("robust_drop_occlude", clean - results["occlude"])
        mlflow.log_artifacts(str(out), artifact_path="robustness")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
