"""Build a leaderboard from MLflow runs and diagnostic plots for the best model:
confusion-matrix heatmap, per-class accuracy vs the BEiT baseline. Outputs to
sketch_clf/report_images/ and prints a markdown leaderboard table.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import setup_mlflow, load_meta, ROOT

IMG = ROOT / "report_images"
IMG.mkdir(parents=True, exist_ok=True)


def main():
    mlflow = setup_mlflow("sketch-classifier")
    exp = mlflow.get_experiment_by_name("sketch-classifier")
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], output_format="list")

    rows = []
    for r in runs:
        m, p = r.data.metrics, r.data.params
        name = r.data.tags.get("mlflow.runName", r.info.run_id[:8])
        if "robustness" in name:
            continue
        rows.append({
            "run": name,
            "model": p.get("model", ""),
            "mode": p.get("mode", ""),
            "test_top1": m.get("test_top1", m.get("test_top1_open", float("nan"))),
            "test_top1_restricted": m.get("test_top1_restricted", float("nan")),
            "test_top3": m.get("test_top3", m.get("test_top3_open", float("nan"))),
            "test_macro_f1": m.get("test_macro_f1", m.get("test_macro_f1_restricted", float("nan"))),
            "best_val_top1": m.get("best_val_top1", float("nan")),
            "train_time_sec": m.get("train_time_sec", float("nan")),
            "trainable_params": p.get("trainable_params", ""),
            "max_vram_gb": m.get("max_vram_gb", float("nan")),
        })
    rows.sort(key=lambda x: (-(x["test_top1"] if x["test_top1"] == x["test_top1"] else -1)))

    # markdown table
    hdr = "| run | model | mode | test_top1 | test_top3 | macro_f1 | val_top1 | train_s | vram_G |"
    sep = "|---|---|---|---|---|---|---|---|---|"
    lines = [hdr, sep]
    for x in rows:
        def f(v):
            return f"{v:.4f}" if isinstance(v, float) and v == v else "-"
        lines.append(f"| {x['run']} | {x['model'].split('.')[0]} | {x['mode']} | "
                     f"{f(x['test_top1'])} | {f(x['test_top3'])} | {f(x['test_macro_f1'])} | "
                     f"{f(x['best_val_top1'])} | {f(x['train_time_sec'])} | {f(x['max_vram_gb'])} |")
    table = "\n".join(lines)
    print(table)
    (IMG / "leaderboard.md").write_text(table + "\n")
    (ROOT / "leaderboard.json").write_text(json.dumps(rows, indent=2))

    meta = load_meta()
    classes = meta["classes"]

    # best trained model (exclude baseline)
    trained = [x for x in rows if x["mode"] not in ("pretrained_eval_only",) and x["test_top1"] == x["test_top1"]]
    best = trained[0] if trained else None

    # ---- confusion matrix heatmap for best model ----
    if best:
        cm_path = ROOT / "artifacts" / best["run"] / "confusion_matrix.npy"
        if cm_path.exists():
            cm = np.load(cm_path)
            cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
            fig, ax = plt.subplots(figsize=(11, 9.5))
            im = ax.imshow(cmn, cmap="viridis", vmin=0, vmax=1)
            ax.set_xticks(range(len(classes)))
            ax.set_yticks(range(len(classes)))
            ax.set_xticklabels(classes, rotation=90, fontsize=7)
            ax.set_yticklabels(classes, fontsize=7)
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
            ax.set_title(f"Confusion matrix (row-normalized) — {best['run']} (top1={best['test_top1']:.3f})")
            fig.colorbar(im, fraction=0.046)
            plt.tight_layout()
            fig.savefig(IMG / "confusion_best.png", dpi=110)
            plt.close(fig)

    # ---- per-class accuracy: best model vs baseline ----
    bestpc = ROOT / "artifacts" / (best["run"] if best else "") / "per_class_acc.json"
    basepc = ROOT / "artifacts" / "baseline_beit" / "per_class_acc_open.json"
    if best and bestpc.exists() and basepc.exists():
        b = json.loads(bestpc.read_text())
        a = json.loads(basepc.read_text())
        order = sorted(classes, key=lambda c: b.get(c, 0))
        x = np.arange(len(order))
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(x - 0.2, [a.get(c, 0) for c in order], width=0.4, label="BEiT baseline (open)", color="#c66")
        ax.bar(x + 0.2, [b.get(c, 0) for c in order], width=0.4, label=f"{best['run']}", color="#37a")
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=90, fontsize=8)
        ax.set_ylabel("per-class top-1 accuracy")
        ax.set_title("Per-class accuracy: trained model vs BEiT baseline")
        ax.legend()
        plt.tight_layout()
        fig.savefig(IMG / "per_class_compare.png", dpi=110)
        plt.close(fig)

    print(f"\nplots saved to {IMG}")
    if best:
        print(f"best trained model: {best['run']} test_top1={best['test_top1']:.4f}")


if __name__ == "__main__":
    main()
