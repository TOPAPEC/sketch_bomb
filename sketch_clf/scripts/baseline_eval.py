"""Evaluate the pipeline baseline `kmewhort/beit-sketch-classifier` on our test split
and log it to MLflow as a reference run. Same rendered images as our models -> fair.

We report two numbers:
  - test_top1_open    : argmax over BEiT's full 345-class label space (real pipeline behavior)
  - test_top1_restricted : argmax over only our 30 classes (best case for the baseline)
"""
import json
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

from common import setup_mlflow, load_meta, load_split, ROOT

DEVICE = "cuda"
MODEL_ID = "kmewhort/beit-sketch-classifier"


def norm(s):
    return s.lower().replace("_", " ").strip()


def main():
    meta = load_meta()
    classes = meta["classes"]
    n_classes = len(classes)
    Xte, yte = load_split("test")

    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16).to(DEVICE).eval()
    id2label = model.config.id2label
    label2id = {norm(v): k for k, v in id2label.items()}

    # map our classes -> BEiT indices
    our_to_beit = {}
    missing = []
    for c in classes:
        if norm(c) in label2id:
            our_to_beit[c] = label2id[norm(c)]
        else:
            missing.append(c)
    print(f"matched {len(our_to_beit)}/{n_classes} classes; missing={missing}", flush=True)
    beit_idx = [our_to_beit[c] for c in classes]  # column order = our label ids

    bs = 256
    open_preds_our = []   # mapped to our-id if BEiT's top1 is one of our classes else -1
    restricted_preds = []
    top3_open_correct = []
    beit_to_our = {bidx: i for i, bidx in enumerate(beit_idx)}
    with torch.no_grad():
        for s in range(0, len(Xte), bs):
            batch = Xte[s:s + bs]
            pil = [Image.fromarray(im).convert("RGB") for im in batch]
            inputs = proc(pil, return_tensors="pt").to(DEVICE)
            inputs = {k: v.half() for k, v in inputs.items()}
            logits = model(**inputs).logits.float()
            # open-set top1
            top1 = logits.argmax(-1).cpu().numpy()
            open_preds_our.extend([beit_to_our.get(int(t), -1) for t in top1])
            # open-set top3
            t3 = logits.topk(3, dim=-1).indices.cpu().numpy()
            for row, yt in zip(t3, yte[s:s + bs]):
                mapped = [beit_to_our.get(int(t), -1) for t in row]
                top3_open_correct.append(int(yt) in mapped)
            # restricted to our 30 classes
            sub = logits[:, beit_idx]
            restricted_preds.extend(sub.argmax(-1).cpu().numpy().tolist())
            if s % (bs * 20) == 0:
                print(f"  {s}/{len(Xte)}", flush=True)

    open_preds_our = np.array(open_preds_our)
    restricted_preds = np.array(restricted_preds)
    top1_open = float((open_preds_our == yte).mean())
    top3_open = float(np.mean(top3_open_correct))
    top1_restricted = float((restricted_preds == yte).mean())

    # per-class (open-set) accuracy + macro f1 restricted
    per_class_open = {}
    for c in range(n_classes):
        m = yte == c
        per_class_open[classes[c]] = float((open_preds_our[m] == c).mean()) if m.sum() else 0.0

    from sklearn.metrics import f1_score, confusion_matrix
    macro_f1_restricted = float(f1_score(yte, restricted_preds, average="macro"))
    cm = confusion_matrix(yte, restricted_preds, labels=list(range(n_classes)))

    out = ROOT / "artifacts" / "baseline_beit"
    out.mkdir(parents=True, exist_ok=True)
    (out / "per_class_acc_open.json").write_text(json.dumps(per_class_open, indent=2))
    np.save(out / "confusion_matrix_restricted.npy", cm)
    confusions = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i, j] > 0:
                confusions.append((classes[i], classes[j], int(cm[i, j])))
    confusions.sort(key=lambda x: -x[2])
    (out / "top_confusions_restricted.json").write_text(json.dumps(confusions[:40], indent=2))

    print(f"\nBASELINE BEiT on {len(yte)} test images:")
    print(f"  top1_open       = {top1_open:.4f}")
    print(f"  top3_open       = {top3_open:.4f}")
    print(f"  top1_restricted = {top1_restricted:.4f}")
    print(f"  macro_f1_restr  = {macro_f1_restricted:.4f}")

    mlflow = setup_mlflow("sketch-classifier")
    with mlflow.start_run(run_name="baseline_beit"):
        mlflow.log_params({
            "model": MODEL_ID, "mode": "pretrained_eval_only", "n_classes": n_classes,
            "n_test": len(yte), "matched_classes": len(our_to_beit),
            "beit_label_space": len(id2label),
        })
        mlflow.log_metric("test_top1_open", top1_open)
        mlflow.log_metric("test_top3_open", top3_open)
        mlflow.log_metric("test_top1_restricted", top1_restricted)
        mlflow.log_metric("test_macro_f1_restricted", macro_f1_restricted)
        # alias so it lines up with our trained runs' test_top1
        mlflow.log_metric("test_top1", top1_open)
        mlflow.log_artifacts(str(out), artifact_path="baseline")


if __name__ == "__main__":
    main()
