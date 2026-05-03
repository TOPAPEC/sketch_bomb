"""EDA for 30-class Quick Draw sketch classification."""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "quick_draw"))

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from collections import defaultdict

from v8_tailored import BeitSketchClassifier, SiglipScorer, fetch_quickdraw, drawing_to_img

DEVICE = "cuda"
N_SAMPLES = 16
OUT = Path("eda_30c_out")
OUT.mkdir(exist_ok=True)

CLASSES = [
    "cookie", "broom", "violin", "camel", "steak",
    "trumpet", "bread", "banana", "arm", "flying saucer",
    "asparagus", "drums", "fish", "aircraft carrier", "backpack",
    "hat", "spider", "swing set", "telephone", "parachute",
    "sea turtle", "passport", "fence", "pliers", "hand",
    "fan", "face", "chair", "butterfly", "sword",
]

def get_font(size):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except:
        return ImageFont.load_default()

FONT = get_font(12)
FONT_SM = get_font(10)
FONT_LG = get_font(14)


def stage1_collect_data():
    """Fetch sketches, classify with BEiT, compute stroke stats."""
    print("=== Stage 1: Collecting data ===")
    beit = BeitSketchClassifier(DEVICE)

    all_data = []
    t0 = time.time()

    for ci, cls in enumerate(CLASSES):
        raw_drawings = fetch_quickdraw(cls, N_SAMPLES, offset=0)
        for si, raw in enumerate(raw_drawings):
            sketch = drawing_to_img(raw, sz=256)

            # Stroke stats
            strokes = raw["drawing"]
            n_strokes = len(strokes)
            n_points = sum(len(s[0]) for s in strokes)
            all_x = [x for s in strokes for x in s[0]]
            all_y = [y for s in strokes for y in s[1]]
            bbox_w = max(all_x) - min(all_x) if all_x else 0
            bbox_h = max(all_y) - min(all_y) if all_y else 0
            coverage = (bbox_w * bbox_h) / (255 * 255) if bbox_w and bbox_h else 0

            # BEiT classify
            preds = beit.classify(sketch, top_k=5)

            all_data.append({
                "cls": cls,
                "idx": si,
                "n_strokes": n_strokes,
                "n_points": n_points,
                "bbox_coverage": round(coverage, 3),
                "beit_top1": preds[0][0],
                "beit_top1_conf": round(preds[0][1], 4),
                "beit_top5": [(l, round(s, 4)) for l, s in preds],
                "beit_correct": preds[0][0] == cls,
                "beit_top3_correct": cls in [p[0] for p in preds[:3]],
                "beit_top5_correct": cls in [p[0] for p in preds],
            })

        elapsed = time.time() - t0
        eta = elapsed / (ci + 1) * (len(CLASSES) - ci - 1)
        print(f"  [{ci+1}/{len(CLASSES)}] {cls:20s} acc={sum(1 for d in all_data if d['cls']==cls and d['beit_correct'])}/{N_SAMPLES}  (ETA {eta:.0f}s)")

    print(f"  Total: {time.time()-t0:.0f}s")
    return all_data


def stage2_analysis(data):
    """Compute aggregate stats."""
    print("\n=== Stage 2: Analysis ===")

    # Per-class stats
    cls_stats = {}
    for cls in CLASSES:
        items = [d for d in data if d["cls"] == cls]
        acc = np.mean([d["beit_correct"] for d in items])
        top3_acc = np.mean([d["beit_top3_correct"] for d in items])
        top5_acc = np.mean([d["beit_top5_correct"] for d in items])
        mean_conf = np.mean([d["beit_top1_conf"] for d in items])
        mean_conf_correct = np.mean([d["beit_top1_conf"] for d in items if d["beit_correct"]]) if acc > 0 else 0
        mean_conf_wrong = np.mean([d["beit_top1_conf"] for d in items if not d["beit_correct"]]) if acc < 1 else 0
        mean_strokes = np.mean([d["n_strokes"] for d in items])
        mean_points = np.mean([d["n_points"] for d in items])
        mean_coverage = np.mean([d["bbox_coverage"] for d in items])

        # Confusion: where do errors go?
        confusion = defaultdict(int)
        for d in items:
            if not d["beit_correct"]:
                confusion[d["beit_top1"]] += 1

        cls_stats[cls] = {
            "accuracy": round(acc, 3),
            "top3_accuracy": round(top3_acc, 3),
            "top5_accuracy": round(top5_acc, 3),
            "mean_conf": round(mean_conf, 3),
            "mean_conf_correct": round(mean_conf_correct, 3),
            "mean_conf_wrong": round(mean_conf_wrong, 3),
            "mean_strokes": round(mean_strokes, 1),
            "mean_points": round(mean_points, 1),
            "mean_coverage": round(mean_coverage, 3),
            "top_confusions": dict(sorted(confusion.items(), key=lambda x: -x[1])[:5]),
            "n_correct": sum(1 for d in items if d["beit_correct"]),
            "n_total": len(items),
        }

    # Overall
    overall_acc = np.mean([d["beit_correct"] for d in data])
    overall_top3 = np.mean([d["beit_top3_correct"] for d in data])
    overall_top5 = np.mean([d["beit_top5_correct"] for d in data])

    # Correlation: strokes vs accuracy
    cls_strokes = [cls_stats[c]["mean_strokes"] for c in CLASSES]
    cls_acc = [cls_stats[c]["accuracy"] for c in CLASSES]
    corr_strokes = np.corrcoef(cls_strokes, cls_acc)[0, 1]

    cls_points = [cls_stats[c]["mean_points"] for c in CLASSES]
    corr_points = np.corrcoef(cls_points, cls_acc)[0, 1]

    cls_coverage = [cls_stats[c]["mean_coverage"] for c in CLASSES]
    corr_coverage = np.corrcoef(cls_coverage, cls_acc)[0, 1]

    # Confidence analysis
    correct_confs = [d["beit_top1_conf"] for d in data if d["beit_correct"]]
    wrong_confs = [d["beit_top1_conf"] for d in data if not d["beit_correct"]]
    confident_wrong = [d for d in data if not d["beit_correct"] and d["beit_top1_conf"] > 0.5]

    # Global confusion matrix
    confusion_matrix = defaultdict(int)
    for d in data:
        if not d["beit_correct"]:
            confusion_matrix[(d["cls"], d["beit_top1"])] += 1

    top_confusions = sorted(confusion_matrix.items(), key=lambda x: -x[1])[:20]

    analysis = {
        "overall": {
            "accuracy": round(overall_acc, 3),
            "top3_accuracy": round(overall_top3, 3),
            "top5_accuracy": round(overall_top5, 3),
            "total_samples": len(data),
            "total_correct": sum(1 for d in data if d["beit_correct"]),
        },
        "correlations": {
            "strokes_vs_accuracy": round(corr_strokes, 3),
            "points_vs_accuracy": round(corr_points, 3),
            "coverage_vs_accuracy": round(corr_coverage, 3),
        },
        "confidence": {
            "mean_correct": round(np.mean(correct_confs), 3),
            "mean_wrong": round(np.mean(wrong_confs), 3),
            "median_correct": round(np.median(correct_confs), 3),
            "median_wrong": round(np.median(wrong_confs), 3),
            "confident_wrong_count": len(confident_wrong),
            "confident_wrong_pct": round(len(confident_wrong) / len(data), 3),
        },
        "top_confusions": [(f"{a} -> {b}", n) for (a, b), n in top_confusions],
        "per_class": cls_stats,
    }

    # Print summary
    print(f"  Overall accuracy: {overall_acc:.1%} (top3: {overall_top3:.1%}, top5: {overall_top5:.1%})")
    print(f"  Correlations: strokes={corr_strokes:.2f}, points={corr_points:.2f}, coverage={corr_coverage:.2f}")
    print(f"  Confidence: correct={np.mean(correct_confs):.2f}, wrong={np.mean(wrong_confs):.2f}")
    print(f"  Confident errors (>50%): {len(confident_wrong)}/{len(data)} ({len(confident_wrong)/len(data):.1%})")
    print(f"\n  Top confusions:")
    for pair, n in top_confusions[:10]:
        print(f"    {pair}: {n}")

    return analysis


def stage3_visuals(data, analysis):
    """Generate visual collages."""
    print("\n=== Stage 3: Generating visuals ===")

    # Re-fetch and render sketches (cheap, <1s per class)
    sketch_cache = {}
    for cls in CLASSES:
        raw_drawings = fetch_quickdraw(cls, N_SAMPLES, offset=0)
        for i, raw in enumerate(raw_drawings):
            sketch_cache[(cls, i)] = drawing_to_img(raw, sz=200)

    # 1. Per-class collage: 4x4 grid with BEiT labels
    print("  Building per-class collages...")
    cell = 200
    label_h = 18
    for cls in CLASSES:
        items = [d for d in data if d["cls"] == cls]
        cols, rows = 4, 4
        w = cols * cell
        h = rows * (cell + label_h)
        grid = Image.new("RGB", (w, h), (255, 255, 255))
        draw = ImageDraw.Draw(grid)

        for i, d in enumerate(items):
            col, row = i % cols, i // cols
            x, y = col * cell, row * (cell + label_h)
            sketch = sketch_cache[(cls, i)]
            grid.paste(sketch, (x, y))

            pred = d["beit_top1"]
            conf = d["beit_top1_conf"]
            color = (0, 120, 0) if d["beit_correct"] else (200, 0, 0)
            label = f"{pred} ({conf:.0%})"
            draw.text((x + 2, y + cell + 1), label, fill=color, font=FONT_SM)

        acc = analysis["per_class"][cls]["accuracy"]
        safe = cls.replace(" ", "_")
        grid.save(OUT / f"class_{safe}.png")

    # 2. Accuracy overview: all classes sorted by accuracy
    print("  Building accuracy overview...")
    sorted_cls = sorted(CLASSES, key=lambda c: analysis["per_class"][c]["accuracy"])
    bar_w = 500
    row_h = 22
    label_w = 150
    w = label_w + bar_w + 80
    h = len(CLASSES) * row_h + 40
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((10, 5), "BEiT Top-1 Accuracy by Class", fill=(0, 0, 0), font=FONT_LG)

    for i, cls in enumerate(sorted_cls):
        y = 35 + i * row_h
        acc = analysis["per_class"][cls]["accuracy"]
        bar_len = int(acc * bar_w)
        color = (46, 139, 87) if acc >= 0.75 else (200, 160, 0) if acc >= 0.5 else (200, 50, 50)
        draw.text((5, y + 2), f"{cls}", fill=(0, 0, 0), font=FONT)
        draw.rectangle([(label_w, y + 2), (label_w + bar_len, y + row_h - 4)], fill=color)
        draw.text((label_w + bar_len + 5, y + 2), f"{acc:.0%}", fill=(100, 100, 100), font=FONT_SM)

    img.save(OUT / "accuracy_bars.png")

    # 3. Confusion pairs visual: sketch -> what BEiT thinks
    print("  Building confusion examples...")
    top_conf = analysis["top_confusions"][:12]
    row_h_conf = 210
    header = 30
    w = 220 * 4 + 30
    h = header + len(top_conf) * row_h_conf
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((10, 5), "Top BEiT Confusions (true -> predicted, N times)", fill=(0, 0, 0), font=FONT_LG)

    for ci, (pair_str, count) in enumerate(top_conf):
        y = header + ci * row_h_conf
        true_cls, pred_cls = pair_str.split(" -> ")
        draw.text((5, y + 5), f"{pair_str} (x{count})", fill=(200, 0, 0), font=FONT)

        # Show up to 4 examples of this confusion
        examples = [d for d in data if d["cls"] == true_cls and d["beit_top1"] == pred_cls][:4]
        for ei, d in enumerate(examples):
            sketch = sketch_cache.get((true_cls, d["idx"]))
            if sketch:
                small = sketch.resize((190, 190), Image.LANCZOS)
                img.paste(small, (10 + ei * 220, y + 20))

    img.save(OUT / "confusion_examples.png")

    # 4. Stroke complexity scatter data (for report, text-based)
    print("  Building complexity analysis...")
    cell = 210
    cols = 6
    rows = 5
    w = cols * cell
    h = rows * (cell + 30)
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    sorted_by_strokes = sorted(CLASSES, key=lambda c: analysis["per_class"][c]["mean_strokes"])
    for i, cls in enumerate(sorted_by_strokes):
        col, row = i % cols, i // cols
        x, y = col * cell, row * (cell + 30)
        stats = analysis["per_class"][cls]

        # Show one example sketch
        sketch = sketch_cache[(cls, 0)].resize((cell - 10, cell - 10), Image.LANCZOS)
        img.paste(sketch, (x + 5, y))

        acc = stats["accuracy"]
        strokes = stats["mean_strokes"]
        color = (46, 139, 87) if acc >= 0.75 else (200, 160, 0) if acc >= 0.5 else (200, 50, 50)
        draw.text((x + 2, y + cell - 8), f"{cls}", fill=(0, 0, 0), font=FONT_SM)
        draw.text((x + 2, y + cell + 4), f"{strokes:.0f} strokes, {acc:.0%}", fill=color, font=FONT_SM)

    img.save(OUT / "complexity_grid.png")

    # 5. Confidence histogram (correct vs wrong)
    print("  Building confidence analysis...")
    correct_confs = [d["beit_top1_conf"] for d in data if d["beit_correct"]]
    wrong_confs = [d["beit_top1_conf"] for d in data if not d["beit_correct"]]

    bins = np.arange(0, 1.05, 0.1)
    w, h = 600, 300
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((10, 5), "BEiT Confidence: Correct (green) vs Wrong (red)", fill=(0, 0, 0), font=FONT_LG)

    hist_correct, _ = np.histogram(correct_confs, bins=bins)
    hist_wrong, _ = np.histogram(wrong_confs, bins=bins)
    max_val = max(max(hist_correct), max(hist_wrong)) + 1

    bar_w = 45
    chart_h = 220
    chart_y = 50
    for i in range(len(bins) - 1):
        x = 50 + i * (bar_w + 8)
        # Correct bar (green)
        h_c = int(hist_correct[i] / max_val * chart_h)
        draw.rectangle([(x, chart_y + chart_h - h_c), (x + bar_w // 2 - 1, chart_y + chart_h)],
                       fill=(46, 139, 87))
        # Wrong bar (red)
        h_w = int(hist_wrong[i] / max_val * chart_h)
        draw.rectangle([(x + bar_w // 2, chart_y + chart_h - h_w), (x + bar_w - 1, chart_y + chart_h)],
                       fill=(200, 50, 50))
        # Label
        draw.text((x, chart_y + chart_h + 5), f"{bins[i]:.1f}", fill=(100, 100, 100), font=FONT_SM)
        # Count labels
        if hist_correct[i] > 0:
            draw.text((x, chart_y + chart_h - h_c - 12), str(hist_correct[i]), fill=(46, 139, 87), font=FONT_SM)
        if hist_wrong[i] > 0:
            draw.text((x + bar_w // 2, chart_y + chart_h - h_w - 12), str(hist_wrong[i]), fill=(200, 50, 50), font=FONT_SM)

    img.save(OUT / "confidence_histogram.png")

    # 6. Worst confident errors
    print("  Building worst errors grid...")
    confident_wrong = sorted(
        [d for d in data if not d["beit_correct"]],
        key=lambda d: -d["beit_top1_conf"]
    )[:16]

    cell = 200
    cols, rows = 4, 4
    w = cols * (cell + 5)
    h = rows * (cell + 35)
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    for i, d in enumerate(confident_wrong):
        col, row = i % cols, i // cols
        x, y = col * (cell + 5), row * (cell + 35)
        sketch = sketch_cache[(d["cls"], d["idx"])]
        img.paste(sketch, (x, y))
        draw.text((x + 2, y + cell + 1),
                  f"true: {d['cls']}", fill=(0, 0, 0), font=FONT_SM)
        draw.text((x + 2, y + cell + 13),
                  f"pred: {d['beit_top1']} ({d['beit_top1_conf']:.0%})",
                  fill=(200, 0, 0), font=FONT_SM)

    img.save(OUT / "worst_confident_errors.png")

    print("  Done!")


if __name__ == "__main__":
    t_start = time.time()

    data = stage1_collect_data()
    analysis = stage2_analysis(data)

    # Save raw data
    with open(OUT / "eda_data.json", "w") as f:
        json.dump({"data": data, "analysis": analysis}, f, indent=2)
    print(f"\n  Data saved to {OUT}/eda_data.json")

    stage3_visuals(data, analysis)

    elapsed = time.time() - t_start
    print(f"\nTotal EDA time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Output: {OUT}/")
