"""
BEiT Sketch Classifier Demo
Fetches 20 random QuickDraw sketches, classifies with BEiT, and outputs 2 canvases.
Each cell shows: input sketch with ground truth + predicted label + confidence.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import random
import json
import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# Import from v8_tailored
from v8_tailored import (
    BeitSketchClassifier, drawing_to_img, fetch_quickdraw, _PROMPT_DATA
)

ALL_CLASSES = _PROMPT_DATA.get("classes", [])
DEMO_OUT = Path(__file__).parent.parent / "demonstration"
DEMO_OUT.mkdir(parents=True, exist_ok=True)


def get_font(size):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def make_labeled_cell(sketch, gt_label, pred_label, pred_conf, top3, is_correct, cell_h=320):
    """Build a single cell: sketch + GT header + prediction footer."""
    w = sketch.width
    font_gt = get_font(15)
    font_pred = get_font(13)
    font_sm = get_font(11)

    header_h = 24
    footer_h = 44
    cell = Image.new("RGB", (w, header_h + w + footer_h), (30, 30, 30))
    draw = ImageDraw.Draw(cell)

    # Header: ground truth
    draw.rectangle([0, 0, w, header_h], fill=(40, 40, 40))
    gt_text = f"GT: {gt_label}"
    bb = draw.textbbox((0, 0), gt_text, font=font_gt)
    draw.text(((w - bb[2] + bb[0]) // 2, 4), gt_text, fill="white", font=font_gt)

    # Sketch
    cell.paste(sketch, (0, header_h))

    # Footer: prediction
    y = header_h + w
    color = (0, 200, 0) if is_correct else (220, 40, 40)
    draw.rectangle([0, y, w, y + footer_h], fill=(20, 20, 20))

    check = "Y" if is_correct else "X"
    line1 = f"[{check}] {pred_label} ({pred_conf:.0%})"
    draw.text((4, y + 3), line1, fill=color, font=font_pred)

    if len(top3) > 1:
        line2 = " | ".join(f"{l} ({s:.0%})" for l, s in top3[1:3])
        draw.text((4, y + 22), line2, fill=(160, 160, 160), font=font_sm)

    return cell


def make_canvas(cells, cols=5, padding=4):
    """Arrange cell images into a grid canvas."""
    if not cells:
        return Image.new("RGB", (100, 100), (30, 30, 30))
    cell_w, cell_h = cells[0].size
    rows = (len(cells) + cols - 1) // cols
    w = cols * (cell_w + padding) + padding
    h = rows * (cell_h + padding) + padding
    canvas = Image.new("RGB", (w, h), (30, 30, 30))

    for i, cell in enumerate(cells):
        r, c = divmod(i, cols)
        x = padding + c * (cell_w + padding)
        y = padding + r * (cell_h + padding)
        canvas.paste(cell, (x, y))

    return canvas


def run_demo(n=20, seed=42):
    random.seed(seed)

    # Pick 20 random classes
    if len(ALL_CLASSES) >= n:
        demo_classes = random.sample(ALL_CLASSES, n)
    else:
        demo_classes = random.choices(ALL_CLASSES or ["cat", "car", "flower", "house", "bird"], k=n)

    print(f"Loading BEiT sketch classifier...")
    classifier = BeitSketchClassifier(device="cuda")

    print(f"\nFetching {n} sketches from {n} random classes...")
    cells = []
    correct = 0

    for i, cls in enumerate(demo_classes):
        # Fetch 1 random sketch
        offset = random.randint(0, 500)
        try:
            drawings = fetch_quickdraw(cls, 1, offset)
        except Exception as e:
            print(f"  [{i+1}/{n}] {cls}: fetch failed ({e}), skipping")
            continue

        if not drawings:
            print(f"  [{i+1}/{n}] {cls}: no drawings returned, skipping")
            continue

        # Render sketch
        sketch = drawing_to_img(drawings[0], sz=256)

        # Classify
        preds = classifier.classify(sketch, top_k=3)
        pred_label = preds[0][0]
        pred_conf = preds[0][1]
        is_correct = pred_label == cls

        if is_correct:
            correct += 1

        # Build labeled cell
        labeled = make_labeled_cell(
            sketch, cls, pred_label, pred_conf, preds, is_correct
        )
        cells.append(labeled)

        status = "OK" if is_correct else "MISS"
        print(f"  [{i+1}/{n}] {cls} -> {pred_label} ({pred_conf:.1%}) [{status}]")

    if not cells:
        print("No cells generated!")
        return

    # Split into 2 canvases of 10
    half = len(cells) // 2
    cells1 = cells[:half] if half > 0 else cells
    cells2 = cells[half:] if half > 0 else []

    canvas1 = make_canvas(cells1, cols=5)
    canvas1_path = DEMO_OUT / "beit_demo_1.png"
    canvas1.save(canvas1_path)
    print(f"\nCanvas 1 saved: {canvas1_path}")

    if cells2:
        canvas2 = make_canvas(cells2, cols=5)
        canvas2_path = DEMO_OUT / "beit_demo_2.png"
        canvas2.save(canvas2_path)
        print(f"Canvas 2 saved: {canvas2_path}")

    acc = correct / len(cells) if cells else 0
    print(f"\nAccuracy: {correct}/{len(cells)} ({acc:.1%})")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    run_demo(n=n)
