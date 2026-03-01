"""
BEiT + v8 Pipeline Demo
Fetches N random QuickDraw sketches, classifies with BEiT, runs full pipeline
(DomainNet match → ControlNet generation → refinement),
and outputs a single canvas where each ROW shows all steps for one image.

Columns: QuickDraw Sketch | BEiT Prediction | Matched DomainNet | Generated Result
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import random
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

from v8_tailored import (
    Pipeline, DomainNetMatcher, fetch_quickdraw, drawing_to_img,
    to_lineart, remove_bg, _PROMPT_DATA
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


def add_label(image, text, position="top", color="white", bg_color=(40, 40, 40),
              font_size=14, height=24):
    """Add a label bar to top or bottom of an image."""
    w = image.width
    font = get_font(font_size)
    bar = Image.new("RGB", (w, height), bg_color)
    draw = ImageDraw.Draw(bar)
    bb = draw.textbbox((0, 0), text, font=font)
    text_w = bb[2] - bb[0]
    text_h = bb[3] - bb[1]
    x = (w - text_w) // 2
    y = (height - text_h) // 2
    draw.text((x, y), text, fill=color, font=font)

    if position == "top":
        combined = Image.new("RGB", (w, height + image.height))
        combined.paste(bar, (0, 0))
        combined.paste(image, (0, height))
    else:
        combined = Image.new("RGB", (w, image.height + height))
        combined.paste(image, (0, 0))
        combined.paste(bar, (0, image.height))
    return combined


def make_row(qd_sketch, beit_info, dn_sketch, generated, cell_size=256):
    """Build one row: [QuickDraw Sketch] [BEiT Classification] [DomainNet Match] [Generated Result]"""
    qd = qd_sketch.resize((cell_size, cell_size), Image.LANCZOS)
    dn = dn_sketch.resize((cell_size, cell_size), Image.LANCZOS)
    gen = generated.resize((cell_size, cell_size), Image.LANCZOS)

    beit_cell = qd.copy()
    gt_label = beit_info["gt"]
    pred_label = beit_info["pred"]
    pred_conf = beit_info["conf"]
    top3 = beit_info["top3"]
    is_correct = pred_label == gt_label

    qd_labeled = add_label(qd, f"GT: {gt_label}", position="top")

    check = "Y" if is_correct else "X"
    pred_color = (0, 200, 0) if is_correct else (220, 40, 40)
    line1 = f"[{check}] {pred_label} ({pred_conf:.0%})"
    beit_labeled = add_label(beit_cell, line1, position="top",
                             color=pred_color, font_size=13)
    if len(top3) > 1:
        line2 = " | ".join(f"{l} {s:.0%}" for l, s in top3[1:3])
        beit_labeled = add_label(beit_labeled, line2, position="bottom",
                                 color=(180, 180, 180), bg_color=(25, 25, 25),
                                 font_size=11, height=20)

    dn_labeled = add_label(dn, "DomainNet Match", position="top",
                           color=(200, 200, 255), bg_color=(30, 30, 50))

    gen_labeled = add_label(gen, "Generated", position="top",
                            color=(200, 255, 200), bg_color=(30, 50, 30))

    return [qd_labeled, beit_labeled, dn_labeled, gen_labeled]


def make_canvas(rows, headers, padding=6, bg_color=(30, 30, 30)):
    """
    Build a canvas from rows of cell images.
    Each row is a list of PIL images (same height per row).
    """
    if not rows:
        return Image.new("RGB", (100, 100), bg_color)

    cols = len(rows[0])
    col_widths = [0] * cols
    for row in rows:
        for c, cell in enumerate(row):
            col_widths[c] = max(col_widths[c], cell.width)

    header_font = get_font(16)
    header_h = 30

    total_w = padding + sum(w + padding for w in col_widths)
    total_h = padding + header_h

    row_heights = []
    for row in rows:
        rh = max(cell.height for cell in row)
        row_heights.append(rh)
        total_h += rh + padding

    canvas = Image.new("RGB", (total_w, total_h), bg_color)
    draw = ImageDraw.Draw(canvas)

    x = padding
    for c in range(cols):
        header_text = headers[c] if c < len(headers) else ""
        bb = draw.textbbox((0, 0), header_text, font=header_font)
        text_w = bb[2] - bb[0]
        hx = x + (col_widths[c] - text_w) // 2
        draw.text((hx, padding + 4), header_text, fill=(220, 220, 220), font=header_font)
        x += col_widths[c] + padding

    y = padding + header_h
    for r, row in enumerate(rows):
        x = padding
        for c, cell in enumerate(row):
            cx = x + (col_widths[c] - cell.width) // 2
            cy = y + (row_heights[r] - cell.height) // 2
            canvas.paste(cell, (cx, cy))
            x += col_widths[c] + padding
        y += row_heights[r] + padding

    return canvas


def run_demo(n=10, seed=42, model_key="sd15"):
    random.seed(seed)

    if len(ALL_CLASSES) >= n:
        demo_classes = random.sample(ALL_CLASSES, n)
    else:
        demo_classes = random.choices(ALL_CLASSES or ["cat", "car", "flower", "house", "bird"], k=n)

    print(f"Loading pipeline (model={model_key}, BEiT=on)...")
    pipe = Pipeline(device="cuda", use_beit=True)
    pipe.load(model_key)

    print("Building DomainNet index for demo classes...")
    matcher = DomainNetMatcher(pipe.scorer, demo_classes, batch_size=16)

    print(f"\nProcessing {n} sketches from {n} random classes...")
    headers = ["QuickDraw Sketch", "BEiT Classification", "Matched DomainNet", "Generated Result"]
    rows = []
    correct = 0

    for i, cls in enumerate(demo_classes):
        offset = random.randint(0, 500)
        try:
            drawings = fetch_quickdraw(cls, 1, offset)
        except Exception as e:
            print(f"  [{i+1}/{n}] {cls}: fetch failed ({e}), skipping")
            continue
        if not drawings:
            print(f"  [{i+1}/{n}] {cls}: no drawings, skipping")
            continue

        # 1. Render QuickDraw sketch
        qd_sketch = drawing_to_img(drawings[0], sz=256)

        # 2. BEiT classification
        preds = pipe.classify_sketch(qd_sketch, top_k=3)
        pred_label = preds[0][0]
        pred_conf = preds[0][1]
        is_correct = pred_label == cls
        if is_correct:
            correct += 1

        beit_info = {
            "gt": cls,
            "pred": pred_label,
            "conf": pred_conf,
            "top3": preds,
        }

        # 3. DomainNet match via SigLIP2 embeddings
        dn_sketch, dn_path, sim = matcher.match(qd_sketch, cls)

        # 4. Generate single image with ControlNet + refine
        ctrl = to_lineart(dn_sketch)
        gen_seed = random.randint(0, 999999)
        gen_imgs = pipe.generate(ctrl, cls, seed=gen_seed, batch=1)
        gen_img = pipe.refine(gen_imgs[0], cls, seed=gen_seed)
        gen_img = remove_bg(gen_img)

        status = "OK" if is_correct else "MISS"
        print(f"  [{i+1}/{n}] {cls} -> BEiT:{pred_label}({pred_conf:.0%}) "
              f"DN:{dn_path.name}(sim={sim:.3f}) [{status}]")

        row = make_row(qd_sketch, beit_info, dn_sketch, gen_img, cell_size=256)
        rows.append(row)

    if not rows:
        print("No rows generated!")
        return

    canvas = make_canvas(rows, headers=headers)
    canvas_path = DEMO_OUT / "pipeline_beit_demo.png"
    canvas.save(canvas_path)
    print(f"\nCanvas saved: {canvas_path}")

    acc = correct / len(rows) if rows else 0
    print(f"BEiT Accuracy: {correct}/{len(rows)} ({acc:.1%})")
    print(f"Pipeline complete!")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    model = sys.argv[2] if len(sys.argv) > 2 else "sd15"
    run_demo(n=n, model_key=model)
