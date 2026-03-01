"""
BEiT + SDXL Pipeline Demo (no background removal)
Same as demo_pipeline_beit_sdxl.py but skips rembg to keep original backgrounds.

Columns: QuickDraw Sketch | BEiT Classification | Matched DomainNet | Generated Result
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import random
import json
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from diffusers import (StableDiffusionXLControlNetPipeline, StableDiffusionXLImg2ImgPipeline,
                       ControlNetModel, AutoencoderKL, EulerDiscreteScheduler)
from rembg import remove as rembg_remove
import cv2

from v8_tailored import (
    BeitSketchClassifier, SiglipScorer, DomainNetMatcher,
    fetch_quickdraw, _PROMPT_DATA
)

ALL_CLASSES = _PROMPT_DATA.get("classes", [])
PROMPTS = _PROMPT_DATA.get("prompts", {})
NEGATIVES = _PROMPT_DATA.get("negatives", {})
# SDXL doesn't use easynegative (SD1.5-only textual inversion)
DEFAULT_NEG = (
    "(worst quality, low quality:1.4), people, human, text, watermark, "
    "ugly, blurry, deformed, extra limbs, bad anatomy"
)

DEMO_OUT = Path(__file__).parent.parent / "demonstration"
DEMO_OUT.mkdir(parents=True, exist_ok=True)


def get_prompt(label):
    return PROMPTS.get(label, f"a single {label}, solo, centered, (digital illustration:1.2), detailed, sharp focus, vivid colors")


def get_negative(label):
    neg = NEGATIVES.get(label, DEFAULT_NEG)
    # Strip easynegative token from SD1.5 negatives since SDXL can't use it
    return neg.replace("easynegative, ", "").replace("easynegative,", "").replace("easynegative", "")


def remove_bg(img, bg=(255, 255, 255)):
    rgba = rembg_remove(img)
    out = Image.new('RGBA', rgba.size, (*bg, 255))
    out.paste(rgba, mask=rgba.split()[3])
    return out.convert('RGB')


def drawing_to_img(d, sz=1024):
    """Render QuickDraw drawing at SDXL native resolution."""
    img = Image.new('RGB', (sz, sz), 'white')
    draw = ImageDraw.Draw(img)
    strokes = d['drawing']
    xs = [x for s in strokes for x in s[0]]
    ys = [y for s in strokes for y in s[1]]
    if not xs:
        return img
    mnx, mxx, mny, mxy = min(xs), max(xs), min(ys), max(ys)
    sc = sz * 0.8 / max(mxx - mnx, mxy - mny, 1)
    ox = (sz - (mxx - mnx) * sc) / 2 - mnx * sc
    oy = (sz - (mxy - mny) * sc) / 2 - mny * sc
    for s in strokes:
        pts = [(x * sc + ox, y * sc + oy) for x, y in zip(s[0], s[1])]
        if len(pts) > 1:
            draw.line(pts, fill='black', width=max(3, sz // 170))
    return img


def to_lineart(image):
    """Convert to lineart with thicker dilation for SDXL."""
    arr = np.array(image.convert('L'))
    _, b = cv2.threshold(arr, 200, 255, cv2.THRESH_BINARY_INV)
    b = cv2.dilate(b, np.ones((3, 3), np.uint8), iterations=1)
    return Image.fromarray(b).convert('RGB')


class SDXLPipeline:
    """SDXL + MistoLine ControlNet pipeline (from v9_sdxl.py)."""

    def __init__(self, device="cuda"):
        self.device = device
        self.pipe = None
        self.refiner = None

    def load(self):
        print("Loading SDXL + MistoLine ControlNet...")
        cn = ControlNetModel.from_pretrained(
            "TheMistoAI/MistoLine", torch_dtype=torch.float16, variant="fp16").to(self.device)
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", controlnet=cn, vae=vae,
            torch_dtype=torch.float16, variant="fp16", safety_checker=None).to(self.device)
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_attention_slicing()
        comps = {k: v for k, v in self.pipe.components.items() if k != "controlnet"}
        self.refiner = StableDiffusionXLImg2ImgPipeline(**comps)
        print("SDXL Ready")

    def generate(self, ctrl, label, seed=42, steps=30, cn_scale=0.5, guidance=7.0):
        return self.pipe(
            prompt=get_prompt(label), negative_prompt=get_negative(label),
            image=ctrl, num_inference_steps=steps,
            guidance_scale=guidance, controlnet_conditioning_scale=cn_scale,
            control_guidance_start=0.0, control_guidance_end=min(1.0, 12 / steps),
            generator=torch.Generator(device=self.device).manual_seed(seed),
            num_images_per_prompt=1,
        ).images[0]

    def refine_img(self, img, label, seed=42, strength=0.25):
        return self.refiner(
            prompt=get_prompt(label), negative_prompt=get_negative(label),
            image=img, strength=strength, num_inference_steps=15, guidance_scale=4.0,
            generator=torch.Generator(device=self.device).manual_seed(seed),
        ).images[0]


# ── Canvas helpers (same as demo_pipeline_beit.py) ──

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
    gen_labeled = add_label(gen, "Generated (SDXL)", position="top",
                            color=(200, 255, 200), bg_color=(30, 50, 30))

    return [qd_labeled, beit_labeled, dn_labeled, gen_labeled]


def make_canvas(rows, headers, padding=6, bg_color=(30, 30, 30)):
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


# ── Main ──

def run_demo(n=10, seed=42):
    random.seed(seed)

    if len(ALL_CLASSES) >= n:
        demo_classes = random.sample(ALL_CLASSES, n)
    else:
        demo_classes = random.choices(ALL_CLASSES or ["cat", "car", "flower", "house", "bird"], k=n)

    # Load BEiT classifier
    print("Loading BEiT classifier...")
    beit = BeitSketchClassifier(device="cuda")

    # Load SigLIP2 scorer (for DomainNet matching)
    print("Loading SigLIP2 scorer...")
    scorer = SiglipScorer(device="cuda")

    # Load SDXL pipeline
    sdxl = SDXLPipeline(device="cuda")
    sdxl.load()

    # Build DomainNet index
    print("Building DomainNet index for demo classes...")
    matcher = DomainNetMatcher(scorer, demo_classes, batch_size=16)

    print(f"\nProcessing {n} sketches from {n} random classes (SDXL)...")
    headers = ["QuickDraw Sketch", "BEiT Classification", "Matched DomainNet", "Generated (SDXL)"]
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

        # 1. Render QuickDraw sketch (256 for BEiT, 1024 for SDXL)
        from v8_tailored import drawing_to_img as drawing_to_img_small
        qd_sketch_small = drawing_to_img_small(drawings[0], sz=256)
        qd_sketch_large = drawing_to_img(drawings[0], sz=1024)

        # 2. BEiT classification (on small sketch)
        preds = beit.classify(qd_sketch_small, top_k=3)
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
        dn_sketch, dn_path, sim = matcher.match(qd_sketch_small, cls)

        # 4. Generate with SDXL ControlNet + refine
        # Resize DomainNet match to 1024 for SDXL
        dn_large = dn_sketch.resize((1024, 1024), Image.LANCZOS)
        ctrl = to_lineart(dn_large)
        gen_seed = random.randint(0, 999999)
        gen_img = sdxl.generate(ctrl, cls, seed=gen_seed)
        gen_img = sdxl.refine_img(gen_img, cls, seed=gen_seed)
        # No background removal in this variant

        status = "OK" if is_correct else "MISS"
        print(f"  [{i+1}/{n}] {cls} -> BEiT:{pred_label}({pred_conf:.0%}) "
              f"DN:{dn_path.name}(sim={sim:.3f}) [{status}]")

        row = make_row(qd_sketch_small, beit_info, dn_sketch, gen_img, cell_size=256)
        rows.append(row)

    if not rows:
        print("No rows generated!")
        return

    canvas = make_canvas(rows, headers=headers)
    canvas_path = DEMO_OUT / "pipeline_beit_sdxl_nobg_demo.png"
    canvas.save(canvas_path)
    print(f"\nCanvas saved: {canvas_path}")

    acc = correct / len(rows) if rows else 0
    print(f"BEiT Accuracy: {correct}/{len(rows)} ({acc:.1%})")
    print(f"SDXL Pipeline complete!")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run_demo(n=n)
