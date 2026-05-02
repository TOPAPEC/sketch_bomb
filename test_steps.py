"""Compare generation quality at 10/20/30 steps. 10 images, 2x5 grid per step count."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "quick_draw"))

import torch
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from demo_beit_compare import (
    load_sd15, to_lineart_sd15, get_prompt, get_negative_sd15, drawing_to_img_small,
)
from v8_tailored import fetch_quickdraw, SiglipScorer, score_prompt

DEVICE = "cuda"
SEED = 42
N = 10
STEPS_LIST = [10, 20, 30]

CLASSES = ["cat", "car", "flower", "house", "bird", "apple", "fish", "dog", "tree", "butterfly"]

print("Fetching sketches...")
samples = []
for cls in CLASSES[:N]:
    drawings = fetch_quickdraw(cls, 1, offset=5)
    sketch = drawing_to_img_small(drawings[0], 256)
    ctrl = to_lineart_sd15(sketch.resize((512, 512), Image.LANCZOS))
    samples.append({"cls": cls, "sketch": sketch, "ctrl": ctrl})

print("Loading SD1.5...")
pipe, refiner = load_sd15(DEVICE)

results = {}
for steps in STEPS_LIST:
    print(f"\n=== {steps} STEPS ===")
    imgs = []
    t0 = time.time()
    for i, s in enumerate(samples):
        img = pipe(
            prompt=get_prompt(s["cls"]),
            negative_prompt=get_negative_sd15(s["cls"]),
            image=s["ctrl"],
            num_inference_steps=steps,
            guidance_scale=7.5,
            controlnet_conditioning_scale=0.7,
            control_guidance_start=0.0,
            control_guidance_end=0.8,
            generator=torch.Generator(device=DEVICE).manual_seed(SEED),
            num_images_per_prompt=1,
        ).images[0]
        # Refiner with proportional steps
        ref_steps = max(5, steps // 2)
        img = refiner(
            prompt=get_prompt(s["cls"]),
            negative_prompt=get_negative_sd15(s["cls"]),
            image=img,
            strength=0.25,
            num_inference_steps=ref_steps,
            guidance_scale=4.0,
            generator=torch.Generator(device=DEVICE).manual_seed(SEED),
        ).images[0]
        imgs.append(img)
    elapsed = time.time() - t0
    results[steps] = {"images": imgs, "time": elapsed}
    print(f"  {elapsed:.1f}s total, {elapsed/N:.2f}s/img")

# Score with SigLIP2
print("\nScoring with SigLIP2...")
del pipe, refiner
torch.cuda.empty_cache()
scorer = SiglipScorer(device=DEVICE)
for steps in STEPS_LIST:
    scores = []
    for i, s in enumerate(samples):
        sc = scorer.score_single(results[steps]["images"][i], score_prompt(s["cls"]))
        scores.append(sc)
    results[steps]["scores"] = scores
    mean_sc = np.mean(scores)
    print(f"  {steps} steps: mean SigLIP2 = {mean_sc:.4f} ({results[steps]['time']/N:.2f}s/img)")
del scorer
torch.cuda.empty_cache()

# Build comparison grid: rows = samples, cols = step counts
def get_font(size):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except:
        return ImageFont.load_default()

cell_w, cell_h = 256, 256
header = 35
label_w = 80
cols = len(STEPS_LIST)
rows = N
font = get_font(16)
font_sm = get_font(12)

grid = Image.new("RGB",
    (label_w + cols * cell_w, header + rows * cell_h),
    (255, 255, 255))
draw = ImageDraw.Draw(grid)

# Header
for j, steps in enumerate(STEPS_LIST):
    x = label_w + j * cell_w + cell_w // 2
    mean_sc = np.mean(results[steps]["scores"])
    t_per = results[steps]["time"] / N
    draw.text((x - 60, 5), f"{steps} steps", fill="black", font=font)
    draw.text((x - 60, 20), f"SigLIP {mean_sc:.3f} | {t_per:.1f}s", fill="gray", font=font_sm)

# Cells
for i, s in enumerate(samples):
    y = header + i * cell_h
    draw.text((5, y + cell_h // 2 - 8), s["cls"], fill="black", font=font)
    for j, steps in enumerate(STEPS_LIST):
        x = label_w + j * cell_w
        img = results[steps]["images"][i].resize((cell_w, cell_h), Image.LANCZOS)
        grid.paste(img, (x, y))
        sc = results[steps]["scores"][i]
        draw.text((x + 3, y + cell_h - 15), f"{sc:.3f}", fill="yellow", font=font_sm)

grid.save("test_steps_comparison.png")
print(f"\nSaved: test_steps_comparison.png ({grid.size[0]}x{grid.size[1]})")
print("Done!")
