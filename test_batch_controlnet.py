"""Quick test: batch ControlNet generation with different control images + different seeds."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "quick_draw"))

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from demo_beit_compare import (
    load_sd15, generate_sd15, to_lineart_sd15,
    get_prompt, get_negative_sd15, drawing_to_img_small,
)
from v8_tailored import fetch_quickdraw

DEVICE = "cuda"
BATCH = 4
LABEL = "cat"
SEED = 42

print("Fetching 4 different cat sketches from QuickDraw...")
drawings = fetch_quickdraw("cat", BATCH, offset=0)
sketches = [drawing_to_img_small(d, 256) for d in drawings]

# Different control images from different sketches
ctrls = []
for sk in sketches:
    big = sk.resize((512, 512), Image.LANCZOS)
    ctrl = to_lineart_sd15(big)
    ctrls.append(ctrl)

print("Loading SD1.5...")
pipe, refiner = load_sd15(DEVICE)

# ── Test 1: Sequential (baseline) ──
print("\n=== SEQUENTIAL (4 separate calls) ===")
seq_results = []
import time
t0 = time.time()
for i in range(BATCH):
    img = generate_sd15(pipe, refiner, ctrls[i], LABEL, SEED + i, DEVICE)
    seq_results.append(img)
t_seq = time.time() - t0
print(f"Sequential: {t_seq:.1f}s ({t_seq/BATCH:.1f}s per image)")

# ── Test 2: Batched pipeline call ──
print("\n=== BATCHED (single pipe call, 4 different controls + seeds) ===")
prompt = get_prompt(LABEL)
neg = get_negative_sd15(LABEL)

t0 = time.time()
# Each image gets its own generator for a different seed
generators = [torch.Generator(device=DEVICE).manual_seed(SEED + i) for i in range(BATCH)]

batch_results = pipe(
    prompt=[prompt] * BATCH,
    negative_prompt=[neg] * BATCH,
    image=ctrls,
    num_inference_steps=30,
    guidance_scale=7.5,
    controlnet_conditioning_scale=0.7,
    control_guidance_start=0.0,
    control_guidance_end=0.8,
    generator=generators,
    num_images_per_prompt=1,
).images

# Refine each (refiner doesn't take controlnet, so we batch it separately)
ref_generators = [torch.Generator(device=DEVICE).manual_seed(SEED + i) for i in range(BATCH)]
batch_refined = refiner(
    prompt=[prompt] * BATCH,
    negative_prompt=[neg] * BATCH,
    image=batch_results,
    strength=0.25,
    num_inference_steps=15,
    guidance_scale=4.0,
    generator=ref_generators,
).images

t_batch = time.time() - t0
print(f"Batched: {t_batch:.1f}s ({t_batch/BATCH:.1f}s per image)")
print(f"Speedup: {t_seq/t_batch:.2f}x")

# ── Compare: are batch results == sequential results? ──
print("\n=== COMPARISON ===")
for i in range(BATCH):
    arr_s = np.array(seq_results[i])
    arr_b = np.array(batch_refined[i])
    diff = np.abs(arr_s.astype(float) - arr_b.astype(float)).mean()
    print(f"  Image {i}: mean pixel diff = {diff:.2f} (0 = identical)")

# ── Check batch images are different from each other ──
print("\n=== DIVERSITY CHECK (batch images differ from each other) ===")
for i in range(BATCH):
    for j in range(i+1, BATCH):
        arr_i = np.array(batch_refined[i])
        arr_j = np.array(batch_refined[j])
        diff = np.abs(arr_i.astype(float) - arr_j.astype(float)).mean()
        print(f"  Batch {i} vs {j}: mean pixel diff = {diff:.2f}")

# ── Build 2x2 grid ──
def make_grid(images, labels, cols=2):
    w, h = images[0].size
    pad = 30
    rows_n = (len(images) + cols - 1) // cols
    grid = Image.new("RGB", (cols * w, rows_n * (h + pad)), (255, 255, 255))
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except:
        font = ImageFont.load_default()
    for idx, (img, lbl) in enumerate(zip(images, labels)):
        r, c = divmod(idx, cols)
        x, y = c * w, r * (h + pad)
        grid.paste(img, (x, y + pad))
        draw.text((x + 5, y + 5), lbl, fill="black", font=font)
    return grid

# Grid 1: Sequential results
grid_seq = make_grid(seq_results,
    [f"Seq #{i} seed={SEED+i}" for i in range(BATCH)])
grid_seq.save("test_sequential_2x2.png")

# Grid 2: Batch results
grid_batch = make_grid(batch_refined,
    [f"Batch #{i} seed={SEED+i}" for i in range(BATCH)])
grid_batch.save("test_batch_2x2.png")

# Grid 3: Controls
grid_ctrl = make_grid(ctrls,
    [f"Control #{i}" for i in range(BATCH)])
grid_ctrl.save("test_controls_2x2.png")

# Grid 4: Side-by-side seq vs batch for each image
combined = []
combined_labels = []
for i in range(BATCH):
    combined.append(seq_results[i])
    combined_labels.append(f"Seq #{i}")
    combined.append(batch_refined[i])
    combined_labels.append(f"Batch #{i}")
grid_compare = make_grid(combined, combined_labels, cols=2)
grid_compare.save("test_seq_vs_batch.png")

print(f"\nSaved: test_controls_2x2.png, test_sequential_2x2.png, test_batch_2x2.png, test_seq_vs_batch.png")
print("Done!")
