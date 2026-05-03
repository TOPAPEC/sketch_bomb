"""Ablation: ControlNet scale (0.4/0.6/0.7/0.9) and refiner on/off."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "quick_draw"))

import torch
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from demo_beit_compare import (
    load_sd15, to_lineart_sd15, get_prompt, get_negative_sd15,
    drawing_to_img_small, zoom_to_content,
)
from v8_tailored import fetch_quickdraw, SiglipScorer, DomainNetMatcher, score_prompt

DEVICE = "cuda"
SEED = 42
N = 5
CLASSES = ["cat", "car", "flower", "bird", "house"]
CN_SCALES = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
STEPS = 20

# Fetch and prepare with DomainNet
print("Fetching sketches + DomainNet matching...")
scorer = SiglipScorer(device=DEVICE)
samples = []
for cls in CLASSES[:N]:
    drawings = fetch_quickdraw(cls, 1, offset=5)
    sketch = drawing_to_img_small(drawings[0], 256)
    samples.append({"cls": cls, "sketch": sketch})

labels = [s["cls"] for s in samples]
matcher = DomainNetMatcher(scorer, labels, batch_size=16)
for s in samples:
    match = matcher.match_topk(s["sketch"], s["cls"], k=1)
    dn_img = zoom_to_content(match[0][0], 1024, fill_pct=0.9)
    s["ctrl"] = to_lineart_sd15(dn_img.resize((512, 512), Image.LANCZOS))
del matcher

print("Loading SD1.5...")
pipe, refiner = load_sd15(DEVICE)

results = []

# Test CN scales with refiner
for cn_scale in CN_SCALES:
    label = f"cn={cn_scale}"
    print(f"\n=== {label} ===")
    imgs = []
    t0 = time.time()
    for s in samples:
        img = pipe(
            prompt=get_prompt(s["cls"]), negative_prompt=get_negative_sd15(s["cls"]),
            image=s["ctrl"], num_inference_steps=STEPS, guidance_scale=7.5,
            controlnet_conditioning_scale=cn_scale,
            control_guidance_start=0.0, control_guidance_end=0.8,
            generator=torch.Generator(device=DEVICE).manual_seed(SEED),
        ).images[0]
        img = refiner(
            prompt=get_prompt(s["cls"]), negative_prompt=get_negative_sd15(s["cls"]),
            image=img, strength=0.25, num_inference_steps=10, guidance_scale=4.0,
            generator=torch.Generator(device=DEVICE).manual_seed(SEED),
        ).images[0]
        imgs.append(img)
    elapsed = time.time() - t0
    results.append({"name": label, "images": imgs, "time": elapsed})
    print(f"  {elapsed:.1f}s")

# Test no refiner at best CN scale (0.7)
print(f"\n=== cn=0.7 NO REFINER ===")
imgs = []
t0 = time.time()
for s in samples:
    img = pipe(
        prompt=get_prompt(s["cls"]), negative_prompt=get_negative_sd15(s["cls"]),
        image=s["ctrl"], num_inference_steps=STEPS, guidance_scale=7.5,
        controlnet_conditioning_scale=0.7,
        control_guidance_start=0.0, control_guidance_end=0.8,
        generator=torch.Generator(device=DEVICE).manual_seed(SEED),
    ).images[0]
    imgs.append(img)
elapsed = time.time() - t0
results.append({"name": "cn=0.7_noref", "images": imgs, "time": elapsed})
print(f"  {elapsed:.1f}s")

# Test different control_guidance_end
for end in [0.5, 0.7, 1.0]:
    label = f"cn=0.7_end={end}"
    print(f"\n=== {label} ===")
    imgs = []
    for s in samples:
        img = pipe(
            prompt=get_prompt(s["cls"]), negative_prompt=get_negative_sd15(s["cls"]),
            image=s["ctrl"], num_inference_steps=STEPS, guidance_scale=7.5,
            controlnet_conditioning_scale=0.7,
            control_guidance_start=0.0, control_guidance_end=end,
            generator=torch.Generator(device=DEVICE).manual_seed(SEED),
        ).images[0]
        img = refiner(
            prompt=get_prompt(s["cls"]), negative_prompt=get_negative_sd15(s["cls"]),
            image=img, strength=0.25, num_inference_steps=10, guidance_scale=4.0,
            generator=torch.Generator(device=DEVICE).manual_seed(SEED),
        ).images[0]
        imgs.append(img)
    results.append({"name": label, "images": imgs, "time": 0})

# Score
print("\nScoring...")
del pipe, refiner
torch.cuda.empty_cache()
for r in results:
    scores = []
    for i, s in enumerate(samples):
        sc = scorer.score_single(r["images"][i], score_prompt(s["cls"]))
        scores.append(sc)
    r["scores"] = scores
    r["mean"] = float(np.mean(scores))
    print(f"  {r['name']:20s}: {r['mean']:.4f}")
del scorer
torch.cuda.empty_cache()

# Grid
def get_font(sz):
    try: return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", sz)
    except: return ImageFont.load_default()

cell = 200
header = 40
label_w = 80
font = get_font(12)
font_sm = get_font(9)
cols = len(results)
rows = N

grid = Image.new("RGB", (label_w + cols * cell, header + rows * cell), (255, 255, 255))
draw = ImageDraw.Draw(grid)
for j, r in enumerate(results):
    x = label_w + j * cell + 2
    draw.text((x, 2), r["name"], fill="black", font=font)
    draw.text((x, 16), f"{r['mean']:.3f}", fill="gray", font=font_sm)
for i, s in enumerate(samples):
    y = header + i * cell
    draw.text((5, y + cell//2 - 8), s["cls"], fill="black", font=font)
    for j, r in enumerate(results):
        img = r["images"][i].resize((cell, cell), Image.LANCZOS)
        grid.paste(img, (label_w + j * cell, y))

grid.save("test_cn_ablation_grid.png")
print(f"\nSaved: test_cn_ablation_grid.png")

print("\n=== SUMMARY (sorted by score) ===")
for r in sorted(results, key=lambda x: x["mean"], reverse=True):
    print(f"  {r['name']:20s}: SigLIP2={r['mean']:.4f}")
print("Done!")
