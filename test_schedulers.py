"""Benchmark schedulers: quality vs speed at various step counts."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "quick_draw"))

import torch
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from diffusers import (
    EulerDiscreteScheduler, DPMSolverMultistepScheduler,
    UniPCMultistepScheduler, DEISMultistepScheduler, LCMScheduler,
    EulerAncestralDiscreteScheduler,
)
from demo_beit_compare import load_sd15, to_lineart_sd15, get_prompt, get_negative_sd15, drawing_to_img_small
from v8_tailored import fetch_quickdraw, SiglipScorer, score_prompt

DEVICE = "cuda"
SEED = 42
N = 5
CLASSES = ["cat", "car", "flower", "bird", "house"]

print("Fetching sketches...")
samples = []
for cls in CLASSES[:N]:
    drawings = fetch_quickdraw(cls, 1, offset=5)
    sketch = drawing_to_img_small(drawings[0], 256)
    ctrl = to_lineart_sd15(sketch.resize((512, 512), Image.LANCZOS))
    samples.append({"cls": cls, "ctrl": ctrl})

print("Loading SD1.5...")
pipe, refiner = load_sd15(DEVICE)

CONFIGS = [
    ("Euler", EulerDiscreteScheduler, 30, {}),
    ("Euler", EulerDiscreteScheduler, 20, {}),
    ("Euler", EulerDiscreteScheduler, 15, {}),
    ("EulerA", EulerAncestralDiscreteScheduler, 20, {}),
    ("DPM++2M", DPMSolverMultistepScheduler, 20, {"algorithm_type": "dpmsolver++", "use_karras_sigmas": True}),
    ("DPM++2M", DPMSolverMultistepScheduler, 15, {"algorithm_type": "dpmsolver++", "use_karras_sigmas": True}),
    ("DPM++2M", DPMSolverMultistepScheduler, 10, {"algorithm_type": "dpmsolver++", "use_karras_sigmas": True}),
    ("UniPC", UniPCMultistepScheduler, 20, {}),
    ("UniPC", UniPCMultistepScheduler, 15, {}),
    ("UniPC", UniPCMultistepScheduler, 10, {}),
    ("DEIS", DEISMultistepScheduler, 15, {}),
    ("DEIS", DEISMultistepScheduler, 10, {}),
]

results = []
for name, sched_cls, steps, kwargs in CONFIGS:
    label = f"{name}_{steps}"
    print(f"\n=== {label} ===")
    pipe.scheduler = sched_cls.from_config(pipe.scheduler.config, **kwargs)

    imgs = []
    t0 = time.time()
    for s in samples:
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
        # Refiner
        ref_steps = max(5, steps // 2)
        img = refiner(
            prompt=get_prompt(s["cls"]),
            negative_prompt=get_negative_sd15(s["cls"]),
            image=img, strength=0.25, num_inference_steps=ref_steps, guidance_scale=4.0,
            generator=torch.Generator(device=DEVICE).manual_seed(SEED),
        ).images[0]
        imgs.append(img)
    elapsed = time.time() - t0
    results.append({"name": label, "images": imgs, "time": elapsed, "steps": steps})
    print(f"  {elapsed:.1f}s, {elapsed/N:.2f}s/img")

# Score all
print("\nScoring...")
del pipe, refiner
torch.cuda.empty_cache()
scorer = SiglipScorer(device=DEVICE)
for r in results:
    scores = []
    for i, s in enumerate(samples):
        sc = scorer.score_single(r["images"][i], score_prompt(s["cls"]))
        scores.append(sc)
    r["scores"] = scores
    r["mean_score"] = float(np.mean(scores))
    print(f"  {r['name']:15s}: SigLIP2={r['mean_score']:.4f}, {r['time']/N:.2f}s/img")
del scorer
torch.cuda.empty_cache()

# Build comparison grid
def get_font(size):
    try: return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except: return ImageFont.load_default()

cell = 200
header = 40
label_w = 90
font = get_font(13)
font_sm = get_font(10)
cols = len(results)
rows = N

grid = Image.new("RGB", (label_w + cols * cell, header + rows * cell), (255, 255, 255))
draw = ImageDraw.Draw(grid)

for j, r in enumerate(results):
    x = label_w + j * cell + 5
    draw.text((x, 2), r["name"], fill="black", font=font)
    draw.text((x, 16), f"{r['mean_score']:.3f} | {r['time']/N:.1f}s", fill="gray", font=font_sm)
    draw.text((x, 28), f"steps={r['steps']}", fill="gray", font=font_sm)

for i, s in enumerate(samples):
    y = header + i * cell
    draw.text((5, y + cell//2 - 8), s["cls"], fill="black", font=font)
    for j, r in enumerate(results):
        x = label_w + j * cell
        img = r["images"][i].resize((cell, cell), Image.LANCZOS)
        grid.paste(img, (x, y))

grid.save("test_schedulers_grid.png")
print(f"\nSaved: test_schedulers_grid.png ({grid.size[0]}x{grid.size[1]})")

# Print summary table
print("\n=== SUMMARY ===")
print(f"{'Config':15s} {'Steps':>5s} {'s/img':>6s} {'SigLIP2':>8s}")
for r in sorted(results, key=lambda x: x["mean_score"], reverse=True):
    print(f"{r['name']:15s} {r['steps']:5d} {r['time']/N:6.2f} {r['mean_score']:8.4f}")
