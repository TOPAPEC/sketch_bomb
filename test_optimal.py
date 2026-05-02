"""Benchmark optimal config: BS=4 + torch.compile + 20 steps. Full pipeline timing."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "quick_draw"))

import torch
import time
import numpy as np
from PIL import Image
from demo_beit_compare import (
    load_sd15, to_lineart_sd15, get_prompt, get_negative_sd15, drawing_to_img_small,
)
from v8_tailored import fetch_quickdraw, BeitSketchClassifier, SiglipScorer, score_prompt, remove_bg

DEVICE = "cuda"
SEED = 42
N_CLASSES = 5
N_SAMPLES = 5
BEST_OF = 4
STEPS = 20
BS = 4

CLASSES = ["cat", "car", "flower", "house", "bird"]

print(f"Config: {N_CLASSES}c × {N_SAMPLES}s × best_of={BEST_OF}, steps={STEPS}, BS={BS}, compile=True")
total_images = N_CLASSES * N_SAMPLES * BEST_OF
print(f"Total generations: {total_images}")

timings = {}

# Stage 1: Fetch + BEiT
print("\n── Stage 1: Fetch + BEiT ──")
t0 = time.time()
beit = BeitSketchClassifier(device=DEVICE)
samples = []
for cls in CLASSES:
    drawings = fetch_quickdraw(cls, N_SAMPLES, offset=0)
    for i, d in enumerate(drawings):
        sketch = drawing_to_img_small(d, 256)
        preds = beit.classify(sketch, top_k=5)
        label = preds[0][0]
        ctrl = to_lineart_sd15(sketch.resize((512, 512), Image.LANCZOS))
        samples.append({"cls": cls, "label": label, "sketch": sketch, "ctrl": ctrl,
                         "seed": SEED + hash(f"{cls}_{i}") % 100000})
del beit
torch.cuda.empty_cache()
timings["fetch_beit"] = time.time() - t0
print(f"  {timings['fetch_beit']:.1f}s — {len(samples)} samples")

# Stage 2: skip DomainNet (speed test), use sketch as control
print("\n── Stage 2: DomainNet SKIPPED (speed test) ──")
for s in samples:
    s["ctrls"] = [s["ctrl"]] * BEST_OF

# Stage 3: SD1.5 — compile + batch
print("\n── Stage 3: SD1.5 generate (compile + BS=4, 20 steps) ──")
t0 = time.time()
pipe, refiner = load_sd15(DEVICE)
t_load = time.time() - t0
print(f"  Model load: {t_load:.1f}s")

t0 = time.time()
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead")
pipe.controlnet = torch.compile(pipe.controlnet, mode="reduce-overhead")
refiner.unet = torch.compile(refiner.unet, mode="reduce-overhead")

# Warmup compile
print("  Compiling (warmup)...")
dummy_ctrls = [samples[0]["ctrl"]] * BS
prompt = get_prompt(samples[0]["label"])
neg = get_negative_sd15(samples[0]["label"])
gens = [torch.Generator(device=DEVICE).manual_seed(SEED + i) for i in range(BS)]
_ = pipe(prompt=[prompt]*BS, negative_prompt=[neg]*BS, image=dummy_ctrls,
         num_inference_steps=STEPS, guidance_scale=7.5, controlnet_conditioning_scale=0.7,
         control_guidance_start=0.0, control_guidance_end=0.8,
         generator=gens, num_images_per_prompt=1).images
gens = [torch.Generator(device=DEVICE).manual_seed(SEED + i) for i in range(BS)]
_ = refiner(prompt=[prompt]*BS, negative_prompt=[neg]*BS, image=_,
            strength=0.25, num_inference_steps=max(5, STEPS//2), guidance_scale=4.0,
            generator=gens).images
t_compile = time.time() - t0
print(f"  Compile warmup: {t_compile:.1f}s")

# Actual generation
print("  Generating...")
t0 = time.time()
for s in samples:
    prompt = get_prompt(s["label"])
    neg = get_negative_sd15(s["label"])
    ctrls_batch = s["ctrls"][:BEST_OF]

    gens = [torch.Generator(device=DEVICE).manual_seed(s["seed"] + i) for i in range(BEST_OF)]
    imgs = pipe(
        prompt=[prompt] * BEST_OF,
        negative_prompt=[neg] * BEST_OF,
        image=ctrls_batch,
        num_inference_steps=STEPS,
        guidance_scale=7.5,
        controlnet_conditioning_scale=0.7,
        control_guidance_start=0.0,
        control_guidance_end=0.8,
        generator=gens,
        num_images_per_prompt=1,
    ).images

    ref_gens = [torch.Generator(device=DEVICE).manual_seed(s["seed"] + i) for i in range(BEST_OF)]
    ref_steps = max(5, STEPS // 2)
    refined = refiner(
        prompt=[prompt] * BEST_OF,
        negative_prompt=[neg] * BEST_OF,
        image=imgs,
        strength=0.25,
        num_inference_steps=ref_steps,
        guidance_scale=4.0,
        generator=ref_gens,
    ).images

    s["candidates"] = refined

timings["generate"] = time.time() - t0
per_img = timings["generate"] / total_images
print(f"  {timings['generate']:.1f}s — {total_images} images — {per_img:.2f}s/img")

del pipe, refiner
torch.cuda.empty_cache()

# Stage 4: SigLIP2 scoring
print("\n── Stage 4: SigLIP2 scoring ──")
t0 = time.time()
scorer = SiglipScorer(device=DEVICE)
for s in samples:
    prompt = score_prompt(s["label"])
    s["scores"] = [scorer.score_single(c, prompt) for c in s["candidates"]]
    s["pick"] = int(np.argmax(s["scores"]))
timings["scoring"] = time.time() - t0
print(f"  {timings['scoring']:.1f}s — {total_images} scores")
del scorer
torch.cuda.empty_cache()

# Stage 5: rembg
print("\n── Stage 5: Background removal ──")
t0 = time.time()
for s in samples:
    s["final"] = remove_bg(s["candidates"][s["pick"]])
timings["rembg"] = time.time() - t0
print(f"  {timings['rembg']:.1f}s — {len(samples)} finals")

# Summary
print(f"\n{'='*50}")
print("PIPELINE TIMING (optimized: compile + BS=4 + 20 steps)")
print(f"{'='*50}")
total = sum(timings.values())
for k, v in timings.items():
    print(f"  {k:20s}: {v:7.1f}s  ({v*100/total:4.1f}%)")
print(f"  {'TOTAL':20s}: {total:7.1f}s")
print(f"\n  Per image (gen only): {per_img:.2f}s")
print(f"  Per sample (full pipeline): {total/len(samples):.2f}s")

# Extrapolate
full_35_16 = total * (35 * 16) / (N_CLASSES * N_SAMPLES)
print(f"\n  ESTIMATE 35c × 16s × best_of=4:")
print(f"    {full_35_16:.0f}s = {full_35_16/60:.0f}min = {full_35_16/3600:.1f}h")

# Compare to baseline (3.1s/img, 30 steps, no compile, BS=1)
baseline_gen = 3.1 * total_images
print(f"\n  vs baseline (30 steps, BS=1, no compile):")
print(f"    generation: {baseline_gen:.0f}s → {timings['generate']:.0f}s ({baseline_gen/timings['generate']:.1f}x faster)")
