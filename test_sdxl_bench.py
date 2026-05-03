"""Benchmark SDXL with compile + batching + 20 steps. Find max BS."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "quick_draw"))

import torch
import time
from PIL import Image
from demo_beit_compare import (
    load_sdxl_base, to_lineart_sdxl, get_prompt, get_negative_sdxl, drawing_to_img_small,
)
from v8_tailored import fetch_quickdraw

DEVICE = "cuda"
SEED = 42
STEPS = 20

print("Fetching sketches...")
classes = ["cat", "car", "flower", "bird", "house", "apple", "fish", "dog"]
ctrls = []
for cls in classes:
    drawings = fetch_quickdraw(cls, 1, offset=0)
    sketch = drawing_to_img_small(drawings[0], 256)
    ctrl = to_lineart_sdxl(sketch.resize((1024, 1024), Image.LANCZOS))
    ctrls.append((cls, ctrl))

print("Loading SDXL...")
pipe, refiner = load_sdxl_base(DEVICE)

prompt = get_prompt("cat")
neg = get_negative_sdxl("cat")

# Test batch sizes
print("\n=== BATCH SIZE SCALING (no compile, 20 steps) ===")
for bs in [1, 2, 4]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    batch_ctrls = [c for _, c in ctrls[:bs]]
    batch_prompts = [get_prompt(cls) for cls, _ in ctrls[:bs]]
    batch_negs = [get_negative_sdxl(cls) for cls, _ in ctrls[:bs]]
    try:
        # Warmup
        gens = [torch.Generator(device=DEVICE).manual_seed(SEED + i) for i in range(bs)]
        _ = pipe(prompt=batch_prompts, negative_prompt=batch_negs, image=batch_ctrls,
                 num_inference_steps=STEPS, guidance_scale=7.0, controlnet_conditioning_scale=0.5,
                 control_guidance_start=0.0, control_guidance_end=0.9,
                 generator=gens, num_images_per_prompt=1).images
        # Timed run
        torch.cuda.reset_peak_memory_stats()
        gens = [torch.Generator(device=DEVICE).manual_seed(SEED + i) for i in range(bs)]
        t0 = time.time()
        imgs = pipe(prompt=batch_prompts, negative_prompt=batch_negs, image=batch_ctrls,
                    num_inference_steps=STEPS, guidance_scale=7.0, controlnet_conditioning_scale=0.5,
                    control_guidance_start=0.0, control_guidance_end=0.9,
                    generator=gens, num_images_per_prompt=1).images
        # Refine
        ref_gens = [torch.Generator(device=DEVICE).manual_seed(SEED + i) for i in range(bs)]
        imgs = refiner(prompt=batch_prompts, negative_prompt=batch_negs, image=imgs,
                       strength=0.25, num_inference_steps=max(5, STEPS//2), guidance_scale=4.0,
                       generator=ref_gens).images
        elapsed = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1e6
        print(f"  BS={bs}: {elapsed:.1f}s total, {elapsed/bs:.2f}s/img, peak {peak:.0f}MB")
    except torch.cuda.OutOfMemoryError:
        print(f"  BS={bs}: OOM!")
        torch.cuda.empty_cache()
        break

# Compile test at best BS
print("\n=== torch.compile (BS=2, 20 steps) ===")
torch.cuda.empty_cache()
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead")
pipe.controlnet = torch.compile(pipe.controlnet, mode="reduce-overhead")
refiner.unet = torch.compile(refiner.unet, mode="reduce-overhead")

bs = 2
batch_ctrls = [c for _, c in ctrls[:bs]]
batch_prompts = [get_prompt(cls) for cls, _ in ctrls[:bs]]
batch_negs = [get_negative_sdxl(cls) for cls, _ in ctrls[:bs]]

print("  Compiling (warmup)...")
t0 = time.time()
gens = [torch.Generator(device=DEVICE).manual_seed(SEED + i) for i in range(bs)]
_ = pipe(prompt=batch_prompts, negative_prompt=batch_negs, image=batch_ctrls,
         num_inference_steps=STEPS, guidance_scale=7.0, controlnet_conditioning_scale=0.5,
         control_guidance_start=0.0, control_guidance_end=0.9,
         generator=gens, num_images_per_prompt=1).images
ref_gens = [torch.Generator(device=DEVICE).manual_seed(SEED + i) for i in range(bs)]
_ = refiner(prompt=batch_prompts, negative_prompt=batch_negs, image=_,
            strength=0.25, num_inference_steps=max(5, STEPS//2), guidance_scale=4.0,
            generator=ref_gens).images
warmup = time.time() - t0
print(f"  Warmup: {warmup:.0f}s")

# Timed
torch.cuda.reset_peak_memory_stats()
gens = [torch.Generator(device=DEVICE).manual_seed(SEED + i) for i in range(bs)]
t0 = time.time()
imgs = pipe(prompt=batch_prompts, negative_prompt=batch_negs, image=batch_ctrls,
            num_inference_steps=STEPS, guidance_scale=7.0, controlnet_conditioning_scale=0.5,
            control_guidance_start=0.0, control_guidance_end=0.9,
            generator=gens, num_images_per_prompt=1).images
ref_gens = [torch.Generator(device=DEVICE).manual_seed(SEED + i) for i in range(bs)]
imgs = refiner(prompt=batch_prompts, negative_prompt=batch_negs, image=imgs,
               strength=0.25, num_inference_steps=max(5, STEPS//2), guidance_scale=4.0,
               generator=ref_gens).images
elapsed = time.time() - t0
peak = torch.cuda.max_memory_allocated() / 1e6
print(f"  BS={bs} compiled: {elapsed:.1f}s total, {elapsed/bs:.2f}s/img, peak {peak:.0f}MB")

# Also test BS=1 compiled
bs = 1
batch_ctrls = [c for _, c in ctrls[:bs]]
batch_prompts = [get_prompt(cls) for cls, _ in ctrls[:bs]]
batch_negs = [get_negative_sdxl(cls) for cls, _ in ctrls[:bs]]
# Need recompile for different BS
print("  Compiling BS=1 (warmup)...")
t0 = time.time()
gens = [torch.Generator(device=DEVICE).manual_seed(SEED)]
_ = pipe(prompt=batch_prompts, negative_prompt=batch_negs, image=batch_ctrls,
         num_inference_steps=STEPS, guidance_scale=7.0, controlnet_conditioning_scale=0.5,
         control_guidance_start=0.0, control_guidance_end=0.9,
         generator=gens, num_images_per_prompt=1).images
ref_gens = [torch.Generator(device=DEVICE).manual_seed(SEED)]
_ = refiner(prompt=batch_prompts, negative_prompt=batch_negs, image=_,
            strength=0.25, num_inference_steps=max(5, STEPS//2), guidance_scale=4.0,
            generator=ref_gens).images
warmup = time.time() - t0
print(f"  Warmup: {warmup:.0f}s")

gens = [torch.Generator(device=DEVICE).manual_seed(SEED)]
t0 = time.time()
imgs = pipe(prompt=batch_prompts, negative_prompt=batch_negs, image=batch_ctrls,
            num_inference_steps=STEPS, guidance_scale=7.0, controlnet_conditioning_scale=0.5,
            control_guidance_start=0.0, control_guidance_end=0.9,
            generator=gens, num_images_per_prompt=1).images
ref_gens = [torch.Generator(device=DEVICE).manual_seed(SEED)]
imgs = refiner(prompt=batch_prompts, negative_prompt=batch_negs, image=imgs,
               strength=0.25, num_inference_steps=max(5, STEPS//2), guidance_scale=4.0,
               generator=ref_gens).images
elapsed = time.time() - t0
peak = torch.cuda.max_memory_allocated() / 1e6
print(f"  BS={bs} compiled: {elapsed:.1f}s total, {elapsed/bs:.2f}s/img, peak {peak:.0f}MB")

print("\nDone!")
