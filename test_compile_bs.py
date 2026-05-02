"""Test torch.compile + batch size scaling on SD1.5 ControlNet."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "quick_draw"))

import torch
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from demo_beit_compare import (
    load_sd15, to_lineart_sd15, get_prompt, get_negative_sd15, drawing_to_img_small,
)
from v8_tailored import fetch_quickdraw

DEVICE = "cuda"
LABEL = "cat"
SEED = 42

print("Fetching 8 cat sketches...")
drawings = fetch_quickdraw("cat", 8, offset=0)
ctrls = [to_lineart_sd15(drawing_to_img_small(d, 256).resize((512, 512), Image.LANCZOS)) for d in drawings]

print("Loading SD1.5...")
pipe, refiner = load_sd15(DEVICE)

prompt = get_prompt(LABEL)
neg = get_negative_sd15(LABEL)

def run_batch(pipe, refiner, ctrls_batch, label, compile_mode=None):
    bs = len(ctrls_batch)
    generators = [torch.Generator(device=DEVICE).manual_seed(SEED + i) for i in range(bs)]
    imgs = pipe(
        prompt=[prompt] * bs,
        negative_prompt=[neg] * bs,
        image=ctrls_batch,
        num_inference_steps=30,
        guidance_scale=7.5,
        controlnet_conditioning_scale=0.7,
        control_guidance_start=0.0,
        control_guidance_end=0.8,
        generator=generators,
        num_images_per_prompt=1,
    ).images
    ref_gens = [torch.Generator(device=DEVICE).manual_seed(SEED + i) for i in range(bs)]
    refined = refiner(
        prompt=[prompt] * bs,
        negative_prompt=[neg] * bs,
        image=imgs,
        strength=0.25,
        num_inference_steps=15,
        guidance_scale=4.0,
        generator=ref_gens,
    ).images
    return refined

# ── Test batch sizes without compile ──
print("\n=== BATCH SIZE SCALING (no compile) ===")
for bs in [1, 2, 4, 8]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    batch = ctrls[:bs]
    try:
        # Warmup
        if bs == 1:
            _ = run_batch(pipe, refiner, batch, LABEL)
        t0 = time.time()
        imgs = run_batch(pipe, refiner, batch, LABEL)
        elapsed = time.time() - t0
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        print(f"  BS={bs}: {elapsed:.1f}s total, {elapsed/bs:.2f}s/img, peak VRAM {peak_mb:.0f}MB")
    except torch.cuda.OutOfMemoryError:
        print(f"  BS={bs}: OOM!")
        torch.cuda.empty_cache()
        break

# ── Test torch.compile ──
print("\n=== torch.compile ===")
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead")
pipe.controlnet = torch.compile(pipe.controlnet, mode="reduce-overhead")
refiner.unet = torch.compile(refiner.unet, mode="reduce-overhead")

for bs in [1, 4]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    batch = ctrls[:bs]
    try:
        # Warmup (compile happens here)
        print(f"  BS={bs}: compiling (warmup)...")
        t_warmup_start = time.time()
        _ = run_batch(pipe, refiner, batch, LABEL)
        t_warmup = time.time() - t_warmup_start
        print(f"  BS={bs}: warmup took {t_warmup:.1f}s")

        t0 = time.time()
        imgs = run_batch(pipe, refiner, batch, LABEL)
        elapsed = time.time() - t0
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        print(f"  BS={bs} compiled: {elapsed:.1f}s total, {elapsed/bs:.2f}s/img, peak VRAM {peak_mb:.0f}MB")
    except torch.cuda.OutOfMemoryError:
        print(f"  BS={bs} compiled: OOM!")
        torch.cuda.empty_cache()
        break
    except Exception as e:
        print(f"  BS={bs} compiled: ERROR {e}")
        break

print("\nDone!")
