"""Test torch.compile speedup for SDXL-Lightning 4-step."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "quick_draw"))

import time
import torch
from PIL import Image
from pathlib import Path

from v8_tailored import SiglipScorer, DomainNetMatcher, fetch_quickdraw, drawing_to_img
from demo_beit_compare import (
    load_sdxl_lightning, generate_lightning,
    to_lineart_sdxl, zoom_to_content,
)

DEVICE = "cuda"
N_WARMUP = 3
N_BENCH = 20
CLASSES = ["trumpet", "butterfly", "camel", "spider", "bread"]

def main():
    # Prepare test data
    print("Preparing test data...")
    scorer = SiglipScorer(device=DEVICE)
    matcher = DomainNetMatcher(scorer, CLASSES, batch_size=16)

    test_ctrls = []
    test_labels = []
    raw = fetch_quickdraw("trumpet", 1, offset=0)
    sketch = drawing_to_img(raw[0], sz=256)
    matches = matcher.match_topk(sketch, "trumpet", k=1)
    dn_img = zoom_to_content(matches[0][0], 1024, fill_pct=0.9)
    ctrl = to_lineart_sdxl(dn_img.resize((1024, 1024), Image.LANCZOS))

    del matcher, scorer
    torch.cuda.empty_cache()

    # Load pipeline
    pipe, _ = load_sdxl_lightning(DEVICE)

    # ── Baseline (no compile) ──
    print("\n=== BASELINE (no compile) ===")
    for i in range(N_WARMUP):
        _ = generate_lightning(pipe, None, ctrl, "trumpet", i, DEVICE)
    print(f"Warmup done ({N_WARMUP} imgs)")

    times = []
    for i in range(N_BENCH):
        t0 = time.time()
        _ = generate_lightning(pipe, None, ctrl, "trumpet", 100 + i, DEVICE)
        times.append(time.time() - t0)
    avg_base = sum(times) / len(times)
    print(f"Baseline: {avg_base:.3f}s/img (min={min(times):.3f}, max={max(times):.3f})")

    # ── With torch.compile ──
    print("\n=== COMPILED (reduce-overhead) ===")
    pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead")
    pipe.controlnet = torch.compile(pipe.controlnet, mode="reduce-overhead")

    print("Compiling (first few runs are slow)...")
    for i in range(N_WARMUP + 2):
        t0 = time.time()
        _ = generate_lightning(pipe, None, ctrl, "trumpet", 200 + i, DEVICE)
        print(f"  warmup {i}: {time.time()-t0:.2f}s")

    times = []
    for i in range(N_BENCH):
        t0 = time.time()
        _ = generate_lightning(pipe, None, ctrl, "trumpet", 300 + i, DEVICE)
        times.append(time.time() - t0)
    avg_comp = sum(times) / len(times)
    print(f"Compiled: {avg_comp:.3f}s/img (min={min(times):.3f}, max={max(times):.3f})")

    speedup = avg_base / avg_comp
    print(f"\n=== SUMMARY ===")
    print(f"Baseline:  {avg_base:.3f}s/img")
    print(f"Compiled:  {avg_comp:.3f}s/img")
    print(f"Speedup:   {speedup:.2f}x ({(1-avg_comp/avg_base)*100:.1f}% faster)")
    print(f"Compare:   SD1.5 20-step = 1.81s/img")


if __name__ == "__main__":
    main()
