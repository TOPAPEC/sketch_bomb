"""Benchmark different rembg models and approaches."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "quick_draw"))

import time
import torch
from PIL import Image
from rembg import remove as rembg_remove, new_session
from v8_tailored import fetch_quickdraw
from demo_beit_compare import drawing_to_img_small, load_sd15, generate_sd15, to_lineart_sd15, get_prompt, get_negative_sd15

DEVICE = "cuda"

# Generate 5 test images first
print("Generating 5 test images...")
pipe, refiner = load_sd15(DEVICE)
test_imgs = []
classes = ["cat", "car", "flower", "bird", "house"]
for cls in classes:
    drawings = fetch_quickdraw(cls, 1, offset=0)
    sketch = drawing_to_img_small(drawings[0], 256)
    ctrl = to_lineart_sd15(sketch.resize((512, 512), Image.LANCZOS))
    img = generate_sd15(pipe, refiner, ctrl, cls, 42, DEVICE)
    test_imgs.append((cls, img))
del pipe, refiner
torch.cuda.empty_cache()
print(f"Generated {len(test_imgs)} test images")

# Test different rembg models
MODELS = ["u2netp", "silueta", "isnet-general-use", "birefnet-general-lite", "u2net"]

for model_name in MODELS:
    try:
        print(f"\n--- {model_name} ---")
        sess = new_session(model_name)
        # Warmup
        _ = rembg_remove(test_imgs[0][1], session=sess)
        t0 = time.time()
        for cls, img in test_imgs:
            _ = rembg_remove(img, session=sess)
        elapsed = time.time() - t0
        print(f"  {elapsed:.1f}s total, {elapsed/len(test_imgs):.2f}s/img")
    except Exception as e:
        print(f"  ERROR: {e}")

# Also test: just skip rembg entirely (white bg threshold)
print("\n--- threshold (no neural net) ---")
import numpy as np
def threshold_bg(img, thresh=240):
    arr = np.array(img).astype(np.float32)
    bg_mask = np.all(arr > thresh, axis=2)
    arr[bg_mask] = 255
    return Image.fromarray(arr.astype(np.uint8))

t0 = time.time()
for cls, img in test_imgs:
    _ = threshold_bg(img)
elapsed = time.time() - t0
print(f"  {elapsed:.3f}s total, {elapsed/len(test_imgs):.4f}s/img")

print("\nDone!")
