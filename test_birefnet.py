"""Test BiRefNet GPU bg removal vs rembg u2netp CPU."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "quick_draw"))

import time
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from rembg import remove as rembg_remove, new_session

DEVICE = "cuda"

# Load test images (pre-saved from earlier test)
test_paths = [
    "test_sequential_2x2.png",  # use parts of existing test images
]

# Generate test images from QuickDraw
from v8_tailored import fetch_quickdraw
from demo_beit_compare import drawing_to_img_small, load_sd15, generate_sd15, to_lineart_sd15

print("Generating 10 test images...")
pipe, refiner = load_sd15(DEVICE)
test_imgs = []
classes = ["cat", "car", "flower", "bird", "house", "apple", "fish", "dog", "tree", "butterfly"]
for cls in classes:
    drawings = fetch_quickdraw(cls, 1, offset=0)
    sketch = drawing_to_img_small(drawings[0], 256)
    ctrl = to_lineart_sd15(sketch.resize((512, 512), Image.LANCZOS))
    img = generate_sd15(pipe, refiner, ctrl, cls, 42, DEVICE)
    test_imgs.append(img)
del pipe, refiner
torch.cuda.empty_cache()
print(f"Generated {len(test_imgs)} images")

# ── BiRefNet GPU ──
print("\n--- BiRefNet GPU ---")
t0 = time.time()
birefnet = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet", trust_remote_code=True
)
birefnet.to(DEVICE).half().eval()
t_load = time.time() - t0
print(f"  Model load: {t_load:.1f}s")

transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def birefnet_remove_bg(img, model, bg=(255, 255, 255)):
    w, h = img.size
    input_tensor = transform(img).unsqueeze(0).half().to(DEVICE)
    with torch.no_grad():
        preds = model(input_tensor)[-1].sigmoid()
    mask = preds.squeeze().cpu()
    mask = (mask * 255).byte().numpy()
    mask = Image.fromarray(mask).resize((w, h), Image.LANCZOS)
    # Apply mask
    rgba = img.copy().convert("RGBA")
    rgba.putalpha(mask)
    out = Image.new('RGBA', (w, h), (*bg, 255))
    out.paste(rgba, mask=rgba.split()[3])
    return out.convert('RGB')

# Warmup
_ = birefnet_remove_bg(test_imgs[0], birefnet)

t0 = time.time()
birefnet_results = []
for img in test_imgs:
    result = birefnet_remove_bg(img, birefnet)
    birefnet_results.append(result)
t_birefnet = time.time() - t0
print(f"  {t_birefnet:.2f}s total, {t_birefnet/len(test_imgs):.3f}s/img")

del birefnet
torch.cuda.empty_cache()

# ── rembg u2netp CPU ──
print("\n--- rembg u2netp CPU ---")
sess = new_session("u2netp")
_ = rembg_remove(test_imgs[0], session=sess)  # warmup

t0 = time.time()
rembg_results = []
for img in test_imgs:
    rgba = rembg_remove(img, session=sess)
    out = Image.new('RGBA', img.size, (255, 255, 255, 255))
    out.paste(rgba, mask=rgba.split()[3])
    rembg_results.append(out.convert('RGB'))
t_rembg = time.time() - t0
print(f"  {t_rembg:.2f}s total, {t_rembg/len(test_imgs):.3f}s/img")

print(f"\n  Speedup: {t_rembg/t_birefnet:.1f}x")

# Save comparison grid
from PIL import ImageDraw, ImageFont
cell = 256
rows = len(test_imgs)
try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
except: font = ImageFont.load_default()

grid = Image.new("RGB", (cell * 3 + 60, 30 + rows * cell), (255, 255, 255))
draw = ImageDraw.Draw(grid)
draw.text((65, 5), "Original", fill="black", font=font)
draw.text((65 + cell, 5), "BiRefNet GPU", fill="black", font=font)
draw.text((65 + 2*cell, 5), "rembg u2netp", fill="black", font=font)

for i in range(rows):
    y = 30 + i * cell
    draw.text((5, y + cell//2 - 8), classes[i][:6], fill="black", font=font)
    grid.paste(test_imgs[i].resize((cell, cell)), (60, y))
    grid.paste(birefnet_results[i].resize((cell, cell)), (60 + cell, y))
    grid.paste(rembg_results[i].resize((cell, cell)), (60 + 2*cell, y))

grid.save("test_birefnet_comparison.png")
print(f"Saved: test_birefnet_comparison.png")
print("Done!")
