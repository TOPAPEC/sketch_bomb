import torch
import numpy as np
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModel
from pathlib import Path

device = "cuda"
model = AutoModel.from_pretrained("google/siglip2-base-patch16-224", torch_dtype=torch.float16).to(device)
proc = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")

@torch.no_grad()
def score(images, texts):
    inputs = proc(text=texts, images=images, padding="max_length", return_tensors="pt").to(device)
    out = model(**inputs)
    return np.diag(out.logits_per_image.cpu().numpy())

@torch.no_grad()
def score_matrix(images, texts):
    inputs = proc(text=texts, images=images, padding="max_length", return_tensors="pt").to(device)
    out = model(**inputs)
    return out.logits_per_image.cpu().numpy()

# load all validation images
val = Path("out3/v7_experiments/validation")
if not val.exists():
    print("No validation images found, checking for any experiment images...")
    val = Path("out3/v7_experiments")

# collect whatever images we have
all_imgs = {}
for p in sorted(Path("out3/v7_experiments").glob("*.png")):
    all_imgs[p.stem] = Image.open(p)
for p in sorted(Path("out3/v7_experiments/validation").glob("*.png")):
    all_imgs[p.stem] = Image.open(p)

print(f"Found {len(all_imgs)} images: {list(all_imgs.keys())[:20]}")

# === TEST 1: Different text prompts for the same image ===
print("\n" + "="*60)
print("TEST 1: How different text prompts score the SAME image")
print("="*60)

# pick a cat image if available
cat_keys = [k for k in all_imgs if 'cat' in k and 'sketch' not in k]
if cat_keys:
    test_img = all_imgs[cat_keys[0]]
    print(f"Using: {cat_keys[0]}")
else:
    test_img = list(all_imgs.values())[0]
    print(f"Using: {list(all_imgs.keys())[0]}")

prompts_to_test = [
    "cat",
    "a cat",
    "a photo of a cat",
    "a drawing of a cat",
    "an illustration of a cat",
    "a high quality illustration of a cat",
    "a single cat, isolated subject, centered, white background",
    "a single cat, isolated, centered composition, solid white background, studio lighting, high quality digital illustration",
    "a beautiful cat with vivid natural colors on white background",
    "orange cat",
    "fire",
    "abstract art",
    "a dog",
    "a car",
    "random noise",
]

for p in prompts_to_test:
    s = float(score([test_img], [p])[0])
    print(f"  {s:8.3f}  |  {p}")

# === TEST 2: Same prompt, different images ===
print("\n" + "="*60)
print("TEST 2: Same prompt scored against different images")
print("="*60)

test_prompt = "a high quality illustration of a cat"
print(f"Prompt: '{test_prompt}'")
for k in sorted(all_imgs.keys()):
    if 'cat' in k or 'sketch' in k:
        s = float(score([all_imgs[k]], [test_prompt])[0])
        print(f"  {s:8.3f}  |  {k}")

# === TEST 3: Simple test - create synthetic images ===
print("\n" + "="*60)
print("TEST 3: Synthetic images - does siglip prefer matching content?")
print("="*60)

red_img = Image.new('RGB', (224, 224), (255, 0, 0))
blue_img = Image.new('RGB', (224, 224), (0, 0, 255))
white_img = Image.new('RGB', (224, 224), (255, 255, 255))
black_img = Image.new('RGB', (224, 224), (0, 0, 0))

# draw a simple circle on white bg
circle_img = Image.new('RGB', (224, 224), (255, 255, 255))
d = ImageDraw.Draw(circle_img)
d.ellipse([50, 50, 174, 174], fill=(255, 128, 0), outline=(0, 0, 0), width=2)

synth = {"red": red_img, "blue": blue_img, "white": white_img, "black": black_img, "orange_circle": circle_img}

test_texts = ["a red image", "a blue image", "a white image", "a cat", "an orange circle"]
imgs_list = list(synth.values())
names = list(synth.keys())

mat = score_matrix(imgs_list, test_texts)
print(f"\n{'':15s}", end="")
for t in test_texts:
    print(f"{t:>18s}", end="")
print()
for i, n in enumerate(names):
    print(f"{n:15s}", end="")
    for j in range(len(test_texts)):
        print(f"{mat[i,j]:18.3f}", end="")
    print()

# === TEST 4: Score range exploration ===
print("\n" + "="*60)
print("TEST 4: What is the actual score range?")
print("="*60)

if len(all_imgs) >= 2:
    all_images = list(all_imgs.values())
    all_names = list(all_imgs.keys())
    simple_prompt = "a cat"
    scores_all = []
    for img in all_images:
        s = float(score([img], [simple_prompt])[0])
        scores_all.append(s)

    paired = sorted(zip(scores_all, all_names), reverse=True)
    print(f"Prompt: '{simple_prompt}' — all images ranked:")
    for s, n in paired:
        print(f"  {s:8.3f}  |  {n}")

# === TEST 5: logits vs sigmoid ===
print("\n" + "="*60)
print("TEST 5: Raw logits vs sigmoid probabilities")
print("="*60)
print("SigLIP uses sigmoid, not softmax. Logit of 0 = 50% probability")
print("Negative logits = <50% match, positive = >50% match")

if cat_keys:
    test_img = all_imgs[cat_keys[0]]
    key_prompts = ["a cat", "a dog", "a car", "fire", "noise"]
    for p in key_prompts:
        logit = float(score([test_img], [p])[0])
        prob = 1 / (1 + np.exp(-logit))
        print(f"  logit={logit:8.3f}  prob={prob:.4f}  |  {p}")

print("\n\nDONE")
