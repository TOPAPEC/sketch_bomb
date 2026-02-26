import numpy as np
import requests
import json
from PIL import Image, ImageDraw, ImageFont
import torch
from diffusers import (StableDiffusionXLControlNetPipeline, StableDiffusionXLImg2ImgPipeline,
                       ControlNetModel, AutoencoderKL, EulerDiscreteScheduler)
from transformers import AutoProcessor, AutoModel
from rembg import remove as rembg_remove
import cv2
from pathlib import Path
import random
import os
import base64
import io
from datetime import datetime


def load_env():
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip())

load_env()

CLASSES = ["cat", "car", "flower", "house", "bird", "apple"]

PROMPTS = {
    "cat": (
        "a cute cat, feline, digital painting, detailed fur and whiskers, "
        "cat ears, cat face, expressive eyes, soft shading, best quality, "
        "solo animal, full body, concept art style"
    ),
    "car": (
        "a car, digital illustration, "
        "colorful, detailed wheels and windows, clean lines, "
        "best quality, sharp focus, simple background"
    ),
    "flower": (
        "a beautiful flower, botanical illustration, "
        "detailed petals, vivid colors, sharp focus, "
        "best quality, natural lighting"
    ),
    "house": (
        "a house with roof and windows, children's book illustration, "
        "colorful, detailed facade, charming style, "
        "best quality, clean lines"
    ),
    "bird": (
        "a colorful bird, nature illustration, "
        "detailed feathers, vivid plumage, sharp beak, "
        "best quality, digital art"
    ),
    "apple": (
        "a red apple, digital painting, "
        "glossy, realistic shading, vivid red color, "
        "best quality, sharp focus, still life"
    ),
}

NEGATIVE = (
    "low quality, bad quality, worst quality, blurry, ugly, deformed, "
    "extra limbs, bad anatomy, text, watermark, letters, words, writing"
)

NEGATIVES = {
    "cat": NEGATIVE + ", dog, canine, human, person, multiple cats",
    "car": NEGATIVE + ", deformed, broken wheels",
    "flower": NEGATIVE + ", wilted, dead, brown petals",
    "house": NEGATIVE + ", destroyed, ruins",
    "bird": NEGATIVE + ", human, deformed wings",
    "apple": NEGATIVE + ", rotten, bitten, multiple apples",
}

OPENROUTER_MODEL = "moonshotai/kimi-k2.5"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def get_prompt(label):
    return PROMPTS.get(label, f"a single {label}, centered, digital illustration, detailed, sharp focus, vivid colors, best quality")

def get_negative(label):
    return NEGATIVES.get(label, NEGATIVE)

def score_prompt(label):
    return f"a high quality illustration of a {label}"

def remove_bg(img, bg=(255, 255, 255)):
    rgba = rembg_remove(img)
    out = Image.new('RGBA', rgba.size, (*bg, 255))
    out.paste(rgba, mask=rgba.split()[3])
    return out.convert('RGB')


class SiglipScorer:
    def __init__(self, device="cuda"):
        self.device = device
        self.model = AutoModel.from_pretrained(
            "google/siglip2-base-patch16-224", torch_dtype=torch.float16).to(device)
        self.processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")

    @torch.no_grad()
    def score(self, images, texts):
        inputs = self.processor(text=texts, images=images,
                                padding="max_length", return_tensors="pt").to(self.device)
        return np.diag(self.model(**inputs).logits_per_image.cpu().numpy())

    def score_single(self, image, text):
        return float(self.score([image], [text])[0])


class Pipeline:
    def __init__(self, device="cuda"):
        self.device = device
        self.pipe = None
        self.refiner = None
        self.scorer = SiglipScorer(device)

    def load(self):
        print("Loading SDXL + MistoLine ControlNet...")
        cn = ControlNetModel.from_pretrained(
            "TheMistoAI/MistoLine", torch_dtype=torch.float16, variant="fp16")
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", controlnet=cn, vae=vae,
            torch_dtype=torch.float16, variant="fp16", safety_checker=None)
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_model_cpu_offload()
        comps = {k: v for k, v in self.pipe.components.items() if k != "controlnet"}
        self.refiner = StableDiffusionXLImg2ImgPipeline(**comps)
        print("Ready")

    def generate(self, ctrl, label, seed=42, steps=30, cn_scale=0.5, guidance=7.0, batch=1):
        return self.pipe(
            prompt=get_prompt(label), negative_prompt=get_negative(label),
            image=ctrl, num_inference_steps=steps,
            guidance_scale=guidance, controlnet_conditioning_scale=cn_scale,
            control_guidance_start=0.0, control_guidance_end=min(1.0, 12/steps),
            generator=torch.Generator(device="cpu").manual_seed(seed),
            num_images_per_prompt=batch,
        ).images

    def refine(self, img, label, seed=42, strength=0.25):
        return self.refiner(
            prompt=get_prompt(label), negative_prompt=get_negative(label),
            image=img, strength=strength, num_inference_steps=15, guidance_scale=4.0,
            generator=torch.Generator(device="cpu").manual_seed(seed),
        ).images[0]

    def gen_candidates(self, ctrl, label, n=4, do_refine=True):
        seed = random.randint(0, 999999)
        imgs = self.generate(ctrl, label, seed=seed, batch=n)
        if do_refine:
            imgs = [self.refine(im, label, seed=seed+i) for i, im in enumerate(imgs)]
        return [remove_bg(im) for im in imgs]


class KimiJudge:
    def __init__(self):
        self.apiKey = os.getenv("OPENROUTER_API_KEY")
        if not self.apiKey:
            raise ValueError("Set OPENROUTER_API_KEY in .env")

    def _to_b64(self, img):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def pick_best(self, images, label, retries=3):
        content = [{"type": "text", "text": (
            f"You see {len(images)} candidate images of '{label}'. "
            f"Pick the best one by visual quality, coherence, and how well it depicts a {label}. "
            "Be brief. Respond ONLY with JSON: {\"reasoning\": \"...\", \"best_image\": N} "
            "where N is 1-based index."
        )}]
        for i, img in enumerate(images):
            content.append({"type": "text", "text": f"Image {i+1}:"})
            content.append({"type": "image_url",
                           "image_url": {"url": f"data:image/png;base64,{self._to_b64(img)}"}})

        for attempt in range(retries):
            try:
                r = requests.post(OPENROUTER_URL,
                    headers={"Authorization": f"Bearer {self.apiKey}",
                             "Content-Type": "application/json"},
                    json={"model": OPENROUTER_MODEL,
                          "messages": [{"role": "user", "content": content}],
                          "response_format": {"type": "json_object"},
                          "max_tokens": 4096, "temperature": 0.3},
                    timeout=120)
                r.raise_for_status()
                raw = r.json()["choices"][0]["message"].get("content") or ""
                if not raw.strip():
                    print(f"    empty response, retry {attempt+1}")
                    continue
                parsed = json.loads(raw)
                idx = int(parsed["best_image"]) - 1
                if 0 <= idx < len(images):
                    return idx, parsed.get("reasoning", "")
            except (json.JSONDecodeError, KeyError, ValueError, requests.RequestException) as e:
                print(f"    err: {e}, retry {attempt+1}")
        return 0, "fallback"


def fetch_quickdraw(category, n, offset=0):
    url = f"https://storage.googleapis.com/quickdraw_dataset/full/simplified/{category}.ndjson"
    r = requests.get(url, stream=True)
    out = []
    for i, line in enumerate(r.iter_lines()):
        if i < offset: continue
        if len(out) >= n: break
        if line: out.append(json.loads(line))
    return out


def drawing_to_img(d, sz=1024):
    img = Image.new('RGB', (sz, sz), 'white')
    draw = ImageDraw.Draw(img)
    strokes = d['drawing']
    xs = [x for s in strokes for x in s[0]]
    ys = [y for s in strokes for y in s[1]]
    if not xs: return img
    mnx, mxx, mny, mxy = min(xs), max(xs), min(ys), max(ys)
    sc = sz * 0.8 / max(mxx - mnx, mxy - mny, 1)
    ox = (sz - (mxx-mnx)*sc)/2 - mnx*sc
    oy = (sz - (mxy-mny)*sc)/2 - mny*sc
    for s in strokes:
        pts = [(x*sc+ox, y*sc+oy) for x, y in zip(s[0], s[1])]
        if len(pts) > 1:
            draw.line(pts, fill='black', width=max(3, sz//170))
    return img


def to_lineart(image):
    arr = np.array(image.convert('L'))
    _, b = cv2.threshold(arr, 200, 255, cv2.THRESH_BINARY_INV)
    b = cv2.dilate(b, np.ones((3, 3), np.uint8), iterations=1)
    return Image.fromarray(b).convert('RGB')


def stamp(image, score, label=""):
    img = image.copy()
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        font = ImageFont.load_default()
    txt = f"{score:.3f}"
    if label: txt = f"{label}: {txt}"
    bb = d.textbbox((0, 0), txt, font=font)
    d.rectangle([5, 5, bb[2]-bb[0]+15, bb[3]-bb[1]+15], fill='black')
    d.text((10, 8), txt, fill='white', font=font)
    return img


def log_experiment(path, desc, metrics):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(path, 'a') as f:
        m = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        f.write(f"| {ts} | {desc} | {m} |\n")


def run(classes=None, spc=1):
    if classes is None:
        classes = CLASSES[:3]

    out = Path("out3/v9_sdxl")
    out.mkdir(parents=True, exist_ok=True)
    md = Path("experiments.md")

    pipe = Pipeline()
    pipe.load()

    random.seed(2026)
    print("\nFetching sketches...")
    all_data = []
    for cls in classes:
        ds = fetch_quickdraw(cls, spc, random.randint(0, 500))
        for d in ds:
            all_data.append((cls, d))
        print(f"  {cls}: {len(ds)}")

    N = len(all_data)

    # === EXP 1: baseline + refiner + bg removal ===
    print(f"\n{'='*50}\nEXP 1: SDXL + ControlNet + Refiner + BG removal\n{'='*50}")
    exp1_scores = []
    for i, (lbl, drawing) in enumerate(all_data):
        print(f"  {i+1}/{N} ({lbl})")
        sketch = drawing_to_img(drawing)
        ctrl = to_lineart(sketch)
        raw = pipe.generate(ctrl, lbl, seed=42+i)[0]
        raw = pipe.refine(raw, lbl, seed=42+i)
        img = remove_bg(raw)
        sc = pipe.scorer.score_single(img, score_prompt(lbl))
        exp1_scores.append(sc)
        stamp(img, sc, lbl).save(out / f"exp1_{lbl}_{i}.png")
        stamp(raw, sc, f"{lbl}_raw").save(out / f"exp1_{lbl}_{i}_raw.png")
        sketch.save(out / f"sketch_{lbl}_{i}.png")

    avg1 = np.mean(exp1_scores)
    print(f"  avg={avg1:.4f} min={min(exp1_scores):.4f} max={max(exp1_scores):.4f}")
    log_experiment(md, f"v9 SDXL + MistoLine + refiner, {len(classes)}cls x{spc}spc",
                  {"avg_siglip2": avg1, "min": min(exp1_scores), "max": max(exp1_scores)})

    # === EXP 2: best of 4 by SigLIP2 ===
    print(f"\n{'='*50}\nEXP 2: Best of 4 by SigLIP2\n{'='*50}")
    exp2_scores = []
    for i, (lbl, drawing) in enumerate(all_data):
        print(f"  {i+1}/{N} ({lbl}) generating 4 candidates...")
        sketch = drawing_to_img(drawing)
        ctrl = to_lineart(sketch)
        cands = pipe.gen_candidates(ctrl, lbl, n=4, do_refine=True)
        cand_scores = [pipe.scorer.score_single(c, score_prompt(lbl)) for c in cands]
        print(f"    scores: {[f'{s:.3f}' for s in cand_scores]}")

        best_idx = int(np.argmax(cand_scores))
        print(f"    siglip2 picked #{best_idx+1} (score={cand_scores[best_idx]:.3f})")

        exp2_scores.append(cand_scores[best_idx])
        for j, (c, cs) in enumerate(zip(cands, cand_scores)):
            tag = "_BEST" if j == best_idx else ""
            stamp(c, cs, f"c{j+1}").save(out / f"exp2_{lbl}_{i}_c{j+1}{tag}.png")

    avg2 = np.mean(exp2_scores)
    print(f"  avg={avg2:.4f} min={min(exp2_scores):.4f} max={max(exp2_scores):.4f}")
    log_experiment(md, f"v9 SDXL + BestOf4(SigLIP2)",
                  {"avg_siglip2": avg2, "min": min(exp2_scores), "max": max(exp2_scores)})

    print(f"\n{'='*50}\nSUMMARY (SDXL)\n{'='*50}")
    print(f"  Refined:       {avg1:.4f}")
    print(f"  Best-of-4(S):  {avg2:.4f}")
    print(f"\nResults: {out}/")


if __name__ == "__main__":
    run()
