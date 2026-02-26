import numpy as np
import requests
import json
from PIL import Image, ImageDraw, ImageFont
import torch
from diffusers import (StableDiffusionControlNetPipeline, StableDiffusionImg2ImgPipeline,
                       ControlNetModel, EulerDiscreteScheduler)
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

EXPERIMENT_CLASSES = ["cat", "car", "flower"]

NEGATIVE = (
    "easynegative, (worst quality, low quality:1.4), "
    "people, person, human, multiple objects, "
    "text, watermark, signature, logo, "
    "frame, border, ugly, blurry, deformed"
)

OPENROUTER_MODEL = "moonshotai/kimi-k2.5"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def gen_prompt(label):
    return (
        f"a single {label}, solo, centered, "
        "(digital illustration:1.2), detailed, sharp focus, "
        "vivid natural colors, simple background"
    )

def score_prompt(label):
    return f"a high quality illustration of a {label}"

def remove_bg(img, bg_color=(255, 255, 255)):
    rgba = rembg_remove(img)
    out = Image.new('RGBA', rgba.size, (*bg_color, 255))
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


class GenerationPipeline:
    def __init__(self, device="cuda", imgSize=512):
        self.device = device
        self.imgSize = imgSize
        self.controlnet = None
        self.pipe = None
        self.refiner = None
        self.scorer = SiglipScorer(device)

    def load(self, model_id="runwayml/stable-diffusion-v1-5"):
        print(f"Loading {model_id}...")
        if not self.controlnet:
            self.controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11p_sd15_lineart", torch_dtype=torch.float16
            ).to(self.device)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            model_id, controlnet=self.controlnet,
            torch_dtype=torch.float16, safety_checker=None
        ).to(self.device)
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_attention_slicing()
        self.pipe.load_textual_inversion(
            "EvilEngine/easynegative", weight_name="easynegative.safetensors", token="easynegative")
        print("Loaded EasyNegative embedding")
        components = {k: v for k, v in self.pipe.components.items() if k != "controlnet"}
        self.refiner = StableDiffusionImg2ImgPipeline(**components)
        print("Pipeline ready")

    def generate(self, controlImg, label, seed=42, steps=30,
                 cn_scale=0.7, guidance=7.5, batch=1):
        ctrl_end = min(1.0, 12 / steps)
        gen = torch.Generator(device=self.device).manual_seed(seed)
        imgs = self.pipe(
            prompt=gen_prompt(label), negative_prompt=NEGATIVE,
            image=controlImg, num_inference_steps=steps,
            guidance_scale=guidance, controlnet_conditioning_scale=cn_scale,
            control_guidance_start=0.0, control_guidance_end=ctrl_end,
            generator=gen, num_images_per_prompt=batch,
        ).images
        return imgs[0] if batch == 1 else imgs

    def refine_image(self, img, label, seed=42,
                     strength=0.25, steps=15, guidance=4.0):
        return self.refiner(
            prompt=gen_prompt(label),
            negative_prompt="blurry, artifacts, noise, ugly",
            image=img, strength=strength,
            num_inference_steps=steps, guidance_scale=guidance,
            generator=torch.Generator(device=self.device).manual_seed(seed),
        ).images[0]

    def gen_candidates(self, controlImg, label, n=4, steps=30,
                       cn_scale=0.7, do_refine=True, refine_strength=0.25):
        seed = random.randint(0, 999999)
        candidates = self.generate(controlImg, label, seed=seed, steps=steps,
                                   cn_scale=cn_scale, batch=n)
        if do_refine:
            candidates = [self.refine_image(c, label, seed=seed+i, strength=refine_strength)
                          for i, c in enumerate(candidates)]
        return candidates


class KimiJudge:
    def __init__(self):
        self.apiKey = os.getenv("OPENROUTER_API_KEY")
        if not self.apiKey:
            raise ValueError("Set OPENROUTER_API_KEY in .env")
        self.model_id = OPENROUTER_MODEL
        self.url = OPENROUTER_URL

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
                r = requests.post(self.url,
                    headers={"Authorization": f"Bearer {self.apiKey}",
                             "Content-Type": "application/json"},
                    json={"model": self.model_id,
                          "messages": [{"role": "user", "content": content}],
                          "response_format": {"type": "json_object"},
                          "max_tokens": 4096, "temperature": 0.3},
                    timeout=120)
                r.raise_for_status()
                resp = r.json()["choices"][0]["message"]
                raw = resp.get("content") or ""
                if not raw.strip():
                    print(f"    empty content (reasoning model used all tokens?), retry {attempt+1}")
                    continue
                parsed = json.loads(raw)
                idx = int(parsed["best_image"]) - 1
                if 0 <= idx < len(images):
                    return idx, parsed.get("reasoning", "")
                print(f"    bad index {idx}, retry {attempt+1}")
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"    parse err: {e}, retry {attempt+1}")
            except requests.RequestException as e:
                print(f"    api err: {e}, retry {attempt+1}")
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


def drawing_to_img(d, sz=512):
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
            draw.line(pts, fill='black', width=max(2, sz//170))
    return img


def to_lineart(image):
    arr = np.array(image.convert('L'))
    _, b = cv2.threshold(arr, 200, 255, cv2.THRESH_BINARY_INV)
    b = cv2.dilate(b, np.ones((2, 2), np.uint8), iterations=1)
    return Image.fromarray(b).convert('RGB')


def stamp(image, score, label=""):
    img = image.copy()
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except:
        font = ImageFont.load_default()
    txt = f"{score:.3f}"
    if label: txt = f"{label}: {txt}"
    bb = d.textbbox((0, 0), txt, font=font)
    d.rectangle([5, 5, bb[2]-bb[0]+15, bb[3]-bb[1]+15], fill='black')
    d.text((10, 8), txt, fill='white', font=font)
    return img


def build_grid(cols_data, row_labels, img_size=256):
    n_cols = len(cols_data)
    n_rows = len(row_labels)
    header = 35
    lbl_w = 100
    w = lbl_w + n_cols * img_size
    h = header + n_rows * img_size
    grid = Image.new('RGB', (w, h), 'white')
    draw = ImageDraw.Draw(grid)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except:
        f = ImageFont.load_default()

    for j, (name, _, _) in enumerate(cols_data):
        draw.text((lbl_w + j*img_size + 10, 8), name, fill='black', font=f)

    for i, lbl in enumerate(row_labels):
        y = header + i * img_size
        draw.text((5, y + img_size//2 - 8), lbl, fill='black', font=f)
        for j, (_, imgs, scores) in enumerate(cols_data):
            x = lbl_w + j * img_size
            im = imgs[i].resize((img_size, img_size))
            im = stamp(im, scores[i])
            grid.paste(im, (x, y))
    return grid


def log_experiment(path, desc, metrics):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(path, 'a') as f:
        m = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        f.write(f"| {ts} | {desc} | {m} |\n")


def validate_metric(pipe):
    print("\n" + "="*50)
    print("STAGE 0: Metric Validation")
    print("="*50)

    out = Path("out3/v7_experiments/validation")
    out.mkdir(parents=True, exist_ok=True)

    test_classes = ["cat", "flower", "car"]
    all_ok = True

    for cls in test_classes:
        drawings = fetch_quickdraw(cls, 1, offset=42)
        sketch = drawing_to_img(drawings[0])
        ctrl = to_lineart(sketch)
        sprompt = score_prompt(cls)

        sketch_s = pipe.scorer.score_single(sketch, sprompt)

        bad_raw = pipe.generate(ctrl, cls, seed=42, steps=5, guidance=2.0, cn_scale=1.0)
        bad = remove_bg(bad_raw)
        bad_s = pipe.scorer.score_single(bad, sprompt)

        good_raw = pipe.generate(ctrl, cls, seed=42, steps=30)
        good = remove_bg(good_raw)
        good_s = pipe.scorer.score_single(good, sprompt)

        refined_raw = pipe.refine_image(good_raw, cls, seed=42)
        refined = remove_bg(refined_raw)
        refined_s = pipe.scorer.score_single(refined, sprompt)

        ok = good_s > bad_s
        all_ok = all_ok and ok

        print(f"\n  {cls}: sketch={sketch_s:.3f}  bad={bad_s:.3f}  good={good_s:.3f}  refined={refined_s:.3f}  {'OK' if ok else 'FAIL'}")

        stamp(sketch, sketch_s, "sketch").save(out / f"{cls}_sketch.png")
        stamp(bad, bad_s, "bad_5steps").save(out / f"{cls}_bad.png")
        stamp(good, good_s, "good_30steps").save(out / f"{cls}_good.png")
        stamp(refined, refined_s, "refined").save(out / f"{cls}_refined.png")

    print(f"\nMetric validation: {'PASSED' if all_ok else 'NEEDS REVIEW'}")
    print(f"Images saved to {out}/")
    return all_ok


def run_experiments(pipe, classes=None, samples_per_class=2):
    if classes is None:
        classes = EXPERIMENT_CLASSES

    out = Path("out3/v7_experiments")
    out.mkdir(parents=True, exist_ok=True)
    md = Path("experiments.md")

    if not md.exists():
        with open(md, 'w') as f:
            f.write("# Experiment Results\n\n")
            f.write("| Timestamp | Description | Metrics |\n")
            f.write("|-----------|-------------|----------|\n")

    random.seed(2026)
    print("\nDownloading sketches...")
    all_drawings, all_labels = [], []
    for cls in classes:
        ds = fetch_quickdraw(cls, samples_per_class, random.randint(0, 500))
        all_drawings.extend(ds)
        all_labels.extend([cls] * len(ds))
        print(f"  {cls}: {len(ds)}")

    sketches = [drawing_to_img(d) for d in all_drawings]
    controls = [to_lineart(s) for s in sketches]
    N = len(controls)

    # ======= EXP 1 =======
    print(f"\n{'='*50}\nEXP 1: Baseline SD1.5 + ControlNet + BG removal\n{'='*50}")
    baseline_imgs, baseline_scores = [], []
    for i in range(N):
        print(f"  {i+1}/{N} ({all_labels[i]})")
        raw = pipe.generate(controls[i], all_labels[i], seed=42+i)
        img = remove_bg(raw)
        sc = pipe.scorer.score_single(img, score_prompt(all_labels[i]))
        baseline_imgs.append(img)
        baseline_scores.append(sc)
        stamp(img, sc, all_labels[i]).save(out / f"exp1_baseline_{all_labels[i]}_{i}.png")

    avg1 = np.mean(baseline_scores)
    print(f"  avg={avg1:.4f} min={min(baseline_scores):.4f} max={max(baseline_scores):.4f}")
    log_experiment(md, f"Baseline SD1.5+ControlNet, {len(classes)}cls x{samples_per_class}spc",
                  {"avg_siglip2": avg1, "min": min(baseline_scores), "max": max(baseline_scores)})

    # ======= EXP 2 =======
    print(f"\n{'='*50}\nEXP 2: + Refiner + BG removal\n{'='*50}")
    refined_imgs, refined_scores = [], []
    for i in range(N):
        print(f"  {i+1}/{N} ({all_labels[i]})")
        raw = pipe.generate(controls[i], all_labels[i], seed=42+i)
        raw = pipe.refine_image(raw, all_labels[i], seed=42+i)
        img = remove_bg(raw)
        sc = pipe.scorer.score_single(img, score_prompt(all_labels[i]))
        refined_imgs.append(img)
        refined_scores.append(sc)
        stamp(img, sc, all_labels[i]).save(out / f"exp2_refined_{all_labels[i]}_{i}.png")

    avg2 = np.mean(refined_scores)
    print(f"  avg={avg2:.4f} min={min(refined_scores):.4f} max={max(refined_scores):.4f}")
    log_experiment(md, f"SD1.5+ControlNet+Refiner(s=0.25), same sketches as exp1",
                  {"avg_siglip2": avg2, "min": min(refined_scores), "max": max(refined_scores)})

    # ======= EXP 3 =======
    print(f"\n{'='*50}\nEXP 3: Best of 4 + Qwen\n{'='*50}")
    try:
        judge = KimiJudge()
    except ValueError as e:
        print(f"  SKIP: {e}")
        judge = None

    avg3 = None
    best_imgs, best_scores = [], []
    if judge:
        for i in range(N):
            lbl = all_labels[i]
            print(f"  {i+1}/{N} ({lbl}) generating 4 candidates...")
            cands_raw = pipe.gen_candidates(controls[i], lbl, n=4, do_refine=True)
            cands = [remove_bg(c) for c in cands_raw]
            cand_scores = [pipe.scorer.score_single(c, score_prompt(lbl)) for c in cands]
            print(f"    scores: {[f'{s:.3f}' for s in cand_scores]}")

            idx, reason = judge.pick_best(cands, lbl)
            print(f"    kimi picked #{idx+1}: {reason[:60]}")

            best_imgs.append(cands[idx])
            best_scores.append(cand_scores[idx])

            for j, (c, cs) in enumerate(zip(cands, cand_scores)):
                sel = "_BEST" if j == idx else ""
                stamp(c, cs, f"c{j+1}").save(out / f"exp3_{lbl}_{i}_c{j+1}{sel}.png")

        avg3 = np.mean(best_scores)
        print(f"  avg={avg3:.4f} min={min(best_scores):.4f} max={max(best_scores):.4f}")
        log_experiment(md, f"SD1.5+CN+Refiner+BestOf4(Kimi), 4 candidates per sample",
                      {"avg_siglip2": avg3, "min": min(best_scores), "max": max(best_scores)})

    # ======= comparison grid =======
    print("\nBuilding comparison grid...")
    spc = samples_per_class
    cls_b_scores = [np.mean(baseline_scores[i*spc:(i+1)*spc]) for i in range(len(classes))]
    cls_r_scores = [np.mean(refined_scores[i*spc:(i+1)*spc]) for i in range(len(classes))]
    cls_b_imgs = [baseline_imgs[i*spc] for i in range(len(classes))]
    cls_r_imgs = [refined_imgs[i*spc] for i in range(len(classes))]

    cols = [("baseline", cls_b_imgs, cls_b_scores),
            ("refined", cls_r_imgs, cls_r_scores)]
    if judge and best_imgs:
        cls_best_scores = [np.mean(best_scores[i*spc:(i+1)*spc]) for i in range(len(classes))]
        cls_best_imgs = [best_imgs[i*spc] for i in range(len(classes))]
        cols.append(("best_of_4", cls_best_imgs, cls_best_scores))

    grid = build_grid(cols, classes)
    grid.save(out / "comparison.png")

    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"  Baseline:  {avg1:.4f}")
    print(f"  Refined:   {avg2:.4f}")
    if avg3 is not None:
        print(f"  Best-of-4: {avg3:.4f}")
    print(f"\nResults: {out}/")
    print(f"Log: {md}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = GenerationPipeline(device=device)
    pipe.load()

    validate_metric(pipe)
    run_experiments(pipe, samples_per_class=1)


if __name__ == "__main__":
    main()
