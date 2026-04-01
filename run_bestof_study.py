#!/usr/bin/env python3
import numpy as np
import requests, json, random, gc, io, os, base64, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageDraw, ImageFont
import torch
from diffusers import (
    StableDiffusionControlNetPipeline, StableDiffusionImg2ImgPipeline,
    StableDiffusionXLControlNetPipeline, StableDiffusionXLImg2ImgPipeline,
    ControlNetModel, AutoencoderKL, EulerDiscreteScheduler
)
from transformers import AutoModel, SiglipProcessor, SiglipImageProcessor, GemmaTokenizerFast
import cv2
from pathlib import Path
from datetime import datetime

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "moonshotai/kimi-k2.5"
OPENROUTER_KEY = "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"

TEST_CLASSES = ["apple", "cat", "house", "bird", "bicycle"]
SAMPLES_PER_CLASS = {"apple": 4, "cat": 3, "house": 3, "bird": 3, "bicycle": 3}
QD_OFFSETS = {"apple": 100, "cat": 200, "house": 300, "bird": 400, "bicycle": 500}

OUT = Path("bestof_study")
OUT.mkdir(parents=True, exist_ok=True)

with open("quick_draw/prompts.json") as f:
    _pdata = json.load(f)
PROMPTS = _pdata["prompts"]
NEGATIVES = _pdata["negatives"]
DEFAULT_NEG = _pdata.get("default_negative",
    "easynegative, (worst quality, low quality:1.4), people, human, text, watermark, ugly, blurry, deformed")


def get_prompt(label):
    return PROMPTS.get(label, f"a single {label}, solo, centered, (digital illustration:1.2), detailed, sharp focus, vivid colors")

def get_negative(label):
    return NEGATIVES.get(label, DEFAULT_NEG)

def score_prompt(label):
    return f"a high quality illustration of a {label}"


def get_font(size=18):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except:
        return ImageFont.load_default()


def fetch_quickdraw(category, n, offset=0):
    url = f"https://storage.googleapis.com/quickdraw_dataset/full/simplified/{category}.ndjson"
    r = requests.get(url, stream=True)
    out = []
    for i, line in enumerate(r.iter_lines()):
        if i < offset:
            continue
        if len(out) >= n:
            break
        if line:
            out.append(json.loads(line))
    return out


def drawing_to_img(d, sz=512):
    img = Image.new('RGB', (sz, sz), 'white')
    draw = ImageDraw.Draw(img)
    strokes = d['drawing']
    xs = [x for s in strokes for x in s[0]]
    ys = [y for s in strokes for y in s[1]]
    if not xs:
        return img
    mnx, mxx, mny, mxy = min(xs), max(xs), min(ys), max(ys)
    sc = sz * 0.8 / max(mxx - mnx, mxy - mny, 1)
    ox = (sz - (mxx - mnx) * sc) / 2 - mnx * sc
    oy = (sz - (mxy - mny) * sc) / 2 - mny * sc
    for s in strokes:
        pts = [(x * sc + ox, y * sc + oy) for x, y in zip(s[0], s[1])]
        if len(pts) > 1:
            draw.line(pts, fill='black', width=max(2, sz // 170))
    return img


def to_lineart(image):
    arr = np.array(image.convert('L'))
    _, b = cv2.threshold(arr, 200, 255, cv2.THRESH_BINARY_INV)
    b = cv2.dilate(b, np.ones((2, 2), np.uint8), iterations=1)
    return Image.fromarray(b).convert('RGB')


def stamp(image, text):
    img = image.copy()
    d = ImageDraw.Draw(img)
    font = get_font(14)
    bb = d.textbbox((0, 0), text, font=font)
    d.rectangle([3, 3, bb[2] - bb[0] + 13, bb[3] - bb[1] + 13], fill='black')
    d.text((8, 6), text, fill='white', font=font)
    return img


def make_grid(images, labels=None, cols=4, cell=256, title=""):
    n = len(images)
    rows = (n + cols - 1) // cols
    title_h = 40 if title else 0
    grid = Image.new('RGB', (cols * cell, rows * cell + title_h), 'white')
    if title:
        d = ImageDraw.Draw(grid)
        d.text((10, 8), title, fill='black', font=get_font(20))
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        resized = im.resize((cell, cell), Image.LANCZOS)
        if labels and i < len(labels):
            resized = stamp(resized, labels[i])
        grid.paste(resized, (c * cell, r * cell + title_h))
    return grid


class SiglipScorer:
    MODEL_ID = "google/siglip2-base-patch16-224"

    def __init__(self, device="cuda"):
        self.device = device
        self.model = AutoModel.from_pretrained(self.MODEL_ID, torch_dtype=torch.float16).to(device)
        self.processor = SiglipProcessor(
            image_processor=SiglipImageProcessor.from_pretrained(self.MODEL_ID),
            tokenizer=GemmaTokenizerFast.from_pretrained(self.MODEL_ID),
        )
        self.model.eval()

    @torch.no_grad()
    def score(self, images, texts):
        inputs = self.processor(text=texts, images=images, padding="max_length", return_tensors="pt").to(self.device)
        return np.diag(self.model(**inputs).logits_per_image.cpu().numpy())

    def score_single(self, image, text):
        return float(self.score([image], [text])[0])


class KimiScorer:
    def __init__(self):
        self.api_key = OPENROUTER_KEY

    def _b64(self, img):
        buf = io.BytesIO()
        img.resize((512, 512)).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def expert_score(self, image, label, retries=3):
        prompt = (
            f"You are an expert evaluating an AI-generated image that should depict: '{label}'.\n"
            f"Score from 1 to 10 on each criterion:\n"
            f"1) PROMPT_MATCH: does it clearly show a {label}?\n"
            f"2) ARTIFACTS: absence of AI artifacts (extra limbs, melting, smearing). 10=clean\n"
            f"3) CONCEPT_CLARITY: is the object instantly recognizable?\n"
            f"4) BG_SEPARABILITY: can the subject be cleanly cut from background?\n"
            f"5) CLEANLINESS: overall sharpness, no noise, professional look\n\n"
            f"Respond ONLY with JSON:\n"
            f"{{\"prompt_match\": N, \"artifacts\": N, \"concept_clarity\": N, "
            f"\"bg_separability\": N, \"cleanliness\": N, \"avg\": N, \"note\": \"brief\"}}\n"
            f"where avg = mean of all 5 scores, rounded to 1 decimal."
        )
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self._b64(image)}"}}
        ]
        for attempt in range(retries):
            try:
                r = requests.post(OPENROUTER_URL,
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    json={"model": OPENROUTER_MODEL,
                          "messages": [{"role": "user", "content": content}],
                          "response_format": {"type": "json_object"},
                          "max_tokens": 512, "temperature": 0.2},
                    timeout=120)
                r.raise_for_status()
                raw = r.json()["choices"][0]["message"].get("content", "")
                if raw.strip():
                    parsed = json.loads(raw)
                    if "avg" not in parsed:
                        vals = [parsed.get(k, 5) for k in
                                ["prompt_match", "artifacts", "concept_clarity", "bg_separability", "cleanliness"]]
                        parsed["avg"] = round(np.mean(vals), 1)
                    return parsed
            except Exception as e:
                print(f"    kimi err: {e}")
                time.sleep(2)
        return {"prompt_match": 0, "artifacts": 0, "concept_clarity": 0,
                "bg_separability": 0, "cleanliness": 0, "avg": 0, "note": "api_fail"}


class SD15Pipeline:
    def __init__(self, device="cuda"):
        self.device = device
        self.pipe = None
        self.refiner = None

    def load(self):
        print("Loading SD1.5 (dreamshaper)...")
        cn = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_lineart", torch_dtype=torch.float16).to(self.device)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "Lykon/dreamshaper-8", controlnet=cn, torch_dtype=torch.float16, safety_checker=None
        ).to(self.device)
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_attention_slicing()
        try:
            self.pipe.load_textual_inversion(
                "EvilEngine/easynegative", weight_name="easynegative.safetensors", token="easynegative")
        except Exception as e:
            print(f"  easyneg warn: {e}")
        comps = {k: v for k, v in self.pipe.components.items() if k != "controlnet"}
        self.refiner = StableDiffusionImg2ImgPipeline(**comps)
        print("SD1.5 ready")

    def generate(self, ctrl, label, seed=42, batch=1):
        return self.pipe(
            prompt=get_prompt(label), negative_prompt=get_negative(label),
            image=ctrl, num_inference_steps=30,
            guidance_scale=7.5, controlnet_conditioning_scale=0.7,
            control_guidance_start=0.0, control_guidance_end=0.4,
            generator=torch.Generator(device=self.device).manual_seed(seed),
            num_images_per_prompt=batch,
        ).images

    def refine_img(self, img, label, seed=42):
        return self.refiner(
            prompt=get_prompt(label), negative_prompt=get_negative(label),
            image=img, strength=0.25, num_inference_steps=15, guidance_scale=4.0,
            generator=torch.Generator(device=self.device).manual_seed(seed),
        ).images[0]

    def unload(self):
        del self.pipe, self.refiner
        self.pipe = self.refiner = None
        gc.collect()
        torch.cuda.empty_cache()


class SDXLPipeline:
    def __init__(self, device="cuda"):
        self.device = device
        self.pipe = None
        self.refiner = None

    def load(self):
        print("Loading SDXL...")
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
        print("SDXL ready")

    def generate(self, ctrl, label, seed=42, batch=1):
        ctrl_xl = ctrl.resize((1024, 1024), Image.LANCZOS) if ctrl.size[0] < 1024 else ctrl
        return self.pipe(
            prompt=get_prompt(label), negative_prompt=get_negative(label),
            image=ctrl_xl, num_inference_steps=30,
            guidance_scale=7.0, controlnet_conditioning_scale=0.5,
            control_guidance_start=0.0, control_guidance_end=0.4,
            generator=torch.Generator(device="cpu").manual_seed(seed),
            num_images_per_prompt=batch,
        ).images

    def refine_img(self, img, label, seed=42):
        return self.refiner(
            prompt=get_prompt(label), negative_prompt=get_negative(label),
            image=img, strength=0.25, num_inference_steps=15, guidance_scale=4.0,
            generator=torch.Generator(device="cpu").manual_seed(seed),
        ).images[0]

    def unload(self):
        del self.pipe, self.refiner
        self.pipe = self.refiner = None
        gc.collect()
        torch.cuda.empty_cache()


def load_test_set():
    cache = Path("experiments_report/test_set.json")
    if cache.exists():
        with open(cache) as f:
            cached = json.load(f)
        test_set = []
        for item in cached:
            img = drawing_to_img(item["drawing"])
            test_set.append({"class": item["class"], "drawing": item["drawing"],
                             "image": img, "idx": item["idx"]})
        print(f"Loaded {len(test_set)} sketches from cache")
        return test_set

    print("Fetching test set from QuickDraw...")
    random.seed(2026)
    test_set = []
    idx = 0
    for cls in TEST_CLASSES:
        n = SAMPLES_PER_CLASS[cls]
        offset = QD_OFFSETS[cls]
        drawings = fetch_quickdraw(cls, n, offset)
        for d in drawings:
            img = drawing_to_img(d)
            test_set.append({"class": cls, "drawing": d, "image": img, "idx": idx})
            idx += 1
        print(f"  {cls}: {len(drawings)} sketches")

    cache.parent.mkdir(parents=True, exist_ok=True)
    save_data = [{"class": t["class"], "drawing": t["drawing"], "idx": t["idx"]} for t in test_set]
    with open(cache, 'w') as f:
        json.dump(save_data, f)
    return test_set


def run_bestof(pipe, scorer, test_set, best_of, model_tag):
    tag = f"{model_tag}_bestof{best_of}"
    print(f"\n{'=' * 50}")
    print(f"  {tag}")
    print(f"{'=' * 50}")

    results = []
    images = []
    labels = []

    for item in test_set:
        cls = item["class"]
        idx = item["idx"]
        ctrl = to_lineart(item["image"])
        seed_base = 42 + idx * 100

        if best_of > 1:
            candidates = pipe.generate(ctrl, cls, seed=seed_base, batch=best_of)
            candidates = [pipe.refine_img(im, cls, seed=seed_base + j) for j, im in enumerate(candidates)]
            cand_scores = [scorer.score_single(im, score_prompt(cls)) for im in candidates]
            pick = int(np.argmax(cand_scores))
            final = candidates[pick]
        else:
            final = pipe.generate(ctrl, cls, seed=seed_base)[0]
            final = pipe.refine_img(final, cls, seed=seed_base)

        siglip = scorer.score_single(final, score_prompt(cls))
        results.append({"class": cls, "idx": idx, "siglip2": siglip, "image": final})
        images.append(final)
        labels.append(f"{cls}: {siglip:.2f}")
        print(f"  [{idx}] {cls} siglip={siglip:.3f}")

    avg = np.mean([r["siglip2"] for r in results])
    mn = min(r["siglip2"] for r in results)
    mx = max(r["siglip2"] for r in results)
    print(f"  => avg={avg:.3f} min={mn:.3f} max={mx:.3f}")

    grid = make_grid(images, labels, cols=4, title=f"{tag} | avg_siglip={avg:.2f}")
    grid.save(OUT / f"{tag}.png")

    return {"tag": tag, "model": model_tag, "best_of": best_of,
            "results": results, "avg_siglip2": avg, "min_siglip2": mn, "max_siglip2": mx}


def kimi_score_all(kimi, experiments, max_workers=16):
    all_tasks = []
    for exp in experiments:
        for r in exp["results"]:
            all_tasks.append((exp["tag"], r))

    print(f"\nScoring {len(all_tasks)} images with Kimi K2.5 ({max_workers} threads)...")

    def _score(task):
        tag, r = task
        ks = kimi.expert_score(r["image"], r["class"])
        return tag, r["idx"], r["class"], ks

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_score, t): t for t in all_tasks}
        done = 0
        for fut in as_completed(futures):
            tag, idx, cls, ks = fut.result()
            _, r = futures[fut]
            r["kimi"] = ks
            done += 1
            if done % 8 == 0 or done == len(all_tasks):
                print(f"  scored {done}/{len(all_tasks)}")

    for exp in experiments:
        kimi_scores = [r["kimi"].get("avg", 0) for r in exp["results"]]
        exp["kimi_avg"] = float(np.mean(kimi_scores))
        exp["kimi_scores"] = kimi_scores


def print_report(experiments):
    print("\n" + "=" * 70)
    print("  RESULTS: best-of-N study")
    print("=" * 70)

    header = f"{'Config':<25} {'SigLIP2 avg':>12} {'SigLIP2 min':>12} {'SigLIP2 max':>12} {'Kimi avg':>10}"
    print(header)
    print("-" * len(header))

    by_model = {}
    for exp in experiments:
        m = exp["model"]
        if m not in by_model:
            by_model[m] = []
        by_model[m].append(exp)

    for model in sorted(by_model.keys()):
        exps = sorted(by_model[model], key=lambda e: e["best_of"])
        for exp in exps:
            kimi = exp.get("kimi_avg", 0)
            line = f"{exp['tag']:<25} {exp['avg_siglip2']:>12.3f} {exp['min_siglip2']:>12.3f} {exp['max_siglip2']:>12.3f} {kimi:>10.1f}"
            print(line)
        print()

    print("\nPer-class breakdown (Kimi avg):")
    for exp in experiments:
        class_kimi = {}
        for r in exp["results"]:
            cls = r["class"]
            if cls not in class_kimi:
                class_kimi[cls] = []
            class_kimi[cls].append(r["kimi"].get("avg", 0))
        class_avgs = {cls: np.mean(vals) for cls, vals in class_kimi.items()}
        parts = [f"{cls}={v:.1f}" for cls, v in sorted(class_avgs.items())]
        print(f"  {exp['tag']}: {', '.join(parts)}")

    print("\n" + "=" * 70)
    print("  CONCLUSIONS")
    print("=" * 70)

    for model in sorted(by_model.keys()):
        exps = sorted(by_model[model], key=lambda e: e["best_of"])
        if len(exps) < 2:
            continue
        print(f"\n{model}:")
        base = exps[0]
        for exp in exps[1:]:
            ds = exp["avg_siglip2"] - base["avg_siglip2"]
            dk = exp.get("kimi_avg", 0) - base.get("kimi_avg", 0)
            print(f"  best-of-{exp['best_of']} vs best-of-{base['best_of']}: "
                  f"SigLIP2 {'+' if ds >= 0 else ''}{ds:.3f}, "
                  f"Kimi {'+' if dk >= 0 else ''}{dk:.1f}")
        best_s = max(exps, key=lambda e: e["avg_siglip2"])
        best_k = max(exps, key=lambda e: e.get("kimi_avg", 0))
        print(f"  best by SigLIP2: best-of-{best_s['best_of']} ({best_s['avg_siglip2']:.3f})")
        print(f"  best by Kimi: best-of-{best_k['best_of']} ({best_k.get('kimi_avg', 0):.1f})")


def save_results(experiments):
    summary = []
    for exp in experiments:
        entry = {
            "tag": exp["tag"], "model": exp["model"], "best_of": exp["best_of"],
            "avg_siglip2": exp["avg_siglip2"], "min_siglip2": exp["min_siglip2"],
            "max_siglip2": exp["max_siglip2"], "kimi_avg": exp.get("kimi_avg"),
            "per_sample": []
        }
        for r in exp["results"]:
            entry["per_sample"].append({
                "class": r["class"], "idx": r["idx"],
                "siglip2": r["siglip2"], "kimi": r.get("kimi", {})
            })
        summary.append(entry)

    path = OUT / "results.json"
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved to {path}")


def main():
    scorer = SiglipScorer()
    test_set = load_test_set()
    kimi = KimiScorer()

    all_experiments = []

    sd15 = SD15Pipeline()
    sd15.load()
    for bo in [1, 4, 8]:
        exp = run_bestof(sd15, scorer, test_set, best_of=bo, model_tag="SD15")
        all_experiments.append(exp)
    sd15.unload()

    sdxl = SDXLPipeline()
    sdxl.load()
    for bo in [1, 4, 8]:
        exp = run_bestof(sdxl, scorer, test_set, best_of=bo, model_tag="SDXL")
        all_experiments.append(exp)
    sdxl.unload()

    kimi_score_all(kimi, all_experiments, max_workers=16)

    print_report(all_experiments)
    save_results(all_experiments)

    print("\nDone.")


if __name__ == "__main__":
    main()
