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

OUT = Path("bestof_multi")
OUT.mkdir(parents=True, exist_ok=True)

with open("quick_draw/prompts.json") as f:
    _pdata = json.load(f)
PROMPTS = _pdata["prompts"]
NEGATIVES = _pdata["negatives"]
DEFAULT_NEG = _pdata.get("default_negative",
    "easynegative, (worst quality, low quality:1.4), people, human, text, watermark, ugly, blurry, deformed")


QUALITY_AXES = {
    "prompt_match": {
        "template": "a high quality illustration of a {label}",
        "weight": 2.0,
    },
    "no_artifacts": {
        "template": "a clean flawless image without any artifacts, distortions, or deformations",
        "weight": 1.5,
    },
    "sharpness": {
        "template": "a sharp detailed crisp image with clear edges and fine details",
        "weight": 1.0,
    },
    "anatomy": {
        "template": "correct natural proportions and anatomy, physically plausible structure",
        "weight": 1.5,
    },
    "composition": {
        "template": "a single centered {label} on a clean simple background, isolated subject",
        "weight": 1.0,
    },
    "colors": {
        "template": "vibrant natural realistic colors with good lighting and contrast",
        "weight": 0.5,
    },
    "professional": {
        "template": "professional commercial product photography, studio quality render",
        "weight": 0.5,
    },
}


def get_prompt(label):
    return PROMPTS.get(label, f"a single {label}, solo, centered, (digital illustration:1.2), detailed, sharp focus, vivid colors")

def get_negative(label):
    return NEGATIVES.get(label, DEFAULT_NEG)


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
    font = get_font(12)
    bb = d.textbbox((0, 0), text, font=font)
    d.rectangle([2, 2, bb[2] - bb[0] + 10, bb[3] - bb[1] + 10], fill='black')
    d.text((6, 4), text, fill='white', font=font)
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
        return self.model(**inputs).logits_per_image.cpu().numpy()

    def score_single(self, image, text):
        return float(self.score([image], [text])[0, 0])

    @torch.no_grad()
    def score_multi(self, images, texts):
        n_img = len(images)
        n_txt = len(texts)
        all_images = images * n_txt
        all_texts = [t for t in texts for _ in range(n_img)]
        inputs = self.processor(
            text=all_texts, images=all_images, padding="max_length", return_tensors="pt"
        ).to(self.device)
        logits = self.model(**inputs).logits_per_image.cpu().numpy()
        matrix = np.zeros((n_img, n_txt))
        for t_idx in range(n_txt):
            for i_idx in range(n_img):
                flat = t_idx * n_img + i_idx
                matrix[i_idx, t_idx] = logits[flat, flat]
        return matrix

    @torch.no_grad()
    def score_batch_multi(self, images, texts):
        n_img = len(images)
        n_txt = len(texts)
        matrix = np.zeros((n_img, n_txt))
        for t_idx, txt in enumerate(texts):
            batch_texts = [txt] * n_img
            inputs = self.processor(
                text=batch_texts, images=images, padding="max_length", return_tensors="pt"
            ).to(self.device)
            logits = self.model(**inputs).logits_per_image.cpu().numpy()
            for i_idx in range(n_img):
                matrix[i_idx, t_idx] = logits[i_idx, i_idx]
        return matrix


class MultiCriteriaSelector:
    def __init__(self, scorer, axes=None):
        self.scorer = scorer
        self.axes = axes or QUALITY_AXES

    def build_prompts(self, label):
        names = []
        texts = []
        weights = []
        for name, cfg in self.axes.items():
            names.append(name)
            texts.append(cfg["template"].format(label=label))
            weights.append(cfg["weight"])
        return names, texts, np.array(weights)

    def _normalize_columns(self, matrix):
        normed = np.zeros_like(matrix)
        for j in range(matrix.shape[1]):
            col = matrix[:, j]
            mn, mx = col.min(), col.max()
            if mx - mn > 1e-6:
                normed[:, j] = (col - mn) / (mx - mn)
            else:
                normed[:, j] = 0.5
        return normed

    def pick_best(self, candidates, label):
        names, texts, weights = self.build_prompts(label)
        raw = self.scorer.score_batch_multi(candidates, texts)
        normed = self._normalize_columns(raw)
        weighted = normed * weights[None, :]
        composite = weighted.sum(axis=1)
        best_idx = int(np.argmax(composite))

        per_axis = {}
        for j, name in enumerate(names):
            per_axis[name] = {
                "raw_scores": raw[:, j].tolist(),
                "normed_scores": normed[:, j].tolist(),
                "best_raw": float(raw[best_idx, j]),
                "weight": float(weights[j]),
            }

        return best_idx, {
            "composite_scores": composite.tolist(),
            "best_composite": float(composite[best_idx]),
            "per_axis": per_axis,
            "raw_matrix": raw.tolist(),
        }

    def score_single(self, image, label):
        names, texts, weights = self.build_prompts(label)
        raw = self.scorer.score_batch_multi([image], texts)
        scores = raw[0]
        per_axis = {}
        for j, name in enumerate(names):
            per_axis[name] = float(scores[j])
        return {
            "raw_scores": scores.tolist(),
            "per_axis": per_axis,
        }


class KimiScorer:
    def __init__(self):
        self.api_key = OPENROUTER_KEY

    def _b64(self, img):
        buf = io.BytesIO()
        img.resize((512, 512)).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def expert_score(self, image, label, retries=5):
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
                    timeout=180)
                r.raise_for_status()
                data = r.json()
                if "choices" not in data or not data["choices"]:
                    print(f"    kimi: no choices, attempt {attempt+1}")
                    time.sleep(3)
                    continue
                raw = data["choices"][0]["message"].get("content", "")
                if not raw.strip():
                    print(f"    kimi: empty response, attempt {attempt+1}")
                    time.sleep(3)
                    continue
                parsed = json.loads(raw)
                if "avg" not in parsed:
                    vals = [parsed.get(k, 5) for k in
                            ["prompt_match", "artifacts", "concept_clarity", "bg_separability", "cleanliness"]]
                    parsed["avg"] = round(np.mean(vals), 1)
                return parsed
            except Exception as e:
                print(f"    kimi err (attempt {attempt+1}/{retries}): {e}")
                time.sleep(3 + attempt * 2)
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
    return test_set


def run_experiment(pipe, scorer, multi_selector, test_set, best_of, selector_type, model_tag):
    tag = f"{model_tag}_bo{best_of}_{selector_type}"
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

            if selector_type == "multi":
                pick, selection_info = multi_selector.pick_best(candidates, cls)
            else:
                cand_scores = [scorer.score_single(im, f"a high quality illustration of a {cls}") for im in candidates]
                pick = int(np.argmax(cand_scores))
                selection_info = {"scores": cand_scores}

            final = candidates[pick]
        else:
            final = pipe.generate(ctrl, cls, seed=seed_base)[0]
            final = pipe.refine_img(final, cls, seed=seed_base)
            selection_info = {}

        siglip_single = scorer.score_single(final, f"a high quality illustration of a {cls}")
        multi_score = multi_selector.score_single(final, cls)

        results.append({
            "class": cls, "idx": idx, "image": final,
            "siglip2": siglip_single,
            "multi_axes": multi_score["per_axis"],
            "selection": selection_info,
        })
        images.append(final)
        axes_summary = " ".join(f"{k[:3]}={v:.1f}" for k, v in list(multi_score["per_axis"].items())[:3])
        lbl = f"{cls}: s={siglip_single:.1f} {axes_summary}"
        labels.append(lbl)
        print(f"  [{idx}] {cls} siglip={siglip_single:.2f} | {axes_summary}")

    avg_s = np.mean([r["siglip2"] for r in results])
    print(f"  => siglip_avg={avg_s:.3f}")

    grid = make_grid(images, labels, cols=4, title=f"{tag} | avg_siglip={avg_s:.2f}")
    grid.save(OUT / f"{tag}.png")

    return {
        "tag": tag, "model": model_tag, "best_of": best_of, "selector": selector_type,
        "results": results,
        "avg_siglip2": float(avg_s),
        "min_siglip2": float(min(r["siglip2"] for r in results)),
        "max_siglip2": float(max(r["siglip2"] for r in results)),
    }


def kimi_score_all(kimi, experiments, max_workers=16):
    all_tasks = []
    for exp in experiments:
        for r in exp["results"]:
            all_tasks.append((exp["tag"], r))

    print(f"\nScoring {len(all_tasks)} images via Kimi ({max_workers} threads)...")

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


def print_report(experiments):
    print("\n" + "=" * 80)
    print("  RESULTS")
    print("=" * 80)

    axis_names = list(QUALITY_AXES.keys())

    header = f"{'Config':<30} {'SigLIP':>7} {'Kimi':>6}"
    for ax in axis_names:
        header += f" {ax[:6]:>7}"
    print(header)
    print("-" * len(header))

    by_model = {}
    for exp in experiments:
        m = exp["model"]
        if m not in by_model:
            by_model[m] = []
        by_model[m].append(exp)

    for model in sorted(by_model.keys()):
        exps = sorted(by_model[model], key=lambda e: (e["best_of"], e["selector"]))
        for exp in exps:
            kimi = exp.get("kimi_avg", 0)
            line = f"{exp['tag']:<30} {exp['avg_siglip2']:>7.2f} {kimi:>6.1f}"

            for ax in axis_names:
                vals = [r["multi_axes"].get(ax, 0) for r in exp["results"]]
                line += f" {np.mean(vals):>7.2f}"
            print(line)
        print()

    print("\n" + "=" * 80)
    print("  SINGLE vs MULTI selector comparison")
    print("=" * 80)

    for model in sorted(by_model.keys()):
        exps = by_model[model]
        for bo in [4, 8]:
            bo_single = [e for e in exps if e["best_of"] == bo and e["selector"] == "single"]
            bo_multi = [e for e in exps if e["best_of"] == bo and e["selector"] == "multi"]
            if not bo_single or not bo_multi:
                continue
            s = bo_single[0]
            m = bo_multi[0]
            print(f"\n{model} best-of-{bo}:")
            print(f"  single: siglip={s['avg_siglip2']:.3f} kimi={s.get('kimi_avg', 0):.1f}")
            print(f"  multi:  siglip={m['avg_siglip2']:.3f} kimi={m.get('kimi_avg', 0):.1f}")
            ds = m['avg_siglip2'] - s['avg_siglip2']
            dk = m.get('kimi_avg', 0) - s.get('kimi_avg', 0)
            print(f"  delta:  siglip={ds:+.3f} kimi={dk:+.1f}")

            print(f"  per-axis (avg raw scores):")
            for ax in axis_names:
                s_vals = np.mean([r["multi_axes"].get(ax, 0) for r in s["results"]])
                m_vals = np.mean([r["multi_axes"].get(ax, 0) for r in m["results"]])
                delta = m_vals - s_vals
                marker = "^" if delta > 0.1 else ("v" if delta < -0.1 else "=")
                print(f"    {ax:<16}: single={s_vals:>6.2f}  multi={m_vals:>6.2f}  {marker}{delta:+.2f}")

    print("\n" + "=" * 80)
    print("  PER-CLASS BREAKDOWN (best-of-4)")
    print("=" * 80)

    for exp in experiments:
        if exp["best_of"] != 4:
            continue
        print(f"\n{exp['tag']}:")
        by_cls = {}
        for r in exp["results"]:
            cls = r["class"]
            if cls not in by_cls:
                by_cls[cls] = []
            by_cls[cls].append(r)

        for cls in TEST_CLASSES:
            if cls not in by_cls:
                continue
            items = by_cls[cls]
            avg_s = np.mean([r["siglip2"] for r in items])
            avg_k = np.mean([r["kimi"].get("avg", 0) for r in items]) if "kimi" in items[0] else 0
            axes_str = ", ".join(
                f"{ax[:4]}={np.mean([r['multi_axes'].get(ax, 0) for r in items]):.1f}"
                for ax in axis_names
            )
            print(f"  {cls:>10}: siglip={avg_s:.2f} kimi={avg_k:.1f} | {axes_str}")


def save_results(experiments):
    summary = []
    for exp in experiments:
        entry = {
            "tag": exp["tag"], "model": exp["model"], "best_of": exp["best_of"],
            "selector": exp["selector"],
            "avg_siglip2": exp["avg_siglip2"],
            "kimi_avg": exp.get("kimi_avg"),
            "per_sample": []
        }
        for r in exp["results"]:
            sample = {
                "class": r["class"], "idx": r["idx"],
                "siglip2": r["siglip2"],
                "multi_axes": r["multi_axes"],
                "kimi": r.get("kimi", {}),
            }
            entry["per_sample"].append(sample)
        summary.append(entry)

    path = OUT / "results.json"
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved to {path}")


def generate_md_report(experiments):
    axis_names = list(QUALITY_AXES.keys())
    lines = []
    lines.append("# Best-of-N: Single vs Multi-criteria SigLIP2")
    lines.append(f"\n*{datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")

    lines.append("## Идея\n")
    lines.append("Вместо одного промпта `a high quality illustration of a {label}` скорим кандидатов")
    lines.append("по 7 независимым осям качества через SigLIP2. Скоры нормализуются per-axis")
    lines.append("(min-max среди кандидатов), затем суммируются с весами:\n")
    for name, cfg in QUALITY_AXES.items():
        lines.append(f"- **{name}** (w={cfg['weight']}): `{cfg['template']}`")
    lines.append("")

    lines.append("---\n")
    lines.append("## Результаты\n")

    by_model = {}
    for exp in experiments:
        m = exp["model"]
        if m not in by_model:
            by_model[m] = []
        by_model[m].append(exp)

    for model in sorted(by_model.keys()):
        lines.append(f"### {model}\n")
        exps = sorted(by_model[model], key=lambda e: (e["best_of"], e["selector"]))
        for exp in exps:
            lines.append(f"#### {exp['tag']}\n")
            lines.append(f"![{exp['tag']}]({exp['tag']}.png)\n")
            kimi = exp.get('kimi_avg', 0)
            lines.append(f"| Metric | Value |")
            lines.append(f"|--------|-------|")
            lines.append(f"| SigLIP2 avg | **{exp['avg_siglip2']:.3f}** |")
            lines.append(f"| Kimi avg | **{kimi:.1f}**/10 |")

            lines.append(f"\nPer-axis raw scores (avg across samples):\n")
            lines.append(f"| Axis | Avg score |")
            lines.append(f"|------|-----------|")
            for ax in axis_names:
                vals = [r["multi_axes"].get(ax, 0) for r in exp["results"]]
                lines.append(f"| {ax} | {np.mean(vals):.2f} |")
            lines.append("")

    lines.append("---\n")
    lines.append("## Сводная таблица\n")
    header = "| Config | SigLIP2 | Kimi |"
    for ax in axis_names:
        header += f" {ax} |"
    lines.append(header)
    sep = "|--------|---------|------|"
    for ax in axis_names:
        sep += "------|"
    lines.append(sep)
    for exp in experiments:
        kimi = exp.get('kimi_avg', 0)
        row = f"| {exp['tag']} | {exp['avg_siglip2']:.3f} | {kimi:.1f} |"
        for ax in axis_names:
            vals = [r["multi_axes"].get(ax, 0) for r in exp["results"]]
            row += f" {np.mean(vals):.2f} |"
        lines.append(row)

    lines.append("\n---\n")
    lines.append("## Выводы\n")
    lines.append("*(заполняется по результатам)*\n")

    path = OUT / "report.md"
    path.write_text("\n".join(lines))
    print(f"Report: {path}")


def main():
    scorer = SiglipScorer()
    multi = MultiCriteriaSelector(scorer)
    test_set = load_test_set()
    kimi = KimiScorer()

    all_experiments = []

    sd15 = SD15Pipeline()
    sd15.load()

    for bo in [4, 8]:
        exp = run_experiment(sd15, scorer, multi, test_set,
                             best_of=bo, selector_type="single", model_tag="SD15")
        all_experiments.append(exp)

        exp = run_experiment(sd15, scorer, multi, test_set,
                             best_of=bo, selector_type="multi", model_tag="SD15")
        all_experiments.append(exp)

    sd15.unload()

    sdxl = SDXLPipeline()
    sdxl.load()

    for bo in [4, 8]:
        exp = run_experiment(sdxl, scorer, multi, test_set,
                             best_of=bo, selector_type="single", model_tag="SDXL")
        all_experiments.append(exp)

        exp = run_experiment(sdxl, scorer, multi, test_set,
                             best_of=bo, selector_type="multi", model_tag="SDXL")
        all_experiments.append(exp)

    sdxl.unload()

    kimi_score_all(kimi, all_experiments, max_workers=16)

    print_report(all_experiments)
    save_results(all_experiments)
    generate_md_report(all_experiments)

    print("\nDone.")


if __name__ == "__main__":
    main()
