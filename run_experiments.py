#!/usr/bin/env python3
import numpy as np
import requests, json, random, gc, io, os, base64, time, shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageDraw, ImageFont
import torch
from diffusers import (
    StableDiffusionControlNetPipeline, StableDiffusionImg2ImgPipeline,
    StableDiffusionXLControlNetPipeline, StableDiffusionXLImg2ImgPipeline,
    ControlNetModel, AutoencoderKL, EulerDiscreteScheduler
)
from transformers import (
    AutoModel, AutoImageProcessor, AutoModelForImageClassification,
    SiglipProcessor, SiglipImageProcessor, GemmaTokenizerFast
)
from rembg import remove as rembg_remove
import cv2
from pathlib import Path
from datetime import datetime

def load_env():
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip())

load_env()

TEST_CLASSES = ["apple", "cat", "house", "bird", "bicycle"]
SAMPLES_PER_CLASS = {"apple": 4, "cat": 3, "house": 3, "bird": 3, "bicycle": 3}
QD_OFFSETS = {"apple": 100, "cat": 200, "house": 300, "bird": 400, "bicycle": 500}

DOMAINNET_ROOT = Path("data/domainnet/sketch")
FILTERED_ROOT = Path("data/domainnet_filtered/sketch")
OUT = Path("experiments_report")
OUT.mkdir(parents=True, exist_ok=True)

PROMPTS_PATH = Path("quick_draw/prompts.json")
with open(PROMPTS_PATH) as f:
    _pdata = json.load(f)
PROMPTS = _pdata["prompts"]
NEGATIVES = _pdata["negatives"]
DEFAULT_NEG = _pdata.get("default_negative",
    "easynegative, (worst quality, low quality:1.4), people, human, text, watermark, ugly, blurry, deformed")

SD_MODELS = {
    "sd15": "runwayml/stable-diffusion-v1-5",
    "dreamshaper": "Lykon/dreamshaper-8",
}

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "moonshotai/kimi-k2.5"
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "")


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


def remove_bg(img, bg=(255, 255, 255)):
    rgba = rembg_remove(img)
    out = Image.new('RGBA', rgba.size, (*bg, 255))
    out.paste(rgba, mask=rgba.split()[3])
    return out.convert('RGB')


def stamp(image, text):
    img = image.copy()
    d = ImageDraw.Draw(img)
    font = get_font(16)
    bb = d.textbbox((0, 0), text, font=font)
    d.rectangle([3, 3, bb[2]-bb[0]+13, bb[3]-bb[1]+13], fill='black')
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

    @torch.no_grad()
    def embed_images(self, images, batch_size=16):
        all_vecs = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            out = self.model.vision_model(**{k: v for k, v in inputs.items() if k.startswith("pixel")})
            feats = out.pooler_output
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_vecs.append(feats.cpu())
        return torch.cat(all_vecs, dim=0)


class DomainNetMatcher:
    def __init__(self, scorer, classes, root=None, batch_size=16):
        self.scorer = scorer
        self.root = root or DOMAINNET_ROOT
        self.db = {}
        for cls in classes:
            cls_dir = self.root / cls.replace(" ", "_")
            if not cls_dir.exists():
                print(f"  warn: no domainnet dir for {cls}")
                continue
            paths = sorted([p for p in cls_dir.iterdir() if p.suffix in ('.jpg', '.png', '.jpeg')])
            if not paths:
                continue
            imgs = [Image.open(p).convert("RGB") for p in paths]
            vecs = scorer.embed_images(imgs, batch_size)
            self.db[cls] = (vecs, paths)
            for im in imgs:
                im.close()
            print(f"  indexed {cls}: {len(paths)} imgs")

    @torch.no_grad()
    def match(self, query_img, cls):
        if cls not in self.db:
            return None, None, -999
        db_vecs, db_paths = self.db[cls]
        q_vec = self.scorer.embed_images([query_img])
        sims = (q_vec @ db_vecs.T).squeeze(0)
        best_idx = int(sims.argmax())
        best_img = Image.open(db_paths[best_idx]).convert("RGB")
        return best_img, db_paths[best_idx], float(sims[best_idx])


class KimiJudge:
    def __init__(self):
        self.api_key = OPENROUTER_KEY

    def _b64(self, img):
        buf = io.BytesIO()
        img.resize((512, 512)).save(buf, format="PNG")
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
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self._b64(img)}"}})

        for attempt in range(retries):
            try:
                r = requests.post(OPENROUTER_URL,
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    json={"model": OPENROUTER_MODEL,
                          "messages": [{"role": "user", "content": content}],
                          "response_format": {"type": "json_object"},
                          "max_tokens": 1024, "temperature": 0.3},
                    timeout=120)
                r.raise_for_status()
                raw = r.json()["choices"][0]["message"].get("content", "")
                if not raw.strip():
                    continue
                parsed = json.loads(raw)
                idx = int(parsed["best_image"]) - 1
                if 0 <= idx < len(images):
                    return idx, parsed.get("reasoning", "")
            except Exception as e:
                print(f"    kimi pick err: {e}")
                time.sleep(2)
        return 0, "fallback"

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
                        vals = [parsed.get(k, 5) for k in ["prompt_match", "artifacts", "concept_clarity", "bg_separability", "cleanliness"]]
                        parsed["avg"] = round(np.mean(vals), 1)
                    return parsed
            except Exception as e:
                print(f"    kimi score err: {e}")
                time.sleep(2)
        return {"prompt_match": 0, "artifacts": 0, "concept_clarity": 0, "bg_separability": 0, "cleanliness": 0, "avg": 0, "note": "api_fail"}


class SD15Pipeline:
    def __init__(self, scorer, device="cuda"):
        self.device = device
        self.scorer = scorer
        self.pipe = None
        self.refiner = None
        self.model_name = None

    def load(self, model_key="dreamshaper"):
        sd_model = SD_MODELS.get(model_key, model_key)
        self.model_name = model_key
        print(f"Loading SD1.5 ({model_key})...")
        cn = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_lineart", torch_dtype=torch.float16).to(self.device)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            sd_model, controlnet=cn, torch_dtype=torch.float16, safety_checker=None).to(self.device)
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_attention_slicing()
        try:
            self.pipe.load_textual_inversion("EvilEngine/easynegative", weight_name="easynegative.safetensors", token="easynegative")
        except Exception as e:
            print(f"  easyneg warn: {e}")
        comps = {k: v for k, v in self.pipe.components.items() if k != "controlnet"}
        self.refiner = StableDiffusionImg2ImgPipeline(**comps)
        print("SD1.5 ready")

    def generate(self, ctrl, label, seed=42, steps=30, cn_scale=0.7, guidance=7.5, batch=1):
        return self.pipe(
            prompt=get_prompt(label), negative_prompt=get_negative(label),
            image=ctrl, num_inference_steps=steps,
            guidance_scale=guidance, controlnet_conditioning_scale=cn_scale,
            control_guidance_start=0.0, control_guidance_end=min(1.0, 12/steps),
            generator=torch.Generator(device=self.device).manual_seed(seed),
            num_images_per_prompt=batch,
        ).images

    def refine_img(self, img, label, seed=42, strength=0.25):
        return self.refiner(
            prompt=get_prompt(label), negative_prompt=get_negative(label),
            image=img, strength=strength, num_inference_steps=15, guidance_scale=4.0,
            generator=torch.Generator(device=self.device).manual_seed(seed),
        ).images[0]

    def unload(self):
        del self.pipe, self.refiner
        self.pipe = self.refiner = None
        gc.collect()
        torch.cuda.empty_cache()


class SDXLPipeline:
    def __init__(self, scorer, device="cuda"):
        self.device = device
        self.scorer = scorer
        self.pipe = None
        self.refiner = None

    def load(self):
        print("Loading SDXL...")
        cn = ControlNetModel.from_pretrained("TheMistoAI/MistoLine", torch_dtype=torch.float16, variant="fp16")
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", controlnet=cn, vae=vae,
            torch_dtype=torch.float16, variant="fp16", safety_checker=None)
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_model_cpu_offload()
        comps = {k: v for k, v in self.pipe.components.items() if k != "controlnet"}
        self.refiner = StableDiffusionXLImg2ImgPipeline(**comps)
        print("SDXL ready")

    def generate(self, ctrl, label, seed=42, steps=30, cn_scale=0.5, guidance=7.0, batch=1):
        ctrl_xl = ctrl.resize((1024, 1024), Image.LANCZOS) if ctrl.size[0] < 1024 else ctrl
        return self.pipe(
            prompt=get_prompt(label), negative_prompt=get_negative(label),
            image=ctrl_xl, num_inference_steps=steps,
            guidance_scale=guidance, controlnet_conditioning_scale=cn_scale,
            control_guidance_start=0.0, control_guidance_end=min(1.0, 12/steps),
            generator=torch.Generator(device="cpu").manual_seed(seed),
            num_images_per_prompt=batch,
        ).images

    def refine_img(self, img, label, seed=42, strength=0.25):
        return self.refiner(
            prompt=get_prompt(label), negative_prompt=get_negative(label),
            image=img, strength=strength, num_inference_steps=15, guidance_scale=4.0,
            generator=torch.Generator(device="cpu").manual_seed(seed),
        ).images[0]

    def unload(self):
        del self.pipe, self.refiner
        self.pipe = self.refiner = None
        gc.collect()
        torch.cuda.empty_cache()


def filter_domainnet(scorer, classes):
    if FILTERED_ROOT.exists() and any((FILTERED_ROOT / c).exists() for c in classes):
        print("Filtered domainnet already exists, skipping")
        return
    print("Filtering DomainNet (keeping top 50% by SigLIP2)...")
    FILTERED_ROOT.mkdir(parents=True, exist_ok=True)
    for cls in classes:
        cls_dir = DOMAINNET_ROOT / cls.replace(" ", "_")
        out_dir = FILTERED_ROOT / cls.replace(" ", "_")
        out_dir.mkdir(parents=True, exist_ok=True)
        if not cls_dir.exists():
            continue
        paths = sorted([p for p in cls_dir.iterdir() if p.suffix in ('.jpg', '.png', '.jpeg')])
        prompt = f"a sketch drawing of a {cls}"
        scores = []
        for p in paths:
            try:
                im = Image.open(p).convert("RGB")
                s = scorer.score_single(im, prompt)
                scores.append(s)
                im.close()
            except:
                scores.append(float('-inf'))
        scores = np.array(scores)
        median = float(np.median(scores[scores > float('-inf')]))
        kept = 0
        for i, p in enumerate(paths):
            if scores[i] >= median:
                shutil.copy2(p, out_dir / p.name)
                kept += 1
        print(f"  {cls}: {kept}/{len(paths)} kept (median={median:.2f})")


def fetch_fixed_test_set():
    cache_path = OUT / "test_set.json"
    if cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        test_set = []
        for item in cached:
            img = drawing_to_img(item["drawing"])
            test_set.append({"class": item["class"], "drawing": item["drawing"], "image": img, "idx": item["idx"]})
        print(f"Loaded cached test set: {len(test_set)} sketches")
        return test_set

    print("Fetching fixed test set from QuickDraw...")
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
        print(f"  {cls}: {len(drawings)} sketches (offset={offset})")

    save_data = [{"class": t["class"], "drawing": t["drawing"], "idx": t["idx"]} for t in test_set]
    with open(cache_path, 'w') as f:
        json.dump(save_data, f)

    grid_imgs = [t["image"] for t in test_set]
    grid_labels = [f"{t['class']}#{t['idx']}" for t in test_set]
    grid = make_grid(grid_imgs, grid_labels, cols=4, title="Fixed Test Set (16 QuickDraw sketches)")
    grid.save(OUT / "00_test_set.png")
    print(f"Test set: {len(test_set)} sketches saved")
    return test_set


def run_experiment(pipe, test_set, exp_name, use_domainnet=None, do_refine=True,
                   best_of=1, selector="siglip2", do_bg_remove=False, scorer=None, kimi=None):
    print(f"\n{'='*60}")
    print(f"EXP: {exp_name}")
    print(f"{'='*60}")

    results = []
    all_imgs = []
    all_labels = []

    for item in test_set:
        cls = item["class"]
        sketch = item["image"]
        idx = item["idx"]

        if use_domainnet is not None:
            dn_img, dn_path, sim = use_domainnet.match(sketch, cls)
            if dn_img is not None:
                ctrl = to_lineart(dn_img)
            else:
                ctrl = to_lineart(sketch)
        else:
            ctrl = to_lineart(sketch)

        seed_base = 42 + idx * 100

        if best_of > 1:
            imgs = pipe.generate(ctrl, cls, seed=seed_base, batch=best_of)
            if do_refine:
                imgs = [pipe.refine_img(im, cls, seed=seed_base+j) for j, im in enumerate(imgs)]
            if do_bg_remove:
                imgs = [remove_bg(im) for im in imgs]

            if selector == "kimi" and kimi is not None:
                pick_idx, reason = kimi.pick_best(imgs, cls)
            else:
                cand_scores = [scorer.score_single(im, score_prompt(cls)) for im in imgs]
                pick_idx = int(np.argmax(cand_scores))
            final = imgs[pick_idx]
        else:
            final = pipe.generate(ctrl, cls, seed=seed_base)[0]
            if do_refine:
                final = pipe.refine_img(final, cls, seed=seed_base)
            if do_bg_remove:
                final = remove_bg(final)

        siglip_score = scorer.score_single(final, score_prompt(cls))
        results.append({"class": cls, "idx": idx, "siglip2": siglip_score, "image": final})
        all_imgs.append(final)
        all_labels.append(f"{cls}: {siglip_score:.2f}")
        print(f"  [{idx}] {cls} -> siglip2={siglip_score:.3f}")

    avg_s = np.mean([r["siglip2"] for r in results])
    min_s = min(r["siglip2"] for r in results)
    max_s = max(r["siglip2"] for r in results)
    print(f"  => avg={avg_s:.3f} min={min_s:.3f} max={max_s:.3f}")

    grid = make_grid(all_imgs, all_labels, cols=4, title=f"{exp_name} | avg={avg_s:.2f}")
    safe_name = exp_name.replace(" ", "_").replace("+", "_").replace("(", "").replace(")", "").replace("/", "_")
    grid.save(OUT / f"{safe_name}.png")

    return {"name": exp_name, "results": results, "avg": avg_s, "min": min_s, "max": max_s, "grid_path": f"{safe_name}.png"}


def score_with_kimi(kimi, experiment_data, max_workers=16):
    print(f"Scoring '{experiment_data['name']}' with Kimi K2.5 ({max_workers} threads)...")

    def _score_one(r):
        ks = kimi.expert_score(r["image"], r["class"])
        return r["idx"], r["class"], ks

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_score_one, r): r for r in experiment_data["results"]}
        for fut in as_completed(futures):
            idx, cls, ks = fut.result()
            r = futures[fut]
            r["kimi"] = ks
            print(f"  [{idx}] {cls}: kimi_avg={ks.get('avg', 0)}")

    kimi_scores = [r["kimi"].get("avg", 0) for r in experiment_data["results"]]
    experiment_data["kimi_avg"] = float(np.mean(kimi_scores))
    return experiment_data


def model_selection(scorer, test_set):
    print("\n" + "="*60)
    print("PHASE 0: MODEL SELECTION")
    print("="*60)

    subset = test_set[:6]

    results = {}
    for model_key in ["sd15", "dreamshaper"]:
        pipe = SD15Pipeline(scorer)
        pipe.load(model_key)
        scores = []
        for item in subset:
            ctrl = to_lineart(item["image"])
            img = pipe.generate(ctrl, item["class"], seed=42+item["idx"])[0]
            img = pipe.refine_img(img, item["class"], seed=42+item["idx"])
            s = scorer.score_single(img, score_prompt(item["class"]))
            scores.append(s)
        avg = np.mean(scores)
        results[model_key] = avg
        print(f"  {model_key}: avg_siglip2={avg:.3f}")
        pipe.unload()

    best = max(results, key=results.get)
    print(f"  => best SD1.5 model: {best} (avg={results[best]:.3f})")
    return best


def generate_report(all_experiments, test_set):
    print("\nGenerating report...")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = []
    lines.append(f"# Sketch-to-Image: эволюция подходов")
    lines.append(f"\n*Сгенерировано {ts}*\n")
    lines.append(f"Фиксированный тестовый набор: 16 скетчей из QuickDraw, 5 классов разной сложности:")
    lines.append(f"- **apple** (4 шт) — простая геометрия")
    lines.append(f"- **cat** (3 шт) — органика, средняя сложность")
    lines.append(f"- **house** (3 шт) — геометрия + детали")
    lines.append(f"- **bird** (3 шт) — органика, сложнее")
    lines.append(f"- **bicycle** (3 шт) — сложная структура, много частей\n")
    lines.append(f"![Тестовый набор](experiments_report/00_test_set.png)\n")

    lines.append(f"## Метрики\n")
    lines.append(f"- **SigLIP2**: косинусная близость к промпту `a high quality illustration of a {{label}}`. Выше = лучше.")
    lines.append(f"- **Kimi K2.5**: экспертная оценка 1-10 по 5 критериям (соответствие промпту, артефакты, понятность, отделяемость фона, чистота). Выше = лучше.\n")

    lines.append(f"---\n")

    exp_groups = {}
    for exp in all_experiments:
        if "SD1.5" in exp["name"]:
            model = "SD1.5"
        elif "SDXL" in exp["name"]:
            model = "SDXL"
        else:
            model = "other"
        stage = exp["name"].split("] ")[-1] if "] " in exp["name"] else exp["name"]
        if stage not in exp_groups:
            exp_groups[stage] = {}
        exp_groups[stage][model] = exp

    stage_order = [
        "Baseline (raw sketch)",
        "DomainNet sketch refinement",
        "Filtered DomainNet",
        "Best-of-4 (SigLIP2)",
        "Best-of-4 (Kimi K2.5)",
        "Best-of-4 + BG removal",
    ]

    stage_descriptions = {
        "Baseline (raw sketch)": (
            "## Stage 1: Baseline — сырой скетч\n\n"
            "Самый простой подход: берем скетч из QuickDraw как есть, конвертируем в lineart, "
            "прогоняем через ControlNet + диффузию + refiner. Одна генерация, никакой фильтрации."
        ),
        "DomainNet sketch refinement": (
            "## Stage 2: Улучшение скетча через DomainNet\n\n"
            "QuickDraw скетчи очень грубые (3-10 штрихов). Идея: находим ближайший скетч из DomainNet "
            "(через SigLIP2 эмбеддинги) и используем его вместо сырого. DomainNet скетчи показывают "
            "анатомию и структуру объекта значительно лучше."
        ),
        "Filtered DomainNet": (
            "## Stage 3: Фильтрация DomainNet\n\n"
            "DomainNet содержит мусор: фотографии, тёмные фоны, цветные иллюстрации, коллажи. "
            "Фильтруем через SigLIP2 (промпт `a sketch drawing of a {class}`), оставляем верхние 50% "
            "по скору. Результат: чистая база скетчей."
        ),
        "Best-of-4 (SigLIP2)": (
            "## Stage 4a: Best-of-4 с отбором через SigLIP2\n\n"
            "Генерируем 4 кандидата вместо одного. Скорим каждого через SigLIP2 "
            "(`a high quality illustration of a {label}`), выбираем лучшего. "
            "Простая и быстрая стратегия — всё локально, без API."
        ),
        "Best-of-4 (Kimi K2.5)": (
            "## Stage 4b: Best-of-4 с отбором через Kimi K2.5\n\n"
            "Те же 4 кандидата, но выбор делает VLM — Kimi K2.5 через OpenRouter API. "
            "Промпт просит выбрать лучшего по визуальному качеству, когерентности и соответствию классу."
        ),
        "Best-of-4 + BG removal": (
            "## Stage 5: Background removal\n\n"
            "Финальный штрих: убираем фон через rembg (U2-Net). Объект на чистом белом фоне. "
            "Это последний этап пайплайна — полный путь от каракули до финального изображения."
        ),
    }

    for stage_name in stage_order:
        if stage_name not in exp_groups:
            continue
        desc = stage_descriptions.get(stage_name, f"## {stage_name}\n")
        lines.append(desc + "\n")

        models_data = exp_groups[stage_name]
        for model_name in ["SD1.5", "SDXL"]:
            if model_name not in models_data:
                continue
            exp = models_data[model_name]
            lines.append(f"### {model_name}\n")
            lines.append(f"![{stage_name} {model_name}](experiments_report/{exp['grid_path']})\n")
            lines.append(f"| Метрика | Значение |")
            lines.append(f"|---------|----------|")
            lines.append(f"| SigLIP2 avg | **{exp['avg']:.3f}** |")
            lines.append(f"| SigLIP2 min | {exp['min']:.3f} |")
            lines.append(f"| SigLIP2 max | {exp['max']:.3f} |")
            if "kimi_avg" in exp:
                lines.append(f"| Kimi K2.5 avg | **{exp['kimi_avg']:.1f}**/10 |")
            lines.append("")

        lines.append("---\n")

    lines.append("## Сводная таблица\n")
    lines.append("| Этап | SD1.5 SigLIP2 | SD1.5 Kimi | SDXL SigLIP2 | SDXL Kimi |")
    lines.append("|------|--------------|------------|-------------|-----------|")
    for stage_name in stage_order:
        if stage_name not in exp_groups:
            continue
        row = f"| {stage_name} "
        for model_name in ["SD1.5", "SDXL"]:
            exp = exp_groups.get(stage_name, {}).get(model_name)
            if exp:
                row += f"| {exp['avg']:.3f} "
                kimi_val = f"{exp['kimi_avg']:.1f}" if "kimi_avg" in exp else "—"
                row += f"| {kimi_val} "
            else:
                row += "| — | — "
        row += "|"
        lines.append(row)

    lines.append("\n## Выводы\n")

    best_exp = max(all_experiments, key=lambda e: e["avg"])
    worst_exp = min(all_experiments, key=lambda e: e["avg"])
    lines.append(f"- **Лучший результат**: {best_exp['name']} — avg SigLIP2 = **{best_exp['avg']:.3f}**")
    lines.append(f"- **Худший результат**: {worst_exp['name']} — avg SigLIP2 = **{worst_exp['avg']:.3f}**")
    lines.append(f"- **Дельта**: {best_exp['avg'] - worst_exp['avg']:.3f} пунктов SigLIP2\n")

    if len(all_experiments) >= 4:
        baselines = [e for e in all_experiments if "Baseline" in e["name"]]
        finals = [e for e in all_experiments if "BG removal" in e["name"]]
        if baselines and finals:
            b_avg = np.mean([e["avg"] for e in baselines])
            f_avg = np.mean([e["avg"] for e in finals])
            lines.append(f"- Полный пайплайн даёт **+{f_avg - b_avg:.3f}** к SigLIP2 скору по сравнению с baseline")

    report_text = "\n".join(lines)
    report_path = Path("experiments_report.md")
    report_path.write_text(report_text)
    print(f"Report saved to {report_path}")
    return report_path


def main():
    scorer = SiglipScorer()

    test_set = fetch_fixed_test_set()

    filter_domainnet(scorer, TEST_CLASSES)

    # best_sd15 = model_selection(scorer, test_set)
    # SD1.5 already done — dreamshaper won

    kimi = KimiJudge()
    all_experiments = []

    dn_raw = DomainNetMatcher(scorer, TEST_CLASSES, root=DOMAINNET_ROOT)
    dn_filtered = DomainNetMatcher(scorer, TEST_CLASSES, root=FILTERED_ROOT)

    # --- SD1.5 DONE, skipping ---
    # sd15 = SD15Pipeline(scorer)
    # sd15.load("dreamshaper")
    # ... all 6 SD1.5 experiments already ran ...
    # sd15.unload()

    print("\n\n" + "#"*60)
    print("RUNNING SDXL EXPERIMENTS")
    print("#"*60)

    sdxl = SDXLPipeline(scorer)
    sdxl.load()

    exp = run_experiment(sdxl, test_set, f"01_SDXL] Baseline (raw sketch)",
                         use_domainnet=None, do_refine=True, best_of=1, scorer=scorer)
    all_experiments.append(exp)

    exp = run_experiment(sdxl, test_set, f"02_SDXL] DomainNet sketch refinement",
                         use_domainnet=dn_raw, do_refine=True, best_of=1, scorer=scorer)
    all_experiments.append(exp)

    exp = run_experiment(sdxl, test_set, f"03_SDXL] Filtered DomainNet",
                         use_domainnet=dn_filtered, do_refine=True, best_of=1, scorer=scorer)
    all_experiments.append(exp)

    exp = run_experiment(sdxl, test_set, f"04a_SDXL] Best-of-4 (SigLIP2)",
                         use_domainnet=dn_filtered, do_refine=True, best_of=4, selector="siglip2", scorer=scorer)
    all_experiments.append(exp)

    exp = run_experiment(sdxl, test_set, f"04b_SDXL] Best-of-4 (Kimi K2.5)",
                         use_domainnet=dn_filtered, do_refine=True, best_of=4, selector="kimi", scorer=scorer, kimi=kimi)
    all_experiments.append(exp)

    exp = run_experiment(sdxl, test_set, f"05_SDXL] Best-of-4 + BG removal",
                         use_domainnet=dn_filtered, do_refine=True, best_of=4, selector="siglip2",
                         do_bg_remove=True, scorer=scorer)
    all_experiments.append(exp)

    sdxl.unload()

    print("\n\n" + "#"*60)
    print("KIMI EXPERT SCORING (SDXL only for now)")
    print("#"*60)

    for exp in all_experiments:
        score_with_kimi(kimi, exp)

    generate_report(all_experiments, test_set)

    summary_path = OUT / "summary.json"
    summary = []
    for exp in all_experiments:
        entry = {
            "name": exp["name"], "avg_siglip2": exp["avg"],
            "min_siglip2": exp["min"], "max_siglip2": exp["max"],
            "kimi_avg": exp.get("kimi_avg", None),
            "per_class": {}
        }
        for r in exp["results"]:
            cls = r["class"]
            if cls not in entry["per_class"]:
                entry["per_class"][cls] = {"siglip2": [], "kimi": []}
            entry["per_class"][cls]["siglip2"].append(r["siglip2"])
            if "kimi" in r:
                entry["per_class"][cls]["kimi"].append(r["kimi"].get("avg", 0))
        summary.append(entry)

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "="*60)
    print("ALL DONE")
    print("="*60)
    print(f"Report: experiments_report.md")
    print(f"Images: {OUT}/")
    print(f"Data: {summary_path}")


if __name__ == "__main__":
    main()
