import numpy as np
import requests
import json
from PIL import Image, ImageDraw, ImageFont
import torch
from diffusers import (StableDiffusionControlNetPipeline, StableDiffusionImg2ImgPipeline,
                       StableDiffusionPipeline, ControlNetModel, EulerDiscreteScheduler)
from transformers import (AutoProcessor, AutoModel, AutoImageProcessor,
                          AutoModelForImageClassification, SiglipImageProcessor,
                          SiglipProcessor, GemmaTokenizerFast)
from struct import unpack
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

_PROMPTS_PATH = Path(__file__).parent / "prompts.json"

def _load_prompts():
    if _PROMPTS_PATH.exists():
        with open(_PROMPTS_PATH) as f:
            return json.load(f)
    return {"prompts": {}, "negatives": {}, "default_negative": ""}

_PROMPT_DATA = _load_prompts()
PROMPTS = _PROMPT_DATA.get("prompts", {})
NEGATIVES = _PROMPT_DATA.get("negatives", {})
DEFAULT_NEG = _PROMPT_DATA.get("default_negative",
    "easynegative, (worst quality, low quality:1.4), "
    "people, human, text, watermark, ugly, blurry, deformed"
)

OPENROUTER_MODEL = "moonshotai/kimi-k2.5"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def get_prompt(label):
    return PROMPTS.get(label, f"a single {label}, solo, centered, (digital illustration:1.2), detailed, sharp focus, vivid colors")

def get_negative(label):
    return NEGATIVES.get(label, DEFAULT_NEG)

def score_prompt(label):
    return f"a high quality illustration of a {label}"

def remove_bg(img, bg=(255, 255, 255)):
    rgba = rembg_remove(img)
    out = Image.new('RGBA', rgba.size, (*bg, 255))
    out.paste(rgba, mask=rgba.split()[3])
    return out.convert('RGB')


class SiglipScorer:
    MODEL_ID = "google/siglip2-base-patch16-224"

    def __init__(self, device="cuda"):
        self.device = device
        self.model = AutoModel.from_pretrained(
            self.MODEL_ID, torch_dtype=torch.float16).to(device)
        self.processor = SiglipProcessor(
            image_processor=SiglipImageProcessor.from_pretrained(self.MODEL_ID),
            tokenizer=GemmaTokenizerFast.from_pretrained(self.MODEL_ID),
        )

    @torch.no_grad()
    def score(self, images, texts):
        inputs = self.processor(text=texts, images=images,
                                padding="max_length", return_tensors="pt").to(self.device)
        return np.diag(self.model(**inputs).logits_per_image.cpu().numpy())

    def score_single(self, image, text):
        return float(self.score([image], [text])[0])


class BeitSketchClassifier:
    """Classify raw sketch images using kmewhort/beit-sketch-classifier (345 QuickDraw classes)."""

    MODEL_ID = "kmewhort/beit-sketch-classifier"

    def __init__(self, device="cuda"):
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(self.MODEL_ID)
        self.model = AutoModelForImageClassification.from_pretrained(
            self.MODEL_ID, torch_dtype=torch.float16
        ).to(device).eval()

    @staticmethod
    def bin_to_pil(packed_drawing):
        """Convert QuickDraw packed binary bytes to 224x224 RGB PIL image."""
        padding = 8
        radius = 7
        scale = (224.0 - (2 * padding)) / 256

        fh = io.BytesIO(packed_drawing)
        _key_id, = unpack('Q', fh.read(8))
        _country, = unpack('2s', fh.read(2))
        _recognized, = unpack('b', fh.read(1))
        _timestamp, = unpack('I', fh.read(4))
        n_strokes, = unpack('H', fh.read(2))
        strokes = []
        for _ in range(n_strokes):
            n_points, = unpack('H', fh.read(2))
            fmt = str(n_points) + 'B'
            x = unpack(fmt, fh.read(n_points))
            y = unpack(fmt, fh.read(n_points))
            strokes.append((x, y))

        image = np.full((224, 224), 255, np.uint8)
        for stroke in strokes:
            prev_x = round(stroke[0][0] * scale)
            prev_y = round(stroke[1][0] * scale)
            for i in range(1, len(stroke[0])):
                x = round(stroke[0][i] * scale)
                y = round(stroke[1][i] * scale)
                cv2.line(image, (padding + prev_x, padding + prev_y),
                         (padding + x, padding + y), 0, radius, -1)
                prev_x, prev_y = x, y
        return Image.fromarray(image).convert("RGB")

    @torch.no_grad()
    def classify(self, image, top_k=5):
        """Classify a PIL RGB sketch image. Returns list of (label, score) tuples."""
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        logits = self.model(**{k: v.half() for k, v in inputs.items()}).logits
        probs = torch.softmax(logits, dim=-1)
        topk = torch.topk(probs, top_k)
        results = []
        for score, idx in zip(topk.values[0], topk.indices[0]):
            label = self.model.config.id2label[idx.item()]
            results.append((label, float(score)))
        return results

    def predict(self, image):
        """Return the single best predicted label for a sketch image."""
        results = self.classify(image, top_k=1)
        return results[0][0]


SD_MODELS = {
    "sd15": "runwayml/stable-diffusion-v1-5",
    "dreamshaper": "Lykon/dreamshaper-8",
    "deliberate": "XpucT/Deliberate",
    "revanimated": "stablediffusionapi/rev-animated",
}


class Pipeline:
    def __init__(self, device="cuda", use_beit=True):
        self.device = device
        self.pipe = None
        self.refiner = None
        self.txt2img = None
        self.scorer = SiglipScorer(device)
        self.classifier = BeitSketchClassifier(device) if use_beit else None
        self.model_name = None

    def load(self, model_key="sd15"):
        sd_model = SD_MODELS.get(model_key, model_key)
        self.model_name = model_key
        print(f"Loading {sd_model} + ControlNet...")
        cn = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_lineart", torch_dtype=torch.float16).to(self.device)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            sd_model, controlnet=cn,
            torch_dtype=torch.float16, safety_checker=None).to(self.device)
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_attention_slicing()
        try:
            self.pipe.load_textual_inversion(
                "EvilEngine/easynegative", weight_name="easynegative.safetensors", token="easynegative")
        except Exception as e:
            print(f"  warn: easynegative failed: {e}")
        comps = {k: v for k, v in self.pipe.components.items() if k != "controlnet"}
        self.refiner = StableDiffusionImg2ImgPipeline(**comps)
        self.txt2img = StableDiffusionPipeline(**comps)
        print("Ready")

    def generate(self, ctrl, label, seed=42, steps=30, cn_scale=0.7, guidance=7.5, batch=1):
        return self.pipe(
            prompt=get_prompt(label), negative_prompt=get_negative(label),
            image=ctrl, num_inference_steps=steps,
            guidance_scale=guidance, controlnet_conditioning_scale=cn_scale,
            control_guidance_start=0.0, control_guidance_end=min(1.0, 12/steps),
            generator=torch.Generator(device=self.device).manual_seed(seed),
            num_images_per_prompt=batch,
        ).images

    def refine(self, img, label, seed=42, strength=0.25):
        return self.refiner(
            prompt=get_prompt(label), negative_prompt=get_negative(label),
            image=img, strength=strength, num_inference_steps=15, guidance_scale=4.0,
            generator=torch.Generator(device=self.device).manual_seed(seed),
        ).images[0]

    def generate_free(self, label, seed=42, steps=30, guidance=7.5, batch=1):
        return self.txt2img(
            prompt=get_prompt(label), negative_prompt=get_negative(label),
            num_inference_steps=steps, guidance_scale=guidance,
            generator=torch.Generator(device=self.device).manual_seed(seed),
            num_images_per_prompt=batch,
        ).images

    def gen_candidates(self, ctrl, label, n=4, do_refine=True):
        seed = random.randint(0, 999999)
        imgs = self.generate(ctrl, label, seed=seed, batch=n)
        if do_refine:
            imgs = [self.refine(im, label, seed=seed+i) for i, im in enumerate(imgs)]
        return [remove_bg(im) for im in imgs]

    def classify_sketch(self, sketch_img, top_k=5):
        """Classify a raw sketch image using BEiT. Returns list of (label, score)."""
        if self.classifier is None:
            raise RuntimeError("BEiT classifier not loaded (use_beit=False)")
        return self.classifier.classify(sketch_img, top_k=top_k)

    def classify_and_generate(self, sketch_img, n=4, do_refine=True, top_k=5):
        """Classify a raw sketch, then generate images for the predicted class.
        Returns (predicted_label, top_k_predictions, candidate_images).
        """
        predictions = self.classify_sketch(sketch_img, top_k=top_k)
        label = predictions[0][0]
        ctrl = to_lineart(sketch_img)
        candidates = self.gen_candidates(ctrl, label, n=n, do_refine=do_refine)
        return label, predictions, candidates


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


DOMAINNET_ROOT = Path(__file__).parent.parent / "data" / "domainnet" / "sketch"


class DomainNetMatcher:
    """Match QuickDraw sketches to closest DomainNet sketches via SigLIP2 embeddings."""

    def __init__(self, scorer, classes, batch_size=16):
        self.scorer = scorer
        self.db = {}  # class -> (vectors, paths)
        self._build_index(classes, batch_size)

    @torch.no_grad()
    def _embed_batch(self, images):
        inputs = self.scorer.processor(images=images, return_tensors="pt").to(self.scorer.device)
        out = self.scorer.model.vision_model(**{k: v for k, v in inputs.items() if k.startswith("pixel")})
        feats = out.pooler_output
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu()

    def _build_index(self, classes, batch_size):
        if not DOMAINNET_ROOT.exists():
            raise FileNotFoundError(
                f"DomainNet sketch directory not found: {DOMAINNET_ROOT}\n"
                f"Download DomainNet sketch split and place it at: {DOMAINNET_ROOT}"
            )
        print("Building DomainNet SigLIP2 index...")
        missing = []
        for cls in classes:
            # DomainNet uses underscores, QuickDraw uses spaces
            cls_dir = DOMAINNET_ROOT / cls.replace(" ", "_")
            if not cls_dir.exists():
                missing.append(cls)
                continue
            paths = sorted([p for p in cls_dir.iterdir() if p.suffix in ('.jpg', '.png', '.jpeg')])
            if not paths:
                raise FileNotFoundError(f"DomainNet class '{cls}' directory exists but contains no images: {cls_dir}")
            vecs = []
            for i in range(0, len(paths), batch_size):
                batch_paths = paths[i:i+batch_size]
                imgs = [Image.open(p).convert("RGB") for p in batch_paths]
                vecs.append(self._embed_batch(imgs))
            self.db[cls] = (torch.cat(vecs, dim=0), paths)
            print(f"  {cls}: {len(paths)} images indexed")
        if missing:
            raise FileNotFoundError(
                f"DomainNet class directories not found for: {missing}\n"
                f"Expected at: {DOMAINNET_ROOT}/<class_name>/"
            )

    @torch.no_grad()
    def match(self, query_img, cls):
        """Return (best_domainnet_image, best_path, similarity_score)."""
        if cls not in self.db:
            raise KeyError(f"Class '{cls}' not in DomainNet index. Available: {list(self.db.keys())}")
        db_vecs, db_paths = self.db[cls]
        q_vec = self._embed_batch([query_img])  # (1, dim)
        sims = (q_vec @ db_vecs.T).squeeze(0)   # (N,)
        best_idx = int(sims.argmax())
        best_path = db_paths[best_idx]
        best_img = Image.open(best_path).convert("RGB")
        return best_img, best_path, float(sims[best_idx])

    @torch.no_grad()
    def match_topk(self, query_img, cls, k=4):
        """Return list of (image, path, similarity) for top-k matches."""
        if cls not in self.db:
            raise KeyError(f"Class '{cls}' not in DomainNet index. Available: {list(self.db.keys())}")
        db_vecs, db_paths = self.db[cls]
        q_vec = self._embed_batch([query_img])
        sims = (q_vec @ db_vecs.T).squeeze(0)
        k = min(k, len(db_paths))
        top_indices = sims.argsort(descending=True)[:k]
        results = []
        for idx in top_indices:
            idx = int(idx)
            img = Image.open(db_paths[idx]).convert("RGB")
            results.append((img, db_paths[idx], float(sims[idx])))
        return results


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


def log_experiment(path, desc, metrics):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(path, 'a') as f:
        m = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        f.write(f"| {ts} | {desc} | {m} |\n")


def run(classes=None, spc=1, model_key="sd15", use_beit=True):
    if classes is None:
        classes = CLASSES[:3]

    out = Path(f"out3/v8_{model_key}")
    out.mkdir(parents=True, exist_ok=True)
    md = Path("experiments.md")

    pipe = Pipeline(use_beit=use_beit)
    pipe.load(model_key)

    random.seed(2026)
    print("\nFetching sketches...")
    all_data = []
    for cls in classes:
        ds = fetch_quickdraw(cls, spc, random.randint(0, 500))
        for d in ds:
            all_data.append((cls, d))
        print(f"  {cls}: {len(ds)}")

    N = len(all_data)

    # === EXP 6: DomainNet-matched sketch + refiner + best-of-4 (SigLIP2) ===
    print(f"\n{'='*50}\nEXP 6: DomainNet-matched sketch + refiner + best-of-4\n{'='*50}")
    matcher = DomainNetMatcher(pipe.scorer, classes, batch_size=16)
    exp6_scores = []
    beit_correct = 0
    for i, (lbl, drawing) in enumerate(all_data):
        print(f"  {i+1}/{N} ({lbl})")
        qd_sketch = drawing_to_img(drawing)

        # BEiT classification on raw sketch
        if pipe.classifier is not None:
            preds = pipe.classify_sketch(qd_sketch, top_k=5)
            beit_label = preds[0][0]
            beit_conf = preds[0][1]
            if beit_label == lbl:
                beit_correct += 1
            print(f"    beit: {beit_label} ({beit_conf:.3f}) | top5: {[(l, f'{s:.3f}') for l, s in preds]}")

        dn_sketch, dn_path, sim = matcher.match(qd_sketch, lbl)
        print(f"    matched: {dn_path.name if dn_path else 'none'} (sim={sim:.3f})")
        qd_sketch.save(out / f"exp6_{lbl}_{i}_qd.png")
        dn_sketch.save(out / f"exp6_{lbl}_{i}_dn.png")

        ctrl = to_lineart(dn_sketch)
        cands = pipe.gen_candidates(ctrl, lbl, n=4, do_refine=True)
        cand_scores = [pipe.scorer.score_single(c, score_prompt(lbl)) for c in cands]
        print(f"    scores: {[f'{s:.3f}' for s in cand_scores]}")

        best_idx = int(np.argmax(cand_scores))
        print(f"    siglip2 picked #{best_idx+1} (score={cand_scores[best_idx]:.3f})")
        exp6_scores.append(cand_scores[best_idx])

        for j, (c, cs) in enumerate(zip(cands, cand_scores)):
            tag = "_BEST" if j == best_idx else ""
            stamp(c, cs, f"c{j+1}").save(out / f"exp6_{lbl}_{i}_c{j+1}{tag}.png")

    avg6 = np.mean(exp6_scores)
    print(f"  avg={avg6:.4f} min={min(exp6_scores):.4f} max={max(exp6_scores):.4f}")
    if pipe.classifier is not None:
        print(f"  beit accuracy: {beit_correct}/{N} ({beit_correct/N:.1%})")
    log_experiment(md, f"v8 {model_key} DomainNet-matched + BestOf4(SigLIP2)",
                  {"avg_siglip2": avg6, "min": min(exp6_scores), "max": max(exp6_scores)})

    print(f"\n{'='*50}\nSUMMARY ({model_key})\n{'='*50}")
    print(f"  DomainNet+B4S: {avg6:.4f}")
    if pipe.classifier is not None:
        print(f"  BEiT accuracy: {beit_correct}/{N} ({beit_correct/N:.1%})")
    print(f"\nResults: {out}/")


if __name__ == "__main__":
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else "sd15"
    run(model_key=model)
