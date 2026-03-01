"""
BEiT Pipeline Comparison: SD1.5 vs SDXL-base vs WAI-Illustrious-SDXL
Runs the same 10 fixed sketches through each model, scores with SigLIP2,
outputs per-model canvases + comparison metrics table.

Columns per row:
  QuickDraw | BEiT | DomainNet | Raw (no BG) | rembg | GrabCut | Threshold
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import random
import json
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from diffusers import (
    StableDiffusionControlNetPipeline, StableDiffusionImg2ImgPipeline,
    StableDiffusionXLControlNetPipeline, StableDiffusionXLImg2ImgPipeline,
    ControlNetModel, AutoencoderKL, EulerDiscreteScheduler,
)
from rembg import remove as rembg_remove
import cv2
import requests
import base64
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from v8_tailored import (
    BeitSketchClassifier, SiglipScorer, DomainNetMatcher,
    fetch_quickdraw, _PROMPT_DATA,
)

ALL_CLASSES = _PROMPT_DATA.get("classes", [])
PROMPTS = _PROMPT_DATA.get("prompts", {})
NEGATIVES = _PROMPT_DATA.get("negatives", {})
DEFAULT_NEG_SD15 = _PROMPT_DATA.get("default_negative",
    "easynegative, (worst quality, low quality:1.4), people, human, text, watermark, ugly, blurry, deformed")
DEFAULT_NEG_SDXL = (
    "(worst quality, low quality:1.4), people, human, text, watermark, "
    "ugly, blurry, deformed, extra limbs, bad anatomy"
)

WAI_CKPT = Path(__file__).parent.parent / "models" / "waiIllustriousSDXL_v160.safetensors"
DEMO_OUT = Path(__file__).parent.parent / "demonstration"
DEMO_OUT.mkdir(parents=True, exist_ok=True)
EXP_MD = Path(__file__).parent.parent / "experiments.md"

# ── Load .env for API keys ──
def _load_env():
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip())

_load_env()

OPENROUTER_MODEL = "moonshotai/kimi-k2.5"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


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
            f"Rank {len(images)} images of '{label}' from best to worst. "
            "Compare ALL side by side — do NOT score independently.\n\n"
            "Criteria (priority order):\n"
            f"1) OBJECT CORRECTNESS: Does it clearly depict a single '{label}' with correct structure? "
            "Right number of parts (4 legs on a chair, 2 wings on a bird, 5 fingers on a hand), "
            "proper proportions, no merged/duplicated/extra objects.\n"
            "2) AI ARTIFACT CHECK — penalize heavily for these common AI generation flaws:\n"
            "   - Melted/warped regions (faces melting, objects bending unnaturally)\n"
            "   - Extra or missing limbs/appendages (6 fingers, 3 legs, missing tail)\n"
            "   - Texture smearing (skin looking like plastic, fur merging into background)\n"
            "   - Anatomical impossibilities (joints bending wrong way, eyes at wrong position)\n"
            "   - Gibberish text or symbols floating in the image\n"
            "   - Repetition/tiling artifacts (same pattern repeating unnaturally)\n"
            "   - Color banding or unnatural color transitions\n"
            "   - Uncanny valley distortions (almost-right but clearly wrong proportions)\n"
            "   - Blurry or out-of-focus regions that should be sharp\n"
            "   - Background objects bleeding into foreground\n"
            "3) AESTHETICS: Clean composition, good colors, professional quality.\n"
            "4) CLEAN BACKGROUND: Subject clearly separated from background.\n\n"
            "Keep your reasoning to 1-2 sentences. Be brief but specific about WHY #1 won.\n"
            "JSON ONLY: {\"ranking\": [best,...,worst], \"reasoning\": \"...\"}"
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
                          "max_tokens": 6144, "temperature": 0.3},
                    timeout=120)
                r.raise_for_status()
                raw = r.json()["choices"][0]["message"].get("content") or ""
                if not raw.strip():
                    print(f"    Kimi empty response, retry {attempt+1}")
                    continue
                parsed = json.loads(raw)
                ranking = parsed.get("ranking")
                if ranking and len(ranking) >= 1:
                    idx = int(ranking[0]) - 1
                    if 0 <= idx < len(images):
                        rank_str = " > ".join(f"Img{r}" for r in ranking)
                        print(f"      Ranking: {rank_str}")
                        print(f"      Reason: {parsed.get('reasoning', '')[:120]}")
                        return idx, parsed.get("reasoning", "")
            except (json.JSONDecodeError, KeyError, ValueError, requests.RequestException) as e:
                print(f"    Kimi err: {e}, retry {attempt+1}")
        raise RuntimeError(f"Kimi failed after {retries} retries")


# ── Fixed test set ──
FIXED_SEED = 42
FIXED_N = 10


def get_fixed_test_set(n=FIXED_N, seed=FIXED_SEED):
    """Return deterministic list of (class, quickdraw_drawing) pairs."""
    rng = random.Random(seed)
    classes = rng.sample(ALL_CLASSES, n)
    offsets = [rng.randint(0, 500) for _ in range(n)]
    gen_seeds = [rng.randint(0, 999999) for _ in range(n)]

    test_set = []
    for cls, offset, gseed in zip(classes, offsets, gen_seeds):
        drawings = fetch_quickdraw(cls, 1, offset)
        if not drawings:
            raise RuntimeError(f"No drawings returned for class '{cls}' at offset {offset}")
        test_set.append({"cls": cls, "drawing": drawings[0], "gen_seed": gseed})
    return test_set


# ── Prompt helpers ──

def get_prompt(label):
    return PROMPTS.get(label, f"a single {label}, solo, centered, (digital illustration:1.2), detailed, sharp focus, vivid colors")

def get_negative_sd15(label):
    return NEGATIVES.get(label, DEFAULT_NEG_SD15)

def get_negative_sdxl(label):
    neg = NEGATIVES.get(label, DEFAULT_NEG_SDXL)
    return neg.replace("easynegative, ", "").replace("easynegative,", "").replace("easynegative", "")

def score_prompt(label):
    return f"a high quality illustration of a {label}"


# ── Image utils ──

def remove_bg_rembg(img, bg=(255, 255, 255)):
    """Neural network bg removal via rembg (U2-Net)."""
    rgba = rembg_remove(img)
    out = Image.new('RGBA', rgba.size, (*bg, 255))
    out.paste(rgba, mask=rgba.split()[3])
    return out.convert('RGB')


def remove_bg_grabcut(img, bg=(255, 255, 255), margin=20):
    """OpenCV GrabCut bg removal."""
    arr = np.array(img)
    h, w = arr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    rect = (margin, margin, w - 2 * margin, h - 2 * margin)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(arr, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    result = np.full_like(arr, bg)
    result[fg_mask == 255] = arr[fg_mask == 255]
    return Image.fromarray(result)


def remove_bg_threshold(img, bg=(255, 255, 255), thresh=230):
    """Replace near-white/near-gray background pixels with solid white."""
    arr = np.array(img).astype(np.float32)
    # Pixels where all channels are above threshold -> background
    bg_mask = np.all(arr > thresh, axis=2)
    arr[bg_mask] = bg
    return Image.fromarray(arr.astype(np.uint8))


BG_METHODS = [
    ("Raw", None),
    ("rembg", remove_bg_rembg),
    ("GrabCut", remove_bg_grabcut),
    ("Threshold", remove_bg_threshold),
]

def zoom_to_content(img, target_size, fill_pct=0.9):
    """Crop to content bounding box, pad to square, resize. Content fills fill_pct of output."""
    arr = np.array(img.convert('L'))
    mask = arr < 240  # non-white pixels = content
    if not mask.any():
        return img.resize((target_size, target_size), Image.LANCZOS)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # Crop to content
    cropped = img.crop((cmin, rmin, cmax + 1, rmax + 1))
    cw, ch = cropped.size
    # Pad to square
    side = max(cw, ch)
    padded = Image.new('RGB', (side, side), (255, 255, 255))
    padded.paste(cropped, ((side - cw) // 2, (side - ch) // 2))
    # Add margin so content fills fill_pct
    final_side = int(side / fill_pct)
    final = Image.new('RGB', (final_side, final_side), (255, 255, 255))
    final.paste(padded, ((final_side - side) // 2, (final_side - side) // 2))
    return final.resize((target_size, target_size), Image.LANCZOS)


def drawing_to_img_small(d, sz=256):
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

def drawing_to_img_large(d, sz=1024):
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

def to_lineart_sd15(image):
    arr = np.array(image.convert('L'))
    _, b = cv2.threshold(arr, 200, 255, cv2.THRESH_BINARY_INV)
    b = cv2.dilate(b, np.ones((2, 2), np.uint8), iterations=1)
    return Image.fromarray(b).convert('RGB')

def to_lineart_sdxl(image):
    arr = np.array(image.convert('L'))
    _, b = cv2.threshold(arr, 200, 255, cv2.THRESH_BINARY_INV)
    b = cv2.dilate(b, np.ones((3, 3), np.uint8), iterations=1)
    return Image.fromarray(b).convert('RGB')


# ── Model loaders ──

def load_sd15(device="cuda"):
    print("Loading SD1.5 + lineart ControlNet...")
    cn = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_lineart", torch_dtype=torch.float16).to(device)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=cn,
        torch_dtype=torch.float16, safety_checker=None).to(device)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()
    try:
        pipe.load_textual_inversion(
            "EvilEngine/easynegative", weight_name="easynegative.safetensors", token="easynegative")
    except Exception as e:
        print(f"  warn: easynegative failed: {e}")
    comps = {k: v for k, v in pipe.components.items() if k != "controlnet"}
    refiner = StableDiffusionImg2ImgPipeline(**comps)
    print("SD1.5 Ready")
    return pipe, refiner


def load_sdxl_base(device="cuda"):
    print("Loading SDXL-base + MistoLine ControlNet...")
    cn = ControlNetModel.from_pretrained(
        "TheMistoAI/MistoLine", torch_dtype=torch.float16, variant="fp16").to(device)
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", controlnet=cn, vae=vae,
        torch_dtype=torch.float16, variant="fp16", safety_checker=None).to(device)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()
    comps = {k: v for k, v in pipe.components.items() if k != "controlnet"}
    refiner = StableDiffusionXLImg2ImgPipeline(**comps)
    print("SDXL-base Ready")
    return pipe, refiner


def load_wai(device="cuda"):
    if not WAI_CKPT.exists():
        raise FileNotFoundError(f"WAI checkpoint not found: {WAI_CKPT}")
    print(f"Loading WAI-Illustrious-SDXL from {WAI_CKPT.name}...")
    cn = ControlNetModel.from_pretrained(
        "TheMistoAI/MistoLine", torch_dtype=torch.float16, variant="fp16").to(device)
    pipe = StableDiffusionXLControlNetPipeline.from_single_file(
        str(WAI_CKPT), controlnet=cn,
        torch_dtype=torch.float16, safety_checker=None).to(device)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()
    comps = {k: v for k, v in pipe.components.items() if k != "controlnet"}
    refiner = StableDiffusionXLImg2ImgPipeline(**comps)
    print("WAI-Illustrious Ready")
    return pipe, refiner


# ── Generation helpers ──

def generate_sd15(pipe, refiner, ctrl, label, seed, device="cuda"):
    img = pipe(
        prompt=get_prompt(label), negative_prompt=get_negative_sd15(label),
        image=ctrl, num_inference_steps=30,
        guidance_scale=7.5, controlnet_conditioning_scale=0.7,
        control_guidance_start=0.0, control_guidance_end=0.8,
        generator=torch.Generator(device=device).manual_seed(seed),
        num_images_per_prompt=1,
    ).images[0]
    img = refiner(
        prompt=get_prompt(label), negative_prompt=get_negative_sd15(label),
        image=img, strength=0.25, num_inference_steps=15, guidance_scale=4.0,
        generator=torch.Generator(device=device).manual_seed(seed),
    ).images[0]
    return img


def generate_sdxl(pipe, refiner, ctrl, label, seed, device="cuda"):
    img = pipe(
        prompt=get_prompt(label), negative_prompt=get_negative_sdxl(label),
        image=ctrl, num_inference_steps=30,
        guidance_scale=7.0, controlnet_conditioning_scale=0.5,
        control_guidance_start=0.0, control_guidance_end=0.9,
        generator=torch.Generator(device=device).manual_seed(seed),
        num_images_per_prompt=1,
    ).images[0]
    img = refiner(
        prompt=get_prompt(label), negative_prompt=get_negative_sdxl(label),
        image=img, strength=0.25, num_inference_steps=15, guidance_scale=4.0,
        generator=torch.Generator(device=device).manual_seed(seed),
    ).images[0]
    return img


def generate_wai(pipe, refiner, ctrl, label, seed, device="cuda"):
    img = pipe(
        prompt=get_prompt(label), negative_prompt=get_negative_sdxl(label),
        image=ctrl, num_inference_steps=25,
        guidance_scale=6.0, controlnet_conditioning_scale=0.5,
        control_guidance_start=0.0, control_guidance_end=0.9,
        generator=torch.Generator(device=device).manual_seed(seed),
        num_images_per_prompt=1,
    ).images[0]
    img = refiner(
        prompt=get_prompt(label), negative_prompt=get_negative_sdxl(label),
        image=img, strength=0.25, num_inference_steps=15, guidance_scale=4.0,
        generator=torch.Generator(device=device).manual_seed(seed),
    ).images[0]
    return img


# ── Canvas helpers ──

def get_font(size):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def add_label(image, text, position="top", color="white", bg_color=(40, 40, 40),
              font_size=14, height=24):
    w = image.width
    font = get_font(font_size)
    bar = Image.new("RGB", (w, height), bg_color)
    draw = ImageDraw.Draw(bar)
    bb = draw.textbbox((0, 0), text, font=font)
    text_w = bb[2] - bb[0]
    text_h = bb[3] - bb[1]
    x = (w - text_w) // 2
    y = (height - text_h) // 2
    draw.text((x, y), text, fill=color, font=font)
    if position == "top":
        combined = Image.new("RGB", (w, height + image.height))
        combined.paste(bar, (0, 0))
        combined.paste(image, (0, height))
    else:
        combined = Image.new("RGB", (w, image.height + height))
        combined.paste(image, (0, 0))
        combined.paste(bar, (0, image.height))
    return combined


def img_to_b64(img, fmt="JPEG", quality=85):
    """Convert PIL Image to base64 data URI."""
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    return f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode()}"


def build_html_report(all_model_data, test_set, beit_correct, models, run_ts):
    """Build a self-contained HTML report with embedded thumbnails and click-to-zoom."""

    html = []
    html.append(f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>BEiT Pipeline Comparison — {run_ts}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #1a1a2e; color: #eee; font-family: 'Segoe UI', system-ui, sans-serif; padding: 20px; }}
  h1 {{ text-align: center; color: #e94560; margin-bottom: 6px; }}
  .meta {{ text-align: center; color: #888; margin-bottom: 20px; font-size: 14px; }}
  .model-section {{ background: #16213e; border-radius: 12px; padding: 20px; margin-bottom: 30px; }}
  .model-title {{ font-size: 22px; color: #0f3460; background: #e94560; display: inline-block;
                  padding: 4px 18px; border-radius: 6px; color: #fff; margin-bottom: 14px; }}
  .sample {{ background: #0f3460; border-radius: 8px; padding: 14px; margin-bottom: 16px; }}
  .sample-header {{ font-size: 15px; margin-bottom: 10px; display: flex; gap: 12px; align-items: center; }}
  .tag {{ padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }}
  .tag-ok {{ background: #2d6a4f; color: #b7e4c7; }}
  .tag-miss {{ background: #9b2226; color: #ffd7ba; }}
  .stage {{ margin-bottom: 10px; }}
  .stage-label {{ font-size: 13px; color: #aaa; margin-bottom: 4px; font-weight: 600;
                  border-left: 3px solid #e94560; padding-left: 8px; }}
  .thumbs {{ display: flex; gap: 8px; flex-wrap: wrap; align-items: flex-start; }}
  .thumb-wrap {{ text-align: center; }}
  .thumb-wrap img {{ width: 150px; height: 150px; object-fit: contain; border-radius: 6px;
                     cursor: pointer; border: 2px solid #233; transition: border 0.15s; background: #fff; }}
  .thumb-wrap img:hover {{ border-color: #e94560; }}
  .thumb-cap {{ font-size: 11px; color: #999; margin-top: 2px; max-width: 150px; word-wrap: break-word; }}
  .kimi-reason {{ font-size: 12px; color: #c9a227; margin-top: 4px; font-style: italic;
                  max-width: 800px; }}
  .winner {{ border-color: #2d6a4f !important; box-shadow: 0 0 8px #2d6a4f; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
  th, td {{ padding: 6px 12px; border: 1px solid #333; text-align: center; font-size: 13px; }}
  th {{ background: #0f3460; }}
  /* Lightbox */
  .lightbox {{ display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
               background: rgba(0,0,0,0.92); z-index: 9999; justify-content: center; align-items: center; cursor: zoom-out; }}
  .lightbox.active {{ display: flex; }}
  .lightbox img {{ max-width: 90vw; max-height: 90vh; border-radius: 8px; }}
</style></head><body>
<div class="lightbox" id="lb" onclick="this.classList.remove('active')">
  <img id="lb-img" src="">
</div>
<script>
function zoom(el) {{
  document.getElementById('lb-img').src = el.src;
  document.getElementById('lb').classList.add('active');
}}
</script>
<h1>BEiT Pipeline Comparison</h1>
<div class="meta">Run: {run_ts} &bull; Samples: {len(test_set)} &bull;
BEiT accuracy: {beit_correct}/{len(test_set)} ({beit_correct/len(test_set):.0%}) &bull;
Models: {', '.join(models)}</div>
""")

    for model_name in models:
        mdata = all_model_data[model_name]
        html.append(f'<div class="model-section">')
        html.append(f'<div class="model-title">{model_name.upper()}</div>')

        # Summary table
        bg_names = [name for name, _ in BG_METHODS]
        html.append('<table><tr><th>BG Method</th><th>Avg SigLIP2</th><th>Min</th><th>Max</th></tr>')
        for bn in bg_names:
            scores = mdata["scores"][bn]
            if scores:
                html.append(f'<tr><td>{bn}</td><td>{np.mean(scores):.4f}</td>'
                            f'<td>{min(scores):.4f}</td><td>{max(scores):.4f}</td></tr>')
        html.append('</table>')

        for si, sample in enumerate(mdata["samples"]):
            t = test_set[si]
            gt = t["cls"]
            pred = t["pred_label"]
            correct = t["beit_correct"]
            tag_cls = "tag-ok" if correct else "tag-miss"
            tag_txt = "OK" if correct else "MISS"

            html.append(f'<div class="sample">')
            html.append(f'<div class="sample-header">'
                        f'<span>#{si+1}</span>'
                        f'<span><b>GT:</b> {gt}</span>'
                        f'<span><b>Pred:</b> {pred} ({t["beit_preds"][0][1]:.0%})</span>'
                        f'<span class="tag {tag_cls}">{tag_txt}</span></div>')

            # Stage 1: Input sketch
            html.append('<div class="stage"><div class="stage-label">1. Input QuickDraw Sketch</div>')
            html.append('<div class="thumbs">')
            uri = img_to_b64(sample["qd_sketch"])
            html.append(f'<div class="thumb-wrap"><img src="{uri}" onclick="zoom(this)"></div>')
            html.append('</div></div>')

            # Stage 2: Top-4 DomainNet matches
            html.append('<div class="stage"><div class="stage-label">2. Top-4 DomainNet Matches (SigLIP2 similarity)</div>')
            html.append('<div class="thumbs">')
            for di, (dn_img, dn_path, dn_sim) in enumerate(t["dn_top4"]):
                dn_thumb = dn_img.resize((256, 256), Image.LANCZOS)
                uri = img_to_b64(dn_thumb)
                html.append(f'<div class="thumb-wrap"><img src="{uri}" onclick="zoom(this)">'
                            f'<div class="thumb-cap">#{di+1} sim={dn_sim:.3f}</div></div>')
            html.append('</div></div>')

            # Stage 3: ControlNet inputs (lineart)
            html.append('<div class="stage"><div class="stage-label">3. ControlNet Lineart Inputs</div>')
            html.append('<div class="thumbs">')
            for ci, ctrl_img in enumerate(sample.get("ctrl_imgs", [])):
                ctrl_thumb = ctrl_img.resize((256, 256), Image.LANCZOS)
                uri = img_to_b64(ctrl_thumb)
                html.append(f'<div class="thumb-wrap"><img src="{uri}" onclick="zoom(this)">'
                            f'<div class="thumb-cap">ctrl #{ci+1}</div></div>')
            html.append('</div></div>')

            # Stage 4: 4 raw candidates
            best_idx = sample["kimi_best_idx"]
            html.append('<div class="stage"><div class="stage-label">'
                        f'4. Generated Candidates (Kimi picked #{best_idx+1})</div>')
            html.append('<div class="thumbs">')
            for ci, cand in enumerate(sample["candidates"]):
                cand_thumb = cand.resize((256, 256), Image.LANCZOS)
                uri = img_to_b64(cand_thumb)
                winner = " winner" if ci == best_idx else ""
                html.append(f'<div class="thumb-wrap"><img class="{winner}" src="{uri}" onclick="zoom(this)">'
                            f'<div class="thumb-cap">cand #{ci+1}</div></div>')
            html.append('</div>')
            if sample.get("kimi_reason"):
                html.append(f'<div class="kimi-reason">Kimi: {sample["kimi_reason"]}</div>')
            html.append('</div>')

            # Stage 5: BG removal variants with scores
            html.append('<div class="stage"><div class="stage-label">5. BG Removal Variants + SigLIP2 Scores</div>')
            html.append('<div class="thumbs">')
            for method_name, img, sc in sample["bg_variants"]:
                bg_thumb = img.resize((256, 256), Image.LANCZOS)
                uri = img_to_b64(bg_thumb)
                html.append(f'<div class="thumb-wrap"><img src="{uri}" onclick="zoom(this)">'
                            f'<div class="thumb-cap">{method_name}: {sc:.3f}</div></div>')
            html.append('</div></div>')

            html.append('</div>')  # sample

        html.append('</div>')  # model-section

    html.append('</body></html>')
    return '\n'.join(html)


# ── Main ──

def run(models=None, n_samples=None):
    device = "cuda"
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if models is None:
        models = ["sd15", "sdxl", "wai"]
    if n_samples is None:
        n_samples = FIXED_N

    # 1. Load shared components
    print("Loading BEiT classifier...")
    beit = BeitSketchClassifier(device=device)
    print("Loading SigLIP2 scorer...")
    scorer = SiglipScorer(device=device)

    # 2. Fetch fixed test set (always generate full set for reproducibility, then slice)
    print(f"\nFetching {n_samples} fixed test sketches (seed={FIXED_SEED})...")
    test_set = get_fixed_test_set(n=n_samples)
    classes = [t["cls"] for t in test_set]
    print(f"  Classes: {classes}")

    # 3. BEiT classification + DomainNet matching (shared across models)
    print("\nRunning BEiT classification...")
    beit_correct = 0
    for t in test_set:
        sketch = drawing_to_img_small(t["drawing"])
        preds = beit.classify(sketch, top_k=3)
        t["beit_preds"] = preds
        t["beit_correct"] = preds[0][0] == t["cls"]
        if t["beit_correct"]:
            beit_correct += 1
        print(f"  {t['cls']} -> {preds[0][0]} ({preds[0][1]:.0%}) {'OK' if t['beit_correct'] else 'MISS'}")
    print(f"BEiT accuracy: {beit_correct}/{len(test_set)} ({beit_correct/len(test_set):.0%})")

    # Use BEiT PREDICTED labels (not GT) for DomainNet matching and generation
    pred_classes = list(set(t["beit_preds"][0][0] for t in test_set))
    print(f"\nBuilding DomainNet index for predicted classes: {pred_classes}")
    matcher = DomainNetMatcher(scorer, pred_classes, batch_size=16)
    for t in test_set:
        pred_label = t["beit_preds"][0][0]
        t["pred_label"] = pred_label
        sketch = drawing_to_img_small(t["drawing"])
        top4 = matcher.match_topk(sketch, pred_label, k=4)
        t["dn_top4"] = [(zoom_to_content(img, 1024, fill_pct=0.9), path, sim)
                         for img, path, sim in top4]
        t["dn_sketch"] = t["dn_top4"][0][0]
        t["dn_path"] = t["dn_top4"][0][1]
        t["dn_sim"] = t["dn_top4"][0][2]
        names = [p.name for _, p, _ in top4]
        print(f"  {t['cls']} (pred:{pred_label}) -> top4: {names}")

    # Load Kimi judge for best-of-4 selection
    print("\nLoading Kimi judge...")
    kimi = KimiJudge()

    # 4. Run each model — collect stage data for HTML report
    all_model_data = {}
    bg_method_names = [name for name, _ in BG_METHODS]

    for model_name in models:
        print(f"\n{'='*50}")
        print(f"Model: {model_name}")
        print(f"{'='*50}")

        if model_name == "sd15":
            pipe, refiner = load_sd15(device)
            gen_fn = lambda ctrl, lbl, seed: generate_sd15(pipe, refiner, ctrl, lbl, seed, device)
            lineart_fn = to_lineart_sd15
            resize_for_ctrl = 512
        elif model_name == "sdxl":
            pipe, refiner = load_sdxl_base(device)
            gen_fn = lambda ctrl, lbl, seed: generate_sdxl(pipe, refiner, ctrl, lbl, seed, device)
            lineart_fn = to_lineart_sdxl
            resize_for_ctrl = 1024
        elif model_name == "wai":
            pipe, refiner = load_wai(device)
            gen_fn = lambda ctrl, lbl, seed: generate_wai(pipe, refiner, ctrl, lbl, seed, device)
            lineart_fn = to_lineart_sdxl
            resize_for_ctrl = 1024
        else:
            raise ValueError(f"Unknown model: {model_name}")

        model_scores = {name: [] for name in bg_method_names}
        model_samples = []

        # Phase 1: Generate ALL candidates (GPU-bound, sequential)
        print(f"  Phase 1: Generating 4 candidates x {len(test_set)} samples...")
        all_candidates = []
        all_ctrl_imgs = []
        for i, t in enumerate(test_set):
            pred_label = t["pred_label"]
            candidates = []
            ctrl_imgs = []
            for j, (dn_img, dn_path, dn_sim) in enumerate(t["dn_top4"]):
                dn_resized = dn_img.resize((resize_for_ctrl, resize_for_ctrl), Image.LANCZOS)
                ctrl = lineart_fn(dn_resized)
                ctrl_imgs.append(ctrl)
                raw_img = gen_fn(ctrl, pred_label, t["gen_seed"])
                candidates.append(raw_img)
            all_candidates.append(candidates)
            all_ctrl_imgs.append(ctrl_imgs)
            print(f"    [{i+1}/{len(test_set)}] generated 4 candidates for {pred_label}")

        # Phase 2: Kimi ranking ALL samples in parallel (I/O-bound)
        print(f"  Phase 2: Kimi ranking {len(test_set)} samples in parallel (8 threads)...")
        kimi_results = [None] * len(test_set)

        def _kimi_task(idx):
            cands = all_candidates[idx]
            lbl = test_set[idx]["pred_label"]
            best_idx, reason = kimi.pick_best(cands, lbl)
            return idx, best_idx, reason

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(_kimi_task, i): i for i in range(len(test_set))}
            for fut in as_completed(futures):
                idx, best_idx, reason = fut.result()
                kimi_results[idx] = (best_idx, reason)
                print(f"    Kimi [{idx+1}/{len(test_set)}] {test_set[idx]['pred_label']}: "
                      f"picked #{best_idx+1} — {reason[:60]}")

        # Phase 3: BG removal + scoring (GPU-bound, sequential)
        print(f"  Phase 3: BG removal + SigLIP2 scoring...")
        for i, t in enumerate(test_set):
            gt_cls = t["cls"]
            pred_label = t["pred_label"]
            best_idx, reason = kimi_results[i]
            raw_img = all_candidates[i][best_idx]

            bg_variants = []
            scores_str = []
            for method_name, method_fn in BG_METHODS:
                processed = raw_img if method_fn is None else method_fn(raw_img)
                sc = scorer.score_single(processed, score_prompt(pred_label))
                bg_variants.append((method_name, processed, sc))
                model_scores[method_name].append(sc)
                scores_str.append(f"{method_name}={sc:.3f}")

            status = "OK" if t["beit_correct"] else "MISS"
            print(f"  [{i+1}/{len(test_set)}] gt={gt_cls} pred={pred_label} -> {' | '.join(scores_str)} [{status}]")

            qd_sketch = drawing_to_img_small(t["drawing"])
            model_samples.append({
                "qd_sketch": qd_sketch,
                "ctrl_imgs": all_ctrl_imgs[i],
                "candidates": all_candidates[i],
                "kimi_best_idx": best_idx,
                "kimi_reason": reason,
                "bg_variants": bg_variants,
            })

        all_model_data[model_name] = {"scores": model_scores, "samples": model_samples}

        for method_name in bg_method_names:
            scores = model_scores[method_name]
            print(f"  {method_name}: avg={np.mean(scores):.4f} min={min(scores):.4f} max={max(scores):.4f}")

        # Unload model to free VRAM for next
        del pipe, refiner
        torch.cuda.empty_cache()

    # 5. Build and save HTML report
    print("\nBuilding HTML report...")
    html = build_html_report(all_model_data, test_set, beit_correct, models, run_ts)
    html_path = DEMO_OUT / f"report_{run_ts}.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"HTML report: {html_path}")

    # 6. Summary to experiments.md
    ts_pretty = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(EXP_MD, "a") as f:
        f.write(f"\n## BEiT Pipeline Comparison ({ts_pretty}, run={run_ts})\n")
        f.write(f"Fixed test set: {n_samples} sketches, seed={FIXED_SEED}\n")
        f.write(f"BEiT accuracy: {beit_correct}/{len(test_set)} ({beit_correct/len(test_set):.0%})\n")
        f.write(f"HTML report: `{html_path.name}`\n\n")
        f.write(f"| Model | BG Method | Avg SigLIP2 | Min | Max |\n")
        f.write(f"|-------|-----------|-------------|-----|-----|\n")
        for model_name in models:
            for method_name in bg_method_names:
                scores = all_model_data[model_name]["scores"][method_name]
                avg, mn, mx = np.mean(scores), min(scores), max(scores)
                print(f"  {model_name}/{method_name}: avg={avg:.4f} min={mn:.4f} max={mx:.4f}")
                f.write(f"| {model_name} | {method_name} | {avg:.4f} | {mn:.4f} | {mx:.4f} |\n")

    print(f"\nDone. Report: {html_path}")


if __name__ == "__main__":
    import argparse as _ap
    p = _ap.ArgumentParser()
    p.add_argument("models", nargs="*", default=["sd15", "sdxl", "wai"])
    p.add_argument("-n", type=int, default=FIXED_N, help="number of test sketches")
    args = p.parse_args()
    run(models=args.models, n_samples=args.n)
