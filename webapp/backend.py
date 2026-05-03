import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "quick_draw"))

import io
import json
import base64
import random
import traceback
import zipfile
import numpy as np
import torch
from torchvision import transforms as T
import requests
from PIL import Image
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from transformers import AutoModelForImageSegmentation

from v8_tailored import (
    BeitSketchClassifier, SiglipScorer, DomainNetMatcher,
    get_prompt, get_negative, score_prompt,
)
from demo_beit_compare import (
    load_sd15, load_sdxl_base, load_sdxl_lightning,
    generate_sd15, generate_sdxl, generate_lightning,
    to_lineart_sd15, to_lineart_sdxl,
    zoom_to_content, KimiJudge,
)

DEVICE = "cuda"
S = {}

BIREFNET_TRANSFORM = T.Compose([
    T.Resize((1024, 1024)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def remove_bg_birefnet(img, bg=(255, 255, 255)):
    """GPU-accelerated background removal using BiRefNet (33x faster than rembg)."""
    birefnet = S["birefnet"]
    w, h = img.size
    inp = BIREFNET_TRANSFORM(img).unsqueeze(0).to(DEVICE, dtype=torch.float16)
    with torch.no_grad():
        pred = birefnet(inp)[-1].sigmoid().cpu()
    mask = (pred[0, 0] * 255).byte().numpy()
    mask = Image.fromarray(mask).resize((w, h), Image.BILINEAR)
    rgba = img.copy().convert("RGBA")
    rgba.putalpha(mask)
    out = Image.new("RGBA", rgba.size, (*bg, 255))
    out.paste(rgba, mask=rgba.split()[3])
    return out.convert("RGB")


PROJECT_ROOT = Path(__file__).parent.parent
DOMAINNET_ROOT = PROJECT_ROOT / "data" / "domainnet" / "sketch"
DOMAINNET_URL = "http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip"


def ensure_domainnet():
    """Download and extract DomainNet sketch split if not present."""
    if DOMAINNET_ROOT.exists() and any(DOMAINNET_ROOT.iterdir()):
        return
    print("DomainNet data not found. Downloading (~1.2GB)...")
    DOMAINNET_ROOT.parent.mkdir(parents=True, exist_ok=True)
    zip_path = PROJECT_ROOT / "data" / "domainnet" / "sketch.zip"
    if not zip_path.exists():
        r = requests.get(DOMAINNET_URL, stream=True, timeout=600)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192 * 16):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 / total
                    print(f"\r  {downloaded // (1024**2)}/{total // (1024**2)} MB ({pct:.0f}%)", end="", flush=True)
        print()
    print("Extracting DomainNet...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(PROJECT_ROOT / "data" / "domainnet")
    zip_path.unlink(missing_ok=True)
    print("DomainNet ready.")


# ── Multi-criteria SigLIP2 selector ──
QUALITY_AXES = {
    "prompt_match":  {"template": "a high quality illustration of a {label}", "weight": 2.0},
    "no_artifacts":  {"template": "a clean flawless image without any artifacts, distortions, or deformations", "weight": 1.5},
    "anatomy":       {"template": "correct natural proportions and anatomy, physically plausible structure", "weight": 1.5},
    "sharpness":     {"template": "a sharp detailed crisp image with clear edges and fine details", "weight": 1.0},
    "composition":   {"template": "a single centered {label} on a clean simple background, isolated subject", "weight": 1.0},
    "colors":        {"template": "vibrant natural realistic colors with good lighting and contrast", "weight": 0.5},
    "professional":  {"template": "professional commercial product photography, studio quality render", "weight": 0.5},
}


def img_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def b64_img(s):
    return Image.open(io.BytesIO(base64.b64decode(s.split(",")[-1]))).convert("RGB")


def siglip_pick_best(scorer, candidates, label):
    """Single-prompt SigLIP2 selector. Returns (best_idx, scores, reason)."""
    prompt = score_prompt(label)
    scores = [scorer.score_single(c, prompt) for c in candidates]
    best_idx = int(np.argmax(scores))
    return best_idx, scores, f"SigLIP2 score {scores[best_idx]:.3f}"


def siglip_multi_pick_best(scorer, candidates, label):
    """Multi-criteria SigLIP2 selector. Returns (best_idx, single_scores, reason)."""
    axes = list(QUALITY_AXES.keys())
    texts = [QUALITY_AXES[a]["template"].format(label=label) for a in axes]
    weights = np.array([QUALITY_AXES[a]["weight"] for a in axes])

    # Score each candidate on each axis
    raw = np.zeros((len(candidates), len(axes)))
    for i, cand in enumerate(candidates):
        for j, t in enumerate(texts):
            raw[i, j] = scorer.score_single(cand, t)

    # Normalize per axis, compute weighted composite
    normed = np.zeros_like(raw)
    for j in range(raw.shape[1]):
        col = raw[:, j]
        mn, mx = col.min(), col.max()
        normed[:, j] = (col - mn) / (mx - mn + 1e-8)

    composite = (normed * weights[None, :]).sum(axis=1)
    best_idx = int(np.argmax(composite))

    # Also compute single-prompt scores for display
    prompt = score_prompt(label)
    single_scores = [scorer.score_single(c, prompt) for c in candidates]

    top_axes = sorted(range(len(axes)), key=lambda j: normed[best_idx, j], reverse=True)[:3]
    reason = f"Multi-criteria winner (top: {', '.join(axes[j] for j in top_axes)})"
    return best_idx, single_scores, reason


@asynccontextmanager
async def lifespan(app):
    ensure_domainnet()
    print("Loading models...")
    S["beit"] = BeitSketchClassifier(device=DEVICE)
    S["scorer"] = SiglipScorer(device=DEVICE)
    S["birefnet"] = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/BiRefNet", trust_remote_code=True
    ).to(DEVICE).half().eval()
    print("BiRefNet loaded (GPU bg removal)")
    S["sd15_pipe"], S["sd15_refiner"] = load_sd15(DEVICE)
    S["lightning_pipe"], _ = load_sdxl_lightning(DEVICE)
    S["sdxl_loaded"] = False
    S["dn"] = {}
    try:
        S["kimi"] = KimiJudge()
    except ValueError:
        print("WARNING: Kimi judge not available (no OPENROUTER_API_KEY)")
        S["kimi"] = None
    print("All models loaded. SD1.5 + Lightning ready. SDXL will load on first use.")
    yield


app = FastAPI(lifespan=lifespan)


def ensure_sdxl():
    """Lazy-load SDXL on first request."""
    if not S.get("sdxl_loaded"):
        print("Loading SDXL (first request)...")
        S["sdxl_pipe"], S["sdxl_refiner"] = load_sdxl_base(DEVICE)
        S["sdxl_loaded"] = True
        print("SDXL ready.")


def get_matcher(cls):
    if cls not in S["dn"]:
        S["dn"][cls] = DomainNetMatcher(S["scorer"], [cls], batch_size=16)
    return S["dn"][cls]


class Req(BaseModel):
    image: str
    model: str = "lightning"
    best_of: int = 4
    selector: str = "siglip"
    remove_bg: bool = True
    seed: int = -1


def sse_event(data):
    """Format a Server-Sent Event line."""
    return f"data: {json.dumps(data)}\n\n"


def run_pipeline(req: Req):
    """Generator that yields SSE events with intermediate previews per stage."""
    try:
        sketch = b64_img(req.image)
        seed = req.seed if req.seed >= 0 else random.randint(0, 999999)
        model = req.model if req.model in ("sd15", "sdxl", "lightning") else "lightning"
        best_of = max(1, min(req.best_of, 8))
        selector = req.selector if req.selector in ("siglip", "siglip_multi", "kimi") else "siglip"
        do_remove_bg = req.remove_bg

        # Send config so frontend knows what to expect
        yield sse_event({"type": "config", "model": model, "best_of": best_of,
                         "selector": selector, "remove_bg": do_remove_bg, "seed": seed})

        # ── Stage 1: Classification ──
        yield sse_event({"type": "stage_start", "stage": "classify"})
        small = sketch.resize((256, 256), Image.LANCZOS)
        preds = S["beit"].classify(small, top_k=5)
        label = preds[0][0]
        yield sse_event({"type": "stage_done", "stage": "classify",
                         "data": {
                             "sketch": img_b64(small),
                             "label": label,
                             "predictions": [{"label": l, "score": round(s, 3)} for l, s in preds],
                         }})

        # ── Stage 2: DomainNet matching ──
        yield sse_event({"type": "stage_start", "stage": "domainnet"})
        matcher = get_matcher(label)
        top_matches = matcher.match_topk(small, label, k=best_of)
        dn_imgs = [zoom_to_content(img, 1024, fill_pct=0.9) for img, _, _ in top_matches]
        dn_previews = [img_b64(d.resize((512, 512), Image.LANCZOS)) for d in dn_imgs]
        yield sse_event({"type": "stage_done", "stage": "domainnet",
                         "data": {"images": dn_previews}})

        # ── Stage 3: Generation ──
        yield sse_event({"type": "stage_start", "stage": "generate"})

        if model == "lightning":
            pipe = S["lightning_pipe"]
            gen_fn = lambda ctrl, lbl, sd: generate_lightning(pipe, None, ctrl, lbl, sd, DEVICE)
            lineart_fn = to_lineart_sdxl
            ctrl_size = 1024
        elif model == "sdxl":
            ensure_sdxl()
            pipe, refiner = S["sdxl_pipe"], S["sdxl_refiner"]
            gen_fn = lambda ctrl, lbl, sd: generate_sdxl(pipe, refiner, ctrl, lbl, sd, DEVICE)
            lineart_fn = to_lineart_sdxl
            ctrl_size = 1024
        else:
            pipe, refiner = S["sd15_pipe"], S["sd15_refiner"]
            gen_fn = lambda ctrl, lbl, sd: generate_sd15(pipe, refiner, ctrl, lbl, sd, DEVICE)
            lineart_fn = to_lineart_sd15
            ctrl_size = 512

        candidates = []
        ctrls = []

        # Generate from DomainNet-matched sketches — one candidate per match
        # If fewer matches than best_of, reuse top match with varied seeds
        for i in range(best_of):
            dn_img = dn_imgs[i] if i < len(dn_imgs) else dn_imgs[0]
            dn_resized = dn_img.resize((ctrl_size, ctrl_size), Image.LANCZOS)
            ctrl = lineart_fn(dn_resized)
            ctrls.append(ctrl)
            cand = gen_fn(ctrl, label, seed + i)
            candidates.append(cand)
            yield sse_event({"type": "candidate", "index": i,
                             "lineart": img_b64(ctrl), "image": img_b64(cand)})

        yield sse_event({"type": "stage_done", "stage": "generate",
                         "data": {"count": len(candidates)}})

        # ── Stage 4: Scoring & Selection ──
        yield sse_event({"type": "stage_start", "stage": "score"})

        if best_of == 1:
            pick_idx = 0
            siglip_scores = [S["scorer"].score_single(candidates[0], score_prompt(label))]
            reason = "Single candidate"
        elif selector == "kimi" and S.get("kimi"):
            pick_idx, reason = S["kimi"].pick_best(candidates, label)
            siglip_scores = [S["scorer"].score_single(c, score_prompt(label)) for c in candidates]
        elif selector == "siglip_multi":
            pick_idx, siglip_scores, reason = siglip_multi_pick_best(S["scorer"], candidates, label)
        else:
            pick_idx, siglip_scores, reason = siglip_pick_best(S["scorer"], candidates, label)

        yield sse_event({"type": "stage_done", "stage": "score",
                         "data": {
                             "pick_idx": pick_idx,
                             "siglip_scores": [round(float(s), 3) for s in siglip_scores],
                             "selector": selector,
                             "reason": reason,
                         }})

        # ── Stage 5: Background removal ──
        yield sse_event({"type": "stage_start", "stage": "background"})
        if do_remove_bg:
            final = remove_bg_birefnet(candidates[pick_idx])
        else:
            final = candidates[pick_idx]

        yield sse_event({"type": "stage_done", "stage": "background",
                         "data": {"final": img_b64(final), "remove_bg": do_remove_bg}})

        yield "data: [DONE]\n\n"

    except Exception as e:
        traceback.print_exc()
        yield sse_event({"type": "error", "message": str(e)})
        yield "data: [DONE]\n\n"


@app.post("/api/generate")
def generate(req: Req):
    return StreamingResponse(
        run_pipeline(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/status")
def status():
    models = ["sd15", "lightning"]
    if S.get("sdxl_loaded"):
        models.append("sdxl")
    kimi_ok = S.get("kimi") is not None
    return {
        "status": "ok",
        "models_loaded": models,
        "kimi_available": kimi_ok,
    }


app.mount("/", StaticFiles(directory=Path(__file__).parent / "frontend", html=True))
