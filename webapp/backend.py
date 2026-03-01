import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "quick_draw"))

import io
import base64
import random
import torch
from PIL import Image
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from contextlib import asynccontextmanager

from v8_tailored import BeitSketchClassifier, SiglipScorer, DomainNetMatcher
from demo_beit_compare import (
    load_sd15, generate_sd15, to_lineart_sd15,
    zoom_to_content, remove_bg_rembg, KimiJudge,
)

DEVICE = "cuda"
S = {}


def img_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def b64_img(s):
    return Image.open(io.BytesIO(base64.b64decode(s.split(",")[-1]))).convert("RGB")


@asynccontextmanager
async def lifespan(app):
    print("Loading models...")
    S["beit"] = BeitSketchClassifier(device=DEVICE)
    S["scorer"] = SiglipScorer(device=DEVICE)
    S["pipe"], S["refiner"] = load_sd15(DEVICE)
    S["dn"] = {}
    S["kimi"] = KimiJudge()
    print("All models loaded. Ready!")
    yield


app = FastAPI(lifespan=lifespan)


def get_matcher(cls):
    if cls not in S["dn"]:
        try:
            S["dn"][cls] = DomainNetMatcher(S["scorer"], [cls], batch_size=16)
        except FileNotFoundError:
            return None
    return S["dn"][cls]


class Req(BaseModel):
    image: str
    seed: int = -1


@app.post("/api/generate")
def generate(req: Req):
    sketch = b64_img(req.image)
    seed = req.seed if req.seed >= 0 else random.randint(0, 999999)

    small = sketch.resize((256, 256), Image.LANCZOS)
    preds = S["beit"].classify(small, top_k=5)
    label = preds[0][0]

    matcher = get_matcher(label)
    if matcher:
        top4 = matcher.match_topk(small, label, k=4)
        dn_imgs = [zoom_to_content(img, 1024, fill_pct=0.9) for img, _, _ in top4]
    else:
        dn_imgs = [sketch.resize((1024, 1024), Image.LANCZOS)]

    candidates = []
    ctrls = []
    for dn_img in dn_imgs:
        dn_512 = dn_img.resize((512, 512), Image.LANCZOS)
        ctrl = to_lineart_sd15(dn_512)
        ctrls.append(ctrl)
        candidates.append(generate_sd15(S["pipe"], S["refiner"], ctrl, label, seed, DEVICE))

    best_idx, reason = S["kimi"].pick_best(candidates, label)
    final = remove_bg_rembg(candidates[best_idx])

    return {
        "label": label,
        "predictions": [{"label": l, "score": round(s, 3)} for l, s in preds],
        "seed": seed,
        "kimi": {"pick": best_idx, "reason": reason},
        "stages": {
            "sketch": img_b64(small),
            "domainnet": [img_b64(d.resize((512, 512), Image.LANCZOS)) for d in dn_imgs],
            "lineart": [img_b64(c) for c in ctrls],
            "candidates": [img_b64(c) for c in candidates],
            "final": img_b64(final),
        },
    }


app.mount("/", StaticFiles(directory=Path(__file__).parent / "frontend", html=True))
