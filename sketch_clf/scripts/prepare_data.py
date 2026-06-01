"""Prepare a balanced QuickDraw 30-class dataset with honest disjoint splits.

The 30 classes are the deliberately-hard / confusable set used in EDA_30C.md, where
the pipeline baseline `kmewhort/beit-sketch-classifier` scores 64.8% top-1. Using the
same classes makes our comparison direct.

Rendering replicates `quick_draw/v8_tailored.drawing_to_img` EXACTLY (the renderer that
produced the documented baseline number) so train/val/test and the BEiT eval are all
apples-to-apples. We render grayscale uint8 224x224 and cache as .npy arrays.
"""
import io
import json
import sys
import time
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import numpy as np
import requests
from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent.parent / "data"
OUT.mkdir(parents=True, exist_ok=True)

# Same 30 classes as eda_30c.py (baseline BEiT = 64.8% top-1 here)
CLASSES = [
    "cookie", "broom", "violin", "camel", "steak",
    "trumpet", "bread", "banana", "arm", "flying saucer",
    "asparagus", "drums", "fish", "aircraft carrier", "backpack",
    "hat", "spider", "swing set", "telephone", "parachute",
    "sea turtle", "passport", "fence", "pliers", "hand",
    "fan", "face", "chair", "butterfly", "sword",
]
CLASSES.sort()  # deterministic label ordering

N_TRAIN, N_VAL, N_TEST = 3000, 500, 1000
N_TOTAL = N_TRAIN + N_VAL + N_TEST
IMG_SZ = 224
SEED = 42


def drawing_to_img(d, sz=IMG_SZ):
    """EXACT copy of v8_tailored.drawing_to_img (RGB), returned as grayscale uint8."""
    img = Image.new("RGB", (sz, sz), "white")
    draw = ImageDraw.Draw(img)
    strokes = d["drawing"]
    xs = [x for s in strokes for x in s[0]]
    ys = [y for s in strokes for y in s[1]]
    if not xs:
        return np.full((sz, sz), 255, np.uint8)
    mnx, mxx, mny, mxy = min(xs), max(xs), min(ys), max(ys)
    sc = sz * 0.8 / max(mxx - mnx, mxy - mny, 1)
    ox = (sz - (mxx - mnx) * sc) / 2 - mnx * sc
    oy = (sz - (mxy - mny) * sc) / 2 - mny * sc
    for s in strokes:
        pts = [(x * sc + ox, y * sc + oy) for x, y in zip(s[0], s[1])]
        if len(pts) > 1:
            draw.line(pts, fill="black", width=max(2, sz // 170))
    return np.array(img.convert("L"), dtype=np.uint8)


def fetch_quickdraw(category, n):
    url = f"https://storage.googleapis.com/quickdraw_dataset/full/simplified/{category}.ndjson"
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    out = []
    for line in r.iter_lines():
        if len(out) >= n:
            break
        if line:
            out.append(json.loads(line))
    return out


def fetch_one(args):
    ci, cls = args
    t0 = time.time()
    drawings = fetch_quickdraw(cls, N_TOTAL)
    print(f"  [{ci+1}/{len(CLASSES)}] {cls:18s} fetched {len(drawings):5d}  ({time.time()-t0:.1f}s)", flush=True)
    return cls, drawings


def render_one(d):
    return drawing_to_img(d)


def main():
    print(f"Fetching {N_TOTAL}/class x {len(CLASSES)} classes ...", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=16) as ex:
        fetched = dict(ex.map(fetch_one, list(enumerate(CLASSES))))
    print(f"Fetch done in {time.time()-t0:.1f}s", flush=True)

    rng = random.Random(SEED)
    splits = {"train": ([], []), "val": ([], []), "test": ([], [])}

    # Render all in a process pool (per class to keep memory bounded & ordering clear)
    with Pool(processes=32) as pool:
        for label, cls in enumerate(CLASSES):
            drawings = fetched[cls]
            if len(drawings) < N_TOTAL:
                print(f"WARNING: {cls} only has {len(drawings)} < {N_TOTAL}", flush=True)
            rng.shuffle(drawings)  # remove ordering bias before splitting
            imgs = pool.map(render_one, drawings, chunksize=64)
            imgs = imgs[:N_TOTAL]
            tr = imgs[:N_TRAIN]
            va = imgs[N_TRAIN:N_TRAIN + N_VAL]
            te = imgs[N_TRAIN + N_VAL:N_TRAIN + N_VAL + N_TEST]
            for split, chunk in (("train", tr), ("val", va), ("test", te)):
                splits[split][0].extend(chunk)
                splits[split][1].extend([label] * len(chunk))
            print(f"  rendered {cls:18s} tr={len(tr)} va={len(va)} te={len(te)}", flush=True)

    for split, (imgs, labels) in splits.items():
        X = np.stack(imgs).astype(np.uint8)
        y = np.array(labels, dtype=np.int64)
        # shuffle the split as a whole (deterministic)
        idx = np.arange(len(y))
        np.random.RandomState(SEED).shuffle(idx)
        X, y = X[idx], y[idx]
        np.save(OUT / f"{split}_images.npy", X)
        np.save(OUT / f"{split}_labels.npy", y)
        print(f"saved {split}: X={X.shape} y={y.shape}", flush=True)

    meta = {
        "classes": CLASSES,
        "n_classes": len(CLASSES),
        "n_train": N_TRAIN, "n_val": N_VAL, "n_test": N_TEST,
        "img_sz": IMG_SZ, "seed": SEED,
        "renderer": "v8_tailored.drawing_to_img (sz=224, grayscale)",
        "source": "QuickDraw simplified ndjson",
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print("meta saved. Done in %.1fs total." % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
