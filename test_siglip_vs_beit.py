"""Quick A/B test: BEiT vs SigLIP2 zero-shot for Quick Draw sketch classification."""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "quick_draw"))

import torch
import numpy as np
from v8_tailored import BeitSketchClassifier, SiglipScorer, fetch_quickdraw, drawing_to_img

DEVICE = "cuda"
N_SAMPLES = 8
CLASSES = [
    "cookie", "broom", "violin", "camel", "steak",
    "trumpet", "bread", "banana", "arm", "flying saucer",
    "asparagus", "drums", "fish", "aircraft carrier", "backpack",
    "hat", "spider", "swing set", "telephone", "parachute",
    "sea turtle", "passport", "fence", "pliers", "hand",
    "fan", "face", "chair", "butterfly", "sword",
]

# Load all 345 class names for SigLIP2 zero-shot
with open("quick_draw/prompts.json") as f:
    ALL_CLASSES = json.load(f)["classes"]

print(f"Testing {len(CLASSES)} classes x {N_SAMPLES} samples = {len(CLASSES)*N_SAMPLES} sketches")
print(f"SigLIP2 zero-shot over {len(ALL_CLASSES)} classes\n")

# Load models
print("Loading BEiT...", end=" ", flush=True)
beit = BeitSketchClassifier(device=DEVICE)
print("done")

print("Loading SigLIP2...", end=" ", flush=True)
scorer = SiglipScorer(device=DEVICE)
print("done")

# Pre-compute SigLIP2 text embeddings for all 345 classes
print("Caching SigLIP2 text embeddings for 345 classes...", end=" ", flush=True)
t0 = time.time()

# Process in batches to avoid OOM
text_embeds_list = []
prompt_templates = ["a sketch of a {}", "a drawing of a {}", "a {} sketch"]
# Use simple template first
prompts = [f"a sketch of a {c}" for c in ALL_CLASSES]
BATCH = 64
for i in range(0, len(prompts), BATCH):
    batch_prompts = prompts[i:i+BATCH]
    inputs = scorer.processor(text=batch_prompts, padding="max_length", return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = scorer.model.get_text_features(**{k: v for k, v in inputs.items() if k in ["input_ids", "attention_mask"]})
    text_embeds_list.append(out.pooler_output)

text_embeds = torch.cat(text_embeds_list, dim=0)  # (345, dim)
text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
print(f"done ({time.time()-t0:.1f}s, shape={text_embeds.shape})")


def siglip_classify(image, top_k=5):
    """Zero-shot classify using SigLIP2."""
    inputs = scorer.processor(images=[image], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = scorer.model.get_image_features(**{k: v for k, v in inputs.items() if k in ["pixel_values"]})
    img_embed = out.pooler_output
    img_embed = img_embed / img_embed.norm(dim=-1, keepdim=True)
    sims = (img_embed @ text_embeds.T).squeeze(0)
    topk = torch.topk(sims, top_k)
    return [(ALL_CLASSES[i], float(s)) for s, i in zip(topk.values, topk.indices)]


# Test
beit_correct = 0
siglip_correct = 0
siglip_top3_correct = 0
beit_top3_correct = 0
total = 0
per_class = {}

for cls in CLASSES:
    raw_drawings = fetch_quickdraw(cls, N_SAMPLES, offset=0)
    cls_beit = 0
    cls_siglip = 0

    for i, raw in enumerate(raw_drawings):
        sketch = drawing_to_img(raw, sz=256)
        total += 1

        # BEiT
        beit_preds = beit.classify(sketch, top_k=5)
        beit_top1 = beit_preds[0][0]
        beit_top3_labels = [p[0] for p in beit_preds[:3]]
        if beit_top1 == cls:
            beit_correct += 1
            cls_beit += 1
        if cls in beit_top3_labels:
            beit_top3_correct += 1

        # SigLIP2
        siglip_preds = siglip_classify(sketch, top_k=5)
        siglip_top1 = siglip_preds[0][0]
        siglip_top3_labels = [p[0] for p in siglip_preds[:3]]
        if siglip_top1 == cls:
            siglip_correct += 1
            cls_siglip += 1
        if cls in siglip_top3_labels:
            siglip_top3_correct += 1

        # Print disagreements
        if beit_top1 != siglip_top1:
            b_mark = "✓" if beit_top1 == cls else "✗"
            s_mark = "✓" if siglip_top1 == cls else "✗"
            print(f"  {cls:20s}[{i}]: BEiT={beit_top1:20s}{b_mark}  SigLIP={siglip_top1:20s}{s_mark}")

    per_class[cls] = {"beit": cls_beit, "siglip": cls_siglip, "n": N_SAMPLES}

print(f"\n{'='*60}")
print(f"{'RESULTS':^60}")
print(f"{'='*60}")
print(f"Total samples: {total}")
print(f"")
print(f"{'Metric':<30} {'BEiT':>10} {'SigLIP2':>10}")
print(f"{'-'*50}")
print(f"{'Top-1 accuracy':<30} {beit_correct/total:>10.1%} {siglip_correct/total:>10.1%}")
print(f"{'Top-3 accuracy':<30} {beit_top3_correct/total:>10.1%} {siglip_top3_correct/total:>10.1%}")
print(f"{'Correct count':<30} {beit_correct:>10d} {siglip_correct:>10d}")

print(f"\n{'Per-class comparison':}")
print(f"{'Class':<22} {'BEiT':>6} {'SigLIP':>6} {'Winner':>8}")
print(f"{'-'*44}")
for cls in sorted(per_class.keys(), key=lambda c: per_class[c]["siglip"] - per_class[c]["beit"], reverse=True):
    d = per_class[cls]
    winner = "SigLIP" if d["siglip"] > d["beit"] else "BEiT" if d["beit"] > d["siglip"] else "tie"
    print(f"{cls:<22} {d['beit']:>5}/{d['n']} {d['siglip']:>5}/{d['n']} {winner:>8}")

# Summary
siglip_wins = sum(1 for d in per_class.values() if d["siglip"] > d["beit"])
beit_wins = sum(1 for d in per_class.values() if d["beit"] > d["siglip"])
ties = sum(1 for d in per_class.values() if d["beit"] == d["siglip"])
print(f"\nClass-level: SigLIP wins {siglip_wins}, BEiT wins {beit_wins}, ties {ties}")
