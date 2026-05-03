"""Evaluate SDXL-Lightning with best-of-4 selection.
Uses same test set as eval_domainnet.py for fair comparison.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "quick_draw"))

import json
import time
import random
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from datetime import datetime

from v8_tailored import (
    BeitSketchClassifier, SiglipScorer, DomainNetMatcher,
    fetch_quickdraw, score_prompt, _PROMPT_DATA,
)
from demo_beit_compare import (
    load_sdxl_lightning, generate_lightning,
    to_lineart_sdxl, zoom_to_content, drawing_to_img_small,
)

DEVICE = "cuda"
ALL_CLASSES = _PROMPT_DATA.get("classes", [])
OUT_DIR = Path(__file__).parent / "eval_results"


def timer():
    class T:
        def __enter__(self): self.start = time.time(); return self
        def __exit__(self, *_): self.elapsed = time.time() - self.start
    return T()


def run_eval(n_classes=10, n_samples=8, best_of=4, seed=42):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUT_DIR / f"lightning_eval_{run_id}_{n_classes}c_{n_samples}s"
    run_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    classes = rng.sample(ALL_CLASSES, min(n_classes, len(ALL_CLASSES)))
    total_gens = n_classes * n_samples * best_of

    print(f"\n{'='*60}")
    print(f"SDXL-LIGHTNING EVAL: {n_classes}c × {n_samples}s × best_of={best_of}")
    print(f"Total generations: {total_gens}")
    print(f"Classes: {classes}")
    print(f"Output: {run_dir}")
    print(f"{'='*60}\n")

    timings = {}

    # Stage 1: Fetch + BEiT
    print("STAGE 1: Fetch QuickDraw + BEiT")
    with timer() as t1:
        beit = BeitSketchClassifier(device=DEVICE)
        samples = []
        for cls in classes:
            print(f"  Fetching '{cls}'...")
            drawings = fetch_quickdraw(cls, n_samples, offset=0)
            for i, d in enumerate(drawings):
                sketch = drawing_to_img_small(d, 256)
                preds = beit.classify(sketch, top_k=5)
                label = preds[0][0]
                sample_seed = seed + hash(f"{cls}_{i}") % 100000
                samples.append({
                    "cls": cls, "idx": i, "sketch": sketch, "label": label,
                    "beit_correct": label == cls,
                    "beit_preds": [(l, round(s, 4)) for l, s in preds],
                    "seed": sample_seed,
                })
        del beit
        torch.cuda.empty_cache()
    timings["fetch_beit"] = t1.elapsed
    beit_acc = sum(1 for s in samples if s["beit_correct"]) / len(samples)
    print(f"  {t1.elapsed:.1f}s, BEiT accuracy: {beit_acc:.1%}")

    # Stage 2: DomainNet matching
    print("\nSTAGE 2: DomainNet matching")
    with timer() as t2:
        scorer = SiglipScorer(device=DEVICE)
        unique_labels = list(set(s["label"] for s in samples))
        print(f"  Building index for {len(unique_labels)} labels...")
        matcher = DomainNetMatcher(scorer, unique_labels, batch_size=16)
        for s in samples:
            top_matches = matcher.match_topk(s["sketch"], s["label"], k=best_of)
            s["dn_matches"] = [zoom_to_content(img, 1024, fill_pct=0.9) for img, _, _ in top_matches]
            s["dn_sims"] = [round(sim, 4) for _, _, sim in top_matches]
        del matcher
    timings["domainnet"] = t2.elapsed
    print(f"  {t2.elapsed:.1f}s")

    # Stage 3: Lightning generation
    print("\nSTAGE 3: SDXL-Lightning generation")
    with timer() as t3:
        pipe, _ = load_sdxl_lightning(DEVICE)

        # Warmup
        dummy_ctrl = to_lineart_sdxl(samples[0]["dn_matches"][0].resize((1024, 1024), Image.LANCZOS))
        _ = generate_lightning(pipe, None, dummy_ctrl, samples[0]["label"], 0, DEVICE)
        print("  Warmup done.")

        print(f"  Generating {total_gens} images...")
        gen_count = 0
        for s in samples:
            s["candidates"] = []
            for i in range(best_of):
                dn_img = s["dn_matches"][i] if i < len(s["dn_matches"]) else s["dn_matches"][0]
                ctrl = to_lineart_sdxl(dn_img.resize((1024, 1024), Image.LANCZOS))
                img = generate_lightning(pipe, None, ctrl, s["label"], s["seed"] + i, DEVICE)
                s["candidates"].append(img)
                gen_count += 1
                if gen_count % 20 == 0:
                    print(f"    {gen_count}/{total_gens}")

        del pipe
        torch.cuda.empty_cache()
    timings["lightning_gen"] = t3.elapsed
    print(f"  {t3.elapsed:.1f}s ({t3.elapsed/total_gens:.2f}s/img)")

    # Stage 4: SigLIP2 scoring
    print("\nSTAGE 4: SigLIP2 scoring")
    with timer() as t4:
        for s in samples:
            prompt = score_prompt(s["label"])
            s["scores"] = [scorer.score_single(c, prompt) for c in s["candidates"]]
            s["pick"] = int(np.argmax(s["scores"]))
        del scorer
        torch.cuda.empty_cache()
    timings["scoring"] = t4.elapsed
    print(f"  {t4.elapsed:.1f}s")

    # Stage 5: Save
    print("\nSTAGE 5: Saving results")
    results_json = []
    for si, s in enumerate(samples):
        sdir = run_dir / f"{s['cls']}_{s['idx']}"
        sdir.mkdir(exist_ok=True)
        s["sketch"].save(sdir / "sketch.png")
        for ci, c in enumerate(s["candidates"]):
            c.save(sdir / f"cand_{ci}.png")
        s["candidates"][s["pick"]].save(sdir / "winner.png")

        r = {
            "cls": s["cls"], "label": s["label"], "beit_correct": s["beit_correct"],
            "beit_preds": s["beit_preds"], "seed": s["seed"], "dn_sims": s["dn_sims"],
            "pick_idx": s["pick"],
            "scores": [round(float(x), 4) for x in s["scores"]],
            "best_score": round(float(s["scores"][s["pick"]]), 4),
            "mean_score": round(float(np.mean(s["scores"])), 4),
        }
        results_json.append(r)

    # Metrics
    best_scores = [r["best_score"] for r in results_json]
    all_scores = []
    for r in results_json:
        all_scores.extend(r["scores"])

    metrics = {
        "config": {
            "n_classes": n_classes, "n_samples": n_samples, "best_of": best_of,
            "seed": seed, "model": "sdxl-lightning-4step",
            "classes": classes, "total_samples": len(samples),
            "timestamp": datetime.now().isoformat(),
        },
        "timings": {k: round(v, 2) for k, v in timings.items()},
        "timings_total": round(sum(timings.values()), 2),
        "beit_accuracy": round(beit_acc, 4),
        "mean_best": round(float(np.mean(best_scores)), 4),
        "std_best": round(float(np.std(best_scores)), 4),
        "median_best": round(float(np.median(best_scores)), 4),
        "mean_all": round(float(np.mean(all_scores)), 4),
        "std_all": round(float(np.std(all_scores)), 4),
        "per_sample": results_json,
    }

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Print results
    print(f"\n{'='*60}")
    print("SDXL-LIGHTNING RESULTS")
    print(f"{'='*60}")
    print(f"  Mean best (best-of-{best_of}): {np.mean(best_scores):+.4f} ± {np.std(best_scores):.4f}")
    print(f"  Mean all (no selection):  {np.mean(all_scores):+.4f} ± {np.std(all_scores):.4f}")
    print(f"  Best-of gain:             {np.mean(best_scores) - np.mean(all_scores):+.4f}")
    print(f"  Speed: {timings['lightning_gen']/total_gens:.2f}s/img")
    print()
    print("  COMPARISON (from previous eval, same seed/classes):")
    print(f"  SD1.5 best-of-4:   +1.4382 @ 1.81s/img")
    print(f"  SDXL  best-of-4:   +1.1917 @ 13.79s/img")
    print(f"  Lightning best-of-{best_of}: {np.mean(best_scores):+.4f} @ {timings['lightning_gen']/total_gens:.2f}s/img")

    # Per-class
    print(f"\n  Per-class breakdown:")
    by_cls = {}
    for r in results_json:
        by_cls.setdefault(r["cls"], []).append(r["best_score"])
    for cls in sorted(by_cls.keys()):
        print(f"    {cls:15s}: {np.mean(by_cls[cls]):+.4f}")

    print(f"\n  Results saved to: {run_dir}")
    return metrics, run_dir


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--classes", type=int, default=10)
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--best-of", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run_eval(args.classes, args.samples, args.best_of, args.seed)
