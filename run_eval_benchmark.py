"""
Evaluation benchmark: SD1.5 vs SDXL × Single vs Multi SigLIP2 selector.

Batch-by-stage architecture to minimize model loading/unloading:
  Stage 1: Fetch QuickDraw + BEiT classify     [BEiT loaded]
  Stage 2: DomainNet index + match              [SigLIP2 loaded]
  Stage 3a: SD1.5 generate (all samples × N)    [SD1.5 loaded]
  Stage 3b: SDXL generate (all samples × N)     [SDXL loaded]
  Stage 4: Score all candidates                 [SigLIP2 already loaded]
  Stage 5: Select + rembg                       [CPU]

Usage:
  # Quick benchmark: 5 classes × 5 samples
  python run_eval_benchmark.py --mode benchmark

  # Full evaluation: 35 classes × 16 samples (10% of QuickDraw)
  python run_eval_benchmark.py --mode full

  # Custom
  python run_eval_benchmark.py --n_classes 10 --n_samples 8 --best_of 4
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "quick_draw"))

import argparse
import json
import time
import random
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from datetime import datetime

from v8_tailored import (
    BeitSketchClassifier, SiglipScorer, DomainNetMatcher,
    fetch_quickdraw, score_prompt, remove_bg, _PROMPT_DATA,
)
from demo_beit_compare import (
    load_sd15, load_sdxl_base,
    generate_sd15, generate_sdxl,
    to_lineart_sd15, to_lineart_sdxl,
    zoom_to_content, drawing_to_img_small,
)

DEVICE = "cuda"
ALL_CLASSES = _PROMPT_DATA.get("classes", [])
OUT_DIR = Path(__file__).parent / "eval_results"

QUALITY_AXES = {
    "prompt_match":  {"template": "a high quality illustration of a {label}", "weight": 2.0},
    "no_artifacts":  {"template": "a clean flawless image without any artifacts, distortions, or deformations", "weight": 1.5},
    "anatomy":       {"template": "correct natural proportions and anatomy, physically plausible structure", "weight": 1.5},
    "sharpness":     {"template": "a sharp detailed crisp image with clear edges and fine details", "weight": 1.0},
    "composition":   {"template": "a single centered {label} on a clean simple background, isolated subject", "weight": 1.0},
    "colors":        {"template": "vibrant natural realistic colors with good lighting and contrast", "weight": 0.5},
    "professional":  {"template": "professional commercial product photography, studio quality render", "weight": 0.5},
}


def timer():
    """Context manager that returns elapsed seconds."""
    class T:
        def __enter__(self):
            self.start = time.time()
            return self
        def __exit__(self, *_):
            self.elapsed = time.time() - self.start
    return T()


def select_classes(n, seed=42):
    rng = random.Random(seed)
    return rng.sample(ALL_CLASSES, min(n, len(ALL_CLASSES)))


def siglip_multi_score(scorer, candidates, label):
    """Multi-criteria scoring. Returns (best_idx, single_scores, composite_scores, reason)."""
    axes = list(QUALITY_AXES.keys())
    texts = [QUALITY_AXES[a]["template"].format(label=label) for a in axes]
    weights = np.array([QUALITY_AXES[a]["weight"] for a in axes])

    raw = np.zeros((len(candidates), len(axes)))
    for i, cand in enumerate(candidates):
        for j, t in enumerate(texts):
            raw[i, j] = scorer.score_single(cand, t)

    normed = np.zeros_like(raw)
    for j in range(raw.shape[1]):
        col = raw[:, j]
        mn, mx = col.min(), col.max()
        normed[:, j] = (col - mn) / (mx - mn + 1e-8)

    composite = (normed * weights[None, :]).sum(axis=1)
    best_idx = int(np.argmax(composite))

    single_prompt = score_prompt(label)
    single_scores = [scorer.score_single(c, single_prompt) for c in candidates]

    return best_idx, single_scores, composite.tolist(), "multi-criteria"


def run_eval(n_classes, n_samples, best_of, seed=42, skip_domainnet=False):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "speed" if skip_domainnet else "eval"
    run_dir = OUT_DIR / f"{tag}_{run_id}_{n_classes}c_{n_samples}s"
    run_dir.mkdir(parents=True, exist_ok=True)

    classes = select_classes(n_classes, seed)
    print(f"\n{'='*60}")
    print(f"EVALUATION: {n_classes} classes × {n_samples} samples × best_of={best_of}")
    print(f"Classes: {classes}")
    print(f"Output: {run_dir}")
    print(f"{'='*60}\n")

    timings = {}
    results = []

    # ── Stage 1: Fetch QuickDraw + BEiT classification ──
    print("STAGE 1: Fetch QuickDraw + BEiT classification")
    with timer() as t1:
        beit = BeitSketchClassifier(device=DEVICE)
        samples = []
        for cls in classes:
            print(f"  Fetching {n_samples} drawings for '{cls}'...")
            drawings = fetch_quickdraw(cls, n_samples, offset=0)
            if len(drawings) < n_samples:
                print(f"    WARNING: only got {len(drawings)} drawings")
            for i, d in enumerate(drawings):
                sketch = drawing_to_img_small(d, 256)
                preds = beit.classify(sketch, top_k=5)
                label = preds[0][0]
                sample_seed = seed + hash(f"{cls}_{i}") % 100000
                samples.append({
                    "cls": cls,
                    "idx": i,
                    "sketch": sketch,
                    "label": label,
                    "beit_top1_correct": label == cls,
                    "beit_preds": [(l, round(s, 4)) for l, s in preds],
                    "seed": sample_seed,
                })
                if label != cls:
                    print(f"    [{cls}#{i}] BEiT says '{label}' (not '{cls}')")
        del beit
        torch.cuda.empty_cache()
    timings["stage1_classify"] = t1.elapsed
    print(f"  Done: {len(samples)} samples, {t1.elapsed:.1f}s")

    beit_accuracy = sum(1 for s in samples if s["beit_top1_correct"]) / len(samples)
    print(f"  BEiT top-1 accuracy: {beit_accuracy:.1%}")

    # ── Stage 2: DomainNet matching (or QuickDraw fallback for speed test) ──
    if skip_domainnet:
        print("\nSTAGE 2: SKIPPED (using QuickDraw sketches as control images — SPEED TEST ONLY)")
        with timer() as t2:
            scorer = SiglipScorer(device=DEVICE)
            for s in samples:
                sketch_1024 = zoom_to_content(s["sketch"].resize((1024, 1024), Image.LANCZOS), 1024, fill_pct=0.9)
                s["dn_matches"] = [sketch_1024] * best_of
                s["dn_sims"] = [0.0] * best_of
        timings["stage2_domainnet"] = t2.elapsed
        print(f"  SigLIP2 loaded in {t2.elapsed:.1f}s (no DomainNet matching)")
    else:
        print("\nSTAGE 2: DomainNet matching")
        with timer() as t2:
            scorer = SiglipScorer(device=DEVICE)
            unique_labels = list(set(s["label"] for s in samples))
            print(f"  Building index for {len(unique_labels)} unique labels...")
            matcher = DomainNetMatcher(scorer, unique_labels, batch_size=16)
            for s in samples:
                top_matches = matcher.match_topk(s["sketch"], s["label"], k=best_of)
                s["dn_matches"] = [
                    zoom_to_content(img, 1024, fill_pct=0.9)
                    for img, _, _ in top_matches
                ]
                s["dn_sims"] = [round(sim, 4) for _, _, sim in top_matches]
            del matcher
        timings["stage2_domainnet"] = t2.elapsed
        print(f"  Done: {t2.elapsed:.1f}s")

    # ── Stage 3a: SD1.5 generation ──
    print("\nSTAGE 3a: SD1.5 generation")
    with timer() as t3a:
        sd15_pipe, sd15_refiner = load_sd15(DEVICE)
        total_gens = len(samples) * best_of
        done = 0
        for s in samples:
            s["sd15_candidates"] = []
            s["sd15_linearts"] = []
            for i in range(best_of):
                dn_img = s["dn_matches"][i] if i < len(s["dn_matches"]) else s["dn_matches"][0]
                dn_resized = dn_img.resize((512, 512), Image.LANCZOS)
                ctrl = to_lineart_sd15(dn_resized)
                cand = generate_sd15(sd15_pipe, sd15_refiner, ctrl, s["label"], s["seed"] + i, DEVICE)
                s["sd15_candidates"].append(cand)
                s["sd15_linearts"].append(ctrl)
                done += 1
                if done % 10 == 0 or done == total_gens:
                    print(f"  SD1.5: {done}/{total_gens} ({done*100//total_gens}%)")
        del sd15_pipe, sd15_refiner
        torch.cuda.empty_cache()
    timings["stage3a_sd15"] = t3a.elapsed
    print(f"  Done: {t3a.elapsed:.1f}s ({t3a.elapsed/total_gens:.1f}s per image)")

    # ── Stage 3b: SDXL generation ──
    print("\nSTAGE 3b: SDXL generation")
    with timer() as t3b:
        sdxl_pipe, sdxl_refiner = load_sdxl_base(DEVICE)
        done = 0
        for s in samples:
            s["sdxl_candidates"] = []
            s["sdxl_linearts"] = []
            for i in range(best_of):
                dn_img = s["dn_matches"][i] if i < len(s["dn_matches"]) else s["dn_matches"][0]
                dn_resized = dn_img.resize((1024, 1024), Image.LANCZOS)
                ctrl = to_lineart_sdxl(dn_resized)
                cand = generate_sdxl(sdxl_pipe, sdxl_refiner, ctrl, s["label"], s["seed"] + i, DEVICE)
                s["sdxl_candidates"].append(cand)
                s["sdxl_linearts"].append(ctrl)
                done += 1
                if done % 10 == 0 or done == total_gens:
                    print(f"  SDXL: {done}/{total_gens} ({done*100//total_gens}%)")
        del sdxl_pipe, sdxl_refiner
        torch.cuda.empty_cache()
    timings["stage3b_sdxl"] = t3b.elapsed
    print(f"  Done: {t3b.elapsed:.1f}s ({t3b.elapsed/total_gens:.1f}s per image)")

    # ── Stage 4: Scoring all candidates ──
    print("\nSTAGE 4: Scoring (SigLIP2 single + multi)")
    with timer() as t4:
        # scorer is still loaded from stage 2
        for s in samples:
            label = s["label"]

            # SD1.5 - single SigLIP2
            prompt = score_prompt(label)
            s["sd15_single_scores"] = [scorer.score_single(c, prompt) for c in s["sd15_candidates"]]
            s["sd15_single_pick"] = int(np.argmax(s["sd15_single_scores"]))

            # SD1.5 - multi SigLIP2
            pick, _, composite, _ = siglip_multi_score(scorer, s["sd15_candidates"], label)
            s["sd15_multi_pick"] = pick
            s["sd15_multi_composite"] = composite

            # SDXL - single SigLIP2
            s["sdxl_single_scores"] = [scorer.score_single(c, prompt) for c in s["sdxl_candidates"]]
            s["sdxl_single_pick"] = int(np.argmax(s["sdxl_single_scores"]))

            # SDXL - multi SigLIP2
            pick, _, composite, _ = siglip_multi_score(scorer, s["sdxl_candidates"], label)
            s["sdxl_multi_pick"] = pick
            s["sdxl_multi_composite"] = composite

        del scorer
        torch.cuda.empty_cache()
    timings["stage4_scoring"] = t4.elapsed
    print(f"  Done: {t4.elapsed:.1f}s")

    # ── Stage 5: BG removal + save results ──
    print("\nSTAGE 5: Background removal + saving")
    with timer() as t5:
        for si, s in enumerate(samples):
            sample_dir = run_dir / f"{s['cls']}_{s['idx']}"
            sample_dir.mkdir(exist_ok=True)

            s["sketch"].save(sample_dir / "sketch.png")

            configs = [
                ("sd15_single", s["sd15_candidates"], s["sd15_single_pick"], s["sd15_single_scores"]),
                ("sd15_multi", s["sd15_candidates"], s["sd15_multi_pick"], s["sd15_single_scores"]),
                ("sdxl_single", s["sdxl_candidates"], s["sdxl_single_pick"], s["sdxl_single_scores"]),
                ("sdxl_multi", s["sdxl_candidates"], s["sdxl_multi_pick"], s["sdxl_single_scores"]),
            ]

            sample_results = {
                "cls": s["cls"],
                "label": s["label"],
                "beit_correct": s["beit_top1_correct"],
                "beit_preds": s["beit_preds"],
                "seed": s["seed"],
                "dn_sims": s["dn_sims"],
            }

            for name, cands, pick, scores in configs:
                winner = cands[pick]
                final = remove_bg(winner)
                final.save(sample_dir / f"{name}_final.png")
                winner.save(sample_dir / f"{name}_raw.png")

                sample_results[name] = {
                    "pick_idx": pick,
                    "scores": [round(float(x), 4) for x in scores],
                    "best_score": round(float(scores[pick]), 4),
                }

            # Save all candidates for inspection
            for i, c in enumerate(s["sd15_candidates"]):
                c.save(sample_dir / f"sd15_cand_{i}.png")
            for i, c in enumerate(s["sdxl_candidates"]):
                c.save(sample_dir / f"sdxl_cand_{i}.png")

            results.append(sample_results)

            if (si + 1) % 5 == 0 or si == len(samples) - 1:
                print(f"  Saved: {si+1}/{len(samples)}")

    timings["stage5_rembg_save"] = t5.elapsed
    print(f"  Done: {t5.elapsed:.1f}s")

    # ── Aggregate metrics ──
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")

    metrics = {
        "config": {
            "n_classes": n_classes,
            "n_samples": n_samples,
            "best_of": best_of,
            "seed": seed,
            "classes": classes,
            "total_samples": len(samples),
            "timestamp": datetime.now().isoformat(),
        },
        "timings": {k: round(v, 2) for k, v in timings.items()},
        "timings_total": round(sum(timings.values()), 2),
    }

    # Per-config aggregates
    for config_name in ["sd15_single", "sd15_multi", "sdxl_single", "sdxl_multi"]:
        scores_best = [r[config_name]["best_score"] for r in results]
        scores_all = []
        for r in results:
            scores_all.extend(r[config_name]["scores"])
        metrics[config_name] = {
            "mean_best_score": round(float(np.mean(scores_best)), 4),
            "std_best_score": round(float(np.std(scores_best)), 4),
            "median_best_score": round(float(np.median(scores_best)), 4),
            "mean_all_scores": round(float(np.mean(scores_all)), 4),
        }
        print(f"  {config_name:15s}: mean_best={np.mean(scores_best):.4f} ± {np.std(scores_best):.4f}  median={np.median(scores_best):.4f}")

    # Agreement between single and multi selectors
    sd15_agree = sum(1 for r in results if r["sd15_single"]["pick_idx"] == r["sd15_multi"]["pick_idx"])
    sdxl_agree = sum(1 for r in results if r["sdxl_single"]["pick_idx"] == r["sdxl_multi"]["pick_idx"])
    metrics["selector_agreement"] = {
        "sd15": round(sd15_agree / len(results), 4),
        "sdxl": round(sdxl_agree / len(results), 4),
    }
    print(f"\n  Selector agreement (single vs multi):")
    print(f"    SD1.5: {sd15_agree}/{len(results)} ({sd15_agree*100//len(results)}%)")
    print(f"    SDXL:  {sdxl_agree}/{len(results)} ({sdxl_agree*100//len(results)}%)")

    # BEiT accuracy
    metrics["beit_top1_accuracy"] = round(beit_accuracy, 4)
    print(f"\n  BEiT top-1 accuracy: {beit_accuracy:.1%}")

    # Timing summary
    total = sum(timings.values())
    print(f"\n  Timings:")
    for stage, t in timings.items():
        print(f"    {stage:25s}: {t:7.1f}s ({t*100/total:4.1f}%)")
    print(f"    {'TOTAL':25s}: {total:7.1f}s")

    # Estimate full run
    if n_classes < len(ALL_CLASSES):
        full_classes = int(len(ALL_CLASSES) * 0.1)
        full_samples = 16
        scale = (full_classes * full_samples) / (n_classes * n_samples)
        est_hours = total * scale / 3600
        print(f"\n  ESTIMATED full run ({full_classes}c × {full_samples}s): {est_hours:.1f} hours")
        metrics["estimated_full_run_hours"] = round(est_hours, 2)

    # Save
    metrics["per_sample"] = results
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Results saved to: {run_dir}/metrics.json")

    return metrics, run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sketch Bomb evaluation benchmark")
    parser.add_argument("--mode", choices=["benchmark", "full"], default="benchmark")
    parser.add_argument("--n_classes", type=int, default=None)
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--best_of", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_domainnet", action="store_true",
                        help="Use QuickDraw sketches instead of DomainNet (speed test only!)")
    args = parser.parse_args()

    if args.mode == "benchmark":
        nc = args.n_classes or 5
        ns = args.n_samples or 5
    else:
        nc = args.n_classes or 35
        ns = args.n_samples or 16

    run_eval(nc, ns, args.best_of, args.seed, skip_domainnet=args.skip_domainnet)
