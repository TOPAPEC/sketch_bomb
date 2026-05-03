"""Real DomainNet evaluation: SD1.5 vs SDXL × Single vs Multi SigLIP2.
Batch-by-stage, optimized (compile+batch+20steps), skip rembg for speed.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "quick_draw"))

import argparse
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
    load_sd15, load_sdxl_base,
    generate_sd15, generate_sdxl,
    to_lineart_sd15, to_lineart_sdxl,
    zoom_to_content, drawing_to_img_small,
    get_prompt, get_negative_sd15, get_negative_sdxl,
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
    class T:
        def __enter__(self): self.start = time.time(); return self
        def __exit__(self, *_): self.elapsed = time.time() - self.start
    return T()


def select_classes(n, seed=42):
    rng = random.Random(seed)
    return rng.sample(ALL_CLASSES, min(n, len(ALL_CLASSES)))


def siglip_multi_score(scorer, candidates, label):
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
    return int(np.argmax(composite)), composite.tolist()


def batched_generate_sd15(pipe, refiner, all_ctrls, all_labels, all_seeds, bs=4, steps=20, use_refiner=True):
    """Generate images in batches."""
    results = []
    for start in range(0, len(all_ctrls), bs):
        batch_ctrls = all_ctrls[start:start+bs]
        batch_labels = all_labels[start:start+bs]
        batch_seeds = all_seeds[start:start+bs]

        prompts = [get_prompt(l) for l in batch_labels]
        negs = [get_negative_sd15(l) for l in batch_labels]
        gens = [torch.Generator(device=DEVICE).manual_seed(s) for s in batch_seeds]

        imgs = pipe(
            prompt=prompts, negative_prompt=negs, image=batch_ctrls,
            num_inference_steps=steps, guidance_scale=7.5,
            controlnet_conditioning_scale=0.7,
            control_guidance_start=0.0, control_guidance_end=1.0,
            generator=gens, num_images_per_prompt=1,
        ).images

        if use_refiner:
            ref_gens = [torch.Generator(device=DEVICE).manual_seed(s) for s in batch_seeds]
            imgs = refiner(
                prompt=prompts, negative_prompt=negs, image=imgs,
                strength=0.25, num_inference_steps=max(5, steps//2),
                guidance_scale=4.0, generator=ref_gens,
            ).images
        results.extend(imgs)
    return results


def batched_generate_sdxl(pipe, refiner, all_ctrls, all_labels, all_seeds, bs=2, steps=20, use_refiner=True):
    """Generate images in batches (smaller BS for SDXL)."""
    results = []
    for start in range(0, len(all_ctrls), bs):
        batch_ctrls = all_ctrls[start:start+bs]
        batch_labels = all_labels[start:start+bs]
        batch_seeds = all_seeds[start:start+bs]

        prompts = [get_prompt(l) for l in batch_labels]
        negs = [get_negative_sdxl(l) for l in batch_labels]
        gens = [torch.Generator(device=DEVICE).manual_seed(s) for s in batch_seeds]

        imgs = pipe(
            prompt=prompts, negative_prompt=negs, image=batch_ctrls,
            num_inference_steps=steps, guidance_scale=7.0,
            controlnet_conditioning_scale=0.5,
            control_guidance_start=0.0, control_guidance_end=1.0,
            generator=gens, num_images_per_prompt=1,
        ).images

        if use_refiner:
            ref_gens = [torch.Generator(device=DEVICE).manual_seed(s) for s in batch_seeds]
            imgs = refiner(
                prompt=prompts, negative_prompt=negs, image=imgs,
                strength=0.25, num_inference_steps=max(5, steps//2),
                guidance_scale=4.0, generator=ref_gens,
            ).images
        results.extend(imgs)
    return results


def img_b64(img):
    import io, base64
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def run_eval(n_classes, n_samples, best_of=4, seed=42, steps=20, do_compile=True):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUT_DIR / f"dn_eval_{run_id}_{n_classes}c_{n_samples}s"
    run_dir.mkdir(parents=True, exist_ok=True)

    classes = select_classes(n_classes, seed)
    total_gens_per_model = n_classes * n_samples * best_of

    print(f"\n{'='*60}")
    print(f"DOMAINNET EVAL: {n_classes}c × {n_samples}s × best_of={best_of}")
    print(f"Steps={steps}, compile={do_compile}")
    print(f"Total gens per model: {total_gens_per_model}")
    print(f"Classes: {classes}")
    print(f"Output: {run_dir}")
    print(f"{'='*60}\n")

    timings = {}

    # ── Stage 1: Fetch + BEiT ──
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

    # ── Stage 2: DomainNet matching ──
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

    # ── Stage 3a: SD1.5 generation ──
    print("\nSTAGE 3a: SD1.5 generation")
    with timer() as t3a:
        pipe15, ref15 = load_sd15(DEVICE)
        if do_compile:
            print("  Compiling...")
            pipe15.unet = torch.compile(pipe15.unet, mode="reduce-overhead")
            pipe15.controlnet = torch.compile(pipe15.controlnet, mode="reduce-overhead")
            ref15.unet = torch.compile(ref15.unet, mode="reduce-overhead")
            # Warmup
            dummy = [samples[0]["dn_matches"][0].resize((512,512), Image.LANCZOS)]
            dummy_ctrl = [to_lineart_sd15(dummy[0])]
            _ = pipe15(prompt=[get_prompt(samples[0]["label"])],
                       negative_prompt=[get_negative_sd15(samples[0]["label"])],
                       image=dummy_ctrl, num_inference_steps=steps, guidance_scale=7.5,
                       controlnet_conditioning_scale=0.7, generator=[torch.Generator(device=DEVICE).manual_seed(0)],
                       num_images_per_prompt=1).images
            _ = ref15(prompt=[get_prompt(samples[0]["label"])],
                      negative_prompt=[get_negative_sd15(samples[0]["label"])],
                      image=_, strength=0.25, num_inference_steps=max(5,steps//2), guidance_scale=4.0,
                      generator=[torch.Generator(device=DEVICE).manual_seed(0)]).images
            print("  Compiled.")

        # Prepare all controls
        all_ctrls, all_labels, all_seeds, all_sample_idx = [], [], [], []
        for si, s in enumerate(samples):
            for i in range(best_of):
                dn_img = s["dn_matches"][i] if i < len(s["dn_matches"]) else s["dn_matches"][0]
                ctrl = to_lineart_sd15(dn_img.resize((512, 512), Image.LANCZOS))
                all_ctrls.append(ctrl)
                all_labels.append(s["label"])
                all_seeds.append(s["seed"] + i)
                all_sample_idx.append(si)

        print(f"  Generating {len(all_ctrls)} images (BS=4)...")
        all_imgs = batched_generate_sd15(pipe15, ref15, all_ctrls, all_labels, all_seeds, bs=4, steps=steps)

        # Distribute back
        idx = 0
        for s in samples:
            s["sd15_candidates"] = []
            for i in range(best_of):
                s["sd15_candidates"].append(all_imgs[idx])
                idx += 1

        del pipe15, ref15
        torch.cuda.empty_cache()
    timings["sd15_gen"] = t3a.elapsed
    print(f"  {t3a.elapsed:.1f}s ({t3a.elapsed/total_gens_per_model:.2f}s/img incl compile)")

    # ── Stage 3b: SDXL generation ──
    print("\nSTAGE 3b: SDXL generation")
    with timer() as t3b:
        pipexl, refxl = load_sdxl_base(DEVICE)
        if do_compile:
            print("  Compiling...")
            pipexl.unet = torch.compile(pipexl.unet, mode="reduce-overhead")
            pipexl.controlnet = torch.compile(pipexl.controlnet, mode="reduce-overhead")
            refxl.unet = torch.compile(refxl.unet, mode="reduce-overhead")
            dummy = [samples[0]["dn_matches"][0].resize((1024,1024), Image.LANCZOS)]
            dummy_ctrl = [to_lineart_sdxl(dummy[0])]
            _ = pipexl(prompt=[get_prompt(samples[0]["label"])],
                       negative_prompt=[get_negative_sdxl(samples[0]["label"])],
                       image=dummy_ctrl, num_inference_steps=steps, guidance_scale=7.0,
                       controlnet_conditioning_scale=0.5, generator=[torch.Generator(device=DEVICE).manual_seed(0)],
                       num_images_per_prompt=1).images
            _ = refxl(prompt=[get_prompt(samples[0]["label"])],
                      negative_prompt=[get_negative_sdxl(samples[0]["label"])],
                      image=_, strength=0.25, num_inference_steps=max(5,steps//2), guidance_scale=4.0,
                      generator=[torch.Generator(device=DEVICE).manual_seed(0)]).images
            print("  Compiled.")

        all_ctrls, all_labels, all_seeds = [], [], []
        for s in samples:
            for i in range(best_of):
                dn_img = s["dn_matches"][i] if i < len(s["dn_matches"]) else s["dn_matches"][0]
                ctrl = to_lineart_sdxl(dn_img.resize((1024, 1024), Image.LANCZOS))
                all_ctrls.append(ctrl)
                all_labels.append(s["label"])
                all_seeds.append(s["seed"] + i)

        print(f"  Generating {len(all_ctrls)} images (BS=2)...")
        all_imgs = batched_generate_sdxl(pipexl, refxl, all_ctrls, all_labels, all_seeds, bs=2, steps=steps)

        idx = 0
        for s in samples:
            s["sdxl_candidates"] = []
            for i in range(best_of):
                s["sdxl_candidates"].append(all_imgs[idx])
                idx += 1

        del pipexl, refxl
        torch.cuda.empty_cache()
    timings["sdxl_gen"] = t3b.elapsed
    print(f"  {t3b.elapsed:.1f}s ({t3b.elapsed/total_gens_per_model:.2f}s/img incl compile)")

    # ── Stage 4: SigLIP2 scoring ──
    print("\nSTAGE 4: SigLIP2 scoring")
    with timer() as t4:
        for s in samples:
            label = s["label"]
            prompt = score_prompt(label)

            # SD1.5
            s["sd15_single_scores"] = [scorer.score_single(c, prompt) for c in s["sd15_candidates"]]
            s["sd15_single_pick"] = int(np.argmax(s["sd15_single_scores"]))
            pick_m, comp = siglip_multi_score(scorer, s["sd15_candidates"], label)
            s["sd15_multi_pick"] = pick_m
            s["sd15_multi_composite"] = comp

            # SDXL
            s["sdxl_single_scores"] = [scorer.score_single(c, prompt) for c in s["sdxl_candidates"]]
            s["sdxl_single_pick"] = int(np.argmax(s["sdxl_single_scores"]))
            pick_m, comp = siglip_multi_score(scorer, s["sdxl_candidates"], label)
            s["sdxl_multi_pick"] = pick_m
            s["sdxl_multi_composite"] = comp

        del scorer
        torch.cuda.empty_cache()
    timings["scoring"] = t4.elapsed
    print(f"  {t4.elapsed:.1f}s")

    # ── Stage 5: Save results (no rembg for speed) ──
    print("\nSTAGE 5: Saving results")
    with timer() as t5:
        results_json = []
        for si, s in enumerate(samples):
            sdir = run_dir / f"{s['cls']}_{s['idx']}"
            sdir.mkdir(exist_ok=True)
            s["sketch"].save(sdir / "sketch.png")

            r = {
                "cls": s["cls"], "label": s["label"], "beit_correct": s["beit_correct"],
                "beit_preds": s["beit_preds"], "seed": s["seed"], "dn_sims": s["dn_sims"],
            }

            for model in ["sd15", "sdxl"]:
                for selector in ["single", "multi"]:
                    key = f"{model}_{selector}"
                    pick = s[f"{key}_pick"]
                    scores = s[f"{model}_single_scores"]
                    winner = s[f"{model}_candidates"][pick]
                    winner.save(sdir / f"{key}_winner.png")
                    r[key] = {
                        "pick_idx": pick,
                        "scores": [round(float(x), 4) for x in scores],
                        "best_score": round(float(scores[pick]), 4),
                    }

                # Save all candidates
                for ci, c in enumerate(s[f"{model}_candidates"]):
                    c.save(sdir / f"{model}_cand_{ci}.png")

            results_json.append(r)

            if (si + 1) % 10 == 0 or si == len(samples) - 1:
                print(f"  {si+1}/{len(samples)}")

    timings["save"] = t5.elapsed

    # ── Metrics ──
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")

    metrics = {
        "config": {
            "n_classes": n_classes, "n_samples": n_samples, "best_of": best_of,
            "seed": seed, "steps": steps, "compile": do_compile,
            "classes": classes, "total_samples": len(samples),
            "timestamp": datetime.now().isoformat(),
        },
        "timings": {k: round(v, 2) for k, v in timings.items()},
        "timings_total": round(sum(timings.values()), 2),
        "beit_accuracy": round(beit_acc, 4),
    }

    for config_name in ["sd15_single", "sd15_multi", "sdxl_single", "sdxl_multi"]:
        best_scores = [r[config_name]["best_score"] for r in results_json]
        all_scores = []
        for r in results_json:
            all_scores.extend(r[config_name]["scores"])
        metrics[config_name] = {
            "mean_best": round(float(np.mean(best_scores)), 4),
            "std_best": round(float(np.std(best_scores)), 4),
            "median_best": round(float(np.median(best_scores)), 4),
            "mean_all": round(float(np.mean(all_scores)), 4),
        }
        print(f"  {config_name:15s}: mean={np.mean(best_scores):.4f} ± {np.std(best_scores):.4f}")

    # Agreement
    for model in ["sd15", "sdxl"]:
        agree = sum(1 for r in results_json if r[f"{model}_single"]["pick_idx"] == r[f"{model}_multi"]["pick_idx"])
        metrics[f"{model}_selector_agreement"] = round(agree / len(results_json), 4)
        print(f"  {model} single/multi agreement: {agree}/{len(results_json)}")

    # SD1.5 vs SDXL (using single selector)
    sd15_wins = sum(1 for r in results_json if r["sd15_single"]["best_score"] > r["sdxl_single"]["best_score"])
    metrics["sd15_wins_single"] = sd15_wins
    metrics["sdxl_wins_single"] = len(results_json) - sd15_wins
    print(f"\n  SD1.5 vs SDXL (single selector): SD1.5 wins {sd15_wins}/{len(results_json)}")

    total = sum(timings.values())
    print(f"\n  Timings:")
    for k, v in timings.items():
        print(f"    {k:20s}: {v:7.1f}s ({v*100/total:4.1f}%)")
    print(f"    {'TOTAL':20s}: {total:7.1f}s")

    metrics["per_sample"] = results_json
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Build comparison grids (sample of 10)
    print("\n  Building comparison grids...")
    build_grids(samples[:min(20, len(samples))], run_dir)

    print(f"\n  Results saved to: {run_dir}")
    return metrics, run_dir


def build_grids(samples, run_dir):
    """Build visual comparison grids."""
    def get_font(size):
        try: return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
        except: return ImageFont.load_default()

    cell = 256
    header = 35
    label_w = 80
    font = get_font(14)
    font_sm = get_font(10)

    # Grid: rows=samples, cols=[sketch, sd15_single, sd15_multi, sdxl_single, sdxl_multi]
    col_labels = ["Sketch", "SD1.5 Single", "SD1.5 Multi", "SDXL Single", "SDXL Multi"]
    n_cols = len(col_labels)
    n_rows = len(samples)

    grid = Image.new("RGB", (label_w + n_cols * cell, header + n_rows * cell), (255, 255, 255))
    draw = ImageDraw.Draw(grid)

    for j, lbl in enumerate(col_labels):
        draw.text((label_w + j * cell + 5, 10), lbl, fill="black", font=font)

    for i, s in enumerate(samples):
        y = header + i * cell
        draw.text((5, y + cell//2 - 8), s["cls"][:10], fill="black", font=font)

        # Sketch
        sk = s["sketch"].resize((cell, cell), Image.LANCZOS)
        grid.paste(sk, (label_w, y))

        configs = [
            ("sd15", "single"), ("sd15", "multi"),
            ("sdxl", "single"), ("sdxl", "multi"),
        ]
        for j, (model, sel) in enumerate(configs, 1):
            pick = s[f"{model}_{sel}_pick"]
            img = s[f"{model}_candidates"][pick].resize((cell, cell), Image.LANCZOS)
            grid.paste(img, (label_w + j * cell, y))
            score = s[f"{model}_single_scores"][pick]
            draw.text((label_w + j * cell + 3, y + cell - 14),
                      f"#{pick} {score:.3f}", fill="yellow", font=font_sm)

    grid.save(run_dir / "comparison_grid.png")
    print(f"  Grid saved: comparison_grid.png ({grid.size[0]}x{grid.size[1]})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--best_of", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_compile", action="store_true")
    args = parser.parse_args()
    run_eval(args.n_classes, args.n_samples, args.best_of, args.seed, args.steps, not args.no_compile)
