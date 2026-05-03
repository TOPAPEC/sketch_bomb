"""Quick analysis of eval_domainnet.py results."""
import json
import sys
import numpy as np
from pathlib import Path


def analyze(metrics_path):
    with open(metrics_path) as f:
        m = json.load(f)

    cfg = m["config"]
    print(f"Config: {cfg['n_classes']}c × {cfg['n_samples']}s × best_of={cfg['best_of']}")
    print(f"Steps: {cfg['steps']}, Classes: {cfg['classes'][:5]}...")
    print(f"BEiT accuracy: {m['beit_accuracy']:.1%}")
    print()

    # Quality comparison
    print("="*60)
    print("QUALITY (SigLIP2 score, higher = better)")
    print("="*60)
    configs = ["sd15_single", "sd15_multi", "sdxl_single", "sdxl_multi"]
    for c in configs:
        d = m[c]
        print(f"  {c:15s}: mean={d['mean_best']:+.4f} ± {d['std_best']:.4f}  (all={d['mean_all']:+.4f})")

    print()
    print(f"  SD1.5 selector agreement: {m.get('sd15_selector_agreement', 'N/A'):.0%}")
    print(f"  SDXL  selector agreement: {m.get('sdxl_selector_agreement', 'N/A'):.0%}")
    print(f"  SD1.5 vs SDXL (single): SD1.5 wins {m.get('sd15_wins_single', '?')}/{cfg['n_classes']*cfg['n_samples']}")

    # Speed comparison
    print()
    print("="*60)
    print("SPEED")
    print("="*60)
    t = m["timings"]
    total = m["timings_total"]
    for k, v in t.items():
        print(f"  {k:20s}: {v:7.1f}s ({v*100/total:4.1f}%)")
    print(f"  {'TOTAL':20s}: {total:7.1f}s")

    n_gens = cfg["n_classes"] * cfg["n_samples"] * cfg["best_of"]
    if "sd15_gen" in t:
        print(f"\n  SD1.5: {t['sd15_gen']/n_gens:.2f}s/img (incl compile)")
    if "sdxl_gen" in t:
        print(f"  SDXL:  {t['sdxl_gen']/n_gens:.2f}s/img (incl compile)")
        if "sd15_gen" in t:
            ratio = t["sdxl_gen"] / t["sd15_gen"]
            print(f"  SDXL/SD1.5 speed ratio: {ratio:.1f}x slower")

    # Per-class analysis
    print()
    print("="*60)
    print("PER-CLASS BREAKDOWN")
    print("="*60)
    per_sample = m.get("per_sample", [])
    if per_sample:
        by_cls = {}
        for r in per_sample:
            cls = r["cls"]
            if cls not in by_cls:
                by_cls[cls] = {"sd15": [], "sdxl": []}
            by_cls[cls]["sd15"].append(r["sd15_single"]["best_score"])
            by_cls[cls]["sdxl"].append(r["sdxl_single"]["best_score"])

        print(f"  {'Class':15s}  {'SD1.5 mean':>10s}  {'SDXL mean':>10s}  {'Winner':>8s}")
        print("  " + "-"*50)
        for cls in sorted(by_cls.keys()):
            sd15 = np.mean(by_cls[cls]["sd15"])
            sdxl = np.mean(by_cls[cls]["sdxl"])
            winner = "SD1.5" if sd15 > sdxl else "SDXL"
            print(f"  {cls:15s}  {sd15:+10.4f}  {sdxl:+10.4f}  {winner:>8s}")

    # Improvement from best_of selection
    print()
    print("="*60)
    print("BEST-OF SELECTION IMPACT")
    print("="*60)
    for model in ["sd15", "sdxl"]:
        d = m[f"{model}_single"]
        gain = d["mean_best"] - d["mean_all"]
        print(f"  {model.upper():5s}: best_of gain = {gain:+.4f} (best={d['mean_best']:+.4f} vs all={d['mean_all']:+.4f})")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        results_dir = Path("eval_results")
        runs = sorted(results_dir.glob("dn_eval_*"))
        if not runs:
            print("No eval results found")
            sys.exit(1)
        path = runs[-1] / "metrics.json"
    print(f"Analyzing: {path}\n")
    analyze(path)
