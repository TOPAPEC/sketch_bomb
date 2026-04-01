#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
import run_bestof_multi as rbm
from run_bestof_multi import *

rbm.OUT = Path("bestof_bo16")
rbm.OUT.mkdir(parents=True, exist_ok=True)
OUT = rbm.OUT

def main():
    scorer = SiglipScorer()
    multi = MultiCriteriaSelector(scorer)
    test_set = load_test_set()
    kimi = KimiScorer()

    all_experiments = []

    sd15 = SD15Pipeline()
    sd15.load()

    exp = run_experiment(sd15, scorer, multi, test_set,
                         best_of=16, selector_type="single", model_tag="SD15")
    exp["tag"] = "SD15_bo16_single"
    all_experiments.append(exp)

    exp = run_experiment(sd15, scorer, multi, test_set,
                         best_of=16, selector_type="multi", model_tag="SD15")
    exp["tag"] = "SD15_bo16_multi"
    all_experiments.append(exp)

    sd15.unload()

    kimi_score_all(kimi, all_experiments, max_workers=16)

    print_report(all_experiments)

    summary = []
    for exp in all_experiments:
        entry = {
            "tag": exp["tag"], "model": exp["model"], "best_of": exp["best_of"],
            "selector": exp["selector"],
            "avg_siglip2": exp["avg_siglip2"],
            "kimi_avg": exp.get("kimi_avg"),
            "per_sample": []
        }
        for r in exp["results"]:
            sample = {
                "class": r["class"], "idx": r["idx"],
                "siglip2": r["siglip2"],
                "multi_axes": r["multi_axes"],
                "kimi": r.get("kimi", {}),
            }
            entry["per_sample"].append(sample)
        summary.append(entry)

    path = OUT / "results.json"
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    for exp in all_experiments:
        grid_src = Path("bestof_multi") / f"{exp['tag']}.png"
        if not grid_src.exists():
            imgs = [r["image"] for r in exp["results"]]
            lbls = [f"{r['class']}: s={r['siglip2']:.1f}" for r in exp["results"]]
            grid = make_grid(imgs, lbls, cols=4,
                             title=f"{exp['tag']} | avg_siglip={exp['avg_siglip2']:.2f}")
            grid.save(OUT / f"{exp['tag']}.png")

    print(f"\nSaved to {path}")
    print("Done.")


if __name__ == "__main__":
    main()
