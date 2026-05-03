"""Build comparison images for 30-class Lightning eval report."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "quick_draw"))

from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import numpy as np

OUT = Path("report_images_30c")
OUT.mkdir(exist_ok=True)


def get_font(size):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except:
        return ImageFont.load_default()

FONT = get_font(14)
FONT_SM = get_font(11)
FONT_LG = get_font(16)


def find_lightning_dir():
    """Find the latest 30c lightning eval directory."""
    results = Path("eval_results")
    dirs = sorted(results.glob("lightning_eval_*_30c_*"))
    if not dirs:
        raise FileNotFoundError("No 30c eval found")
    return dirs[-1]


def load_img(path, size=256):
    if not path.exists():
        img = Image.new("RGB", (size, size), (40, 40, 40))
        d = ImageDraw.Draw(img)
        d.text((10, size//2), "N/A", fill="red", font=FONT)
        return img
    return Image.open(path).resize((size, size), Image.LANCZOS)


def build_full_grid(eval_dir, metrics, cell=200):
    """Big grid: all classes, sketch + 4 candidates + winner. Sorted by best_score."""
    ps = metrics["per_sample"]
    classes = sorted(set(p["cls"] for p in ps))

    # Compute per-class mean best_score
    cls_scores = {}
    for cls in classes:
        scores = [p["best_score"] for p in ps if p["cls"] == cls]
        cls_scores[cls] = np.mean(scores)

    classes_sorted = sorted(classes, key=lambda c: cls_scores[c], reverse=True)

    label_w = 130
    score_w = 60
    cols = 3  # sketch, winner, best-of comparison (4 small)
    w = label_w + score_w + 2 * cell + cell * 2 + 30
    # Actually simpler: sketch | winner | score
    w = label_w + cell + cell + score_w + 20
    header_h = 28
    h = header_h + len(classes_sorted) * (cell + 4)

    grid = Image.new("RGB", (w, h), (20, 20, 30))
    draw = ImageDraw.Draw(grid)

    draw.text((label_w + 5, 6), "Скетч", fill="#888", font=FONT_SM)
    draw.text((label_w + cell + 10, 6), "Lightning winner", fill="#888", font=FONT_SM)

    for i, cls in enumerate(classes_sorted):
        y = header_h + i * (cell + 4)
        score = cls_scores[cls]
        color = "#2d6a4f" if score > 2 else "#c9a227" if score > 0 else "#f55"
        draw.text((8, y + cell // 2 - 8), cls[:14], fill=color, font=FONT)
        draw.text((8, y + cell // 2 + 8), f"{score:+.2f}", fill="#666", font=FONT_SM)

        # Find first sample for this class
        cls_dir = eval_dir / f"{cls}_0"
        sketch = load_img(cls_dir / "sketch.png", cell)
        winner = load_img(cls_dir / "winner.png", cell)

        grid.paste(sketch, (label_w, y))
        grid.paste(winner, (label_w + cell + 8, y))

    grid.save(OUT / "full_grid_30c.png")
    print(f"  full_grid_30c.png ({w}x{h})")
    return grid


def build_top_bottom(eval_dir, metrics, n=8, cell=256):
    """Top N and bottom N classes side by side."""
    ps = metrics["per_sample"]
    classes = sorted(set(p["cls"] for p in ps))

    cls_scores = {}
    for cls in classes:
        scores = [p["best_score"] for p in ps if p["cls"] == cls]
        cls_scores[cls] = np.mean(scores)

    sorted_cls = sorted(classes, key=lambda c: cls_scores[c], reverse=True)
    top = sorted_cls[:n]
    bottom = sorted_cls[-n:]

    for group_name, group in [("top", top), ("bottom", bottom)]:
        label_w = 120
        w = label_w + cell * 2 + 12
        header_h = 30
        h = header_h + len(group) * (cell + 4)

        grid = Image.new("RGB", (w, h), (20, 20, 30))
        draw = ImageDraw.Draw(grid)

        title = f"{'Лучшие' if group_name == 'top' else 'Худшие'} {n} классов"
        draw.text((10, 6), title, fill="#e94560", font=FONT_LG)
        draw.text((label_w + 5, 6), "Скетч", fill="#666", font=FONT_SM)
        draw.text((label_w + cell + 14, 6), "Winner", fill="#666", font=FONT_SM)

        for i, cls in enumerate(group):
            y = header_h + i * (cell + 4)
            score = cls_scores[cls]
            color = "#2d6a4f" if score > 2 else "#c9a227" if score > 0 else "#f55"
            draw.text((8, y + cell // 2 - 12), cls[:12], fill=color, font=FONT)
            draw.text((8, y + cell // 2 + 6), f"{score:+.2f}", fill="#666", font=FONT_SM)

            cls_dir = eval_dir / f"{cls}_0"
            sketch = load_img(cls_dir / "sketch.png", cell)
            winner = load_img(cls_dir / "winner.png", cell)
            grid.paste(sketch, (label_w, y))
            grid.paste(winner, (label_w + cell + 8, y))

        grid.save(OUT / f"{group_name}_{n}_classes.png")
        print(f"  {group_name}_{n}_classes.png")


def build_bestof_examples(eval_dir, metrics, cell=220):
    """Show best-of-4 candidates for a high-variance and a low-variance class."""
    ps = metrics["per_sample"]

    # Find highest and lowest variance classes
    classes = sorted(set(p["cls"] for p in ps))
    cls_var = {}
    for cls in classes:
        items = [p for p in ps if p["cls"] == cls]
        gains = [p["best_score"] - p["mean_score"] for p in items]
        cls_var[cls] = np.mean(gains)

    high_var_cls = max(cls_var, key=cls_var.get)
    low_var_cls = min(cls_var, key=cls_var.get)

    for cls, label in [(high_var_cls, "high_var"), (low_var_cls, "low_var")]:
        sample = [p for p in ps if p["cls"] == cls][0]
        cls_dir = eval_dir / f"{cls}_0"

        w = 4 * cell + 3 * 8
        header_h = 30
        h = header_h + cell

        grid = Image.new("RGB", (w, h), (20, 20, 30))
        draw = ImageDraw.Draw(grid)

        pick = sample["pick_idx"]
        gain = sample["best_score"] - sample["mean_score"]
        draw.text((5, 6), f"{cls} (gain: {gain:+.2f})", fill="#e94560", font=FONT)

        for i in range(4):
            x = i * (cell + 8)
            cand = load_img(cls_dir / f"cand_{i}.png", cell)
            grid.paste(cand, (x, header_h))
            star = " ★" if i == pick else ""
            sc = sample["scores"][i]
            color = "#2d6a4f" if i == pick else "#666"
            draw.text((x + 5, header_h + cell - 20), f"{sc:+.2f}{star}", fill=color, font=FONT_SM)
            if i == pick:
                draw.rectangle([(x, header_h), (x + cell - 1, header_h + cell - 1)], outline="#2d6a4f", width=3)

        grid.save(OUT / f"bestof_{label}_{cls.replace(' ', '_')}.png")
        print(f"  bestof_{label}_{cls.replace(' ', '_')}.png")


def build_misclass_grid(eval_dir, metrics, cell=200):
    """Show misclassification examples."""
    ps = metrics["per_sample"]
    wrong = [p for p in ps if not p["beit_correct"]]

    # Pick diverse examples
    seen_pairs = set()
    examples = []
    for p in wrong:
        pair = (p["cls"], p["label"])
        if pair not in seen_pairs and len(examples) < 10:
            seen_pairs.add(pair)
            examples.append(p)

    if not examples:
        print("  No misclassifications found")
        return

    label_w = 180
    w = label_w + cell * 2 + 12
    header_h = 28
    h = header_h + len(examples) * (cell + 4)

    grid = Image.new("RGB", (w, h), (20, 20, 30))
    draw = ImageDraw.Draw(grid)

    draw.text((10, 6), "Ошибки BEiT: скетч → результат", fill="#f55", font=FONT)

    for i, p in enumerate(examples):
        y = header_h + i * (cell + 4)
        cls_dir = eval_dir / f"{p['cls']}_{ps.index(p) % 16}"
        # Find correct index
        cls_samples = [pp for pp in ps if pp["cls"] == p["cls"]]
        idx = cls_samples.index(p)
        cls_dir = eval_dir / f"{p['cls']}_{idx}"

        sketch = load_img(cls_dir / "sketch.png", cell)
        winner = load_img(cls_dir / "winner.png", cell)

        conf = p["beit_preds"][0][1]
        draw.text((8, y + cell//2 - 16), f"{p['cls']}", fill="#888", font=FONT)
        draw.text((8, y + cell//2), f"→ {p['label']}", fill="#f55", font=FONT)
        draw.text((8, y + cell//2 + 16), f"({conf:.0%})", fill="#555", font=FONT_SM)

        grid.paste(sketch, (label_w, y))
        grid.paste(winner, (label_w + cell + 8, y))

    grid.save(OUT / "misclass_grid.png")
    print(f"  misclass_grid.png ({len(examples)} examples)")


def build_score_histogram(metrics):
    """Simple text-based score distribution."""
    ps = metrics["per_sample"]
    scores = [p["best_score"] for p in ps]

    bins = {}
    for s in scores:
        b = int(s) if s >= 0 else int(s) - 1
        bins[b] = bins.get(b, 0) + 1

    print("\n  Score distribution:")
    for b in sorted(bins.keys()):
        bar = "█" * bins[b]
        print(f"    [{b:+d} to {b+1:+d}): {bins[b]:3d} {bar}")


if __name__ == "__main__":
    eval_dir = find_lightning_dir()
    print(f"Using: {eval_dir}")

    metrics = json.load(open(eval_dir / "metrics.json"))
    n_classes = len(set(p["cls"] for p in metrics["per_sample"]))
    n_samples = len(metrics["per_sample"]) // n_classes
    print(f"Config: {n_classes}c × {n_samples}s\n")

    print("Building images:")
    build_full_grid(eval_dir, metrics)
    build_top_bottom(eval_dir, metrics)
    build_bestof_examples(eval_dir, metrics)
    build_misclass_grid(eval_dir, metrics)
    build_score_histogram(metrics)

    # Print summary stats
    ps = metrics["per_sample"]
    print(f"\n=== SUMMARY ===")
    print(f"Mean best (best-of-4): {metrics['mean_best']:+.4f} ± {metrics['std_best']:.4f}")
    print(f"Mean all (no selection): {metrics['mean_all']:+.4f}")
    print(f"BEiT accuracy: {metrics['beit_accuracy']:.1%}")
    correct = [p for p in ps if p["beit_correct"]]
    wrong = [p for p in ps if not p["beit_correct"]]
    print(f"Quality when BEiT correct ({len(correct)}): {np.mean([p['best_score'] for p in correct]):+.3f}")
    if wrong:
        print(f"Quality when BEiT wrong ({len(wrong)}): {np.mean([p['best_score'] for p in wrong]):+.3f}")

    print(f"\nImages saved to {OUT}/")
