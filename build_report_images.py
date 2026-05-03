"""Build side-by-side comparison images for the research report."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "quick_draw"))

from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

SD15_DIR = Path("eval_results/dn_eval_20260502_220303_10c_8s")
LIGHTNING_DIR = Path("eval_results/lightning_eval_20260502_235352_10c_8s")
OUT = Path("report_images")
OUT.mkdir(exist_ok=True)

CLASSES = ["trumpet", "camel", "spider", "hand", "bread", "fence", "chair", "butterfly", "arm", "flying saucer"]

def get_font(size):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except:
        return ImageFont.load_default()

FONT = get_font(16)
FONT_SM = get_font(13)
CELL = 300
PAD = 8
HEADER = 30


def load_img(path, size=CELL):
    if not path.exists():
        img = Image.new("RGB", (size, size), (40, 40, 40))
        d = ImageDraw.Draw(img)
        d.text((10, size//2), "N/A", fill="red", font=FONT)
        return img
    return Image.open(path).resize((size, size), Image.LANCZOS)


def build_comparison(cls, sample_idx=0):
    """Build: Sketch | SD1.5 | SDXL | Lightning"""
    cls_dir = f"{cls}_{sample_idx}"

    sketch = load_img(SD15_DIR / cls_dir / "sketch.png")
    sd15 = load_img(SD15_DIR / cls_dir / "sd15_single_winner.png")
    sdxl = load_img(SD15_DIR / cls_dir / "sdxl_single_winner.png")
    lightning = load_img(LIGHTNING_DIR / cls_dir / "winner.png")

    cols = [("Скетч", sketch), ("SD 1.5", sd15), ("SDXL", sdxl), ("Lightning", lightning)]
    w = len(cols) * CELL + (len(cols) - 1) * PAD
    h = HEADER + CELL

    grid = Image.new("RGB", (w, h), (20, 20, 30))
    draw = ImageDraw.Draw(grid)

    for i, (label, img) in enumerate(cols):
        x = i * (CELL + PAD)
        draw.text((x + CELL // 2 - len(label) * 4, 6), label, fill="#aaa", font=FONT)
        grid.paste(img, (x, HEADER))

    return grid


def build_bestof_grid(cls, sample_idx=0):
    """Show all 4 Lightning candidates for a class."""
    cls_dir = f"{cls}_{sample_idx}"
    cands = []
    for i in range(4):
        p = LIGHTNING_DIR / cls_dir / f"cand_{i}.png"
        cands.append(load_img(p, 250))

    w = 4 * 250 + 3 * PAD
    h = HEADER + 250
    grid = Image.new("RGB", (w, h), (20, 20, 30))
    draw = ImageDraw.Draw(grid)

    # Load scores
    import json
    metrics = json.load(open(LIGHTNING_DIR / "metrics.json"))
    sample = [p for p in metrics["per_sample"] if p["cls"] == cls][sample_idx]
    scores = sample["scores"]
    pick = sample["pick_idx"]

    for i, img in enumerate(cands):
        x = i * (250 + PAD)
        star = " ★" if i == pick else ""
        label = f"#{i+1} ({scores[i]:+.2f}){star}"
        color = "#2d6a4f" if i == pick else "#666"
        draw.text((x + 5, 6), label, fill=color, font=FONT_SM)
        grid.paste(img, (x, HEADER))
        if i == pick:
            draw.rectangle([(x, HEADER), (x + 249, HEADER + 249)], outline="#2d6a4f", width=3)

    return grid


def build_misclass_comparison():
    """Show BEiT misclassification examples: bread->bear, arm->waterslide."""
    examples = [
        ("bread_2", "bread → bear"),
        ("arm_0", "arm → waterslide"),
    ]

    rows = []
    for cls_dir, label in examples:
        sketch = load_img(SD15_DIR / cls_dir / "sketch.png", 250)
        sd15 = load_img(SD15_DIR / cls_dir / "sd15_single_winner.png", 250)
        lightning = load_img(LIGHTNING_DIR / cls_dir / "winner.png", 250)
        rows.append((label, sketch, sd15, lightning))

    w = 3 * 250 + 2 * PAD
    row_h = HEADER + 250
    h = len(rows) * row_h + (len(rows) - 1) * PAD
    grid = Image.new("RGB", (w, h), (20, 20, 30))
    draw = ImageDraw.Draw(grid)

    headers = ["Скетч", "SD 1.5", "Lightning"]
    for i, hdr in enumerate(headers):
        draw.text((i * (250 + PAD) + 5, 6), hdr, fill="#aaa", font=FONT_SM)

    for r, (label, sketch, sd15, lightning) in enumerate(rows):
        y = r * (row_h + PAD)
        if r > 0:
            for i, img in enumerate([sketch, sd15, lightning]):
                draw.text((i * (250 + PAD) + 5, y + 6), headers[i] if r == 0 else "", fill="#aaa", font=FONT_SM)

        draw.text((5, y + HEADER + 250 - 22), label, fill="#f55", font=FONT_SM)
        for i, img in enumerate([sketch, sd15, lightning]):
            grid.paste(img, (i * (250 + PAD), y + HEADER))

    return grid


def build_full_grid():
    """One big grid: all classes, sketch | SD1.5 | SDXL | Lightning."""
    cols_per_row = 4
    row_h = HEADER + CELL

    rows = []
    for cls in CLASSES:
        sketch = load_img(SD15_DIR / f"{cls}_0" / "sketch.png")
        sd15 = load_img(SD15_DIR / f"{cls}_0" / "sd15_single_winner.png")
        sdxl = load_img(SD15_DIR / f"{cls}_0" / "sdxl_single_winner.png")
        lightning = load_img(LIGHTNING_DIR / f"{cls}_0" / "winner.png")
        rows.append((cls, sketch, sd15, sdxl, lightning))

    label_w = 120
    w = label_w + cols_per_row * CELL + (cols_per_row - 1) * PAD
    h = HEADER + len(rows) * (CELL + PAD)

    grid = Image.new("RGB", (w, h), (20, 20, 30))
    draw = ImageDraw.Draw(grid)

    headers = ["Скетч", "SD 1.5", "SDXL", "Lightning"]
    for i, hdr in enumerate(headers):
        draw.text((label_w + i * (CELL + PAD) + 5, 6), hdr, fill="#aaa", font=FONT)

    for r, (cls, sketch, sd15, sdxl, lightning) in enumerate(rows):
        y = HEADER + r * (CELL + PAD)
        draw.text((10, y + CELL // 2 - 10), cls, fill="#e94560", font=FONT)
        for i, img in enumerate([sketch, sd15, sdxl, lightning]):
            grid.paste(img, (label_w + i * (CELL + PAD), y))

    return grid


if __name__ == "__main__":
    # Per-class comparisons
    for cls in CLASSES:
        print(f"  {cls}...")
        img = build_comparison(cls, 0)
        safe_name = cls.replace(" ", "_")
        img.save(OUT / f"compare_{safe_name}.png")

    # Best-of-4 example
    print("  best-of camel...")
    img = build_bestof_grid("camel", 0)
    img.save(OUT / "bestof_camel.png")

    print("  best-of arm...")
    img = build_bestof_grid("arm", 0)
    img.save(OUT / "bestof_arm.png")

    # Misclassification examples
    print("  misclass...")
    img = build_misclass_comparison()
    img.save(OUT / "misclass_examples.png")

    # Full comparison grid
    print("  full grid...")
    img = build_full_grid()
    img.save(OUT / "full_comparison.png")

    print(f"Done! Images in {OUT}/")
