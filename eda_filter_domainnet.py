import json, random, shutil, gc
import numpy as np
import torch
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModel, SiglipProcessor, SiglipImageProcessor, GemmaTokenizerFast

random.seed(42)
np.random.seed(42)

DOMAINNET_ROOT = Path("data/domainnet/sketch")
PROMPTS_PATH = Path("quick_draw/prompts.json")
OUT_DIR = Path("eda_out/filtering")
FILTERED_DIR = Path("data/domainnet_filtered/sketch")

CLASSES = ["cat", "car", "flower", "house", "bird", "apple", "bicycle", "face", "tree", "fish"]

CELL = 256
GRID = 4
COLLAGE_SZ = CELL * GRID

with open(PROMPTS_PATH) as f:
    pdata = json.load(f)
PROMPTS = pdata["prompts"]


def get_font(size=20):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except:
        return ImageFont.load_default()


def add_label(img, text):
    d = ImageDraw.Draw(img)
    font = get_font(14)
    bb = d.textbbox((0, 0), text, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    d.rectangle([0, 0, tw + 10, th + 8], fill="black")
    d.text((5, 3), text, fill="white", font=font)
    return img


def make_collage(images, title=""):
    collage = Image.new("RGB", (COLLAGE_SZ, COLLAGE_SZ + 40), "white")
    d = ImageDraw.Draw(collage)
    font = get_font(22)
    d.text((10, 8), title, fill="black", font=font)
    for i, img in enumerate(images[:16]):
        row, col = divmod(i, GRID)
        resized = img.resize((CELL, CELL), Image.LANCZOS)
        collage.paste(resized, (col * CELL, row * CELL + 40))
    return collage


def score_prompt(label):
    return f"a sketch drawing of a {label}"


class SiglipScorer:
    MODEL_ID = "google/siglip2-base-patch16-224"

    def __init__(self, device="cuda"):
        self.device = device
        print(f"Loading SigLIP2 on {device}...")
        self.model = AutoModel.from_pretrained(
            self.MODEL_ID, dtype=torch.float16).to(device)
        self.processor = SiglipProcessor(
            image_processor=SiglipImageProcessor.from_pretrained(self.MODEL_ID),
            tokenizer=GemmaTokenizerFast.from_pretrained(self.MODEL_ID),
        )
        self.model.eval()
        print("SigLIP2 loaded.")

    @torch.no_grad()
    def score_single(self, image, text):
        inputs = self.processor(
            text=[text], images=[image],
            padding="max_length", return_tensors="pt"
        ).to(self.device)
        logits = self.model(**inputs).logits_per_image
        score = logits[0, 0].item()
        del inputs, logits
        return score


def get_class_paths(cls):
    cls_dir = DOMAINNET_ROOT / cls.replace(" ", "_")
    if not cls_dir.exists():
        return []
    return sorted([
        p for p in cls_dir.glob("*.*")
        if p.suffix.lower() in (".jpg", ".png", ".jpeg")
    ])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scorer = SiglipScorer()

    all_stats = {}

    for cls in CLASSES:
        print(f"\n{'='*60}")
        print(f"CLASS: {cls}")
        print(f"{'='*60}")

        paths = get_class_paths(cls)
        if not paths:
            print(f"  NO IMAGES for {cls}")
            continue

        prompt = score_prompt(cls)
        print(f"  Prompt: '{prompt}'")
        print(f"  Scoring {len(paths)} images one by one...")

        scores = []
        for idx, p in enumerate(paths):
            try:
                im = Image.open(p).convert("RGB")
                s = scorer.score_single(im, prompt)
                scores.append(s)
                im.close()
            except Exception as e:
                print(f"  skip {p.name}: {e}")
                scores.append(float("-inf"))
            if (idx + 1) % 50 == 0:
                print(f"    scored {idx+1}/{len(paths)}")

        scores = np.array(scores)
        valid_mask = scores > float("-inf")
        valid_scores = scores[valid_mask]

        sorted_indices = np.argsort(scores)[::-1]
        median = float(np.median(valid_scores))
        mean = float(np.mean(valid_scores))
        std_val = float(np.std(valid_scores))

        print(f"  Scores: min={valid_scores.min():.4f}, max={valid_scores.max():.4f}, "
              f"mean={mean:.4f}, median={median:.4f}, std={std_val:.4f}")

        all_stats[cls] = {
            "count": len(paths),
            "min": float(valid_scores.min()),
            "max": float(valid_scores.max()),
            "mean": mean,
            "median": median,
            "std": std_val,
            "prompt": prompt,
        }

        # --- Top 16 collage ---
        top16_idx = sorted_indices[:16]
        top16_imgs = []
        for i in top16_idx:
            im = Image.open(paths[i]).convert("RGB")
            rank = int(np.where(sorted_indices == i)[0][0]) + 1
            add_label(im, f"#{rank} s={scores[i]:.2f}")
            top16_imgs.append(im)
        make_collage(top16_imgs, f"TOP 16: {cls} (best SigLIP2)").save(OUT_DIR / f"top16_{cls}.png")
        for im in top16_imgs:
            im.close()
        del top16_imgs

        # --- Bottom 16 collage ---
        bot16_idx = sorted_indices[-16:][::-1]
        bot16_imgs = []
        for i in bot16_idx:
            rank = int(np.where(sorted_indices == i)[0][0]) + 1
            im = Image.open(paths[i]).convert("RGB")
            add_label(im, f"#{rank} s={scores[i]:.2f}")
            bot16_imgs.append(im)
        make_collage(bot16_imgs, f"BOTTOM 16: {cls} (worst SigLIP2)").save(OUT_DIR / f"bot16_{cls}.png")
        for im in bot16_imgs:
            im.close()
        del bot16_imgs

        # --- Filter: top 50% (above median) ---
        threshold = median
        keep_indices = [i for i in range(len(paths)) if scores[i] >= threshold]
        drop_indices = [i for i in range(len(paths)) if scores[i] < threshold]

        print(f"  Threshold (median): {threshold:.4f}")
        print(f"  Keep: {len(keep_indices)}/{len(paths)} ({100*len(keep_indices)/len(paths):.0f}%)")

        all_stats[cls]["threshold"] = threshold
        all_stats[cls]["kept"] = len(keep_indices)
        all_stats[cls]["dropped"] = len(drop_indices)

        # --- Copy filtered images ---
        cls_out = FILTERED_DIR / cls.replace(" ", "_")
        cls_out.mkdir(parents=True, exist_ok=True)
        for i in keep_indices:
            shutil.copy2(paths[i], cls_out / paths[i].name)

        # --- Random filtered vs random unfiltered comparison ---
        rng = random.Random(42)

        sample_filt = rng.sample(keep_indices, min(16, len(keep_indices)))
        filtered_imgs = []
        for i in sample_filt:
            im = Image.open(paths[i]).convert("RGB")
            add_label(im, f"s={scores[i]:.2f}")
            filtered_imgs.append(im)
        make_collage(filtered_imgs, f"FILTERED (top50%): {cls}").save(OUT_DIR / f"filtered_{cls}.png")
        for im in filtered_imgs:
            im.close()
        del filtered_imgs

        sample_unfilt = rng.sample(range(len(paths)), min(16, len(paths)))
        unfiltered_imgs = []
        for i in sample_unfilt:
            im = Image.open(paths[i]).convert("RGB")
            add_label(im, f"s={scores[i]:.2f}")
            unfiltered_imgs.append(im)
        make_collage(unfiltered_imgs, f"UNFILTERED (random): {cls}").save(OUT_DIR / f"unfiltered_{cls}.png")
        for im in unfiltered_imgs:
            im.close()
        del unfiltered_imgs

        gc.collect()
        torch.cuda.empty_cache()
        print(f"  Saved collages: top16, bot16, filtered, unfiltered")

    with open(OUT_DIR / "stats.json", "w") as f:
        json.dump(all_stats, f, indent=2)

    print(f"\nAll results saved to {OUT_DIR}/")
    print(f"Filtered dataset saved to {FILTERED_DIR}/")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Class':<12} {'Total':<8} {'Kept':<8} {'Drop':<8} {'Median':<10} {'Min':<10} {'Max':<10}")
    for cls in CLASSES:
        s = all_stats.get(cls, {})
        if not s:
            continue
        print(f"{cls:<12} {s['count']:<8} {s['kept']:<8} {s['dropped']:<8} "
              f"{s['median']:<10.4f} {s['min']:<10.4f} {s['max']:<10.4f}")


if __name__ == "__main__":
    main()
