import json, os, io, random, requests, zipfile, shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

random.seed(42)

EDA_DIR = Path("eda_out")
EDA_DIR.mkdir(exist_ok=True)
(EDA_DIR / "quickdraw").mkdir(exist_ok=True)
(EDA_DIR / "domainnet").mkdir(exist_ok=True)

CLASSES = [
    "cat", "car", "flower", "house", "bird", "apple",
    "bicycle", "face", "tree", "fish",
]

DOMAINNET_URL = "http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip"
DOMAINNET_ROOT = Path("data/domainnet/sketch")

CELL = 256
GRID = 4
COLLAGE_SZ = CELL * GRID


def get_font(size=20):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except:
        return ImageFont.load_default()


def add_label(img, text, pos="top"):
    d = ImageDraw.Draw(img)
    font = get_font(16)
    bb = d.textbbox((0, 0), text, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    if pos == "top":
        d.rectangle([0, 0, tw + 10, th + 8], fill="black")
        d.text((5, 3), text, fill="white", font=font)
    return img


def make_collage(images, title="", cell_size=CELL):
    collage = Image.new("RGB", (COLLAGE_SZ, COLLAGE_SZ + 40), "white")
    d = ImageDraw.Draw(collage)
    font = get_font(24)
    d.text((10, 8), title, fill="black", font=font)

    for i, img in enumerate(images[:16]):
        row, col = divmod(i, GRID)
        resized = img.resize((cell_size, cell_size), Image.LANCZOS)
        collage.paste(resized, (col * cell_size, row * cell_size + 40))
    return collage


def fetch_quickdraw(category, n, offset=0):
    url = f"https://storage.googleapis.com/quickdraw_dataset/full/simplified/{category}.ndjson"
    r = requests.get(url, stream=True, timeout=60)
    out = []
    for i, line in enumerate(r.iter_lines()):
        if i < offset:
            continue
        if len(out) >= n:
            break
        if line:
            out.append(json.loads(line))
    return out


def drawing_to_img(d, sz=256):
    img = Image.new("RGB", (sz, sz), "white")
    draw = ImageDraw.Draw(img)
    strokes = d["drawing"]
    xs = [x for s in strokes for x in s[0]]
    ys = [y for s in strokes for y in s[1]]
    if not xs:
        return img
    mnx, mxx, mny, mxy = min(xs), max(xs), min(ys), max(ys)
    sc = sz * 0.85 / max(mxx - mnx, mxy - mny, 1)
    ox = (sz - (mxx - mnx) * sc) / 2 - mnx * sc
    oy = (sz - (mxy - mny) * sc) / 2 - mny * sc
    for s in strokes:
        pts = [(x * sc + ox, y * sc + oy) for x, y in zip(s[0], s[1])]
        if len(pts) > 1:
            draw.line(pts, fill="black", width=max(2, sz // 128))
    return img


def generate_quickdraw_collages():
    print("=" * 60)
    print("QUICKDRAW EDA")
    print("=" * 60)
    for idx, cls in enumerate(CLASSES):
        print(f"  [{idx+1}/{len(CLASSES)}] Fetching QuickDraw '{cls}'...")
        offset = random.randint(0, 500)
        drawings = fetch_quickdraw(cls, 32, offset=offset)
        random.shuffle(drawings)

        imgs = [drawing_to_img(d) for d in drawings[:16]]
        for i, im in enumerate(imgs):
            add_label(im, f"#{i+1}")

        collage = make_collage(imgs, title=f"QuickDraw: {cls} (offset={offset})")
        path = EDA_DIR / "quickdraw" / f"qd_{idx+1:02d}_{cls}.png"
        collage.save(path)
        print(f"    -> saved {path}")


def download_domainnet():
    if DOMAINNET_ROOT.exists() and any(DOMAINNET_ROOT.iterdir()):
        print("DomainNet already present.")
        return

    DOMAINNET_ROOT.parent.mkdir(parents=True, exist_ok=True)
    zip_path = Path("data/domainnet/sketch.zip")

    if not zip_path.exists():
        print(f"Downloading DomainNet sketch split (~1.2GB)...")
        print(f"  URL: {DOMAINNET_URL}")
        r = requests.get(DOMAINNET_URL, stream=True, timeout=600)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192 * 16):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 / total
                    print(f"\r  {downloaded/(1024**2):.0f}/{total/(1024**2):.0f} MB ({pct:.0f}%)", end="", flush=True)
        print()

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall("data/domainnet/")
    print("Done.")
    zip_path.unlink(missing_ok=True)


def generate_domainnet_collages():
    print("=" * 60)
    print("DOMAINNET EDA")
    print("=" * 60)

    for idx, cls in enumerate(CLASSES):
        cls_dir = DOMAINNET_ROOT / cls.replace(" ", "_")
        if not cls_dir.exists():
            print(f"  [{idx+1}/{len(CLASSES)}] SKIP '{cls}' - dir not found at {cls_dir}")
            continue

        all_imgs = sorted(cls_dir.glob("*.*"))
        all_imgs = [p for p in all_imgs if p.suffix.lower() in (".jpg", ".png", ".jpeg")]
        print(f"  [{idx+1}/{len(CLASSES)}] DomainNet '{cls}': {len(all_imgs)} images")

        if len(all_imgs) < 16:
            print(f"    WARNING: only {len(all_imgs)} images, need 16")
            sample = all_imgs
        else:
            sample = random.sample(all_imgs, 16)

        imgs = []
        for p in sample:
            try:
                im = Image.open(p).convert("RGB")
                imgs.append(im)
            except Exception as e:
                print(f"    Error loading {p}: {e}")

        for i, im in enumerate(imgs):
            add_label(im, f"#{i+1}")

        collage = make_collage(imgs, title=f"DomainNet sketch: {cls} ({len(all_imgs)} total)")
        path = EDA_DIR / "domainnet" / f"dn_{idx+1:02d}_{cls}.png"
        collage.save(path)
        print(f"    -> saved {path}")


if __name__ == "__main__":
    generate_quickdraw_collages()
    download_domainnet()
    generate_domainnet_collages()
    print("\nAll collages saved to eda_out/")
