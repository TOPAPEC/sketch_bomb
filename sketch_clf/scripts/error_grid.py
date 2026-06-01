"""Qualitative error montage: misclassified test sketches from the best model,
annotated with true -> predicted (conf). Saved to report_images/error_examples.png."""
import argparse
import numpy as np
import torch
import timm
from PIL import Image, ImageDraw, ImageFont

from common import load_meta, load_split, ROOT

DEVICE = "cuda"


def font(sz):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:
        try:
            return ImageFont.truetype(p, sz)
        except OSError:
            pass
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=24)
    args = ap.parse_args()
    meta = load_meta()
    classes = meta["classes"]
    Xte, yte = load_split("test")

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = timm.create_model(ck["model"], pretrained=False, num_classes=len(classes))
    model.load_state_dict(ck["state_dict"])
    model = model.to(DEVICE).eval()
    mean = torch.tensor(ck["mean"]).view(3, 1, 1)
    std = torch.tensor(ck["std"]).view(3, 1, 1)

    preds, confs = [], []
    with torch.no_grad():
        for s in range(0, len(Xte), 256):
            b = Xte[s:s + 256]
            t = torch.from_numpy(b.astype(np.float32) / 255.0)[:, None].repeat(1, 3, 1, 1)
            t = ((t - mean) / std).to(DEVICE)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                p = model(t).softmax(-1).float().cpu()
            preds.extend(p.argmax(-1).tolist())
            confs.extend(p.max(-1).values.tolist())
    preds = np.array(preds)
    wrong = np.where(preds != yte)[0]
    rng = np.random.RandomState(1)
    pick = rng.choice(wrong, size=min(args.n, len(wrong)), replace=False)

    cols, cell, hdr = 6, 200, 40
    rows = (len(pick) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cell, rows * (cell + hdr)), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    f = font(13)
    for i, idx in enumerate(pick):
        r, c = divmod(i, cols)
        x0, y0 = c * cell, r * (cell + hdr)
        sk = Image.fromarray(Xte[idx]).convert("RGB").resize((cell, cell))
        canvas.paste(sk, (x0, y0 + hdr))
        d.rectangle([x0, y0, x0 + cell, y0 + hdr], fill=(245, 230, 230))
        d.text((x0 + 3, y0 + 2), f"T:{classes[yte[idx]]}", fill=(0, 0, 0), font=f)
        d.text((x0 + 3, y0 + 20), f"P:{classes[preds[idx]]} {confs[idx]:.0%}",
                fill=(180, 0, 0), font=f)
    out = ROOT / "report_images" / "error_examples.png"
    canvas.save(out)
    print("saved", out, "from", len(wrong), "errors")


if __name__ == "__main__":
    main()
