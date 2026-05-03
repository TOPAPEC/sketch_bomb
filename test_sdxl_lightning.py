"""
Test SDXL-Lightning (4-step) + xinsir scribble ControlNet.
Goal: Get SDXL quality at near-SD1.5 speed (~2s/img vs current ~10s/img).
"""
import time
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, EulerDiscreteScheduler
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

import sys
sys.path.insert(0, str(Path(__file__).parent / "quick_draw"))
from v8_tailored import SiglipScorer, DomainNetMatcher, score_prompt
from demo_beit_compare import to_lineart_sdxl, zoom_to_content, drawing_to_img_small

DEVICE = "cuda"
SEED = 42
CLASSES = ["trumpet", "butterfly", "chair", "spider", "bread"]
N_IMAGES = 2  # per class

OUT_DIR = Path("test_results/sdxl_lightning")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_lightning_pipeline():
    """Load SDXL-Lightning 4-step with xinsir scribble ControlNet."""
    print("Loading xinsir/controlnet-scribble-sdxl-1.0...")
    controlnet = ControlNetModel.from_pretrained(
        "xinsir/controlnet-scribble-sdxl-1.0",
        torch_dtype=torch.float16,
    )

    print("Loading SDXL base + Lightning LoRA...")
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        variant="fp16",
    ).to(DEVICE)

    # Load Lightning LoRA for 4-step
    pipe.load_lora_weights(
        hf_hub_download("ByteDance/SDXL-Lightning", "sdxl_lightning_4step_lora.safetensors")
    )
    pipe.fuse_lora()

    # Use Euler for Lightning (specific scheduler requirement)
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="trailing",
    )

    pipe.set_progress_bar_config(disable=True)
    return pipe


def get_prompt(label):
    return f"a high quality detailed illustration of a {label}, professional artwork, sharp details, vibrant colors"


def get_negative():
    return "ugly, blurry, low quality, distorted, deformed, disfigured, bad anatomy, watermark, text"


def generate_lightning(pipe, ctrl, label, seed):
    """Generate with SDXL-Lightning 4-step."""
    img = pipe(
        prompt=get_prompt(label),
        negative_prompt=get_negative(),
        image=ctrl,
        num_inference_steps=4,
        guidance_scale=1.5,  # Lightning uses low guidance
        controlnet_conditioning_scale=0.7,
        control_guidance_end=1.0,
        generator=torch.Generator(device=DEVICE).manual_seed(seed),
        num_images_per_prompt=1,
    ).images[0]
    return img


def main():
    # Load DomainNet data
    print("Loading SigLIP2 scorer...")
    scorer = SiglipScorer(device=DEVICE)

    print("\nLoading DomainNet matcher...")
    matcher = DomainNetMatcher(scorer, CLASSES, batch_size=16)

    # Get test images from DomainNet
    print("\nGathering test images...")
    test_data = []
    from v8_tailored import fetch_quickdraw, drawing_to_img
    for cls in CLASSES:
        raw_drawings = fetch_quickdraw(cls, N_IMAGES, offset=0)
        for i, raw in enumerate(raw_drawings):
            sketch = drawing_to_img(raw, sz=256)
            matches = matcher.match_topk(sketch, cls, k=1)
            dn_img = zoom_to_content(matches[0][0], 1024, fill_pct=0.9)
            ctrl = to_lineart_sdxl(dn_img.resize((1024, 1024), Image.LANCZOS))
            test_data.append({"cls": cls, "sketch": sketch, "ctrl": ctrl, "seed": SEED + i})

    del matcher
    torch.cuda.empty_cache()

    # Load Lightning pipeline
    pipe = load_lightning_pipeline()

    # Warmup
    print("\nWarmup...")
    _ = generate_lightning(pipe, test_data[0]["ctrl"], test_data[0]["cls"], 0)

    # Generate
    print(f"\nGenerating {len(test_data)} images with SDXL-Lightning 4-step...")
    results = []
    t0 = time.time()
    for d in test_data:
        t1 = time.time()
        img = generate_lightning(pipe, d["ctrl"], d["cls"], d["seed"])
        elapsed = time.time() - t1
        results.append({**d, "img": img, "time": elapsed})
        print(f"  {d['cls']} #{d['seed']}: {elapsed:.2f}s")
    total_gen = time.time() - t0
    avg = total_gen / len(results)
    print(f"\nTotal: {total_gen:.1f}s, Avg: {avg:.2f}s/img")

    del pipe
    torch.cuda.empty_cache()

    # Score with SigLIP2
    print("\nScoring with SigLIP2...")
    scorer2 = SiglipScorer(device=DEVICE)
    for r in results:
        r["siglip"] = scorer2.score_single(r["img"], score_prompt(r["cls"]))
    del scorer2

    scores = [r["siglip"] for r in results]
    print(f"Mean SigLIP2: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    # Build comparison grid
    print("\nBuilding grid...")
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except:
        font = ImageFont.load_default()

    cell = 256
    cols = N_IMAGES + 1  # sketch + N generated
    rows = len(CLASSES)
    grid = Image.new("RGB", (cols * cell + 80, rows * cell + 30), (255, 255, 255))
    draw = ImageDraw.Draw(grid)

    headers = ["Sketch"] + [f"Gen #{i+1}" for i in range(N_IMAGES)]
    for j, h in enumerate(headers):
        draw.text((80 + j * cell + 5, 8), h, fill="black", font=font)

    for i, cls in enumerate(CLASSES):
        y = 30 + i * cell
        draw.text((5, y + cell//2 - 8), cls[:8], fill="black", font=font)
        cls_results = [r for r in results if r["cls"] == cls]

        sk = cls_results[0]["sketch"].resize((cell, cell), Image.LANCZOS)
        grid.paste(sk, (80, y))

        for j, r in enumerate(cls_results):
            img = r["img"].resize((cell, cell), Image.LANCZOS)
            grid.paste(img, (80 + (j + 1) * cell, y))
            draw.text((80 + (j + 1) * cell + 3, y + cell - 16),
                      f"{r['siglip']:.2f} / {r['time']:.1f}s", fill="yellow", font=font)

    grid.save(OUT_DIR / "lightning_grid.png")
    print(f"Grid saved: {OUT_DIR / 'lightning_grid.png'}")

    # Summary
    print(f"\n{'='*50}")
    print(f"SDXL-Lightning 4-step Results")
    print(f"{'='*50}")
    print(f"Avg speed: {avg:.2f}s/img")
    print(f"Mean SigLIP2: {np.mean(scores):.4f}")
    print(f"Compare: SD1.5 20-step = ~1.53s/img, SigLIP2 ≈ -1.64")
    print(f"Compare: SDXL 20-step  = ~10.7s/img")


if __name__ == "__main__":
    main()
