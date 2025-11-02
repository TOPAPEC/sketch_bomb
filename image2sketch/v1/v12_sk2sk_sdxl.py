# sdxl_controlnet_union_infer.py
import argparse, torch, os
from PIL import Image, ImageOps
from diffusers import StableDiffusionXLControlNetUnionPipeline, ControlNetUnionModel, DPMSolverMultistepScheduler

def center_square_resize_invert(im, w, h):
    W,H=im.size
    m=min(W,H)
    x=(W-m)//2
    y=(H-m)//2
    im=im.crop((x,y,x+m,y+m)).resize((w,h), Image.BICUBIC)
    if im.mode!="RGB":
        im=im.convert("RGB")
    return ImageOps.invert(im)
    # return im

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--sketch",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--model",default="Bercraft/Illustrious-XL-v2.0-FP16-Diffusers")
    ap.add_argument("--controlnet",default="xinsir/controlnet-union-sdxl-1.0")
    ap.add_argument("--width",type=int,default=1024)
    ap.add_argument("--height",type=int,default=1024)
    ap.add_argument("--steps",type=int,default=30)
    ap.add_argument("--cfg",type=float,default=7.5)
    ap.add_argument("--strength",type=float,default=0.8)
    ap.add_argument("--seed",type=int,default=1340)
    args=ap.parse_args()

    cn=ControlNetUnionModel.from_pretrained(args.controlnet, torch_dtype=torch.float16)
    pipe=StableDiffusionXLControlNetUnionPipeline.from_pretrained(args.model, controlnet=cn, torch_dtype=torch.float16)
    pipe.scheduler=DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")

    cn_img=center_square_resize_invert(Image.open(args.sketch).convert("RGB"), args.width, args.height)
    cn_img.save(os.path.join(os.path.dirname(args.out), "interm_cn.png"))

    gen=torch.Generator(device="cuda").manual_seed(args.seed)

    prompt="clean lineart illustration, (uniform stroke width:1.55), (pure black ink:1.6), (white background:1.6), outline-only, vector-style inking, sharp hard edges, smooth continuous lines, high-contrast black & white, digital inking, posterized 2-color, manga/ink lineart"
    negative="color, gray, midtones, gradients, shading, halftone, screentone, pencil, graphite, charcoal, sketchy/hairy lines, wobble, tapering strokes, variable line weight, faint/light lines, blur, noise, paper texture, watercolor, fills/solid fill, background tint, glow, bloom, 3d render, photorealistic, text, signature, watermark, compression artifacts"

    out=pipe(
        prompt=prompt,
        negative_prompt=negative,
        control_image=[cn_img],
        control_mode=[4],
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        controlnet_conditioning_scale=args.strength,
        width=args.width,
        height=args.height,
        generator=gen
    ).images[0]
    out.save(args.out)

if __name__=="__main__":
    main()
