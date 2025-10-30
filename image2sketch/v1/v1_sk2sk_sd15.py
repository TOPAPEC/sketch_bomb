# sd15_controlnet_scribble_clear_sketch.py
import argparse, os, torch
from PIL import Image, ImageOps
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DPMSolverMultistepScheduler

def load_scribble(p):
    im=Image.open(p).convert("L")
    im=ImageOps.invert(im)
    im=im.convert("RGB")
    return im

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--sketch",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--model",default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--controlnet",default="lllyasviel/sd-controlnet-scribble")
    ap.add_argument("--width",type=int,default=768)
    ap.add_argument("--height",type=int,default=768)
    ap.add_argument("--steps",type=int,default=30)
    ap.add_argument("--cfg",type=float,default=7.5)
    ap.add_argument("--strength",type=float,default=1.0)
    ap.add_argument("--seed",type=int,default=1234)
    args=ap.parse_args()

    prompt="clean lineart, uniform stroke width, high-contrast edges, no shading, white background"
    negative="color, shading, blur, artifacts, text, watermark, background noise"

    torch.manual_seed(args.seed)
    device="cuda" if torch.cuda.is_available() else "cpu"
    dtype=torch.bfloat16 if device=="cuda" and torch.cuda.is_bf16_supported() else (torch.float16 if device=="cuda" else torch.float32)

    controlnet=ControlNetModel.from_pretrained(args.controlnet, torch_dtype=dtype)
    pipe=StableDiffusionControlNetPipeline.from_pretrained(args.model, controlnet=controlnet, safety_checker=None, torch_dtype=dtype)
    pipe.scheduler=DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    cond=load_scribble(args.sketch).resize((args.width,args.height))
    img=pipe(prompt=prompt, negative_prompt=negative, image=cond, num_inference_steps=args.steps, guidance_scale=args.cfg, controlnet_conditioning_scale=args.strength, width=args.width, height=args.height).images[0]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    img.save(args.out)

if __name__=="__main__":
    main()
