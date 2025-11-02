# sd15_controlnet_scribble_clear_sketch.py
import argparse, os, torch
from PIL import Image, ImageOps, ImageDraw, ImageFont
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DPMSolverMultistepScheduler
from transformers import AutoProcessor, AutoModel

def load_scribble(p):
    im=Image.open(p).convert("L")
    im=ImageOps.invert(im)
    return im.convert("RGB")

def center_square_resize(im, w, h):
    W,H=im.size
    m=min(W,H); x=(W-m)//2; y=(H-m)//2
    sq=im.crop((x,y,x+m,y+m)).resize((w,h), Image.BICUBIC)
    if sq.mode!="RGB": sq=sq.convert("RGB")
    return sq

def siglip2_distance(model, processor, device, img, text):
    inputs=processor(text=[text.lower()], images=img, padding="max_length", max_length=64, return_tensors="pt").to(device)
    with torch.no_grad():
        out=model(**inputs)
    prob=torch.sigmoid(out.logits_per_image)[0,0].item()
    return 1.0-prob

def get_font(size, path=None):
    if path and os.path.exists(path): return ImageFont.truetype(path, size=size)
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf","/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf","/usr/share/fonts/truetype/freefont/FreeSans.ttf","/System/Library/Fonts/Supplemental/Arial Unicode.ttf","/Library/Fonts/Arial.ttf"]:
        if os.path.exists(p): return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--sketch",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--model",default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--controlnet",default="lllyasviel/control_v11p_sd15_lineart")
    ap.add_argument("--width",type=int,default=768)
    ap.add_argument("--height",type=int,default=768)
    ap.add_argument("--steps",type=int,default=30)
    ap.add_argument("--cfg",type=float,default=3.0)
    ap.add_argument("--strength",type=float,default=2.0)
    ap.add_argument("--seed",type=int,default=1337)
    ap.add_argument("--control_steps",type=int,default=5)
    ap.add_argument("--n",type=int,default=9)
    ap.add_argument("--siglip2",default="google/siglip2-base-patch16-224")
    ap.add_argument("--font_size",type=int,default=32)
    ap.add_argument("--font_path",default="")
    args=ap.parse_args()

    prompt="colorful anime artstyle and nature background, rich colors, medieval setting"
    negative=""
    device="cuda" if torch.cuda.is_available() else "cpu"
    dtype=torch.bfloat16 if device=="cuda" and torch.cuda.is_bf16_supported() else (torch.float16 if device=="cuda" else torch.float32)

    controlnet=ControlNetModel.from_pretrained(args.controlnet, torch_dtype=dtype)
    pipe=StableDiffusionControlNetPipeline.from_pretrained(args.model, controlnet=controlnet, safety_checker=None, torch_dtype=dtype)
    pipe.scheduler=DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    try: pipe.enable_xformers_memory_efficient_attention()
    except: pass

    n_ctrl=max(0,min(args.control_steps,args.steps))
    scale = args.strength
    cond=center_square_resize(load_scribble(args.sketch), args.width, args.height)
    cn_path=os.path.join(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", "interm_cn.png")
    cond.save(cn_path)

    sl_model=AutoModel.from_pretrained(args.siglip2, attn_implementation="sdpa", torch_dtype=(torch.float16 if device=="cuda" else torch.float32)).to(device)
    sl_proc=AutoProcessor.from_pretrained(args.siglip2)
    text_query="lineart 2d illustration, Monotonous background, minimal visual details"
    font=get_font(args.font_size, args.font_path if args.font_path else None)

    imgs=[]
    for i in range(args.n):
        gen=torch.Generator(device=device).manual_seed(args.seed+i)
        img=pipe(prompt=prompt, negative_prompt=negative, image=cond, num_inference_steps=args.steps, guidance_scale=args.cfg, controlnet_conditioning_scale=scale, width=args.width, height=args.height, generator=gen).images[0]
        dist=siglip2_distance(sl_model, sl_proc, device, img, text_query)
        rimg=img.resize((args.width//2, args.height//2), Image.BICUBIC)
        draw=ImageDraw.Draw(rimg)
        txt=f"dist={dist:.3f}"
        bbox=draw.textbbox((0,0), txt, font=font)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        draw.rectangle([4,4,8+tw,8+th], fill="white")
        draw.text((6,6), txt, fill="black", font=font)
        imgs.append(rimg)

    cols=3; rows=(args.n+cols-1)//cols
    tile_w=args.width//2; tile_h=args.height//2
    canvas=Image.new("RGB",(tile_w*cols,tile_h*rows),"white")
    for idx,im in enumerate(imgs):
        r,c=divmod(idx,cols)
        canvas.paste(im,(c*tile_w,r*tile_h))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    canvas.save(args.out)

if __name__=="__main__":
    main()
