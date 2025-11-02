# sdxl_controlnet_union_infer.py
import argparse, torch, os, random
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
    ap.add_argument("--controlnet",default="brad-twinkl/controlnet-union-sdxl-1.0-promax")
    ap.add_argument("--width",type=int,default=1024)
    ap.add_argument("--height",type=int,default=1024)
    ap.add_argument("--steps",type=int,default=20)
    ap.add_argument("--cfg",type=float,default=7.5)
    ap.add_argument("--strength",type=float,default=1.0)
    ap.add_argument("--seed",type=int,default=1340)
    args=ap.parse_args()

    cn=ControlNetUnionModel.from_pretrained(args.controlnet, torch_dtype=torch.float16)
    pipe=StableDiffusionXLControlNetUnionPipeline.from_pretrained(args.model, controlnet=cn, torch_dtype=torch.float16)
    pipe.scheduler=DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    pipe.to("cuda")

    cn_img=center_square_resize_invert(Image.open(args.sketch).convert("RGB"), args.width, args.height)
    cn_path=os.path.join(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", "interm_cn.png")
    cn_img.save(cn_path)

    prompt="anime aesthetic style"
    negative=""


    cn_steps = int(0.5 * args.steps) 
    imgs=[]
    with torch.inference_mode():
        for i in range(9):
            print(f"Generating {i}")
            gen=torch.Generator(device="cuda").manual_seed(args.seed+i)
            img=pipe(
                prompt=prompt,
                negative_prompt=negative,
                control_image=[cn_img],
                control_mode=[4],
                num_inference_steps=args.steps,
                guidance_scale=args.cfg,
                controlnet_conditioning_scale=[args.strength if i < cn_steps else 0. for i in range(args.steps)],
                width=args.width,
                height=args.height,
                generator=gen
            ).images[0]
            imgs.append(img)

    grid_w, grid_h = args.width*3, args.height*3
    canvas=Image.new("RGB",(grid_w,grid_h),"white")
    for idx,img in enumerate(imgs):
        r, c = divmod(idx,3)
        canvas.paste(img,(c*args.width, r*args.height))

    out_w, out_h = grid_w//2, grid_h//2
    canvas=canvas.resize((out_w,out_h), Image.LANCZOS)
    canvas.save(args.out)

if __name__=="__main__":
    main()
