# mvp_yolo_sketch_extract.py
import cv2, numpy as np, argparse, os
from ultralytics import YOLO

def xdog(gray, sigma=0.8, k=1.6, tau=0.98, phi=10.0, eps=-0.1):
    g1=cv2.GaussianBlur(gray,(0,0),sigma)
    g2=cv2.GaussianBlur(gray,(0,0),sigma*k)
    d=g1 - tau*g2
    d=(d-np.mean(d))/max(np.std(d),1e-6)
    x=np.tanh(phi*(d-eps))
    x=(x-np.min(x))/(np.max(x)-np.min(x)+1e-6)
    sk=(x<0.5).astype(np.uint8)*255
    return sk

def boost_contrast(sketch):
    p1,p99=np.percentile(sketch,[1,99])
    sc=np.clip((sketch-p1)*(255.0/max(p99-p1,1)),0,255).astype(np.uint8)
    clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    sc=clahe.apply(sc)
    blur=cv2.GaussianBlur(sc,(0,0),1.0)
    sharp=cv2.addWeighted(sc,1.6,blur,-0.6,0)
    return sharp

def edges_reinforce(gray,sketch):
    v=np.median(gray)
    t1=int(max(0,0.66*v)); t2=int(min(255,1.33*v))
    e=cv2.Canny(gray,t1,t2)
    out=cv2.subtract(sketch,(e*0.75).astype(np.uint8))
    return out

def img_to_sketch_contrast(img):
    g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sk=xdog(g,sigma=0.8,k=1.6,tau=0.98,phi=12.0,eps=-0.1)
    sk=boost_contrast(sk)
    sk=edges_reinforce(g,sk)
    sk=cv2.threshold(sk,220,255,cv2.THRESH_BINARY)[1]
    return sk

def thicken_sketch(sketch, scale=3.0):
    if sketch.ndim==3:
        sketch=cv2.cvtColor(sketch,cv2.COLOR_BGR2GRAY)
    bw=(sketch<128).astype(np.uint8)
    ww=(sketch>127).astype(np.uint8)
    ink_is_white = ww.sum()<bw.sum()
    ink = ww if ink_is_white else bw
    if ink.sum()==0:
        return sketch
    k3=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    ink=cv2.morphologyEx(ink,cv2.MORPH_OPEN,k3)
    dt=cv2.distanceTransform(ink,cv2.DIST_L2,3)
    d=cv2.dilate(dt,k3)
    ridge=(dt>=d-1e-6) & (dt>0)
    r=dt[ridge]
    if r.size==0:
        r=dt[dt>0]
    r_avg=float(np.median(r)) if r.size else 1.0
    delta=int(max(1,min(32,round((scale-1.0)*r_avg))))
    ker=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*delta+1,2*delta+1))
    thick=cv2.dilate(ink,ker)
    thick=cv2.morphologyEx(thick,cv2.MORPH_CLOSE,k3)
    if ink_is_white:
        out=(thick*255).astype(np.uint8)
    else:
        out=((1-thick)*255).astype(np.uint8)
    return out

def multiply_random_offsets(sketch, repeats=16, radius=2, seed=None):
    if sketch.ndim==3:
        sketch=cv2.cvtColor(sketch,cv2.COLOR_BGR2GRAY)
    rng=np.random.default_rng(seed)
    bw=(sketch<128).sum()
    ww=(sketch>127).sum()
    ink_is_white = ww<bw
    a=sketch.astype(np.float32)/255.0
    if ink_is_white:
        a=1.0-a
    acc=np.ones_like(a,dtype=np.float32)
    h,w=a.shape
    for _ in range(repeats):
        dx=int(rng.integers(-radius,radius+1))
        dy=int(rng.integers(-radius,radius+1))
        M=np.float32([[1,0,dx],[0,1,dy]])
        sh=cv2.warpAffine(a,M,(w,h),flags=cv2.INTER_NEAREST,borderValue=1.0)
        acc*=sh
    out=acc
    if ink_is_white:
        out=1.0-out
    return np.clip(out*255,0,255).astype(np.uint8)

def combine_masks(masks, mode):
    if masks is None or len(masks)==0:
        return None
    areas=[int(m.sum()) for m in masks]
    if mode=="largest":
        return (masks[int(np.argmax(areas))]>0).astype(np.uint8)
    m=np.zeros_like(masks[0],dtype=np.uint8)
    for mk in masks:
        m|=(mk>0).astype(np.uint8)
    return m

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True)
    ap.add_argument("--outdir",default="out")
    ap.add_argument("--model",default="yolov8n-seg.pt")
    ap.add_argument("--mode",choices=["largest","all"],default="largest")
    ap.add_argument("--classes",type=int,nargs="*",default=None)
    args=ap.parse_args()
    os.makedirs(args.outdir,exist_ok=True)

    img=cv2.imread(args.input,cv2.IMREAD_COLOR)
    h,w=img.shape[:2]

    model=YOLO(args.model)
    res=model(img,verbose=False)[0]

    masks=[]
    for i,box in enumerate(res.boxes):
        if args.classes is not None:
            if int(box.cls.item()) not in args.classes:
                continue
        if res.masks is None:
            continue
        mk=res.masks.data[i].cpu().numpy().astype(np.uint8)
        mk=cv2.resize(mk,(w,h),interpolation=cv2.INTER_NEAREST)
        masks.append(mk)

    mask=combine_masks(masks,args.mode)
    if mask is None:
        white=np.full((h,w,3),255,np.uint8)
        sketch=img_to_sketch_contrast(white)
        thick=thicken_sketch(sketch,3.0)
        mul=multiply_random_offsets(sketch,16,2,None)
        cv2.imwrite(os.path.join(args.outdir,"02_yolo_mask.png"),np.zeros((h,w),np.uint8))
        cv2.imwrite(os.path.join(args.outdir,"03_yolo_masked.png"),white)
        cv2.imwrite(os.path.join(args.outdir,"04_yolo_sketch.png"),sketch)
        cv2.imwrite(os.path.join(args.outdir,"05_yolo_sketch_thick.png"),thick)
        cv2.imwrite(os.path.join(args.outdir,"06_yolo_sketch_mul.png"),mul)
        print(os.path.join(args.outdir,"04_yolo_sketch.png"))
        return

    mask3=np.dstack([mask*255]*3)
    bg=np.full((h,w,3),255,np.uint8)
    masked=np.where(mask3>0,img,bg)
    sketch_region=img_to_sketch_contrast(masked)
    inv_mask=((mask==0)*255).astype(np.uint8)
    inv_mask3=np.dstack([inv_mask]*3)
    composite=np.where(inv_mask3>0,bg,cv2.cvtColor(sketch_region,cv2.COLOR_GRAY2BGR))
    sketch_gray=cv2.cvtColor(composite,cv2.COLOR_BGR2GRAY)
    thick=thicken_sketch(sketch_gray,3.0)
    mul=multiply_random_offsets(sketch_gray,16,3,None)

    cv2.imwrite(os.path.join(args.outdir,"03_yolo_mask.png"),mask*255)
    cv2.imwrite(os.path.join(args.outdir,"03_yolo_masked.png"),masked)
    cv2.imwrite(os.path.join(args.outdir,"04_yolo_sketch.png"),sketch_gray)
    cv2.imwrite(os.path.join(args.outdir,"05_yolo_sketch_thick.png"),thick)
    cv2.imwrite(os.path.join(args.outdir,"06_yolo_sketch_mul.png"),mul)
    print(os.path.join(args.outdir,"04_yolo_sketch.png"))

if __name__=="__main__":
    main()
