# mvp_yolo_sketch_extract.py
import cv2, numpy as np, argparse, os
from ultralytics import YOLO

try:
    from controlnet_aux import HEDdetector, PidiNetDetector, LineartDetector, LineartAnimeDetector
    try:
        from controlnet_aux.mlsd import MLSDdetector
        _HAS_MLSD = True
    except Exception:
        _HAS_MLSD = False
    _HAS_CNAUX = True
except Exception:
    _HAS_CNAUX = False
    _HAS_MLSD = False

def xdog(gray, sigma=0.8, k=1.6, tau=0.98, phi=12.0, eps=-0.1):
    g1=cv2.GaussianBlur(gray,(0,0),sigma)
    g2=cv2.GaussianBlur(gray,(0,0),sigma*k)
    d=g1 - tau*g2
    d=(d-d.mean())/max(d.std(),1e-6)
    x=np.tanh(phi*(d-eps))
    x=(x-x.min())/(x.max()-x.min()+1e-6)
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

def multiply_random_offsets(sketch, repeats=16, radius=3, seed=None):
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

def method_xdog(img):
    g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sk=xdog(g)
    sk=boost_contrast(sk)
    sk=edges_reinforce(g,sk)
    sk=cv2.threshold(sk,220,255,cv2.THRESH_BINARY)[1]
    return sk

def method_canny(img):
    g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    v=np.median(g)
    t1=int(max(0,0.66*v)); t2=int(min(255,1.33*v))
    e=cv2.Canny(g,t1,t2)
    return 255-e

def method_sobel(img):
    g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sx=cv2.Sobel(g,cv2.CV_32F,1,0,ksize=3)
    sy=cv2.Sobel(g,cv2.CV_32F,0,1,ksize=3)
    m=cv2.magnitude(sx,sy)
    m=(m/(m.max()+1e-6)*255).astype(np.uint8)
    return 255-m

def method_scharr(img):
    g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sx=cv2.Scharr(g,cv2.CV_32F,1,0)
    sy=cv2.Scharr(g,cv2.CV_32F,0,1)
    m=cv2.magnitude(sx,sy)
    m=(m/(m.max()+1e-6)*255).astype(np.uint8)
    return 255-m

def method_laplacian(img):
    g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    l=cv2.Laplacian(g,cv2.CV_16S,ksize=3)
    a=cv2.convertScaleAbs(l)
    return 255-a

def method_adaptive(img):
    g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    g=cv2.bilateralFilter(g,9,75,75)
    b=cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,21,5)
    return b

_detectors_cache={}

def _require_cnaux():
    if not _HAS_CNAUX:
        raise RuntimeError("Install controlnet-aux to use deep methods: pip install controlnet-aux")

def _get_detector(name):
    if name in _detectors_cache:
        return _detectors_cache[name]
    _require_cnaux()
    if name=="hed":
        d=HEDdetector.from_pretrained("lllyasviel/Annotators")
    elif name=="pidinet":
        d=PidiNetDetector.from_pretrained("lllyasviel/Annotators")
    elif name=="lineart":
        d=LineartDetector.from_pretrained("lllyasviel/Annotators")
    elif name=="lineart_anime":
        d=LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")
    elif name=="mlsd":
        if not _HAS_MLSD:
            raise RuntimeError("MLSD not available in controlnet-aux version")
        d=MLSDdetector.from_pretrained("lllyasviel/Annotators")
    else:
        raise RuntimeError("Unknown detector")
    _detectors_cache[name]=d
    return d

def _cnaux_to_gray(img, det):
    from PIL import Image
    p=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    o=det(p)
    if hasattr(o,"convert"):
        o=o.convert("L")
        a=np.array(o)
    else:
        a=np.array(o)
        if a.ndim==3:
            a=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY)
    return a

def method_hed(img):
    d=_get_detector("hed")
    a=_cnaux_to_gray(img,d)
    return a

def method_pidinet(img):
    d=_get_detector("pidinet")
    a=_cnaux_to_gray(img,d)
    return a

def method_lineart(img):
    d=_get_detector("lineart")
    a=_cnaux_to_gray(img,d)
    return a

def method_lineart_anime(img):
    d=_get_detector("lineart_anime")
    a=_cnaux_to_gray(img,d)
    return a

def method_mlsd(img):
    d=_get_detector("mlsd")
    a=_cnaux_to_gray(img,d)
    return a

def apply_method(img, method):
    if method=="xdog": return method_xdog(img)
    if method=="canny": return method_canny(img)
    if method=="sobel": return method_sobel(img)
    if method=="scharr": return method_scharr(img)
    if method=="laplacian": return method_laplacian(img)
    if method=="adaptive": return method_adaptive(img)
    if method=="hed": return method_hed(img)
    if method=="pidinet": return method_pidinet(img)
    if method=="lineart": return method_lineart(img)
    if method=="lineart_anime": return method_lineart_anime(img)
    if method=="mlsd": return method_mlsd(img)
    return method_xdog(img)

def available_methods():
    base=["xdog","canny","sobel","scharr","laplacian","adaptive"]
    deep=[]
    if _HAS_CNAUX:
        deep+=["hed","pidinet","lineart","lineart_anime"]
        if _HAS_MLSD:
            deep+=["mlsd"]
    return base+deep

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
    ap.add_argument("--method",choices=["all","xdog","canny","sobel","scharr","laplacian","adaptive","hed","pidinet","lineart","lineart_anime","mlsd"],default="all")
    ap.add_argument("--thick_scale",type=float,default=3.0)
    ap.add_argument("--mul_repeats",type=int,default=16)
    ap.add_argument("--mul_radius",type=int,default=3)
    args=ap.parse_args()
    os.makedirs(args.outdir,exist_ok=True)
    img=cv2.imread(args.input,cv2.IMREAD_COLOR)
    h,w=img.shape[:2]
    model=YOLO(args.model)
    res=model(img,verbose=False)[0]
    masks=[]
    if res.masks is not None:
        for i,box in enumerate(res.boxes):
            if args.classes is not None and int(box.cls.item()) not in args.classes:
                continue
            mk=res.masks.data[i].cpu().numpy().astype(np.uint8)
            mk=cv2.resize(mk,(w,h),interpolation=cv2.INTER_NEAREST)
            masks.append(mk)
    mask=combine_masks(masks,args.mode)
    if mask is None:
        mask=np.ones((h,w),np.uint8)
        masked=img.copy()
    else:
        mask3=np.dstack([mask*255]*3)
        bg=np.full((h,w,3),255,np.uint8)
        masked=np.where(mask3>0,img,bg)
    cv2.imwrite(os.path.join(args.outdir,"02_yolo_mask.png"),(mask*255).astype(np.uint8))
    cv2.imwrite(os.path.join(args.outdir,"03_yolo_masked.png"),masked)
    methods=available_methods() if args.method=="all" else [args.method]
    for m in methods:
        sketch=apply_method(masked,m)
        inv_mask=((mask==0)*255).astype(np.uint8)
        inv_mask3=np.dstack([inv_mask]*3)
        composite=np.where(inv_mask3>0,255,cv2.cvtColor(sketch,cv2.COLOR_GRAY2BGR))
        sketch_gray=cv2.cvtColor(composite,cv2.COLOR_BGR2GRAY)
        thick=thicken_sketch(sketch_gray,args.thick_scale)
        mul=multiply_random_offsets(sketch_gray,args.mul_repeats,args.mul_radius,None)
        cv2.imwrite(os.path.join(args.outdir,f"04_{m}_sketch.png"),sketch_gray)
        cv2.imwrite(os.path.join(args.outdir,f"05_{m}_sketch_thick.png"),thick)
        cv2.imwrite(os.path.join(args.outdir,f"06_{m}_sketch_mul.png"),mul)

if __name__=="__main__":
    main()

