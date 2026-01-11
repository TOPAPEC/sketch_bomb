# app.py
import os
import io
import base64
import time
import secrets
from datetime import datetime, timedelta
from typing import Optional
from contextlib import asynccontextmanager

import numpy as np
import torch
import cv2
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Header, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from transformers import MobileViTImageProcessor, MobileViTForImageClassification
from diffusers import StableDiffusionXLControlNetUnionPipeline, ControlNetUnionModel, EulerDiscreteScheduler
import jwt

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./history.db")
JWT_SECRET = os.getenv("JWT_SECRET", secrets.token_hex(32))
JWT_ALGORITHM = "HS256"
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "admin-delete-token-12345")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class RequestHistory(Base):
    __tablename__ = "request_history"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    endpoint = Column(String(50))
    image_width = Column(Integer, nullable=True)
    image_height = Column(Integer, nullable=True)
    predicted_label = Column(String(100), nullable=True)
    processing_time_ms = Column(Float)
    status = Column(String(20))
    error_message = Column(Text, nullable=True)


CLASS_LABELS = {
    "0": "aircraft carrier", "1": "airplane", "2": "alarm clock", "3": "ambulance",
    "4": "angel", "5": "animal migration", "6": "ant", "7": "anvil", "8": "apple",
    "9": "arm", "10": "asparagus", "11": "axe", "12": "backpack", "13": "banana",
    "14": "bandage", "15": "barn", "16": "baseball bat", "17": "baseball",
    "18": "basket", "19": "basketball", "20": "bat", "21": "bathtub", "22": "beach",
    "23": "bear", "24": "beard", "25": "bed", "26": "bee", "27": "belt", "28": "bench",
    "29": "bicycle", "30": "binoculars", "31": "bird", "32": "birthday cake",
    "33": "blackberry", "34": "blueberry", "35": "book", "36": "boomerang",
    "37": "bottlecap", "38": "bowtie", "39": "bracelet", "40": "brain", "41": "bread",
    "42": "bridge", "43": "broccoli", "44": "broom", "45": "bucket", "46": "bulldozer",
    "47": "bus", "48": "bush", "49": "butterfly", "50": "cactus", "51": "cake",
    "52": "calculator", "53": "calendar", "54": "camel", "55": "camera",
    "56": "camouflage", "57": "campfire", "58": "candle", "59": "cannon", "60": "canoe",
    "61": "car", "62": "carrot", "63": "castle", "64": "cat", "65": "ceiling fan",
    "66": "cell phone", "67": "cello", "68": "chair", "69": "chandelier", "70": "church",
    "71": "circle", "72": "clarinet", "73": "clock", "74": "cloud", "75": "coffee cup",
    "76": "compass", "77": "computer", "78": "cookie", "79": "cooler", "80": "couch",
    "81": "cow", "82": "crab", "83": "crayon", "84": "crocodile", "85": "crown",
    "86": "cruise ship", "87": "cup", "88": "diamond", "89": "dishwasher",
    "90": "diving board", "91": "dog", "92": "dolphin", "93": "donut", "94": "door",
    "95": "dragon", "96": "dresser", "97": "drill", "98": "drums", "99": "duck",
    "100": "dumbbell", "101": "ear", "102": "elbow", "103": "elephant", "104": "envelope",
    "105": "eraser", "106": "eye", "107": "eyeglasses", "108": "face", "109": "fan",
    "110": "feather", "111": "fence", "112": "finger", "113": "fire hydrant",
    "114": "fireplace", "115": "firetruck", "116": "fish", "117": "flamingo",
    "118": "flashlight", "119": "flip flops", "120": "floor lamp", "121": "flower",
    "122": "flying saucer", "123": "foot", "124": "fork", "125": "frog", "126": "frying pan",
    "127": "garden hose", "128": "garden", "129": "giraffe", "130": "goatee",
    "131": "golf club", "132": "grapes", "133": "grass", "134": "guitar",
    "135": "hamburger", "136": "hammer", "137": "hand", "138": "harp", "139": "hat",
    "140": "headphones", "141": "hedgehog", "142": "helicopter", "143": "helmet",
    "144": "hexagon", "145": "hockey puck", "146": "hockey stick", "147": "horse",
    "148": "hospital", "149": "hot air balloon", "150": "hot dog", "151": "hot tub",
    "152": "hourglass", "153": "house plant", "154": "house", "155": "hurricane",
    "156": "ice cream", "157": "jacket", "158": "jail", "159": "kangaroo", "160": "key",
    "161": "keyboard", "162": "knee", "163": "knife", "164": "ladder", "165": "lantern",
    "166": "laptop", "167": "leaf", "168": "leg", "169": "light bulb", "170": "lighter",
    "171": "lighthouse", "172": "lightning", "173": "line", "174": "lion",
    "175": "lipstick", "176": "lobster", "177": "lollipop", "178": "mailbox",
    "179": "map", "180": "marker", "181": "matches", "182": "megaphone", "183": "mermaid",
    "184": "microphone", "185": "microwave", "186": "monkey", "187": "moon",
    "188": "mosquito", "189": "motorbike", "190": "mountain", "191": "mouse",
    "192": "moustache", "193": "mouth", "194": "mug", "195": "mushroom", "196": "nail",
    "197": "necklace", "198": "nose", "199": "ocean", "200": "octagon", "201": "octopus",
    "202": "onion", "203": "oven", "204": "owl", "205": "paint can", "206": "paintbrush",
    "207": "palm tree", "208": "panda", "209": "pants", "210": "paper clip",
    "211": "parachute", "212": "parrot", "213": "passport", "214": "peanut",
    "215": "pear", "216": "peas", "217": "pencil", "218": "penguin", "219": "piano",
    "220": "pickup truck", "221": "picture frame", "222": "pig", "223": "pillow",
    "224": "pineapple", "225": "pizza", "226": "pliers", "227": "police car",
    "228": "pond", "229": "pool", "230": "popsicle", "231": "postcard", "232": "potato",
    "233": "power outlet", "234": "purse", "235": "rabbit", "236": "raccoon",
    "237": "radio", "238": "rain", "239": "rainbow", "240": "rake", "241": "remote control",
    "242": "rhinoceros", "243": "rifle", "244": "river", "245": "roller coaster",
    "246": "rollerskates", "247": "sailboat", "248": "sandwich", "249": "saw",
    "250": "saxophone", "251": "school bus", "252": "scissors", "253": "scorpion",
    "254": "screwdriver", "255": "sea turtle", "256": "see saw", "257": "shark",
    "258": "sheep", "259": "shoe", "260": "shorts", "261": "shovel", "262": "sink",
    "263": "skateboard", "264": "skull", "265": "skyscraper", "266": "sleeping bag",
    "267": "smiley face", "268": "snail", "269": "snake", "270": "snorkel",
    "271": "snowflake", "272": "snowman", "273": "soccer ball", "274": "sock",
    "275": "speedboat", "276": "spider", "277": "spoon", "278": "spreadsheet",
    "279": "square", "280": "squiggle", "281": "squirrel", "282": "stairs", "283": "star",
    "284": "steak", "285": "stereo", "286": "stethoscope", "287": "stitches",
    "288": "stop sign", "289": "stove", "290": "strawberry", "291": "streetlight",
    "292": "string bean", "293": "submarine", "294": "suitcase", "295": "sun",
    "296": "swan", "297": "sweater", "298": "swing set", "299": "sword", "300": "syringe",
    "301": "t-shirt", "302": "table", "303": "teapot", "304": "teddy-bear",
    "305": "telephone", "306": "television", "307": "tennis racquet", "308": "tent",
    "309": "The Eiffel Tower", "310": "The Great Wall of China", "311": "The Mona Lisa",
    "312": "tiger", "313": "toaster", "314": "toe", "315": "toilet", "316": "tooth",
    "317": "toothbrush", "318": "toothpaste", "319": "tornado", "320": "tractor",
    "321": "traffic light", "322": "train", "323": "tree", "324": "triangle",
    "325": "trombone", "326": "truck", "327": "trumpet", "328": "umbrella",
    "329": "underwear", "330": "van", "331": "vase", "332": "violin",
    "333": "washing machine", "334": "watermelon", "335": "waterslide", "336": "whale",
    "337": "wheel", "338": "windmill", "339": "wine bottle", "340": "wine glass",
    "341": "wristwatch", "342": "yoga", "343": "zebra", "344": "zigzag"
}

SKETCH_NEGATIVE = (
    "people, person, human, character, face, hands, arms, body, "
    "animal, pet, creature, multiple objects, extra objects, props, "
    "table, desk, shelf, library, room, interior, scenery, landscape, "
    "plants, flowers, cup, mug, pen, pencil, smartphone, food, "
    "text, letters, words, title, logo, watermark, signature, "
    "frame, border, vignette, colored, colorful, blurry, low quality"
)

COLOR_NEGATIVE = (
    "people, person, human, animal, multiple objects, extra objects, "
    "table, desk, shelf, library, room, interior, background scene, "
    "plants, flowers, cup, mug, pen, pencil, smartphone, food, "
    "text, letters, words, title, logo, watermark, signature, "
    "frame, border, sketch, monochrome, black and white, ugly, blurry"
)


def get_sketch_prompt(label: str) -> str:
    return (
        f"a single {label} only, isolated subject, centered composition, "
        "professional graphite pencil sketch, clean background, thick black strokes, "
        "no scene, no props, no extra objects, highly detailed shading, grayscale"
    )


def get_color_prompt(label: str) -> str:
    return (
        f"a single {label} only, isolated subject, centered, "
        "clean plain background, no environment, no props, "
        "high quality digital illustration, detailed, warm colors, beautiful"
    )


class SketchClassifier:
    def __init__(self, device: str = "cuda"):
        self.device = device
        model_name = "Xenova/quickdraw-mobilevit-small"
        self.model = MobileViTForImageClassification.from_pretrained(model_name).to(device)
        self.processor = MobileViTImageProcessor.from_pretrained(model_name)
        self.model.eval()

    def classify(self, image: Image.Image) -> tuple[str, float]:
        image_gray = image.convert("L")
        inputs = self.processor(images=image_gray, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        idx = logits.argmax(-1).item()
        score = torch.softmax(logits, dim=-1)[0, idx].item()
        return self.model.config.id2label[idx], score


class SketchToImagePipeline:
    def __init__(self, device: str = "cuda", img_size: int = 1024):
        self.device = device
        self.img_size = img_size
        self.controlnet = ControlNetUnionModel.from_pretrained(
            "xinsir/controlnet-union-sdxl-1.0", torch_dtype=torch.float16
        ).to(device)
        self.pipe = StableDiffusionXLControlNetUnionPipeline.from_pretrained(
            "cagliostrolab/animagine-xl-4.0",
            controlnet=self.controlnet,
            torch_dtype=torch.float16,
        ).to(device)
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_attention_slicing()

    def preprocess_lineart(self, image: Image.Image) -> Image.Image:
        arr = np.array(image.convert('L'))
        _, binary = cv2.threshold(arr, 200, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=1)
        return Image.fromarray(binary).convert('RGB')

    def preprocess_canny(self, image: Image.Image, low: int = 50, high: int = 150) -> Image.Image:
        gray = np.array(image.convert('L'))
        edges = cv2.Canny(gray, low, high)
        return Image.fromarray(edges).convert('RGB')

    def generate_sketch(self, image: Image.Image, prompt: str, steps: int = 30, strong_steps: int = 30, seed: int = 42) -> Image.Image:
        image = image.resize((self.img_size, self.img_size))
        ctrl_img = self.preprocess_lineart(image)
        control_end = min(1.0, max(0.0, strong_steps / steps))
        generator = torch.Generator(device=self.device).manual_seed(seed)
        return self.pipe(
            prompt=prompt, negative_prompt=SKETCH_NEGATIVE, control_image=[ctrl_img], control_mode=[3],
            num_inference_steps=steps, guidance_scale=8.0, controlnet_conditioning_scale=0.7,
            control_guidance_start=0.0, control_guidance_end=control_end, generator=generator
        ).images[0]

    def colorize(self, sketch: Image.Image, prompt: str, steps: int = 40, strong_steps: int = 10, seed: int = 42) -> Image.Image:
        ctrl_img = self.preprocess_canny(sketch)
        control_end = min(1.0, max(0.0, strong_steps / steps))
        generator = torch.Generator(device=self.device).manual_seed(seed)
        return self.pipe(
            prompt=prompt, negative_prompt=COLOR_NEGATIVE, control_image=[ctrl_img], control_mode=[3],
            num_inference_steps=steps, guidance_scale=8.0, controlnet_conditioning_scale=0.5,
            control_guidance_start=0.0, control_guidance_end=control_end, generator=generator
        ).images[0]


classifier: Optional[SketchClassifier] = None
pipeline: Optional[SketchToImagePipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier, pipeline
    Base.metadata.create_all(bind=engine)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = SketchClassifier(device)
    pipeline = SketchToImagePipeline(device)
    yield
    classifier = None
    pipeline = None


app = FastAPI(title="Sketch Classification & Generation API", lifespan=lifespan)
security = HTTPBearer(auto_error=False)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_jwt_token(user_id: str, is_admin: bool = False) -> str:
    payload = {"sub": user_id, "admin": is_admin, "exp": datetime.utcnow() + timedelta(hours=24)}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_jwt(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    if not credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing token")
    try:
        return jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


def require_admin(payload: dict = Depends(verify_jwt)) -> dict:
    if not payload.get("admin"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin required")
    return payload


def image_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode()


def save_history(db: Session, endpoint: str, width: int, height: int, label: str, time_ms: float, status_str: str, error: str = None):
    record = RequestHistory(
        endpoint=endpoint, image_width=width, image_height=height,
        predicted_label=label, processing_time_ms=time_ms, status=status_str, error_message=error
    )
    db.add(record)
    db.commit()


class TokenRequest(BaseModel):
    user_id: str
    is_admin: bool = False


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@app.post("/token", response_model=TokenResponse)
def get_token(request: TokenRequest):
    return TokenResponse(access_token=create_jwt_token(request.user_id, request.is_admin))


@app.post("/forward")
async def forward(
    image: UploadFile = File(...),
    x_generate_sketch: Optional[str] = Header(default="true"),
    x_colorize: Optional[str] = Header(default="true"),
    x_steps: Optional[int] = Header(default=30),
    x_seed: Optional[int] = Header(default=42),
    db: Session = Depends(get_db)
):
    start_time = time.time()
    width, height, label = None, None, None
    
    try:
        contents = await image.read()
        if not contents:
            raise HTTPException(status_code=400, detail="bad request")
        try:
            pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="bad request")
        width, height = pil_image.size
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="bad request")

    try:
        label, confidence = classifier.classify(pil_image)
        result = {"label": label, "confidence": round(confidence, 4)}
        generate_sketch = x_generate_sketch.lower() == "true"
        colorize = x_colorize.lower() == "true"

        if generate_sketch:
            sketch_image = pipeline.generate_sketch(
                pil_image, get_sketch_prompt(label), steps=x_steps, seed=x_seed
            )
            result["sketch_image"] = image_to_base64(sketch_image)

            if colorize:
                colored_image = pipeline.colorize(
                    sketch_image, get_color_prompt(label), steps=x_steps + 10, seed=x_seed
                )
                result["colored_image"] = image_to_base64(colored_image)

        elapsed_ms = (time.time() - start_time) * 1000
        save_history(db, "/forward", width, height, label, elapsed_ms, "success")
        return result

    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        save_history(db, "/forward", width, height, label, elapsed_ms, "error", str(e))
        raise HTTPException(status_code=403, detail="модель не смогла обработать данные")


@app.get("/history")
def get_history(db: Session = Depends(get_db), _: dict = Depends(verify_jwt)):
    records = db.query(RequestHistory).order_by(RequestHistory.timestamp.desc()).all()
    return [
        {
            "id": r.id, "timestamp": r.timestamp.isoformat(), "endpoint": r.endpoint,
            "image_width": r.image_width, "image_height": r.image_height,
            "predicted_label": r.predicted_label, "processing_time_ms": round(r.processing_time_ms, 2),
            "status": r.status, "error_message": r.error_message
        }
        for r in records
    ]


@app.delete("/history")
def delete_history(x_admin_token: str = Header(...), db: Session = Depends(get_db), _: dict = Depends(require_admin)):
    if x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid admin token")
    count = db.query(RequestHistory).delete()
    db.commit()
    return {"deleted": count}


@app.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    records = db.query(RequestHistory).filter(RequestHistory.status == "success").all()
    if not records:
        return {"message": "No successful requests yet"}
    times = [r.processing_time_ms for r in records]
    widths = [r.image_width for r in records if r.image_width]
    heights = [r.image_height for r in records if r.image_height]
    return {
        "total_requests": len(records),
        "processing_time": {
            "mean": round(float(np.mean(times)), 2),
            "p50": round(float(np.percentile(times, 50)), 2),
            "p95": round(float(np.percentile(times, 95)), 2),
            "p99": round(float(np.percentile(times, 99)), 2)
        },
        "image_dimensions": {
            "width": {"mean": round(float(np.mean(widths)), 2), "min": min(widths), "max": max(widths)} if widths else None,
            "height": {"mean": round(float(np.mean(heights)), 2), "min": min(heights), "max": max(heights)} if heights else None
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)