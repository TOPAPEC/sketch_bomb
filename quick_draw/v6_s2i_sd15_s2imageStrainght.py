import numpy as np
import requests
import json
from PIL import Image, ImageDraw, ImageFont
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, EulerDiscreteScheduler
from transformers import AutoProcessor, AutoModel
import cv2
from pathlib import Path
import random

CLASSES = ["book", "cat", "car", "tree", "house", "guitar", "airplane", "flower",
           "bird", "fish", "bicycle", "clock", "chair", "apple", "sun", "umbrella"]

SD15_DUMPS = [
    "runwayml/stable-diffusion-v1-5",
    "dreamlike-art/dreamlike-photoreal-2.0",
    # Add more SD 1.5 model paths here
]

COLOR_NEGATIVE = (
    "people, person, human, animal, multiple objects, extra objects, "
    "table, desk, shelf, library, room, interior, background scene, "
    "plants, flowers, cup, mug, pen, pencil, smartphone, food, "
    "text, letters, words, title, logo, watermark, signature, "
    "frame, border, sketch, monochrome, black and white, ugly, blurry"
)


def get_color_prompt(label: str) -> str:
    return (
        f"a single {label} only, isolated subject, centered, "
        "clean plain background, no environment, no props, "
        "high quality digital illustration, detailed, warm colors, beautiful"
    )


class SigLIPScorer:
    def __init__(self, device="cuda"):
        self.device = device
        self.model = AutoModel.from_pretrained("google/siglip2-base-patch16-224", torch_dtype=torch.float16).to(device)
        self.processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")
        
    @torch.no_grad()
    def score(self, images: list[Image.Image], texts: list[str]) -> np.ndarray:
        inputs = self.processor(text=texts, images=images, padding="max_length", return_tensors="pt").to(self.device)
        return np.diag(self.model(**inputs).logits_per_image.cpu().numpy())


class Sketch2Image:
    def __init__(self, device="cuda", img_size=512):
        self.device = device
        self.img_size = img_size
        self.controlnet = None
        self.pipe = None
        self.scorer = SigLIPScorer(device)
        
    def setup_models(self, sd_model: str):
        print(f"Loading models with {sd_model}...")
        if self.controlnet is None:
            self.controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11p_sd15_lineart", torch_dtype=torch.float16).to(self.device)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            sd_model, controlnet=self.controlnet, torch_dtype=torch.float16, safety_checker=None).to(self.device)
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_attention_slicing()
        print("Models loaded!")
    
    def download_quickdraw_batch(self, category: str, num_samples: int, random_offset: int = 0):
        url = f"https://storage.googleapis.com/quickdraw_dataset/full/simplified/{category}.ndjson"
        response = requests.get(url, stream=True)
        sketches = []
        for i, line in enumerate(response.iter_lines()):
            if i < random_offset:
                continue
            if len(sketches) >= num_samples:
                break
            if line:
                sketches.append(json.loads(line))
        return sketches
    
    def quickdraw_to_image(self, drawing_data):
        image = Image.new('RGB', (self.img_size, self.img_size), 'white')
        draw = ImageDraw.Draw(image)
        strokes = drawing_data['drawing']
        all_x = [x for stroke in strokes for x in stroke[0]]
        all_y = [y for stroke in strokes for y in stroke[1]]
        if not all_x or not all_y:
            return image
        min_x, max_x, min_y, max_y = min(all_x), max(all_x), min(all_y), max(all_y)
        scale = self.img_size * 0.8 / max(max_x - min_x, max_y - min_y, 1)
        offset_x = (self.img_size - (max_x - min_x) * scale) / 2 - min_x * scale
        offset_y = (self.img_size - (max_y - min_y) * scale) / 2 - min_y * scale
        for stroke in strokes:
            points = [(x * scale + offset_x, y * scale + offset_y) for x, y in zip(stroke[0], stroke[1])]
            if len(points) > 1:
                draw.line(points, fill='black', width=max(2, self.img_size // 170))
        return image
    
    def preprocess_lineart(self, image):
        arr = np.array(image.convert('L'))
        _, binary = cv2.threshold(arr, 200, 255, cv2.THRESH_BINARY_INV)
        binary = cv2.dilate(binary, np.ones((2, 2), np.uint8), iterations=1)
        return Image.fromarray(binary).convert('RGB')
    
    def create_paired_grid(self, sketches: list[Image.Image], results: list[Image.Image], 
                           labels: list[str], scores: list[float], samples_per_class: int = 4):
        img_size = sketches[0].size[0]
        num_classes = len(labels)
        cols = samples_per_class * 2
        rows = num_classes
        label_width = 200
        
        grid = Image.new('RGB', (img_size * cols + label_width, img_size * rows), 'white')
        draw = ImageDraw.Draw(grid)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        for class_idx, label in enumerate(labels):
            for i in range(samples_per_class):
                idx = class_idx * samples_per_class + i
                x_sketch = i * 2 * img_size
                x_result = x_sketch + img_size
                y = class_idx * img_size
                grid.paste(sketches[idx], (x_sketch, y))
                grid.paste(results[idx], (x_result, y))
            
            label_x = cols * img_size + 10
            label_y = class_idx * img_size + img_size // 2 - 20
            draw.text((label_x, label_y), f"{label}", fill='black', font=font)
            draw.text((label_x, label_y + 25), f"score: {scores[class_idx]:.3f}", fill='gray', font=font)
        
        return grid
    
    def batch_colorize(self, control_images, labels, num_inference_steps=30, strong_steps=12):
        control_end = min(1.0, strong_steps / num_inference_steps)
        results = []
        for i, (ctrl_img, label) in enumerate(zip(control_images, labels)):
            print(f"  {i+1}/{len(control_images)} ({label})")
            result = self.pipe(
                prompt=get_color_prompt(label), negative_prompt=COLOR_NEGATIVE,
                image=ctrl_img, num_inference_steps=num_inference_steps, guidance_scale=7.5,
                controlnet_conditioning_scale=0.7, control_guidance_start=0.0, control_guidance_end=control_end,
                generator=torch.Generator(device=self.device).manual_seed(42 + i)
            ).images[0]
            results.append(result)
        return results

    def compute_class_scores(self, images: list[Image.Image], labels: list[str], samples_per_class: int = 4):
        prompts = [get_color_prompt(label) for label in labels]
        scores = self.scorer.score(images, prompts)
        return [np.mean(scores[i*samples_per_class:(i+1)*samples_per_class]) for i in range(len(set(labels)))]


def main():
    samples_per_class = 4
    out_dir = Path("out3/comparingsd15Dumps")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    pipeline = Sketch2Image(device="cuda" if torch.cuda.is_available() else "cpu", img_size=512)
    
    print(f"\nLoading {len(CLASSES)} classes × {samples_per_class} samples")
    all_drawings, all_labels = [], []
    for cls in CLASSES:
        drawings = pipeline.download_quickdraw_batch(cls, samples_per_class, random.randint(0, 1000))
        all_drawings.extend(drawings)
        all_labels.extend([cls] * samples_per_class)
        print(f"  {cls}")
    
    quickdraw_imgs = [pipeline.quickdraw_to_image(d) for d in all_drawings]
    control_imgs = [pipeline.preprocess_lineart(img) for img in quickdraw_imgs]
    
    for sd_model in SD15_DUMPS:
        model_name = sd_model.replace("/", "_")
        print(f"\n{'='*50}\nProcessing: {sd_model}\n{'='*50}")
        
        pipeline.setup_models(sd_model)
        
        print("\nGenerating colored images...")
        colored = pipeline.batch_colorize(control_imgs, all_labels)
        
        print("\nScoring...")
        class_scores = pipeline.compute_class_scores(colored, all_labels, samples_per_class)
        
        grid = pipeline.create_paired_grid(quickdraw_imgs, colored, CLASSES, class_scores, samples_per_class)
        grid.save(out_dir / f"{model_name}.png")
        
        print(f"\nSaved to {out_dir}/{model_name}.png")
        print(f"Average: {np.mean(class_scores):.3f}")
        for cls, score in zip(CLASSES, class_scores):
            print(f"  {cls}: {score:.3f}")


if __name__ == "__main__":
    main()