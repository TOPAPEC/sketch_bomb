import numpy as np
import requests
import json
from PIL import Image, ImageDraw, ImageFont
import torch
from diffusers import StableDiffusionXLControlNetUnionPipeline, ControlNetUnionModel, EulerDiscreteScheduler
from transformers import AutoProcessor, AutoModel
import cv2
from pathlib import Path
import random

CLASSES = ["book", "cat", "car", "tree", "house", "guitar", "airplane", "flower",
           "bird", "fish", "bicycle", "clock", "chair", "apple", "sun", "umbrella"]

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


class SigLIPScorer:
    def __init__(self, device="cuda"):
        self.device = device
        self.model = AutoModel.from_pretrained("google/siglip2-base-patch16-224", torch_dtype=torch.float16).to(device)
        self.processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")
        
    @torch.no_grad()
    def score(self, images: list[Image.Image], texts: list[str]) -> np.ndarray:
        inputs = self.processor(text=texts, images=images, padding="max_length", return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits_per_image.cpu().numpy()
        return np.diag(logits)


class BatchQuickDrawToArt:
    def __init__(self, device="cuda", img_size=1024):
        self.device = device
        self.img_size = img_size
        self.setup_models()
        
    def setup_models(self):
        print("Loading models...")
        self.controlnet = ControlNetUnionModel.from_pretrained(
            "xinsir/controlnet-union-sdxl-1.0", torch_dtype=torch.float16
        ).to(self.device)
        
        self.pipe = StableDiffusionXLControlNetUnionPipeline.from_pretrained(
            "cagliostrolab/animagine-xl-4.0", controlnet=self.controlnet, torch_dtype=torch.float16
        ).to(self.device)
        
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_attention_slicing()
        self.scorer = SigLIPScorer(self.device)
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
        width, height = max_x - min_x, max_y - min_y
        scale = self.img_size * 0.8 / max(width, height, 1)
        offset_x = (self.img_size - width * scale) / 2 - min_x * scale
        offset_y = (self.img_size - height * scale) / 2 - min_y * scale
        line_width = max(3, self.img_size // 170)
        for stroke in strokes:
            points = [(x * scale + offset_x, y * scale + offset_y) for x, y in zip(stroke[0], stroke[1])]
            if len(points) > 1:
                draw.line(points, fill='black', width=line_width)
        return image
    
    def preprocess_lineart(self, image):
        arr = np.array(image.convert('L'))
        _, binary = cv2.threshold(arr, 200, 255, cv2.THRESH_BINARY_INV)
        binary = cv2.dilate(binary, np.ones((2, 2), np.uint8), iterations=1)
        return Image.fromarray(binary).convert('RGB')
    
    def preprocess_canny(self, image, low=50, high=150):
        edges = cv2.Canny(np.array(image.convert('L')), low, high)
        return Image.fromarray(edges).convert('RGB')
    
    def create_labeled_grid(self, images: list[Image.Image], labels: list[str], scores: list[float] = None, 
                            cols: int = 4, samples_per_class: int = 4):
        img_size = images[0].size[0]
        rows = len(images) // cols
        label_height = 60
        grid = Image.new('RGB', (img_size * cols, img_size * rows + label_height * (rows // samples_per_class)), 'white')
        draw = ImageDraw.Draw(grid)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
        except:
            font = ImageFont.load_default()
        
        y_offset = 0
        for i, img in enumerate(images):
            class_idx = i // samples_per_class
            local_idx = i % samples_per_class
            
            if local_idx == 0 and i > 0:
                y_offset += label_height
            
            row, col = i // cols, i % cols
            grid.paste(img, (col * img_size, row * img_size + y_offset))
            
            if local_idx == samples_per_class - 1:
                label_y = (class_idx + 1) * samples_per_class // cols * img_size + y_offset + img_size
                label_text = labels[class_idx]
                if scores:
                    label_text += f" (avg: {scores[class_idx]:.3f})"
                draw.rectangle([0, label_y, img_size * cols, label_y + label_height], fill='lightgray')
                draw.text((10, label_y + 10), label_text, fill='black', font=font)
                
        return grid
    
    def batch_realistic_sketch(self, control_images, labels, num_inference_steps=30, strong_steps=30):
        control_end = min(1.0, strong_steps / num_inference_steps)
        results = []
        for i, (ctrl_img, label) in enumerate(zip(control_images, labels)):
            print(f"  Sketch {i+1}/{len(control_images)} ({label})")
            img = self.pipe(
                prompt=get_sketch_prompt(label),
                negative_prompt=SKETCH_NEGATIVE,
                control_image=[ctrl_img], control_mode=[3],
                num_inference_steps=num_inference_steps, guidance_scale=8.0,
                controlnet_conditioning_scale=0.7, control_guidance_start=0.0, control_guidance_end=control_end,
                generator=torch.Generator(device=self.device).manual_seed(42 + i)
            ).images[0]
            results.append(img)
        return results
        
    def batch_colorize(self, sketch_images, labels, num_inference_steps=40, strong_steps=10):
        control_end = min(1.0, strong_steps / num_inference_steps)
        results = []
        for i, (sketch, label) in enumerate(zip(sketch_images, labels)):
            print(f"  Colorize {i+1}/{len(sketch_images)} ({label})")
            result = self.pipe(
                prompt=get_color_prompt(label),
                negative_prompt=COLOR_NEGATIVE,
                control_image=[self.preprocess_canny(sketch)], control_mode=[3],
                num_inference_steps=num_inference_steps, guidance_scale=8.0,
                controlnet_conditioning_scale=0.5, control_guidance_start=0.0, control_guidance_end=control_end,
                generator=torch.Generator(device=self.device).manual_seed(42 + i)
            ).images[0]
            results.append(result)
        return results

    def compute_class_scores(self, images: list[Image.Image], labels: list[str], samples_per_class: int = 4):
        prompts = [get_color_prompt(label) for label in labels]
        scores = self.scorer.score(images, prompts)
        class_scores = [np.mean(scores[i*samples_per_class:(i+1)*samples_per_class]) 
                        for i in range(len(labels) // samples_per_class)]
        return scores, class_scores


def main():
    samples_per_class = 4
    num_classes = len(CLASSES)
    out_dir = Path("out3/testing_proximity")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    pipeline = BatchQuickDrawToArt(device="cuda" if torch.cuda.is_available() else "cpu", img_size=1024)
    
    print(f"\nLoading {num_classes} classes × {samples_per_class} samples")
    
    all_drawings, all_labels = [], []
    for cls in CLASSES:
        offset = random.randint(0, 1000)
        drawings = pipeline.download_quickdraw_batch(cls, samples_per_class, offset)
        all_drawings.extend(drawings)
        all_labels.extend([cls] * samples_per_class)
        print(f"  Downloaded {cls}")
    
    quickdraw_imgs = [pipeline.quickdraw_to_image(d) for d in all_drawings]
    control_imgs = [pipeline.preprocess_lineart(img) for img in quickdraw_imgs]
    
    pipeline.create_labeled_grid(control_imgs, CLASSES, None, cols=samples_per_class, 
                                  samples_per_class=samples_per_class).save(out_dir / "01_controlnet_input.png")
    
    print("\nGenerating sketches...")
    sketches = pipeline.batch_realistic_sketch(control_imgs, all_labels)
    pipeline.create_labeled_grid(sketches, CLASSES, None, cols=samples_per_class,
                                  samples_per_class=samples_per_class).save(out_dir / "02_sketches.png")
    
    print("\nColorizing...")
    colored = pipeline.batch_colorize(sketches, all_labels)
    
    print("\nScoring with SigLIP2...")
    _, class_scores = pipeline.compute_class_scores(colored, all_labels, samples_per_class)
    
    pipeline.create_labeled_grid(colored, CLASSES, class_scores, cols=samples_per_class,
                                  samples_per_class=samples_per_class).save(out_dir / "03_final_scored.png")
    
    print(f"\nResults saved to {out_dir}/")
    print(f"Average score: {np.mean(class_scores):.3f}")
    for cls, score in zip(CLASSES, class_scores):
        print(f"  {cls}: {score:.3f}")


if __name__ == "__main__":
    main()