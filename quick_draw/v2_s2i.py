import numpy as np
import requests
import json
from PIL import Image, ImageDraw
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, EulerDiscreteScheduler
import cv2


class BatchQuickDrawToArt:
    def __init__(self, device="cuda"):
        self.device = device
        self.setup_models()
        
    def setup_models(self):
        print("Loading models...")
        
        self.scribble_controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_scribble",
            torch_dtype=torch.float16
        ).to(self.device)
        
        self.lineart_controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_lineart",
            torch_dtype=torch.float16
        ).to(self.device)
        
        base_model = "Lykon/dreamshaper-8"
        
        self.sketch_pipe = StableDiffusionControlNetPipeline.from_pretrained(
            base_model,
            controlnet=self.scribble_controlnet,
            torch_dtype=torch.float16,
            safety_checker=None
        ).to(self.device)
        
        self.color_pipe = StableDiffusionControlNetPipeline.from_pretrained(
            base_model,
            controlnet=self.lineart_controlnet,
            torch_dtype=torch.float16,
            safety_checker=None
        ).to(self.device)
        
        self.sketch_pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.sketch_pipe.scheduler.config
        )
        self.color_pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.color_pipe.scheduler.config
        )
        
        self.sketch_pipe.enable_attention_slicing()
        self.color_pipe.enable_attention_slicing()
        
        print("Models loaded!")
    
    def download_quickdraw_batch(self, category="book", num_samples=9):
        url = f"https://storage.googleapis.com/quickdraw_dataset/full/simplified/{category}.ndjson"
        print(f"Downloading {num_samples} Quick Draw '{category}' sketches...")
        
        response = requests.get(url, stream=True)
        sketches = []
        for i, line in enumerate(response.iter_lines()):
            if i >= num_samples:
                break
            if line:
                sketches.append(json.loads(line))
        
        return sketches
    
    def quickdraw_to_image(self, drawing_data, img_size=512):
        image = Image.new('RGB', (img_size, img_size), 'white')
        draw = ImageDraw.Draw(image)
        strokes = drawing_data['drawing']
        
        all_x = [x for stroke in strokes for x in stroke[0]]
        all_y = [y for stroke in strokes for y in stroke[1]]
        
        if not all_x or not all_y:
            return image
        
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        width = max_x - min_x
        height = max_y - min_y
        scale = min(img_size * 0.8 / max(width, height, 1), 1.0)
        offset_x = (img_size - width * scale) / 2 - min_x * scale
        offset_y = (img_size - height * scale) / 2 - min_y * scale
        
        for stroke in strokes:
            points = [(x * scale + offset_x, y * scale + offset_y) 
                     for x, y in zip(stroke[0], stroke[1])]
            if len(points) > 1:
                draw.line(points, fill='black', width=3)
        
        return image
    
    def preprocess_for_controlnet(self, image):
        img_array = np.array(image)
        if img_array.mean() > 127:
            img_array = 255 - img_array
        return Image.fromarray(img_array)
    
    def create_grid(self, images, n):
        img_size = images[0].size[0]
        grid = Image.new('RGB', (img_size * n, img_size * n), 'white')
        for i, img in enumerate(images):
            row = i // n
            col = i % n
            grid.paste(img, (col * img_size, row * img_size))
        return grid
    
    def batch_realistic_sketch(self, control_images, prompt, num_inference_steps=30, strong_steps=15):
        sketch_negative = (
            "people, person, human, character, face, hands, arms, body, "
            "animal, pet, creature, "
            "multiple objects, extra objects, props, items, accessories, "
            "table, desk, shelf, library, room, interior, scenery, landscape, background scene, "
            "plants, flowers, cup, mug, pen, pencil, smartphone, food, "
            "text, letters, words, title, logo, watermark, signature, emblem, "
            "frame, border, vignette, "
            "colored, colorful, blurry, low quality, distorted"
        )

        print(f"Creating {len(control_images)} realistic sketches...")
        results = []

        num_inference_steps = int(num_inference_steps)
        strong_steps = int(strong_steps)

        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be > 0")
        if strong_steps < 0:
            raise ValueError("strong_steps must be >= 0")

        control_end = strong_steps / num_inference_steps
        control_end = 0.0 if control_end < 0.0 else (1.0 if control_end > 1.0 else control_end)

        for i, ctrl_img in enumerate(control_images):
            print(f"  Processing {i+1}/{len(control_images)}")
            generator = torch.Generator(device=self.device).manual_seed(42 + i)

            img = self.sketch_pipe(
                prompt=prompt,
                negative_prompt=sketch_negative,
                image=ctrl_img,
                num_inference_steps=num_inference_steps,
                guidance_scale=8.0,
                controlnet_conditioning_scale=0.9,
                control_guidance_start=0.0,
                control_guidance_end=control_end,
                generator=generator
            ).images[0]

            results.append(img)

        return results

        
    def batch_colorize(self, sketch_images, prompt, num_inference_steps=30, strong_steps=15):
        print(f"Colorizing {len(sketch_images)} sketches...")
        color_negative = (
            "people, person, human, character, face, hands, arms, "
            "animal, "
            "multiple objects, extra objects, props, items, accessories, "
            "table, desk, shelf, library, room, interior, background scene, scenery, "
            "plants, flowers, cup, mug, pen, pencil, smartphone, food, "
            "text, letters, words, title, logo, watermark, signature, "
            "frame, border, vignette, "
            "sketch, monochrome, black and white, ugly, blurry, poor quality"
        )
        
        num_inference_steps = int(num_inference_steps)
        strong_steps = int(strong_steps)
        
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be > 0")
        if strong_steps < 0:
            raise ValueError("strong_steps must be >= 0")
        
        control_end = strong_steps / num_inference_steps
        control_end = 0.0 if control_end < 0.0 else (1.0 if control_end > 1.0 else control_end)
        
        results = []
        for i, sketch in enumerate(sketch_images):
            print(f"  Processing {i+1}/{len(sketch_images)}")
            sketch_array = np.array(sketch.convert('L'))
            sketch_array = cv2.Canny(sketch_array, 50, 150)
            control_image = Image.fromarray(sketch_array)
            
            result = self.color_pipe(
                prompt=prompt,
                negative_prompt=color_negative,
                image=control_image,
                num_inference_steps=num_inference_steps,
                guidance_scale=7.5,
                controlnet_conditioning_scale=0.9,
                control_guidance_start=0.0,
                control_guidance_end=control_end,
                generator=torch.Generator(device=self.device).manual_seed(42 + i)
            ).images[0]
            results.append(result)
        
        return results


def main():
    n = 4
    num_images = n * n
    
    pipeline = BatchQuickDrawToArt(device="cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\nGenerating {num_images} images in {n}x{n} grid\n")
    
    drawings = pipeline.download_quickdraw_batch(category="book", num_samples=num_images)
    
    print("\nConverting to images...")
    quickdraw_imgs = [pipeline.quickdraw_to_image(d, img_size=512) for d in drawings]
    grid = pipeline.create_grid(quickdraw_imgs, n)
    grid.save("grid_01_quickdraw.png")
    print("Saved: grid_01_quickdraw.png")
    
    print("\nPreparing for ControlNet...")
    control_imgs = [pipeline.preprocess_for_controlnet(img) for img in quickdraw_imgs]
    grid = pipeline.create_grid(control_imgs, n)
    grid.save("grid_02_controlnet_input.png")
    print("Saved: grid_02_controlnet_input.png")
    
    sketch_prompt = (
        "a single book only, isolated subject, centered composition, "
        "professional graphite pencil sketch, thick black strokes, realistic, clean background, "
        "no scene, no props, no extra objects, highly detailed shading, grayscale"
    )
 
    realistic_sketches = pipeline.batch_realistic_sketch(control_imgs, sketch_prompt)
    grid = pipeline.create_grid(realistic_sketches, n)
    grid.save("grid_03_realistic_sketches.png")
    print("Saved: grid_03_realistic_sketches.png")
    
    color_prompt = (
        "a single old book only, isolated subject, centered, "
        "clean plain background, no environment, no props, "
        "high quality digital illustration, detailed ornate cover, warm colors"
    )

    colored_images = pipeline.batch_colorize(realistic_sketches, color_prompt)
    grid = pipeline.create_grid(colored_images, n)
    grid.save("grid_04_final_colored.png")
    print("Saved: grid_04_final_colored.png")
    
    print(f"\nDONE! Generated {n}x{n} grid with {num_images} images")


if __name__ == "__main__":
    main()