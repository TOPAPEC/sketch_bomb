import numpy as np
import requests
import json
from PIL import Image, ImageDraw
import torch
from diffusers import StableDiffusionXLControlNetUnionPipeline, ControlNetUnionModel, EulerDiscreteScheduler
import cv2


class BatchQuickDrawToArt:
    def __init__(self, device="cuda", img_size=1024):
        self.device = device
        self.img_size = img_size
        self.setup_models()
        
    def setup_models(self):
        print("Loading models...")
        
        self.controlnet = ControlNetUnionModel.from_pretrained(
            "xinsir/controlnet-union-sdxl-1.0",
            torch_dtype=torch.float16
        ).to(self.device)
        
        base_model = "cagliostrolab/animagine-xl-4.0"
        
        self.pipe = StableDiffusionXLControlNetUnionPipeline.from_pretrained(
            base_model,
            controlnet=self.controlnet,
            torch_dtype=torch.float16,
        ).to(self.device)
        
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe.enable_attention_slicing()
        
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
    
    def quickdraw_to_image(self, drawing_data):
        image = Image.new('RGB', (self.img_size, self.img_size), 'white')
        draw = ImageDraw.Draw(image)
        strokes = drawing_data['drawing']
        
        all_x = [x for stroke in strokes for x in stroke[0]]
        all_y = [y for stroke in strokes for y in stroke[1]]
        
        if not all_x or not all_y:
            return image
        
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        width, height = max_x - min_x, max_y - min_y
        scale = self.img_size * 0.8 / max(width, height, 1)
        offset_x = (self.img_size - width * scale) / 2 - min_x * scale
        offset_y = (self.img_size - height * scale) / 2 - min_y * scale
        
        line_width = max(3, self.img_size // 170)
        for stroke in strokes:
            points = [(x * scale + offset_x, y * scale + offset_y) 
                     for x, y in zip(stroke[0], stroke[1])]
            if len(points) > 1:
                draw.line(points, fill='black', width=line_width)
        
        return image
    
    def preprocess_lineart(self, image):
        """White lines on black background for Union lineart mode"""
        arr = np.array(image.convert('L'))
        _, binary = cv2.threshold(arr, 200, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=1)
        return Image.fromarray(binary).convert('RGB')
    
    def preprocess_canny(self, image, low=50, high=150):
        gray = np.array(image.convert('L'))
        edges = cv2.Canny(gray, low, high)
        return Image.fromarray(edges).convert('RGB')
    
    def create_grid(self, images, n):
        img_size = images[0].size[0]
        grid = Image.new('RGB', (img_size * n, img_size * n), 'white')
        for i, img in enumerate(images):
            row, col = i // n, i % n
            grid.paste(img, (col * img_size, row * img_size))
        return grid
    
    def batch_realistic_sketch(self, control_images, prompt, num_inference_steps=30, strong_steps=30):
        sketch_negative = (
            "people, person, human, character, face, hands, arms, body, "
            "animal, pet, creature, multiple objects, extra objects, props, "
            "table, desk, shelf, library, room, interior, scenery, landscape, "
            "plants, flowers, cup, mug, pen, pencil, smartphone, food, "
            "text, letters, words, title, logo, watermark, signature, "
            "frame, border, vignette, colored, colorful, blurry, low quality"
        )

        print(f"Creating {len(control_images)} realistic sketches...")
        num_inference_steps = int(num_inference_steps)
        strong_steps = int(strong_steps)
        control_end = min(1.0, max(0.0, strong_steps / num_inference_steps))
        
        results = []
        for i, ctrl_img in enumerate(control_images):
            print(f"  Processing {i+1}/{len(control_images)}")
            generator = torch.Generator(device=self.device).manual_seed(42 + i)

            img = self.pipe(
                prompt=prompt,
                negative_prompt=sketch_negative,
                control_image=[ctrl_img],
                control_mode=[3],  # 3 = lineart/canny/mlsd
                num_inference_steps=num_inference_steps,
                guidance_scale=8.0,
                controlnet_conditioning_scale=0.7,
                control_guidance_start=0.0,
                control_guidance_end=control_end,
                generator=generator
            ).images[0]
            results.append(img)

        return results
        
    def batch_colorize(self, sketch_images, prompt, num_inference_steps=40, strong_steps=10):
        print(f"Colorizing {len(sketch_images)} sketches...")
        color_negative = (
            "people, person, human, animal, multiple objects, extra objects, "
            "table, desk, shelf, library, room, interior, background scene, "
            "plants, flowers, cup, mug, pen, pencil, smartphone, food, "
            "text, letters, words, title, logo, watermark, signature, "
            "frame, border, sketch, monochrome, black and white, ugly, blurry"
        )
        
        num_inference_steps = int(num_inference_steps)
        strong_steps = int(strong_steps)
        control_end = min(1.0, max(0.0, strong_steps / num_inference_steps))
        
        results = []
        for i, sketch in enumerate(sketch_images):
            print(f"  Processing {i+1}/{len(sketch_images)}")
            control_image = self.preprocess_canny(sketch)
            
            result = self.pipe(
                prompt=prompt,
                negative_prompt=color_negative,
                control_image=[control_image],
                control_mode=[3],  # canny mode
                num_inference_steps=num_inference_steps,
                guidance_scale=8.0,
                controlnet_conditioning_scale=0.5,
                control_guidance_start=0.0,
                control_guidance_end=control_end,
                generator=torch.Generator(device=self.device).manual_seed(42 + i)
            ).images[0]
            results.append(result)
        
        return results


def main():
    n = 4
    num_images = n * n
    
    pipeline = BatchQuickDrawToArt(
        device="cuda" if torch.cuda.is_available() else "cpu",
        img_size=1024
    )
    
    print(f"\nGenerating {num_images} images in {n}x{n} grid\n")
    
    drawings = pipeline.download_quickdraw_batch(category="book", num_samples=num_images)
    
    print("\nConverting to images...")
    quickdraw_imgs = [pipeline.quickdraw_to_image(d) for d in drawings]
    pipeline.create_grid(quickdraw_imgs, n).save("grid_01_quickdraw.png")
    print("Saved: grid_01_quickdraw.png")
    
    print("\nPreparing for ControlNet (lineart mode)...")
    control_imgs = [pipeline.preprocess_lineart(img) for img in quickdraw_imgs]
    pipeline.create_grid(control_imgs, n).save("grid_02_controlnet_input.png")
    print("Saved: grid_02_controlnet_input.png")
    
    sketch_prompt = (
        "a single book only, isolated subject, centered composition, "
        "professional graphite pencil sketch, clean background, thick black strokes,"
        "no scene, no props, no extra objects, highly detailed shading, grayscale"
    )
 
    realistic_sketches = pipeline.batch_realistic_sketch(control_imgs, sketch_prompt)
    pipeline.create_grid(realistic_sketches, n).save("grid_03_realistic_sketches.png")
    print("Saved: grid_03_realistic_sketches.png")
    
    color_prompt = (
        "a single old book only, isolated subject, centered, "
        "clean plain background, no environment, no props, "
        "high quality digital illustration, detailed ornate cover, warm colors, detailed, beautiful"
    )

    colored_images = pipeline.batch_colorize(realistic_sketches, color_prompt)
    pipeline.create_grid(colored_images, n).save("grid_04_final_colored.png")
    print("Saved: grid_04_final_colored.png")
    
    print(f"\nDONE! Generated {n}x{n} grid with {num_images} images")


if __name__ == "__main__":
    main()