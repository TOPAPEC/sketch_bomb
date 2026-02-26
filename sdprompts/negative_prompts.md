# Negative Prompts Collection for SD1.5

## Universal Negative (Realistic Models)
```
(worst quality, low quality, normal quality, low-res, low details, oversaturated, undersaturated, overexposed, underexposed, grayscale, bw, bad photo, bad art:1.4), (watermark, signature, text font, username, error, logo, words, letters, digits, autograph, trademark, name:1.2), (blur, blurry, grainy), morbid, ugly, asymmetrical, mutated malformed, mutilated, poorly lit, bad shadow, draft, cropped, out of frame, cut off, censored, jpeg artifacts, out of focus, glitch, duplicate
```

## Minimal Negative (General Purpose)
```
low quality, blurry, artifacts, grainy, distorted, ugly, text, watermark, signature
```

## For Isolated Object Generation
```
people, person, human, multiple objects, extra objects, table, desk, shelf, room, interior, background scene, text, letters, words, watermark, signature, frame, border, sketch, monochrome, black and white, ugly, blurry, low quality, deformed
```

## Anti-Artifact (SD1.5 Specific)
```
(bad hands, bad anatomy, bad body, bad face, bad teeth, bad arms, bad legs, deformities:1.3), extra fingers, extra limbs, missing fingers, fused fingers, mutated hands
```

## Style-Specific Anti-Bleed
When SD1.5 bleeds unwanted style elements:
```
(airbrushed, cartoon, anime, semi-realistic, CGI, render, blender, digital art, manga, amateur:1.3)
```

## Key Rules
- Negative prompts are ESSENTIAL for SD1.5 (less so for SDXL)
- Don't put things in positive prompt to avoid them ("no fire" = more fire)
- Use weighting (keyword:1.4) for strong avoidance
- Keep a standard template and add object-specific negatives
- Fire/orange artifacts: add `fire, flames, orange tint, warm lighting`

Sources:
- https://machinelearningmastery.com/prompting-techniques-stable-diffusion/
- https://education.civitai.com/civitais-prompt-crafting-guide-part-1-basics/
- https://www.aiarty.com/stable-diffusion-prompts/stable-diffusion-prompt-guide.htm
