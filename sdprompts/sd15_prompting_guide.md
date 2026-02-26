# SD1.5 Prompting Guide

## Prompt Structure

SD1.5 works best with **keyword-driven, tag-style prompts** (not natural language sentences like SDXL/Flux).

Formula: `[subject], [medium/style], [lighting], [color], [composition], [quality]`

### Token Position Matters
Earlier tokens get stronger weight. Put the most important stuff first.

Bad: `high quality, detailed, a cat sitting on a table`
Good: `a cat, sitting, detailed, high quality digital illustration`

### Grouping
Keep related tokens together. "1girl, woman, pale skin, detailed face" works better than scattering descriptors.

## Quality Boosters (tested)

Strong effect:
- `masterpiece, best quality` (especially on anime/finetuned models)
- `highly detailed, sharp focus`
- `8k, high resolution` (subtle but helps)
- `professional`

Medium effect:
- `studio lighting, dramatic lighting`
- `vivid colors, vibrant`
- `digital painting, concept art`

Weak/style-dependent:
- `trending on artstation` (overused, can bias style)
- `unreal engine` (adds 3D game look)
- `beautiful` (vague)

## Emphasis/Weighting

- `(keyword)` = 1.1x weight
- `((keyword))` = 1.21x
- `(keyword:1.3)` = explicit 1.3x
- `[keyword]` = 0.9x (reduce)
- Stay within 0.5-1.5 range, higher causes artifacts

## Isolated Object Generation

For single objects on clean backgrounds:
- `a single {object}, solo, alone, centered`
- `simple background, plain background`
- `product photography, studio shot` (triggers clean bg)
- `no background, isolated` (sometimes works)

DO NOT say "white background" - SD1.5 interprets this inconsistently and often produces weird gradients. Instead, generate freely and use **background removal** post-processing.

## ControlNet Lineart Tips

- ControlNet weight 0.6-0.8 works best (default 1.0 is too strict)
- `control_guidance_end` of 0.3-0.5 lets the model deviate from sketch in later steps
- Lineart ControlNet works well with longer descriptive prompts
- The prompt should describe the DESIRED output, not the sketch
- Lower `controlnet_conditioning_scale` for more creative freedom

## What Causes Common Artifacts

| Problem | Cause | Fix |
|---------|-------|-----|
| Orange/fire tint | "warm colors" in prompt | Remove, use "vivid colors" or "natural colors" |
| Dark backgrounds | Model's training bias | Use bg removal, not prompt |
| Multiple objects | "a {thing}" is ambiguous | "a single {thing}, solo, alone" |
| Blurry output | Low steps or guidance | 25-30 steps, guidance 7-8 |
| Oversaturated | High guidance scale | Lower to 7.0-7.5 |

## Example Prompts for Objects

Cat:
`a single cat, solo, centered, detailed fur, (digital illustration:1.2), vivid natural colors, sharp focus, simple background`

Car:
`a single car, centered, (concept art:1.1), detailed, studio lighting, professional, simple background`

Flower:
`a single flower, centered, (botanical illustration:1.2), detailed petals, vivid colors, sharp focus, simple background`

## Prompt Categories (in order of impact)

1. **Subject/Object** - what you want. Be specific: "a tabby cat sitting" not just "cat"
2. **Medium** - photograph, digital painting, oil painting, concept art, 3D render
3. **Style** - pop art, impressionist, hyperrealistic, anime
4. **Artist** (optional) - triggers specific styles. Multiple artists blend styles
5. **Resolution/Quality** - highly detailed, 4K, sharp focus
6. **Lighting** - studio lighting, rim lighting, cinematic, volumetric
7. **Color** - specify only if needed, otherwise model picks naturally

## Key Takeaways for Our Pipeline

1. Remove "warm colors" and "white background" from prompts - they cause artifacts
2. Use background removal (rembg/YOLO) post-generation instead of prompting for backgrounds
3. Keep prompts tag-style, not sentences: `a single cat, centered, digital illustration, detailed, sharp focus`
4. Use moderate emphasis: `(keyword:1.2)` max, higher causes artifacts
5. Negative prompt is critical for SD1.5 - see negative_prompts.md
6. ControlNet lineart weight 0.6-0.8, not 1.0
7. 25-30 steps, guidance 7.0-7.5 for best quality
8. Factor value range for emphasis: 0.5-1.5, beyond = artifacts

Sources:
- https://www.aiarty.com/stable-diffusion-prompts/stable-diffusion-prompt-guide.htm
- https://education.civitai.com/civitais-prompt-crafting-guide-part-1-basics/
- https://supagruen.github.io/StableDiffusion-CheatSheet/
- https://sketchbooky.wordpress.com/2023/12/24/on-reading-the-official-stable-diffusion-1-5-prompt-guide/
