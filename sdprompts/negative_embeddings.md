# Negative Embeddings (Textual Inversions) for SD1.5

Negative embeddings are pre-trained vectors that encode "bad" image concepts.
When used in the negative prompt, they steer generation away from those concepts
much more effectively than text-based negative prompts.

## How to Use in Diffusers

```python
# Load after pipeline creation
pipe.load_textual_inversion(
    "EvilEngine/easynegative",
    weight_name="easynegative.safetensors",
    token="easynegative"
)

# Use token in negative prompt
pipe(prompt="a cat", negative_prompt="easynegative, blurry")
```

## Top Negative Embeddings for SD1.5

| Name | Token | Effect | Source |
|------|-------|--------|--------|
| EasyNegative | `easynegative` | General quality improvement | [HF: EvilEngine/easynegative](https://huggingface.co/EvilEngine/easynegative) |
| badhandv4 | `badhandv4` | Fixes bad hands | [Civitai](https://civitai.com/models/16993) |
| Deep Negative V1.75T | `ng_deepnegative_v1_75t` | Quality + vivid colors | [Civitai](https://civitai.com/models/4629) |
| BadDream | `BadDream` | For stylized/illustration work | [Civitai](https://civitai.com/models/72437) |
| UnrealisticDream | `UnrealisticDream` | For realistic styles | [Civitai](https://civitai.com/models/72437) |
| veryBadImageNegative | `verybadimagenegative_v1.3` | Balanced overall quality | [Civitai](https://civitai.com/models/11772) |
| bad_prompt_v2 | `bad_prompt_version2-neg` | General quality | [Civitai](https://civitai.com/models/55700) |

## Recommended Combo

For our pipeline (illustration-style objects):
```python
pipe.load_textual_inversion("EvilEngine/easynegative", weight_name="easynegative.safetensors", token="easynegative")

negative_prompt = "easynegative, low quality, blurry, deformed"
```

EasyNegative alone covers most quality issues. Add badhandv4 only if generating humans/hands.

## File Formats
- `.safetensors` - preferred (safe, fast)
- `.pt` - legacy PyTorch format

## Notes
- These only work with SD1.5 models (not SDXL, Flux etc)
- Very small files (~25KB), negligible memory impact
- Can stack multiple: `"easynegative, BadDream"` in negative prompt
- Lower CFG scale (<=11) recommended with badhandv4 to avoid style drift

Sources:
- https://huggingface.co/docs/diffusers/using-diffusers/textual_inversion_inference
- https://www.digitalcreativeai.net/en/post/recommended-negative-embedding-for-sd15-models
- https://civitai.com/models/7808/easynegative
