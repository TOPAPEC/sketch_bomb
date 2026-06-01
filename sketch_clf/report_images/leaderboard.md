| run | model | mode | test_top1 | test_top3 | macro_f1 | val_top1 | train_s | vram_G |
|---|---|---|---|---|---|---|---|---|
| vit_base_clip_finetune | vit_base_patch16_clip_224 | finetune | 0.9333 | 0.9675 | 0.9333 | 0.9347 | 2166.5008 | 12.3679 |
| vit_base_finetune | vit_base_patch16_224 | finetune | 0.9306 | 0.9672 | 0.9307 | 0.9285 | 2159.7947 | 12.2708 |
| vit_small_finetune | vit_small_patch16_224 | finetune | 0.9260 | 0.9670 | 0.9262 | 0.9237 | 959.8262 | 9.2951 |
| vit_small_linearprobe | vit_small_patch16_224 | linear_probe | 0.8706 | 0.9492 | 0.8705 | 0.8705 | 357.4367 | 1.0353 |
| baseline_beit | kmewhort/beit-sketch-classifier | pretrained_eval_only | 0.6385 | 0.8287 | 0.8907 | - | - | - |
