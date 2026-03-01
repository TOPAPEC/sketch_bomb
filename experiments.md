# Experiment Results

| Timestamp | Description | Metrics |
|-----------|-------------|---------|
| 2026-02-26 16:57 | Baseline SD1.5+ControlNet, 6cls x2spc | avg_siglip2: -2.7363, min: -6.5000, max: -0.2500 |
| 2026-02-26 16:58 | SD1.5+ControlNet+Refiner(s=0.25), same sketches as exp1 | avg_siglip2: -2.4909, min: -5.8047, max: 0.8125 |
| 2026-02-26 17:36 | Baseline SD1.5+ControlNet, 3cls x1spc | avg_siglip2: -1.8125, min: -3.7031, max: -0.4531 |
| 2026-02-26 17:36 | SD1.5+ControlNet+Refiner(s=0.25), same sketches as exp1 | avg_siglip2: -1.8932, min: -4.1172, max: -0.1719 |
| 2026-02-26 17:38 | SD1.5+CN+Refiner+BestOf4(Qwen), 4 candidates per sample | avg_siglip2: -0.5312, min: -1.4219, max: 0.6719 |
| 2026-02-26 18:02 | Baseline SD1.5+ControlNet, 3cls x1spc | avg_siglip2: -2.3281, min: -4.9531, max: -0.8828 |
| 2026-02-26 18:02 | SD1.5+ControlNet+Refiner(s=0.25), same sketches as exp1 | avg_siglip2: -2.5625, min: -5.4062, max: -0.7891 |
| 2026-02-26 18:03 | SD1.5+CN+Refiner+BestOf4(Qwen), 4 candidates per sample | avg_siglip2: -1.7839, min: -2.2891, max: -0.9297 |
| 2026-02-26 18:12 | Baseline SD1.5+ControlNet, 3cls x1spc | avg_siglip2: -2.3073, min: -6.6641, max: 0.7812 |
| 2026-02-26 18:13 | SD1.5+ControlNet+Refiner(s=0.25), same sketches as exp1 | avg_siglip2: -2.3464, min: -6.5938, max: 0.5938 |
| 2026-02-26 18:14 | SD1.5+CN+Refiner+BestOf4(Kimi), 4 candidates per sample | avg_siglip2: -2.2083, min: -4.6875, max: -0.9688 |
| 2026-02-26 18:19 | v8 tailored prompts + BG removal, 3cls x1spc | avg_siglip2: -3.2839, min: -7.4844, max: -0.3438 |
| 2026-02-26 18:19 | v8 tailored + refiner | avg_siglip2: -2.6615, min: -5.9531, max: 0.0312 |
| 2026-02-26 18:21 | v8 tailored + refiner + BestOf4(Kimi) | avg_siglip2: -3.7031, min: -9.4688, max: -0.5469 |
| 2026-02-26 18:23 | v8 tailored prompts + BG removal, 3cls x1spc | avg_siglip2: -1.8047, min: -3.6953, max: -0.5156 |
| 2026-02-26 18:23 | v8 tailored + refiner | avg_siglip2: -1.4271, min: -2.5469, max: -0.2188 |
| 2026-02-26 18:26 | v8 tailored + refiner + BestOf4(Kimi) | avg_siglip2: -2.0729, min: -3.7031, max: -1.2422 |
| 2026-02-26 18:43 | v8 tailored prompts + BG removal, 3cls x1spc | avg_siglip2: -2.7240, min: -5.6172, max: -0.5156 |
| 2026-02-26 18:43 | v8 tailored + refiner | avg_siglip2: -2.7083, min: -5.8984, max: -0.2188 |
| 2026-02-26 18:45 | v8 tailored + refiner + BestOf4(Kimi) | avg_siglip2: -1.4427, min: -1.7500, max: -1.2422 |
| 2026-02-26 18:46 | v8 tailored + refiner + BestOf4(SigLIP2) | avg_siglip2: -1.6797, min: -2.8359, max: -0.4219 |
| 2026-02-26 19:00 | v8 dreamshaper tailored + BG removal, 3cls x1spc | avg_siglip2: -0.9453, min: -1.6641, max: 0.0156 |
| 2026-02-26 19:00 | v8 dreamshaper tailored + refiner | avg_siglip2: -0.8594, min: -1.6406, max: 0.3750 |
| 2026-02-26 19:02 | v8 dreamshaper + BestOf4(Kimi) | avg_siglip2: -1.1302, min: -2.6797, max: 0.4375 |
| 2026-02-26 19:02 | v8 dreamshaper + BestOf4(SigLIP2) | avg_siglip2: -0.1380, min: -1.2422, max: 0.4219 |
| 2026-02-26 19:02 | v8 dreamshaper no-controlnet (txt2img) | avg_siglip2: -1.3594, min: -3.1641, max: 0.0781 |
| 2026-02-26 20:52 | v9 SDXL + MistoLine + refiner, 3cls x1spc | avg_siglip2: -2.4167, min: -5.5703, max: -0.3125 |
| 2026-02-26 20:54 | v9 SDXL + BestOf4(SigLIP2) | avg_siglip2: -1.6536, min: -4.4297, max: 0.2344 |
| 2026-02-26 22:50 | v8 dreamshaper DomainNet-matched + BestOf4(SigLIP2) | avg_siglip2: 0.2604, min: -0.3750, max: 0.6406 |

## BEiT Pipeline Comparison (2026-03-01 15:02)
Fixed test set: 10 sketches, seed=42
BEiT accuracy: 7/10 (70%)

| Model | Avg SigLIP2 | Min | Max |
|-------|-------------|-----|-----|
| sd15 | -16.8648 | -20.4844 | -13.1641 |
| sdxl | -16.6000 | -19.7344 | -13.2734 |
| wai | -17.2023 | -20.1094 | -12.0312 |

## BEiT Pipeline + BG Removal Comparison (2026-03-01 15:28)
Fixed test set: 10 sketches, seed=42
BEiT accuracy: 7/10 (70%)

| Model | BG Method | Avg SigLIP2 | Min | Max |
|-------|-----------|-------------|-----|-----|
| sd15 | Raw | -16.2977 | -19.1406 | -13.6641 |
| sd15 | rembg | -16.8648 | -20.4844 | -13.1641 |
| sd15 | GrabCut | -16.4750 | -19.4375 | -13.9531 |
| sd15 | Threshold | -16.3227 | -19.1562 | -13.6562 |
| sdxl | Raw | -16.4367 | -19.0156 | -13.4219 |
| sdxl | rembg | -16.6000 | -19.7344 | -13.2734 |
| sdxl | GrabCut | -16.6383 | -19.1250 | -13.8281 |
| sdxl | Threshold | -16.4594 | -19.0156 | -13.4375 |
| wai | Raw | -16.8984 | -20.7500 | -12.1797 |
| wai | rembg | -17.2023 | -20.1094 | -12.0312 |
| wai | GrabCut | -16.6773 | -19.2344 | -12.5781 |
| wai | Threshold | -16.9023 | -20.7500 | -12.2344 |

## BEiT Pipeline + BG Removal Comparison (2026-03-01 16:17)
Fixed test set: 10 sketches, seed=42
BEiT accuracy: 7/10 (70%)

| Model | BG Method | Avg SigLIP2 | Min | Max |
|-------|-----------|-------------|-----|-----|
| sd15 | Raw | -17.2820 | -20.5469 | -15.1875 |
| sd15 | rembg | -17.2336 | -20.3594 | -15.8672 |
| sd15 | GrabCut | -17.0367 | -19.3906 | -15.5625 |
| sd15 | Threshold | -17.2766 | -20.5469 | -15.1797 |
| sdxl | Raw | -16.6445 | -19.7031 | -10.7969 |
| sdxl | rembg | -16.4703 | -18.9375 | -10.8594 |
| sdxl | GrabCut | -16.2070 | -18.4375 | -11.8281 |
| sdxl | Threshold | -16.6680 | -19.7031 | -10.8438 |
| wai | Raw | -17.4047 | -20.1562 | -12.9141 |
| wai | rembg | -17.0102 | -20.1250 | -13.8359 |
| wai | GrabCut | -16.6984 | -18.9531 | -13.4375 |
| wai | Threshold | -17.4102 | -20.1562 | -12.9375 |

## BEiT Pipeline + BG Removal Comparison (2026-03-01 16:57)
Fixed test set: 10 sketches, seed=42
BEiT accuracy: 7/10 (70%)

| Model | BG Method | Avg SigLIP2 | Min | Max |
|-------|-----------|-------------|-----|-----|
| sd15 | Raw | -17.2625 | -20.8281 | -14.4688 |
| sd15 | rembg | -17.0461 | -19.7969 | -15.8438 |
| sd15 | GrabCut | -17.0766 | -20.1875 | -15.3906 |
| sd15 | Threshold | -17.2656 | -20.8281 | -14.5000 |
| sdxl | Raw | -16.6445 | -19.7031 | -10.7969 |
| sdxl | rembg | -16.4703 | -18.9375 | -10.8594 |
| sdxl | GrabCut | -16.2070 | -18.4375 | -11.8281 |
| sdxl | Threshold | -16.6680 | -19.7031 | -10.8438 |
| wai | Raw | -17.4047 | -20.1562 | -12.9141 |
| wai | rembg | -17.0102 | -20.1250 | -13.8359 |
| wai | GrabCut | -16.6984 | -18.9531 | -13.4375 |
| wai | Threshold | -17.4102 | -20.1562 | -12.9375 |

## BEiT Pipeline + BG Removal Comparison (2026-03-01 17:29)
Fixed test set: 10 sketches, seed=42
BEiT accuracy: 7/10 (70%)

| Model | BG Method | Avg SigLIP2 | Min | Max |
|-------|-----------|-------------|-----|-----|
| sd15 | Raw | -16.5180 | -20.8281 | -13.1719 |
| sd15 | rembg | -16.1187 | -19.7969 | -11.4609 |
| sd15 | GrabCut | -15.9992 | -20.1875 | -11.8125 |
| sd15 | Threshold | -16.5180 | -20.8281 | -13.1719 |
| sdxl | Raw | -16.3094 | -19.7031 | -10.3125 |
| sdxl | rembg | -16.3328 | -18.9375 | -11.8906 |
| sdxl | GrabCut | -15.8055 | -18.4375 | -11.2812 |
| sdxl | Threshold | -16.3234 | -19.7031 | -10.3125 |
| wai | Raw | -16.8094 | -20.1562 | -12.9609 |
| wai | rembg | -16.5766 | -20.1250 | -14.1094 |
| wai | GrabCut | -16.1273 | -18.9531 | -11.9922 |
| wai | Threshold | -16.8266 | -20.1562 | -12.9766 |

## BEiT Pipeline Comparison (2026-03-01 19:17, run=20260301_190244)
Fixed test set: 2 sketches, seed=42
BEiT accuracy: 2/2 (100%)
HTML report: `report_20260301_190244.html`

| Model | BG Method | Avg SigLIP2 | Min | Max |
|-------|-----------|-------------|-----|-----|
| sd15 | Raw | -16.3125 | -17.6875 | -14.9375 |
| sd15 | rembg | -17.2969 | -18.2031 | -16.3906 |
| sd15 | GrabCut | -16.7578 | -17.7656 | -15.7500 |
| sd15 | Threshold | -16.3125 | -17.6875 | -14.9375 |
| sdxl | Raw | -17.4648 | -19.6562 | -15.2734 |
| sdxl | rembg | -17.2109 | -19.0625 | -15.3594 |
| sdxl | GrabCut | -17.0508 | -18.5625 | -15.5391 |
| sdxl | Threshold | -17.4570 | -19.6406 | -15.2734 |
| wai | Raw | -16.9453 | -18.3438 | -15.5469 |
| wai | rembg | -15.9688 | -17.4375 | -14.5000 |
| wai | GrabCut | -16.4297 | -17.5781 | -15.2812 |
| wai | Threshold | -16.9922 | -18.3438 | -15.6406 |
