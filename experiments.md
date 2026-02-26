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
