# Best-of-N: Single vs Multi-criteria SigLIP2

*2026-04-01 12:49*

## Идея

Вместо одного промпта `a high quality illustration of a {label}` скорим кандидатов
по 7 независимым осям качества через SigLIP2. Скоры нормализуются per-axis
(min-max среди кандидатов), затем суммируются с весами:

- **prompt_match** (w=2.0): `a high quality illustration of a {label}`
- **no_artifacts** (w=1.5): `a clean flawless image without any artifacts, distortions, or deformations`
- **sharpness** (w=1.0): `a sharp detailed crisp image with clear edges and fine details`
- **anatomy** (w=1.5): `correct natural proportions and anatomy, physically plausible structure`
- **composition** (w=1.0): `a single centered {label} on a clean simple background, isolated subject`
- **colors** (w=0.5): `vibrant natural realistic colors with good lighting and contrast`
- **professional** (w=0.5): `professional commercial product photography, studio quality render`

---

## Результаты

### SD15

#### SD15_bo4_multi

![SD15_bo4_multi](SD15_bo4_multi.png)

| Metric | Value |
|--------|-------|
| SigLIP2 avg | **-0.477** |
| Kimi avg | **6.1**/10 |

Per-axis raw scores (avg across samples):

| Axis | Avg score |
|------|-----------|
| prompt_match | -0.48 |
| no_artifacts | -6.39 |
| sharpness | -7.87 |
| anatomy | -6.26 |
| composition | -2.80 |
| colors | -7.52 |
| professional | -9.65 |

#### SD15_bo4_single

![SD15_bo4_single](SD15_bo4_single.png)

| Metric | Value |
|--------|-------|
| SigLIP2 avg | **0.021** |
| Kimi avg | **8.6**/10 |

Per-axis raw scores (avg across samples):

| Axis | Avg score |
|------|-----------|
| prompt_match | 0.02 |
| no_artifacts | -7.09 |
| sharpness | -8.52 |
| anatomy | -6.76 |
| composition | -2.74 |
| colors | -8.07 |
| professional | -10.25 |

#### SD15_bo8_multi

![SD15_bo8_multi](SD15_bo8_multi.png)

| Metric | Value |
|--------|-------|
| SigLIP2 avg | **-0.234** |
| Kimi avg | **5.5**/10 |

Per-axis raw scores (avg across samples):

| Axis | Avg score |
|------|-----------|
| prompt_match | -0.23 |
| no_artifacts | -6.21 |
| sharpness | -7.60 |
| anatomy | -6.32 |
| composition | -2.30 |
| colors | -7.37 |
| professional | -9.59 |

#### SD15_bo8_single

![SD15_bo8_single](SD15_bo8_single.png)

| Metric | Value |
|--------|-------|
| SigLIP2 avg | **0.365** |
| Kimi avg | **4.9**/10 |

Per-axis raw scores (avg across samples):

| Axis | Avg score |
|------|-----------|
| prompt_match | 0.37 |
| no_artifacts | -6.88 |
| sharpness | -8.05 |
| anatomy | -6.82 |
| composition | -2.57 |
| colors | -7.91 |
| professional | -10.10 |

### SDXL

#### SDXL_bo4_multi

![SDXL_bo4_multi](SDXL_bo4_multi.png)

| Metric | Value |
|--------|-------|
| SigLIP2 avg | **0.391** |
| Kimi avg | **4.3**/10 |

Per-axis raw scores (avg across samples):

| Axis | Avg score |
|------|-----------|
| prompt_match | 0.39 |
| no_artifacts | -6.73 |
| sharpness | -6.90 |
| anatomy | -6.92 |
| composition | -3.18 |
| colors | -6.98 |
| professional | -10.37 |

#### SDXL_bo4_single

![SDXL_bo4_single](SDXL_bo4_single.png)

| Metric | Value |
|--------|-------|
| SigLIP2 avg | **0.550** |
| Kimi avg | **3.1**/10 |

Per-axis raw scores (avg across samples):

| Axis | Avg score |
|------|-----------|
| prompt_match | 0.55 |
| no_artifacts | -7.11 |
| sharpness | -7.21 |
| anatomy | -6.94 |
| composition | -3.67 |
| colors | -7.03 |
| professional | -10.96 |

#### SDXL_bo8_multi

![SDXL_bo8_multi](SDXL_bo8_multi.png)

| Metric | Value |
|--------|-------|
| SigLIP2 avg | **0.062** |
| Kimi avg | **3.1**/10 |

Per-axis raw scores (avg across samples):

| Axis | Avg score |
|------|-----------|
| prompt_match | 0.06 |
| no_artifacts | -6.37 |
| sharpness | -6.89 |
| anatomy | -6.39 |
| composition | -2.79 |
| colors | -7.06 |
| professional | -9.98 |

#### SDXL_bo8_single

![SDXL_bo8_single](SDXL_bo8_single.png)

| Metric | Value |
|--------|-------|
| SigLIP2 avg | **0.709** |
| Kimi avg | **4.8**/10 |

Per-axis raw scores (avg across samples):

| Axis | Avg score |
|------|-----------|
| prompt_match | 0.71 |
| no_artifacts | -7.10 |
| sharpness | -7.23 |
| anatomy | -6.95 |
| composition | -3.56 |
| colors | -6.86 |
| professional | -11.12 |

---

## Сводная таблица

| Config | SigLIP2 | Kimi | prompt_match | no_artifacts | sharpness | anatomy | composition | colors | professional |
|--------|---------|------|------|------|------|------|------|------|------|
| SD15_bo4_single | 0.021 | 8.6 | 0.02 | -7.09 | -8.52 | -6.76 | -2.74 | -8.07 | -10.25 |
| SD15_bo4_multi | -0.477 | 6.1 | -0.48 | -6.39 | -7.87 | -6.26 | -2.80 | -7.52 | -9.65 |
| SD15_bo8_single | 0.365 | 4.9 | 0.37 | -6.88 | -8.05 | -6.82 | -2.57 | -7.91 | -10.10 |
| SD15_bo8_multi | -0.234 | 5.5 | -0.23 | -6.21 | -7.60 | -6.32 | -2.30 | -7.37 | -9.59 |
| SDXL_bo4_single | 0.550 | 3.1 | 0.55 | -7.11 | -7.21 | -6.94 | -3.67 | -7.03 | -10.96 |
| SDXL_bo4_multi | 0.391 | 4.3 | 0.39 | -6.73 | -6.90 | -6.92 | -3.18 | -6.98 | -10.37 |
| SDXL_bo8_single | 0.709 | 4.8 | 0.71 | -7.10 | -7.23 | -6.95 | -3.56 | -6.86 | -11.12 |
| SDXL_bo8_multi | 0.062 | 3.1 | 0.06 | -6.37 | -6.89 | -6.39 | -2.79 | -7.06 | -9.98 |

---

## Выводы

*(заполняется по результатам)*
