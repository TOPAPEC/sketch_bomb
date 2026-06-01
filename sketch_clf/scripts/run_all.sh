#!/bin/bash
# Sequentially train several backbones, all logged to MLflow. One GPU -> sequential.
set -e
cd "$(dirname "$0")"
PY=/venv/main/bin/python
LOG=../train_log.txt
: > "$LOG"

run () {
  echo "==================== $* ====================" | tee -a "$LOG"
  stdbuf -oL -eL $PY -u train.py "$@" 2>&1 | grep --line-buffered -viE "warning|deprecat|B/s|safetensors|FutureWarning" | tee -a "$LOG"
}

# 1) ViT-small ImageNet-21k init, full fine-tune (fast main model)
run --model vit_small_patch16_224.augreg_in21k --mode finetune \
    --epochs 12 --bs 256 --lr 1e-3 --backbone-lr 3e-5 --run-name vit_small_finetune

# 2) ViT-small linear probe (frozen backbone) -> shows transfer gap
run --model vit_small_patch16_224.augreg_in21k --mode linear_probe \
    --epochs 10 --bs 256 --lr 2e-3 --run-name vit_small_linearprobe

# 3) ViT-base ImageNet-21k init, full fine-tune (stronger)
run --model vit_base_patch16_224.augreg_in21k --mode finetune \
    --epochs 10 --bs 160 --lr 1e-3 --backbone-lr 2e-5 --run-name vit_base_finetune

# 4) CLIP-init ViT-base, full fine-tune (different pretraining)
run --model vit_base_patch16_clip_224.openai --mode finetune \
    --epochs 10 --bs 160 --lr 1e-3 --backbone-lr 2e-5 --run-name vit_base_clip_finetune

echo "ALL DONE" | tee -a "$LOG"
