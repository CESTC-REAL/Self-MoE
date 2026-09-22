#!/usr/bin/env bash
# SFT the CS-MoE (IMoE) checkpoint.
# Usage: DATASET_PATH="data/sft/*.json" bash scripts/sft_imoe.sh
set -euo pipefail

NUM_GPUS="${NUM_GPUS:-4}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export MASTER_ADDR="${MASTER_ADDR:-localhost}"
export MASTER_PORT="${MASTER_PORT:-9901}"

MODEL_SIZE="${MODEL_SIZE:-0.6B}"
DATASET_PATH="${DATASET_PATH:?Set DATASET_PATH to a glob pattern, e.g. \"data/sft/*.json\"}"

accelerate launch --config_file configs/accelerate/accelerate_ddp.yaml --num_processes "$NUM_GPUS" \
  finetuning.py \
  --train_type imoe \
  --model_size "$MODEL_SIZE" \
  --model_path "${MODEL_PATH:-outputs/qwen-imoe-$MODEL_SIZE-pt}" \
  --dataset_path $DATASET_PATH \
  --batch_size "${BATCH_SIZE:-12}" \
  --grad_accum "${GRAD_ACCUM:-8}" \
  --learning_rate "${LEARNING_RATE:-1e-4}" \
  --precision fp16
