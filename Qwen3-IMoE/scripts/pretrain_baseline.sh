#!/usr/bin/env bash
# Pre-train the dense Qwen3 baseline on a single node.
# Usage: NUM_GPUS=4 CUDA_DEVICES=0,1,2,3 bash scripts/pretrain_baseline.sh
set -euo pipefail

NUM_GPUS="${NUM_GPUS:-4}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export MASTER_ADDR="${MASTER_ADDR:-localhost}"
export MASTER_PORT="${MASTER_PORT:-9901}"

MODEL_SIZE="${MODEL_SIZE:-1.7B}"
TRAIN_DATA="${TRAIN_DATA:?Set TRAIN_DATA to a glob pattern, e.g. \"data/train/*.parquet\"}"

accelerate launch --config_file configs/accelerate/accelerate_ddp.yaml --num_processes "$NUM_GPUS" \
  pretraining.py \
  --train_type bs \
  --model_size "$MODEL_SIZE" \
  --tokenizer_id "${TOKENIZER_ID:-Qwen/Qwen3-$MODEL_SIZE}" \
  --train_data $TRAIN_DATA \
  --batch_size "${BATCH_SIZE:-6}" \
  --grad_accum "${GRAD_ACCUM:-32}" \
  --learning_rate "${LEARNING_RATE:-1e-3}" \
  --precision fp16
