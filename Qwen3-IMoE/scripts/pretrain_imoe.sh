#!/usr/bin/env bash
# Pre-train the CS-MoE (IMoE) model on a single node.
# Usage: TRAIN_DATA="data/train/*.parquet" bash scripts/pretrain_imoe.sh
set -euo pipefail

NUM_GPUS="${NUM_GPUS:-4}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export MASTER_ADDR="${MASTER_ADDR:-localhost}"
export MASTER_PORT="${MASTER_PORT:-9901}"

MODEL_SIZE="${MODEL_SIZE:-0.6B}"
CONFIG="${CONFIG:-configs/qwen3_0_6_A0_6b.json}"
TRAIN_DATA="${TRAIN_DATA:?Set TRAIN_DATA to a glob pattern, e.g. \"data/train/*.parquet\"}"

accelerate launch --config_file configs/accelerate/accelerate_ddp.yaml --num_processes "$NUM_GPUS" \
  pretraining.py \
  --train_type imoe \
  --model_size "$MODEL_SIZE" \
  --config "$CONFIG" \
  --train_data $TRAIN_DATA \
  --batch_size "${BATCH_SIZE:-2}" \
  --grad_accum "${GRAD_ACCUM:-32}" \
  --learning_rate "${LEARNING_RATE:-1e-3}" \
  --precision fp16
