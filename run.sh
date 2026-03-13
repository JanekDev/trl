#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-1.7B}"
HUB_MODEL_ID="${HUB_MODEL_ID:-MWilinski/dro-v-qwen3-1.7b-paperlike}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/dro-v-qwen3-1.7b-paperlike}"
RUN_NAME="${RUN_NAME:-dro-v-paperlike-qwen3-1.7b}"
MAX_STEPS="${MAX_STEPS:-40000}"
TARGET_GLOBAL_BATCH_SIZE="${TARGET_GLOBAL_BATCH_SIZE:-32}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
VAL_SIZE="${VAL_SIZE:-2000}"
TEST_SIZE="${TEST_SIZE:-2000}"
SPLIT_SEED="${SPLIT_SEED:-42}"

NUM_GPUS="$(uv run python -c 'import torch; n = torch.cuda.device_count(); print(n if n > 0 else 1)')"
GRADIENT_ACCUMULATION_STEPS="$(( TARGET_GLOBAL_BATCH_SIZE / (NUM_GPUS * PER_DEVICE_TRAIN_BATCH_SIZE) ))"

echo "MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}"
echo "NUM_GPUS=${NUM_GPUS}"
echo "PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE}"
echo "TARGET_GLOBAL_BATCH_SIZE=${TARGET_GLOBAL_BATCH_SIZE}"
echo "GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}"
echo "VAL_SIZE=${VAL_SIZE}"
echo "TEST_SIZE=${TEST_SIZE}"
echo "SPLIT_SEED=${SPLIT_SEED}"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TRL_EXPERIMENTAL_SILENCE=1 \
uv run python examples/scripts/dro.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --hub_model_id "${HUB_MODEL_ID}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_steps "${MAX_STEPS}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --val_size "${VAL_SIZE}" \
  --test_size "${TEST_SIZE}" \
  --split_seed "${SPLIT_SEED}" \
  --learning_rate 1e-4 \
  --optim adafactor \
  --warmup_steps 150 \
  --tau 1.0 \
  --normalize_rewards True \
  --max_length 1024 \
  --max_prompt_length 512 \
  --dtype bfloat16 \
  --save_strategy steps \
  --save_steps 2000 \
  --eval_strategy no \
  --logging_steps 10 \
  --report_to wandb \
  --push_to_hub True \
  --hub_strategy all_checkpoints \
  --run_name "${RUN_NAME}"
