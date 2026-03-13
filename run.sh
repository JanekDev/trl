#!/usr/bin/env bash

set -euo pipefail

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TRL_EXPERIMENTAL_SILENCE=1 \
uv run python examples/scripts/dro.py \
  --model_name_or_path Qwen/Qwen3-1.7B \
  --hub_model_id MWilinski/dro-v-qwen3-1.7b-paperlike \
  --output_dir /tmp/dro-v-qwen3-1.7b-paperlike \
  --max_steps 40000 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 2 \
  --val_size 2000 \
  --test_size 2000 \
  --split_seed 42 \
  --learning_rate 1e-4 \
  --optim adafactor \
  --warmup_steps 150 \
  --tau 1.0 \
  --normalize_rewards True \
  --max_length 1024 \
  --max_prompt_length 512 \
  --dtype bfloat16 \
  --use_peft \
  --lora_r 32 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_target_modules all-linear \
  --save_strategy steps \
  --save_steps 2000 \
  --eval_strategy no \
  --logging_steps 10 \
  --report_to wandb \
  --push_to_hub True \
  --hub_strategy all_checkpoints \
  --run_name dro-v-paperlike-qwen3-1.7b
