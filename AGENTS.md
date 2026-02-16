# Project Context

This repository is a local TRL fork focused on an experimental DRO-V implementation.

## Current Goal

Keep `DROVTrainer` and `examples/scripts/drov.py` aligned with:

1. The DRO-V paper objective and training setup.
2. Common TRL trainer conventions.
3. Practical reproducibility and evaluation workflow.

## Main Files For This Workstream

- `examples/scripts/drov.py`
- `trl/experimental/drov/drov_trainer.py`
- `trl/experimental/drov/drov_config.py`
- `tests/experimental/test_drov_trainer.py`
- Reference paper source in `dro-paper/neurips_2024.tex`

## What Has Already Been Done

1. Fixed DRO-V eval path by implementing a custom `prediction_step` in `DROVTrainer`.
2. Moved reward normalization to train-split statistics only, then applied to eval.
3. Updated DRO-V `_paper` metadata from placeholders to concrete arXiv info.
4. Added optional policy/value backbone sharing for memory saving:
   - config flag: `share_policy_and_value_backbone`
   - script flag: `--share_policy_and_value_backbone`
5. Added overfit sanity mode in the example script:
   - `--overfit_one_batch`
   - `--overfit_steps`
6. Added both paper-style and overfit run recipes at top of `examples/scripts/drov.py`.
7. Added test coverage for eval path and backbone sharing wiring in `tests/experimental/test_drov_trainer.py`.

## Data Split Protocol

The UltraFeedback HF dataset split used here is typically `train`, then internally split into train/eval.

- Default script mode supports seeded prompt-level split (`split_mode=prompt_seeded`).
- For paper-style reproduction, use:
  - `--split_mode prompt_fold`
  - `--num_eval_folds 5`
  - `--eval_prompts_per_fold 1000`
  - run `--eval_fold_index` in `0..4`.

Split metadata is written to:

- `<output_dir>/prompt_split_manifest.json`

## Run Commands

### Overfit One Batch

```bash
python examples/scripts/drov.py \
  --output_dir drov-overfit-one-batch \
  --dataset_revision <ultrafeedback_commit_or_tag> \
  --model_name_or_path google/flan-t5-large \
  --overfit_one_batch \
  --overfit_steps 300 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1
```

### Paper-Style Single Fold

```bash
python examples/scripts/drov.py \
  --output_dir drov-paper-t5-large-fold0 \
  --dataset_revision <ultrafeedback_commit_or_tag> \
  --model_name_or_path google/flan-t5-large \
  --split_mode prompt_fold \
  --num_eval_folds 5 \
  --eval_fold_index 0 \
  --eval_prompts_per_fold 1000 \
  --tau 1.0 \
  --learning_rate 1e-4 \
  --value_learning_rate 1e-4 \
  --max_steps 40000 \
  --warmup_steps 150 \
  --per_device_train_batch_size 32
```

## Model Comparison / Judging Protocol

Use held-out prompts only (never training prompts), with identical generation settings across models.

Recommended:

1. Evaluate on each fold test set.
2. Do pairwise A/B blind judging per prompt (randomized order).
3. Aggregate win-rate across folds, with confidence intervals.
4. Keep dataset revision and split seed/fold settings fixed for all compared models.

## Environment Notes

- `pytest` may be missing in local env; if so, use `uv run python` smoke checks.
- Use `PYTHONPYCACHEPREFIX=/tmp/pycache` if local pycache path permissions fail.
