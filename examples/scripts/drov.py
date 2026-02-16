"""
DRO-V experiment on UltraFeedback.

Paper-style run:
uv run python examples/scripts/drov.py \
    --output_dir drov-paper-fold0 \
    --model_name_or_path google/flan-t5-large \
    --dataset_revision main \
    --num_folds 5 --eval_fold_index 0 --prompts_per_fold 5000 \
    --tau 1.0 --learning_rate 1e-4 --value_learning_rate 1e-4 \
    --max_steps 40000 --warmup_steps 150 --per_device_train_batch_size 32 \
    --optim adafactor --lr_scheduler_type linear --dtype bfloat16

With judge eval (Gemma 3 27B IT via OpenRouter):
OPENAI_BASE_URL=https://openrouter.ai/api/v1 OPENAI_API_KEY=<your-key> \
uv run python examples/scripts/drov.py \
    --output_dir drov-paper-fold0 \
    --model_name_or_path google/flan-t5-large \
    --dataset_revision main \
    --num_folds 5 --eval_fold_index 0 --prompts_per_fold 5000 \
    --tau 1.0 --learning_rate 1e-5 --value_learning_rate 1e-5 \
    --max_steps 40000 --warmup_steps 150 \
    --per_device_train_batch_size 16 --gradient_accumulation_steps 2 \
    --optim adafactor --lr_scheduler_type linear --dtype bfloat16 \
    --max_prompt_length 512 --max_completion_length 512 \
    --eval_steps 2000 \
    --judge_model google/gemma-3-27b-it \
    --judge_num_prompts 50 --judge_max_new_tokens 1024

Overfit one batch (sanity check):
uv run python examples/scripts/drov.py \
    --output_dir drov-overfit \
    --model_name_or_path google/flan-t5-large \
    --dataset_revision main \
    --overfit_one_batch --per_device_train_batch_size 8 \
    --optim adafactor --dtype bfloat16
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser

from trl.experimental.drov import DROVConfig, DROVTrainer

logger = logging.getLogger(__name__)


DTYPE_MAP: dict[str, torch.dtype | None] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "auto": None,
}


@dataclass
class ScriptArgs:
    """Arguments specific to this experiment script (not part of DROVConfig)."""

    model_name_or_path: str = field(default="google/flan-t5-large")
    dtype: str = field(default="float32")
    trust_remote_code: bool = field(default=False)

    dataset_name: str = field(default="openbmb/UltraFeedback")
    dataset_revision: str = field(default="main")
    dataset_split: str = field(default="train")
    aspect: str = field(default="helpfulness")
    reward_definition: str = field(default="rating")
    thumbs_up_threshold: float = field(default=5.0)

    num_folds: int = field(default=5)
    eval_fold_index: int = field(default=0)
    prompts_per_fold: int | None = field(default=None)

    max_train_samples: int | None = field(default=None)
    max_eval_samples: int | None = field(default=None)

    overfit_one_batch: bool = field(default=False)
    overfit_steps: int = field(default=300)

    judge_model: str | None = field(default=None, metadata={"help": "OpenAI-compatible judge model (e.g. OpenRouter model ID). Enables judge eval when set."})

    judge_num_prompts: int = field(default=256, metadata={"help": "Number of unique prompts for judge eval."})
    judge_max_new_tokens: int = field(default=256, metadata={"help": "Max new tokens during judge generation."})


# ---------------------------------------------------------------------------
# UltraFeedback dataset helpers
# ---------------------------------------------------------------------------


def expand_ultrafeedback_batch(
    batch: dict[str, list],
    aspect: str,
    reward_definition: str,
    thumbs_up_threshold: float,
) -> dict[str, list]:
    prompts: list[str] = []
    completions: list[str] = []
    rewards: list[float] = []
    for instruction, comps in zip(batch["instruction"], batch["completions"], strict=True):
        for c in comps:
            prompts.append(instruction)
            completions.append(c["response"])
            r = c["overall_score"] if aspect in "overall" else float(c["annotations"][aspect]["Rating"])
            if reward_definition == "binarized_threshold":
                r = 1.0 if r >= thumbs_up_threshold else -1.0
            rewards.append(r)
    return {"prompt": prompts, "completion": completions, "reward": rewards}


def split_by_prompt_folds(
    rows: Dataset,
    num_folds: int,
    eval_fold_index: int,
    seed: int,
    prompts_per_fold: int | None = None,
    prompt_column: str = "instruction",
) -> DatasetDict:
    prompts = rows[prompt_column]
    unique = np.asarray(list(dict.fromkeys(prompts)), dtype=object)
    perm = unique[np.random.default_rng(seed).permutation(len(unique))]

    if prompts_per_fold is not None:
        required = num_folds * prompts_per_fold
        if required > len(perm):
            raise ValueError(f"Need >= {required} unique prompts, found {len(perm)}.")
        sel = perm[:required]
        lo, hi = eval_fold_index * prompts_per_fold, (eval_fold_index + 1) * prompts_per_fold
        eval_set = set(sel[lo:hi].tolist())
        train_set = set(np.concatenate((sel[:lo], sel[hi:])).tolist())
    else:
        folds = np.array_split(perm, num_folds)
        eval_set = set(folds[eval_fold_index].tolist())
        train_set = set(np.concatenate([f for i, f in enumerate(folds) if i != eval_fold_index]).tolist())

    eval_mask = np.fromiter((p in eval_set for p in prompts), bool, len(prompts))
    train_mask = np.fromiter((p in train_set for p in prompts), bool, len(prompts))
    return DatasetDict({
        "train": rows.select(np.nonzero(train_mask)[0].tolist()),
        "test": rows.select(np.nonzero(eval_mask)[0].tolist()),
    })


def normalize_reward(row: dict, mean: float, std: float) -> dict:
    return {"reward": (float(row["reward"]) - mean) / std}


def prepare_triplet_dataset(sa: ScriptArgs, ta: DROVConfig) -> DatasetDict:
    raw = load_dataset(sa.dataset_name, split=sa.dataset_split, revision=sa.dataset_revision)
    splits = split_by_prompt_folds(raw, sa.num_folds, sa.eval_fold_index, ta.seed, sa.prompts_per_fold)
    logger.info("Prompt-fold split: train=%d rows, eval=%d rows", len(splits["train"]), len(splits["test"]))

    expand_kw = dict(aspect=sa.aspect, reward_definition=sa.reward_definition, thumbs_up_threshold=sa.thumbs_up_threshold)
    train_ds = splits["train"].map(
        expand_ultrafeedback_batch, batched=True, remove_columns=splits["train"].column_names,
        num_proc=ta.dataset_num_proc, fn_kwargs=expand_kw, desc="Expanding train triplets",
    ).shuffle(seed=ta.seed)
    eval_ds = splits["test"].map(
        expand_ultrafeedback_batch, batched=True, remove_columns=splits["test"].column_names,
        num_proc=ta.dataset_num_proc, fn_kwargs=expand_kw, desc="Expanding eval triplets",
    )

    rewards = np.asarray(train_ds["reward"], dtype=np.float32)
    mu, sigma = float(rewards.mean()), float(rewards.std())
    logger.info(
        "Triplets: train=%d, eval=%d | reward mean=%.4f, std=%.4f",
        len(train_ds), len(eval_ds), mu, sigma,
    )
    norm_kw = dict(mean=mu, std=sigma)
    train_ds = train_ds.map(normalize_reward, num_proc=ta.dataset_num_proc, fn_kwargs=norm_kw)
    eval_ds = eval_ds.map(normalize_reward, num_proc=ta.dataset_num_proc, fn_kwargs=norm_kw)

    if sa.max_train_samples is not None:
        train_ds = train_ds.select(range(min(sa.max_train_samples, len(train_ds))))
    if sa.max_eval_samples is not None:
        eval_ds = eval_ds.select(range(min(sa.max_eval_samples, len(eval_ds))))

    out = Path(ta.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "prompt_split_manifest.json").write_text(json.dumps({
        "dataset": sa.dataset_name, "revision": sa.dataset_revision, "aspect": sa.aspect,
        "reward_definition": sa.reward_definition, "seed": ta.seed,
        "num_folds": sa.num_folds, "eval_fold_index": sa.eval_fold_index,
        "prompts_per_fold": sa.prompts_per_fold,
        "train_rows": len(train_ds), "eval_rows": len(eval_ds),
        "reward_mean": mu, "reward_std": sigma,
    }, indent=2) + "\n")

    return DatasetDict({"train": train_ds, "test": eval_ds})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO)

    os.environ.setdefault("WANDB_PROJECT", "drov-ultrafeedback")

    parser = HfArgumentParser((ScriptArgs, DROVConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()

    # Overfit-one-batch convenience mode
    if script_args.overfit_one_batch:
        one_batch = training_args.per_device_train_batch_size * max(training_args.gradient_accumulation_steps, 1)
        if script_args.max_train_samples is None or script_args.max_train_samples > one_batch:
            script_args.max_train_samples = one_batch
        if script_args.max_eval_samples is None:
            script_args.max_eval_samples = one_batch
        training_args.max_steps = script_args.overfit_steps
        training_args.warmup_steps = 0
        training_args.logging_steps = 1
        training_args.eval_strategy = "no"
    else:
        if training_args.eval_strategy in ("no", "IntervalStrategy.NO") or str(training_args.eval_strategy).endswith("no"):
            training_args.eval_strategy = "steps"
        if training_args.eval_steps is None:
            training_args.eval_steps = training_args.save_steps

    torch_dtype = DTYPE_MAP[script_args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, trust_remote_code=script_args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSeq2SeqLM.from_pretrained(
        script_args.model_name_or_path, trust_remote_code=script_args.trust_remote_code, dtype=torch_dtype,
    )
    ref_model = AutoModelForSeq2SeqLM.from_pretrained(
        script_args.model_name_or_path, trust_remote_code=script_args.trust_remote_code, dtype=torch_dtype,
    )
    value_model = AutoModelForSequenceClassification.from_pretrained(
        script_args.model_name_or_path, trust_remote_code=script_args.trust_remote_code, dtype=torch_dtype, num_labels=1,
    )

    dataset = prepare_triplet_dataset(script_args, training_args)

    trainer = DROVTrainer(
        model=model,
        value_model=value_model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=None if script_args.overfit_one_batch else dataset["test"],
        processing_class=tokenizer,
    )

    if script_args.judge_model is not None and not script_args.overfit_one_batch:
        from transformers import GenerationConfig
        from trl.experimental.judges import OpenAIPairwiseJudge
        from trl.experimental.winrate_callback import WinRateCallback

        judge = OpenAIPairwiseJudge(model=script_args.judge_model, max_requests=None)
        generation_config = GenerationConfig(
            max_new_tokens=script_args.judge_max_new_tokens,
            do_sample=False,
        )
        win_rate_callback = WinRateCallback(
            judge=judge,
            trainer=trainer,
            generation_config=generation_config,
            num_prompts=script_args.judge_num_prompts,
        )
        trainer.add_callback(win_rate_callback)

    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
