"""
Run a basic DRO-V experiment inspired by the paper:
"Offline Regularised Reinforcement Learning for Large Language Models Alignment".

Overfit one batch (sanity check):
python examples/scripts/drov.py \
    --output_dir drov-overfit-one-batch \
    --dataset_revision <ultrafeedback_commit_or_tag> \
    --model_name_or_path google/flan-t5-large \
    --overfit_one_batch \
    --overfit_steps 300 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1

Paper-style training run (single eval fold):
python examples/scripts/drov.py \
    --output_dir drov-paper-t5-large-fold0 \
    --dataset_revision <ultrafeedback_commit_or_tag> \
    --model_name_or_path google/flan-t5-large \
    --num_eval_folds 5 \
    --eval_fold_index 0 \
    --eval_prompts_per_fold 1000 \
    --tau 1.0 \
    --policy_learning_rate 1e-4 \
    --value_learning_rate 1e-4 \
    --max_steps 40000 \
    --warmup_steps 150 \
    --per_device_train_batch_size 32

Quick smoke run:
python examples/scripts/drov.py \
    --output_dir drov-smoke \
    --dataset_revision <ultrafeedback_commit_or_tag> \
    --max_steps 50 \
    --max_train_samples 1024 \
    --max_eval_samples 256 \
    --per_device_train_batch_size 8 \
    --save_steps 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer

from trl.experimental.drov import DROVConfig, DROVTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a basic DRO-V experiment on UltraFeedback.")

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, default="google/flan-t5-large")
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16", "auto"], default="float32")

    parser.add_argument("--dataset_name", type=str, default="openbmb/UltraFeedback")
    parser.add_argument(
        "--dataset_revision",
        type=str,
        required=True,
        help="Pinned dataset revision (commit hash/tag). Required for reproducibility.",
    )
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--aspect", type=str, default="helpfulness")
    parser.add_argument("--source_model", type=str, default=None)
    parser.add_argument(
        "--reward_definition",
        type=str,
        choices=["rating", "binarized_threshold"],
        default="rating",
    )
    parser.add_argument("--thumbs_up_threshold", type=float, default=5.0)
    parser.add_argument("--num_eval_folds", type=int, default=5)
    parser.add_argument("--eval_fold_index", type=int, default=0)
    parser.add_argument("--eval_prompts_per_fold", type=int, default=1000)
    parser.add_argument(
        "--save_prompt_splits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save the selected prompt fold metadata in output_dir for reproducibility.",
    )
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--dataset_num_proc", type=int, default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--policy_learning_rate", type=float, default=1e-4)
    parser.add_argument("--value_learning_rate", type=float, default=1e-4)
    parser.add_argument(
        "--share_policy_and_value_backbone",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Share policy backbone parameters with the value model backbone to reduce memory usage.",
    )

    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=40000)
    parser.add_argument("--warmup_steps", type=int, default=150)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument(
        "--overfit_one_batch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable one-batch overfitting sanity mode. "
            "This limits train samples to a single optimization batch and disables eval."
        ),
    )
    parser.add_argument(
        "--overfit_steps",
        type=int,
        default=300,
        help="Training steps to run when --overfit_one_batch is enabled.",
    )

    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_completion_length", type=int, default=512)
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=torch.cuda.is_available())
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)

    return parser.parse_args()


def _parse_dtype(dtype: str) -> torch.dtype | None:
    if dtype == "auto":
        return None
    if dtype == "float32":
        return torch.float32
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype '{dtype}'.")


def _expand_ultrafeedback_batch(
    batch: dict[str, list],
    aspect: str,
    source_model: str | None,
    reward_definition: str,
    thumbs_up_threshold: float,
) -> dict[str, list]:
    prompts: list[str] = []
    completions: list[str] = []
    rewards: list[float] = []

    for instruction, models, completion_list in zip(
        batch["instruction"], batch["models"], batch["completions"], strict=True
    ):
        for model_name, completion in zip(models, completion_list, strict=True):
            if source_model is not None and model_name != source_model:
                continue
            response = completion.get("response")
            if response is None:
                continue
            annotations = completion.get("annotations", {})
            aspect_annotations = annotations.get(aspect, {})
            rating = aspect_annotations.get("Rating")
            if rating is None:
                continue

            rating_value = float(rating)
            if reward_definition == "rating":
                reward_value = rating_value
            elif reward_definition == "binarized_threshold":
                reward_value = 1.0 if rating_value >= thumbs_up_threshold else -1.0
            else:
                raise ValueError(f"Unsupported reward definition: {reward_definition}")

            prompts.append(instruction)
            completions.append(response)
            rewards.append(reward_value)

    return {"prompt": prompts, "completion": completions, "reward": rewards}


def _split_triplets_by_prompt_folds(triplets: Dataset, args: argparse.Namespace) -> DatasetDict:
    if args.num_eval_folds <= 0:
        raise ValueError("--num_eval_folds must be a positive integer.")
    if args.eval_prompts_per_fold <= 0:
        raise ValueError("--eval_prompts_per_fold must be a positive integer.")
    if args.eval_fold_index < 0 or args.eval_fold_index >= args.num_eval_folds:
        raise ValueError("--eval_fold_index must be in [0, num_eval_folds).")

    prompts = np.asarray(triplets["prompt"], dtype=object)
    unique_prompts = np.unique(prompts)

    required_eval_prompts = args.num_eval_folds * args.eval_prompts_per_fold
    if required_eval_prompts >= len(unique_prompts):
        raise ValueError(
            "Not enough unique prompts for the requested fold setup. "
            f"Need > {required_eval_prompts}, found {len(unique_prompts)}."
        )

    rng = np.random.default_rng(args.seed)
    permuted_prompts = unique_prompts[rng.permutation(len(unique_prompts))]
    eval_prompt_grid = permuted_prompts[:required_eval_prompts].reshape(args.num_eval_folds, args.eval_prompts_per_fold)
    eval_prompt_set = set(eval_prompt_grid[args.eval_fold_index].tolist())

    eval_mask = np.fromiter((prompt in eval_prompt_set for prompt in prompts), dtype=bool, count=len(prompts))
    eval_indices = np.nonzero(eval_mask)[0].tolist()
    train_indices = np.nonzero(~eval_mask)[0].tolist()

    train_dataset = triplets.select(train_indices)
    eval_dataset = triplets.select(eval_indices)
    return DatasetDict({"train": train_dataset, "test": eval_dataset})


def _save_prompt_split_manifest(
    dataset: DatasetDict,
    args: argparse.Namespace,
    reward_mean: float | None = None,
    reward_std: float | None = None,
) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "prompt_split_manifest.json"
    manifest = {
        "dataset_name": args.dataset_name,
        "dataset_revision": args.dataset_revision,
        "dataset_split": args.dataset_split,
        "aspect": args.aspect,
        "source_model": args.source_model,
        "reward_definition": args.reward_definition,
        "thumbs_up_threshold": args.thumbs_up_threshold,
        "seed": args.seed,
        "num_eval_folds": args.num_eval_folds,
        "eval_fold_index": args.eval_fold_index,
        "eval_prompts_per_fold": args.eval_prompts_per_fold,
        "train_num_rows": len(dataset["train"]),
        "eval_num_rows": len(dataset["test"]),
        "train_unique_prompts": len(set(dataset["train"]["prompt"])),
        "eval_unique_prompts": len(set(dataset["test"]["prompt"])),
    }
    if reward_mean is not None:
        manifest["train_reward_mean_before_norm"] = reward_mean
    if reward_std is not None:
        manifest["train_reward_std_before_norm"] = reward_std
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _normalize_reward_row(row: dict[str, float], reward_mean: float, reward_std: float) -> dict[str, float]:
    return {"reward": (float(row["reward"]) - reward_mean) / reward_std}


def _share_policy_backbone_with_value_model(model: torch.nn.Module, value_model: torch.nn.Module) -> None:
    if not hasattr(model, "base_model_prefix") or not hasattr(value_model, "base_model_prefix"):
        raise ValueError("Backbone sharing requires both policy and value models to define `base_model_prefix`.")

    policy_backbone_name = model.base_model_prefix
    value_backbone_name = value_model.base_model_prefix
    policy_backbone = getattr(model, policy_backbone_name, None)
    value_backbone = getattr(value_model, value_backbone_name, None)
    if value_backbone is None:
        raise ValueError("Backbone sharing failed because the value model does not expose its backbone module.")

    if policy_backbone is not None:
        setattr(value_model, value_backbone_name, policy_backbone)
    elif all(hasattr(model, attr) for attr in ("shared", "encoder", "decoder")) and all(
        hasattr(value_backbone, attr) for attr in ("shared", "encoder", "decoder")
    ):
        value_backbone.shared = model.shared
        value_backbone.encoder = model.encoder
        value_backbone.decoder = model.decoder
    else:
        raise ValueError(
            "Backbone sharing failed because the policy does not expose a shareable backbone module "
            f"for `{policy_backbone_name}`."
        )


def prepare_triplet_dataset(args: argparse.Namespace) -> DatasetDict:
    raw_dataset = load_dataset(args.dataset_name, split=args.dataset_split, revision=args.dataset_revision)
    triplets = raw_dataset.map(
        _expand_ultrafeedback_batch,
        batched=True,
        remove_columns=raw_dataset.column_names,
        num_proc=args.dataset_num_proc,
        fn_kwargs={
            "aspect": args.aspect,
            "source_model": args.source_model,
            "reward_definition": args.reward_definition,
            "thumbs_up_threshold": args.thumbs_up_threshold,
        },
        desc="Building (prompt, completion, reward) triplets",
    )
    if len(triplets) == 0:
        raise ValueError("No triplets were produced from the dataset with the current filters.")

    dataset = _split_triplets_by_prompt_folds(triplets, args)
    train_dataset: Dataset = dataset["train"]
    eval_dataset: Dataset = dataset["test"]

    train_reward_values = np.asarray(train_dataset["reward"], dtype=np.float32)
    reward_mean = float(train_reward_values.mean())
    reward_std = float(train_reward_values.std())
    if reward_std < 1e-6:
        reward_std = 1.0
    train_dataset = train_dataset.map(
        _normalize_reward_row,
        num_proc=args.dataset_num_proc,
        fn_kwargs={"reward_mean": reward_mean, "reward_std": reward_std},
        desc="Normalizing train rewards to mean 0 / std 1",
    )
    eval_dataset = eval_dataset.map(
        _normalize_reward_row,
        num_proc=args.dataset_num_proc,
        fn_kwargs={"reward_mean": reward_mean, "reward_std": reward_std},
        desc="Normalizing eval rewards with train statistics",
    )

    if args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(args.max_train_samples, len(train_dataset))))
    if args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(min(args.max_eval_samples, len(eval_dataset))))

    split_dataset = DatasetDict({"train": train_dataset, "test": eval_dataset})
    if args.save_prompt_splits:
        _save_prompt_split_manifest(split_dataset, args, reward_mean=reward_mean, reward_std=reward_std)
    return split_dataset


def _apply_overfit_one_batch_mode(args: argparse.Namespace) -> None:
    if not args.overfit_one_batch:
        return

    if args.overfit_steps <= 0:
        raise ValueError("--overfit_steps must be a positive integer when --overfit_one_batch is enabled.")

    one_batch_size = args.per_device_train_batch_size * max(args.gradient_accumulation_steps, 1)
    if args.max_train_samples is None or args.max_train_samples > one_batch_size:
        args.max_train_samples = one_batch_size
    if args.max_eval_samples is None:
        args.max_eval_samples = one_batch_size

    args.max_steps = args.overfit_steps
    args.warmup_steps = 0
    args.logging_steps = 1
    args.save_steps = max(1, min(args.save_steps, args.overfit_steps))


def main() -> None:
    args = parse_args()
    _apply_overfit_one_batch_mode(args)
    torch_dtype = _parse_dtype(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch_dtype,
    )
    ref_model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch_dtype,
    )
    value_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch_dtype,
        num_labels=1,
    )
    if args.share_policy_and_value_backbone:
        _share_policy_backbone_with_value_model(model, value_model)

    dataset = prepare_triplet_dataset(args)

    training_kwargs = {
        "output_dir": args.output_dir,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_steps": args.warmup_steps,
        "logging_steps": args.logging_steps,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "report_to": args.report_to,
        "remove_unused_columns": False,
        "optim": "adafactor",
        "lr_scheduler_type": "linear",
        "tau": args.tau,
        "policy_learning_rate": args.policy_learning_rate,
        "value_learning_rate": args.value_learning_rate,
        "share_policy_and_value_backbone": args.share_policy_and_value_backbone,
        "learning_rate": args.policy_learning_rate,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "bf16": args.bf16,
        "fp16": args.fp16,
    }
    if args.overfit_one_batch:
        training_kwargs["eval_strategy"] = "no"
    else:
        training_kwargs["eval_strategy"] = "steps"
        training_kwargs["eval_steps"] = args.save_steps
    training_args = DROVConfig(**training_kwargs)

    eval_dataset = None if args.overfit_one_batch else dataset["test"]

    trainer = DROVTrainer(
        model=model,
        value_model=value_model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
