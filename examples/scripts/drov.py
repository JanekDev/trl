"""
Run a basic DRO-V experiment inspired by the paper:
"Offline Regularised Reinforcement Learning for Large Language Models Alignment".

Example:
python examples/scripts/drov.py \
    --output_dir flan-t5-large-drov \
    --dataset_revision <ultrafeedback_commit_or_tag> \
    --model_name_or_path google/flan-t5-large \
    --max_steps 40000 \
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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from torch import nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from trl.experimental.drov import DROVConfig, DROVTrainer


@dataclass
class ValueModelOutput:
    values: torch.Tensor


class EncoderDecoderValueModel(nn.Module):
    """Value model V(x) built from a seq2seq encoder backbone plus scalar head."""

    def __init__(self, backbone: AutoModelForSeq2SeqLM):
        super().__init__()
        self.backbone = backbone
        hidden_size = (
            getattr(backbone.config, "d_model", None)
            or getattr(backbone.config, "hidden_size", None)
            or getattr(backbone.config, "decoder_hidden_size", None)
        )
        if hidden_size is None:
            raise ValueError("Could not infer hidden size for value head from model config.")
        self.value_head = nn.Linear(hidden_size, 1)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        trust_remote_code: bool = False,
        torch_dtype: torch.dtype | None = None,
    ) -> "EncoderDecoderValueModel":
        backbone = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        return cls(backbone)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> ValueModelOutput:
        encoder = self.backbone.get_encoder()
        hidden_states = encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        ).last_hidden_state

        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        values = self.value_head(pooled).squeeze(-1)
        return ValueModelOutput(values=values)


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

    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=40000)
    parser.add_argument("--warmup_steps", type=int, default=150)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=2000)

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


def _save_prompt_split_manifest(dataset: DatasetDict, args: argparse.Namespace) -> None:
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
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


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

    reward_values = np.asarray(triplets["reward"], dtype=np.float32)
    reward_mean = float(reward_values.mean())
    reward_std = float(reward_values.std())
    if reward_std < 1e-6:
        reward_std = 1.0
    triplets = triplets.map(
        lambda row: {"reward": (float(row["reward"]) - reward_mean) / reward_std},
        num_proc=args.dataset_num_proc,
        desc="Normalizing rewards to mean 0 / std 1",
    )

    dataset = _split_triplets_by_prompt_folds(triplets, args)
    train_dataset: Dataset = dataset["train"]
    eval_dataset: Dataset = dataset["test"]

    if args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(args.max_train_samples, len(train_dataset))))
    if args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(min(args.max_eval_samples, len(eval_dataset))))

    split_dataset = DatasetDict({"train": train_dataset, "test": eval_dataset})
    if args.save_prompt_splits:
        _save_prompt_split_manifest(split_dataset, args)
    return split_dataset


def main() -> None:
    args = parse_args()
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
    value_model = EncoderDecoderValueModel.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch_dtype,
    )

    dataset = prepare_triplet_dataset(args)

    training_args = DROVConfig(
        output_dir=args.output_dir,
        seed=args.seed,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        report_to=args.report_to,
        remove_unused_columns=False,
        optim="adafactor",
        lr_scheduler_type="linear",
        tau=args.tau,
        policy_learning_rate=args.policy_learning_rate,
        value_learning_rate=args.value_learning_rate,
        learning_rate=args.policy_learning_rate,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        bf16=args.bf16,
        fp16=args.fp16,
    )

    trainer = DROVTrainer(
        model=model,
        value_model=value_model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
