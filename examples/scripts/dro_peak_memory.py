# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "trl",
#     "peft",
#     "datasets",
#     "transformers",
#     "accelerate",
# ]
# ///

import gc
import random
from dataclasses import dataclass, field

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser

from trl import ModelConfig, ScriptArguments, get_peft_config
from trl.experimental.dro import DROConfig, DROTrainer


@dataclass
class DROMemoryProfilerArguments(ScriptArguments):
    value_model_name_or_path: str | None = field(
        default=None,
        metadata={
            "help": "Checkpoint for the value model. Defaults to model_name_or_path when not set."
        },
    )
    max_samples: int | None = field(
        default=None,
        metadata={"help": "Optional cap on train rows after splitting."},
    )
    val_size: int = field(
        default=2000,
        metadata={"help": "Number of unique prompts reserved for validation."},
    )
    test_size: int = field(
        default=2000,
        metadata={"help": "Number of unique prompts reserved for test."},
    )
    split_seed: int = field(
        default=42,
        metadata={"help": "Seed for deterministic prompt-level splitting."},
    )
    profile_num_batches: int = field(
        default=8,
        metadata={"help": "Number of worst-case candidate batches to profile."},
    )


def _expand_completions(batch: dict) -> dict:
    prompts, completions, rewards = [], [], []
    for instruction, comps in zip(batch["instruction"], batch["completions"], strict=True):
        for comp in comps:
            score = comp.get("overall_score")
            response = (comp.get("response") or "").strip()
            if score is not None and response:
                prompts.append(instruction)
                completions.append(response)
                rewards.append(float(score))
    return {"prompt": prompts, "completion": completions, "reward": rewards}


def make_dro_dataset(raw: Dataset, max_samples: int | None = None, num_proc: int | None = None) -> Dataset:
    ds = raw.map(
        _expand_completions,
        batched=True,
        remove_columns=raw.column_names,
        num_proc=num_proc,
        desc="Expanding UltraFeedback completions -> (prompt, completion, reward)",
    )
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def split_dataset_by_prompt(ds: Dataset, val_size: int, test_size: int, seed: int) -> tuple[Dataset, Dataset, Dataset]:
    all_prompts = ds["prompt"]
    unique_prompts = list(dict.fromkeys(all_prompts))
    shuffled_prompts = unique_prompts[:]
    random.Random(seed).shuffle(shuffled_prompts)

    val_prompt_set = set(shuffled_prompts[:val_size])
    test_prompt_set = set(shuffled_prompts[val_size : val_size + test_size])

    train_indices = []
    val_indices = []
    test_indices = []
    for idx, prompt in enumerate(all_prompts):
        if prompt in val_prompt_set:
            val_indices.append(idx)
        elif prompt in test_prompt_set:
            test_indices.append(idx)
        else:
            train_indices.append(idx)

    return ds.select(train_indices), ds.select(val_indices), ds.select(test_indices)


def build_candidate_batches(train_dataset: Dataset, batch_size: int, num_batches: int) -> list[list[int]]:
    completion_lengths = [len(ids) for ids in train_dataset["completion_input_ids"]]
    prompt_lengths = [len(ids) for ids in train_dataset["prompt_input_ids"]]
    indices = list(range(len(train_dataset)))

    orderings = [
        sorted(indices, key=lambda i: completion_lengths[i], reverse=True),
        sorted(indices, key=lambda i: prompt_lengths[i], reverse=True),
        sorted(indices, key=lambda i: (completion_lengths[i], prompt_lengths[i]), reverse=True),
        sorted(indices, key=lambda i: (completion_lengths[i] + prompt_lengths[i]), reverse=True),
    ]

    seen = set()
    candidates = []
    max_chunks_per_order = max(num_batches * 2, 1)
    for ordering in orderings:
        for offset in range(0, min(len(ordering), max_chunks_per_order * batch_size), batch_size):
            batch_indices = ordering[offset : offset + batch_size]
            if len(batch_indices) < batch_size:
                continue
            key = tuple(sorted(batch_indices))
            if key in seen:
                continue
            seen.add(key)
            candidates.append(batch_indices)

    def batch_score(batch_indices: list[int]) -> tuple[int, int]:
        max_completion = max(completion_lengths[i] for i in batch_indices)
        max_prompt = max(prompt_lengths[i] for i in batch_indices)
        return (batch_size * max_completion + batch_size * max_prompt, max_completion)

    candidates.sort(key=batch_score, reverse=True)
    return candidates[:num_batches]


def gib(value: int) -> float:
    return value / (1024 ** 3)


if __name__ == "__main__":
    parser = HfArgumentParser((DROMemoryProfilerArguments, DROConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()

    if not torch.cuda.is_available():
        raise ValueError("This profiler requires CUDA.")

    training_args.push_to_hub = False
    training_args.report_to = []
    training_args.eval_strategy = "no"

    value_model_path = script_args.value_model_name_or_path or model_args.model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = model_args.dtype if model_args.dtype == "auto" else getattr(torch, model_args.dtype)

    policy = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=model_args.trust_remote_code,
        **({} if training_args.model_init_kwargs is None else training_args.model_init_kwargs),
    )
    value_model = AutoModelForSequenceClassification.from_pretrained(
        value_model_path,
        num_labels=1,
        torch_dtype=torch_dtype,
        trust_remote_code=model_args.trust_remote_code,
        **({} if training_args.value_model_init_kwargs is None else training_args.value_model_init_kwargs),
    )
    if value_model.config.pad_token_id is None:
        value_model.config.pad_token_id = tokenizer.pad_token_id

    raw_train = load_dataset(
        script_args.dataset_name or "openbmb/UltraFeedback",
        name=script_args.dataset_config,
        split=script_args.dataset_train_split,
    )
    full_dataset = make_dro_dataset(raw_train, num_proc=training_args.dataset_num_proc)
    train_dataset, _, _ = split_dataset_by_prompt(
        full_dataset,
        val_size=script_args.val_size,
        test_size=script_args.test_size,
        seed=script_args.split_seed,
    )
    if script_args.max_samples is not None:
        train_dataset = train_dataset.select(range(min(script_args.max_samples, len(train_dataset))))

    trainer = DROTrainer(
        model=policy,
        value_model=value_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )
    trainer.model.train()

    batch_size = training_args.per_device_train_batch_size
    candidate_batches = build_candidate_batches(trainer.train_dataset, batch_size, script_args.profile_num_batches)

    device = trainer.accelerator.device
    baseline_allocated = gib(torch.cuda.memory_allocated(device))
    baseline_reserved = gib(torch.cuda.memory_reserved(device))

    print(f"Profiling {len(candidate_batches)} candidate batches")
    print(f"Baseline allocated: {baseline_allocated:.2f} GiB")
    print(f"Baseline reserved:  {baseline_reserved:.2f} GiB")

    results = []
    for rank, batch_indices in enumerate(candidate_batches, start=1):
        trainer.model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        features = [trainer.train_dataset[i] for i in batch_indices]
        batch = trainer.data_collator(features)
        max_completion = max(len(feature["completion_input_ids"]) for feature in features)
        max_prompt = max(len(feature["prompt_input_ids"]) for feature in features)

        try:
            loss, _ = trainer.get_batch_loss_metrics(trainer.model, batch)
            trainer.accelerator.backward(loss)
            torch.cuda.synchronize(device)
            peak_allocated = gib(torch.cuda.max_memory_allocated(device))
            peak_reserved = gib(torch.cuda.max_memory_reserved(device))
            results.append((peak_allocated, peak_reserved, max_completion, max_prompt, batch_indices))
            print(
                f"[{rank}] completion_max={max_completion} prompt_max={max_prompt} "
                f"peak_allocated={peak_allocated:.2f} GiB peak_reserved={peak_reserved:.2f} GiB"
            )
        except torch.OutOfMemoryError:
            current_reserved = gib(torch.cuda.memory_reserved(device))
            print(
                f"[{rank}] completion_max={max_completion} prompt_max={max_prompt} "
                f"OOM current_reserved={current_reserved:.2f} GiB"
            )
            raise
        finally:
            trainer.model.zero_grad(set_to_none=True)

    worst = max(results, key=lambda item: item[0])
    print(
        "Worst profiled batch: "
        f"completion_max={worst[2]} prompt_max={worst[3]} "
        f"peak_allocated={worst[0]:.2f} GiB peak_reserved={worst[1]:.2f} GiB"
    )
