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

"""
DRO-V (offline RL with a joint policy + value loss) training on UltraFeedback.

The same model checkpoint is loaded twice:
  • Policy  → AutoModelForCausalLM
  • Value   → AutoModelForSequenceClassification  (num_labels=1)

This script defaults to ``Qwen/Qwen2-0.5B-Instruct`` which is a small,
instruction-tuned decoder-only model — a practical substitute for Flan-style
alignment.  Swap ``--model_name_or_path`` for any causal LM you prefer
(e.g. ``google/flan-ul2``, ``meta-llama/Llama-3.1-8B-Instruct``, …).

Note: ``google/flan-t5-*`` models use an encoder-decoder architecture and are
NOT compatible with ``AutoModelForCausalLM``.  Use a decoder-only checkpoint
such as the default above.

Dataset
-------
``openbmb/UltraFeedback`` contains (instruction, completions[]) rows where
each completion carries an ``overall_score`` in [1, 10].  This script flattens
the 4 completions per instruction into individual (prompt, completion, reward)
triples.  Rewards are z-normalised at initialisation time (``normalize_rewards=True``).

Usage
-----
# Quick test (downloads a small portion of UltraFeedback):
python examples/scripts/dro.py \\
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \\
    --output_dir /tmp/dro-ultrafeedback \\
    --max_steps 100 \\
    --per_device_train_batch_size 2 \\
    --gradient_accumulation_steps 8 \\
    --max_samples 2000 \\
    --report_to none

# Full single-GPU training:
python examples/scripts/dro.py \\
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \\
    --output_dir dro-qwen-ultrafeedback \\
    --num_train_epochs 1 \\
    --per_device_train_batch_size 4 \\
    --gradient_accumulation_steps 8 \\
    --learning_rate 1e-4 \\
    --tau 1.0 \\
    --warmup_ratio 0.05 \\
    --lr_scheduler_type cosine \\
    --eval_strategy steps \\
    --eval_steps 500 \\
    --logging_steps 10 \\
    --report_to wandb

# Multi-GPU with QLoRA on the policy:
accelerate launch --num_processes 4 examples/scripts/dro.py \\
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \\
    --output_dir dro-qwen-ultrafeedback-qlora \\
    --num_train_epochs 1 \\
    --per_device_train_batch_size 4 \\
    --gradient_accumulation_steps 4 \\
    --use_peft \\
    --load_in_4bit \\
    --lora_r 16 \\
    --lora_alpha 32 \\
    --lora_target_modules all-linear \\
    --report_to wandb
"""

import os
import signal
import sys
from dataclasses import dataclass, field

from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser

from trl import ModelConfig, ScriptArguments, get_peft_config
from trl.experimental.dro import DROConfig, DROTrainer
from trl.experimental.judges import OpenAIPairwiseJudge
from trl.experimental.winrate_callback import WinRateCallback


os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


# ── Extra script arguments ────────────────────────────────────────────────────


@dataclass
class DROScriptArguments(ScriptArguments):
    """
    Extra arguments for the DRO training script.

    Args:
        value_model_name_or_path (`str`, *optional*):
            Checkpoint to load as the value model.  Defaults to ``model_name_or_path``
            (i.e. the same weights used for the policy).
        max_samples (`int`, *optional*):
            If set, truncate each split to this many examples after flattening.
            Useful for quick iteration / smoke-tests.
    """

    value_model_name_or_path: str | None = field(
        default=None,
        metadata={
            "help": "Checkpoint for the value model (SequenceClassification). "
            "Defaults to model_name_or_path when not set."
        },
    )
    max_samples: int | None = field(
        default=None,
        metadata={"help": "Truncate each split to this many examples after flattening (useful for smoke-tests)."},
    )
    win_rate_num_prompts: int = field(
        default=256,
        metadata={"help": "Number of prompts from the eval split used for win-rate evaluation."},
    )


# ── Dataset helpers ───────────────────────────────────────────────────────────


def _expand_completions(batch: dict) -> dict:
    """
    Expand one UltraFeedback batch into flat (prompt, completion, reward) triples.

    Each row in ``openbmb/UltraFeedback`` has a single ``instruction`` and a list
    of up to 4 ``completions``, each carrying ``response`` (str) and
    ``overall_score`` (float, 1–10).  We emit one triple per valid completion.
    """
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
    """Return a Dataset with columns ``prompt``, ``completion``, ``reward``."""
    ds = raw.map(
        _expand_completions,
        batched=True,
        remove_columns=raw.column_names,
        num_proc=num_proc,
        desc="Expanding UltraFeedback completions → (prompt, completion, reward)",
    )
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


# ── Main ──────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = HfArgumentParser((DROScriptArguments, DROConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()

    # Resolve the value model checkpoint (defaults to same as policy)
    value_model_path = script_args.value_model_name_or_path or model_args.model_name_or_path

    # ── Load tokenizer ────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load policy (CausalLM) ────────────────────────────────────────────────
    policy = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        **({} if training_args.model_init_kwargs is None else training_args.model_init_kwargs),
    )

    # ── Load value model (SequenceClassification, num_labels=1) ──────────────
    value_model = AutoModelForSequenceClassification.from_pretrained(
        value_model_path,
        num_labels=1,
        trust_remote_code=model_args.trust_remote_code,
        **({} if training_args.value_model_init_kwargs is None else training_args.value_model_init_kwargs),
    )
    # Ensure value model can handle padded batches
    if value_model.config.pad_token_id is None:
        value_model.config.pad_token_id = tokenizer.pad_token_id

    # ── Load and preprocess dataset ───────────────────────────────────────────
    raw_train = load_dataset(
        script_args.dataset_name or "openbmb/UltraFeedback",
        name=script_args.dataset_config,
        split=script_args.dataset_train_split,
    )
    full_dataset = make_dro_dataset(raw_train, num_proc=training_args.dataset_num_proc)

    eval_dataset = None
    if training_args.eval_strategy != "no":
        # Carve a fixed-size eval split before capping training samples so
        # that win-rate evaluation has enough prompts regardless of max_samples.
        split = full_dataset.train_test_split(test_size=512, seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]
    else:
        train_dataset = full_dataset

    # Apply training-sample cap after the eval split so the eval pool stays large.
    if script_args.max_samples is not None:
        train_dataset = train_dataset.select(range(min(script_args.max_samples, len(train_dataset))))

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = DROTrainer(
        model=policy,
        value_model=value_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    # ── Win-rate evaluation via Gemma-3-27B judge (OpenRouter) ────────────────
    # Compares the trained policy against the reference/starting model every
    # eval step.  Position bias is cancelled by judging each pair twice (both
    # orderings) via double_judge=True; a win is counted only when both agree.
    if eval_dataset is not None:
        judge = OpenAIPairwiseJudge(
            model="google/gemma-3-27b-it",
            base_url="https://openrouter.ai/api/v1",
            double_judge=True,
        )
        win_rate_callback = WinRateCallback(judge=judge, trainer=trainer, num_prompts=script_args.win_rate_num_prompts)
        trainer.add_callback(win_rate_callback)

    def _save_and_push(signum, frame):
        print(f"\nCaught signal {signum}, saving and pushing model before exit...")
        trainer.save_model(training_args.output_dir)
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        sys.exit(0)

    signal.signal(signal.SIGINT, _save_and_push)
    signal.signal(signal.SIGTERM, _save_and_push)

    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────────
    trainer.save_model(training_args.output_dir)
    trainer.push_to_hub(dataset_name=script_args.dataset_name)
