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

from dataclasses import dataclass, field
from typing import Any

from transformers import TrainingArguments

from ...trainer.base_config import BaseConfig


@dataclass
class DROConfig(BaseConfig):
    r"""
    Configuration class for the [`experimental.dro.DROTrainer`].

    This class includes only the parameters specific to DRO-V training. For a full list of training arguments,
    please refer to the [`~transformers.TrainingArguments`] documentation. Note that default values in this class
    may differ from those in [`~transformers.TrainingArguments`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        tau (`float`, *optional*, defaults to `1.0`):
            KL temperature τ. Controls the strength of the KL regularisation. Higher τ means less policy
            regularisation toward the reference model. Used as 1/τ rescaling in the policy loss.
        max_length (`int` or `None`, *optional*, defaults to `1024`):
            Maximum number of tokens for the concatenated prompt + completion (policy input). Completions are
            truncated from the right to fit.
        max_prompt_length (`int` or `None`, *optional*, defaults to `512`):
            Maximum number of tokens for the prompt passed to the value model. Prompts are truncated from the
            left (keeping the end) to fit.
        normalize_rewards (`bool`, *optional*, defaults to `True`):
            If `True`, z-normalises the train-split rewards at initialisation time (mean 0, std 1). Eval rewards
            are kept on their original scale for interpretable metrics.
        precompute_ref_log_probs (`bool`, *optional*, defaults to `False`):
            Whether to cache reference log-probabilities before training. Reduces memory by avoiding keeping the
            reference model in GPU memory during training steps.
        disable_dropout (`bool`, *optional*, defaults to `True`):
            Whether to disable dropout in the policy, value model, and reference model.
        model_init_kwargs (`dict[str, Any]`, *optional*):
            Keyword arguments passed to `AutoModelForCausalLM.from_pretrained` when instantiating the policy
            from a string.
        value_model_init_kwargs (`dict[str, Any]`, *optional*):
            Keyword arguments passed to `AutoModelForSequenceClassification.from_pretrained` when instantiating
            the value model from a string.
        dataset_num_proc (`int` or `None`, *optional*):
            Number of worker processes for dataset `.map()` calls.

    > [!NOTE]
    > These parameters have default values different from [`~transformers.TrainingArguments`]:
    > - `logging_steps`: Defaults to `10` instead of `500`.
    > - `gradient_checkpointing`: Defaults to `True` instead of `False`.
    > - `bf16`: Defaults to `True` if `fp16` is not set, instead of `False`.
    > - `learning_rate`: Defaults to `1e-4` instead of `5e-5`.
    """

    _VALID_DICT_FIELDS = TrainingArguments._VALID_DICT_FIELDS + ["model_init_kwargs", "value_model_init_kwargs"]

    # Override from BaseConfig/TrainingArguments
    learning_rate: float = field(
        default=1e-4,
        metadata={"help": "The initial learning rate for AdamW. Paper uses 1e-4 for both policy and value model."},
    )

    tau: float = field(
        default=1.0,
        metadata={
            "help": "KL temperature τ. Scales 1/τ policy gradient rescaling. Higher τ = less KL regularisation."
        },
    )
    max_length: int | None = field(
        default=1024,
        metadata={"help": "Maximum number of tokens for the concatenated prompt + completion (policy input)."},
    )
    max_prompt_length: int | None = field(
        default=512,
        metadata={
            "help": "Maximum number of tokens for the prompt passed to the value model. Truncates from the left."
        },
    )
    normalize_rewards: bool = field(
        default=True,
        metadata={
            "help": "If True, z-normalises train-split rewards at init (mean 0, std 1). Eval rewards untouched."
        },
    )
    precompute_ref_log_probs: bool = field(
        default=False,
        metadata={
            "help": "Whether to cache reference model log probabilities before training to save GPU memory."
        },
    )
    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Whether to disable dropout in the policy, value model, and reference model."},
    )
    model_init_kwargs: dict[str, Any] | None = field(
        default=None,
        metadata={
            "help": "Keyword arguments passed to `AutoModelForCausalLM.from_pretrained` when the policy is a string."
        },
    )
    value_model_init_kwargs: dict[str, Any] | None = field(
        default=None,
        metadata={
            "help": "Keyword arguments passed to `AutoModelForSequenceClassification.from_pretrained` when the "
            "value model is a string."
        },
    )
    dataset_num_proc: int | None = field(
        default=None,
        metadata={"help": "Number of worker processes for dataset `.map()` calls."},
    )

    def __post_init__(self):
        self.bf16 = not (self.fp16) if self.bf16 is None else self.bf16
        super().__post_init__()
