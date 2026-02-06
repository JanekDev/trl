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

import transformers
from packaging.version import Version
from transformers import TrainingArguments


@dataclass
class DROVConfig(TrainingArguments):
    r"""
    Configuration class for the [`experimental.drov.DROVTrainer`].

    This class includes only DRO-V specific parameters. For common training arguments, see
    [`~transformers.TrainingArguments`].
    """

    _VALID_DICT_FIELDS = TrainingArguments._VALID_DICT_FIELDS + ["model_init_kwargs"]

    # Parameters whose default values are overridden from TrainingArguments
    learning_rate: float = field(default=1e-4, metadata={"help": "The initial learning rate for AdamW."})
    logging_steps: float = field(
        default=10,
        metadata={
            "help": "Log every X updates steps. Should be an integer or a float in range `[0,1)`. If smaller than "
            "1, it is interpreted as ratio of total training steps."
        },
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Enable gradient checkpointing for the policy model."},
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "Must be false for DRO-V batches with custom keys."},
    )
    bf16: bool | None = field(
        default=None,
        metadata={
            "help": "Whether to use bf16 precision. If unset, defaults to `True` when `fp16` is disabled."
        },
    )
    # Transformers 4.57.0 introduced a bug that made this field unparsable; keep workaround while < 5.0.0 is used.
    lr_scheduler_kwargs: dict | str | None = field(
        default=None,
        metadata={"help": "Additional scheduler kwargs."},
    )

    # DRO-V specific parameters
    tau: float = field(default=1.0, metadata={"help": "DRO-V KL coefficient τ."})
    policy_learning_rate: float = field(
        default=1e-4,
        metadata={"help": "Policy learning rate before applying 1/τ scaling."},
    )
    value_learning_rate: float = field(
        default=1e-4,
        metadata={"help": "Learning rate for the value model."},
    )
    label_pad_token_id: int = field(
        default=-100,
        metadata={"help": "Padding label id for completion tokens."},
    )
    max_prompt_length: int | None = field(
        default=512,
        metadata={"help": "Maximum prompt length during tokenization."},
    )
    max_completion_length: int | None = field(
        default=512,
        metadata={"help": "Maximum completion length during tokenization."},
    )
    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Whether to disable dropout in policy, reference, and value models."},
    )
    dataset_num_proc: int | None = field(
        default=None,
        metadata={"help": "Number of processes for dataset preprocessing."},
    )
    model_init_kwargs: dict[str, Any] | None = field(
        default=None,
        metadata={
            "help": "Keyword arguments passed to `from_pretrained` when `model` is provided as a string."
        },
    )

    def __post_init__(self):
        self.bf16 = not self.fp16 if self.bf16 is None else self.bf16

        if self.gradient_checkpointing and Version(transformers.__version__) < Version("5.0.0"):
            self.gradient_checkpointing_kwargs = self.gradient_checkpointing_kwargs or {}
            self.gradient_checkpointing_kwargs.setdefault("use_reentrant", False)

        super().__post_init__()
