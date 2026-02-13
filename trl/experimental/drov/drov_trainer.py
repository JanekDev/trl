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

import inspect
import json
import textwrap
from collections import defaultdict
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
import torch.nn as nn
from accelerate import PartialState, logging
from datasets import Dataset, IterableDataset
from transformers import (
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
)
from transformers.data.data_collator import DataCollatorMixin
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils import is_peft_available

from ...data_utils import is_conversational, maybe_apply_chat_template, maybe_extract_prompt
from ...models.utils import create_reference_model, prepare_deepspeed
from ...trainer.base_trainer import BaseTrainer
from ...trainer.utils import create_model_from_path, disable_dropout_in_model, pad
from .drov_config import DROVConfig


if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training

if TYPE_CHECKING:
    from transformers import EvalPrediction


logger = logging.get_logger(__name__)


def _sequence_log_probs(logits: torch.Tensor, labels: torch.Tensor, label_pad_token_id: int) -> torch.Tensor:
    """Compute log-probability per sequence from token logits and labels."""
    log_probs = torch.log_softmax(logits, dim=-1)
    safe_labels = labels.masked_fill(labels == label_pad_token_id, 0)
    token_logp = torch.gather(log_probs, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    mask = (labels != label_pad_token_id).to(token_logp.dtype)
    return (token_logp * mask).sum(dim=-1)


def compute_drov_residual_loss(
    rewards: torch.Tensor,
    values: torch.Tensor,
    policy_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    tau: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute DRO-V residual loss and components."""
    log_ratio = policy_logps - ref_logps
    delta = rewards - values - tau * log_ratio
    loss = 0.5 * torch.mean(delta.pow(2))
    return loss, delta, log_ratio


@dataclass
class DataCollatorForDROV(DataCollatorMixin):
    """
    Data collator used for DRO-V triplets.

    Expected fields in each example:
      - `prompt_input_ids`
      - `completion_input_ids` (or `response_input_ids`)
      - `reward` (or `reward_z`)
    """

    pad_token_id: int
    return_tensors: str = "pt"

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        prompt_input_ids = [torch.tensor(example["prompt_input_ids"]) for example in examples]
        prompt_attention_mask = [
            torch.tensor(example["prompt_attention_mask"])
            if "prompt_attention_mask" in example
            else torch.ones_like(input_ids)
            for example, input_ids in zip(examples, prompt_input_ids, strict=True)
        ]

        completion_key = "completion_input_ids" if "completion_input_ids" in examples[0] else "response_input_ids"
        attention_key = (
            "completion_attention_mask" if "completion_attention_mask" in examples[0] else "response_attention_mask"
        )
        completion_input_ids = [torch.tensor(example[completion_key]) for example in examples]
        completion_attention_mask = [
            torch.tensor(example[attention_key]) if attention_key in example else torch.ones_like(input_ids)
            for example, input_ids in zip(examples, completion_input_ids, strict=True)
        ]

        rewards = torch.tensor(
            [
                float(example["reward"])
                if "reward" in example
                else float(example["reward_z"])
                if "reward_z" in example
                else float(example["label"])
                for example in examples
            ],
            dtype=torch.float32,
        )

        output = {
            "prompt_input_ids": pad(prompt_input_ids, padding_value=self.pad_token_id, padding_side="left"),
            "prompt_attention_mask": pad(prompt_attention_mask, padding_value=0, padding_side="left"),
            "completion_input_ids": pad(completion_input_ids, padding_value=self.pad_token_id),
            "completion_attention_mask": pad(completion_attention_mask, padding_value=0),
            "reward": rewards,
        }

        return output


class PolicyAndValueWrapper(nn.Module):
    """Container used so Trainer wraps policy and value modules together."""

    def __init__(self, policy: nn.Module, value_model: nn.Module) -> None:
        super().__init__()
        self.policy = policy
        self.value_model = value_model

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.policy(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.policy, name)


class DROVTrainer(BaseTrainer):
    """
    Trainer for Distributional Regularized Offline Value Learning (DRO-V).

    Objective:
        0.5 * E[(r(x,y) - V(x) - tau * log(pi(y|x)/pi_ref(y|x)))^2]
    """

    _tag_names = ["trl", "drov"]
    _name = "DRO-V"
    _paper = {
        "title": "Offline Regularised Reinforcement Learning for Large Language Models Alignment",
        "id": "2405.19107",
        # docstyle-ignore
        "citation": textwrap.dedent("""\
            @article{richemond2024offline,
                title        = {{Offline Regularised Reinforcement Learning for Large Language Models Alignment}},
                author       = {Pierre Harvey Richemond and Yunhao Tang and Daniel Guo and Daniele Calandriello and Mohammad Gheshlaghi Azar and Rafael Rafailov and Bernardo Avila Pires and Eugene Tarassov and Lucas Spangher and Will Ellsworth and Aliaksei Severyn and Jonathan Mallinson and Lior Shani and Gil Shamir and Rishabh Joshi and Tianqi Liu and Remi Munos and Bilal Piot},
                year         = 2024,
                eprint       = {arXiv:2405.19107},
            }"""),
    }

    def __init__(
        self,
        model: str | nn.Module | PreTrainedModel,
        value_model: nn.Module,
        ref_model: PreTrainedModel | nn.Module | str | None = None,
        args: DROVConfig | None = None,
        data_collator: DataCollator | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None = None,
        processing_class: PreTrainedTokenizerBase
        | BaseImageProcessor
        | FeatureExtractionMixin
        | ProcessorMixin
        | None = None,
        compute_metrics: Callable[["EvalPrediction"], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        peft_config: Any | None = None,
    ) -> None:
        if args is None:
            args = DROVConfig(output_dir="tmp_drov_trainer")

        if isinstance(model, str):
            model_init_kwargs = args.model_init_kwargs or {}
            model = create_model_from_path(model, **model_init_kwargs)

        if isinstance(ref_model, str):
            ref_model = create_model_from_path(ref_model)

        if ref_model is model:
            raise ValueError(
                "`model` and `ref_model` cannot be the same object. Pass `ref_model=None` to auto-create a reference."
            )

        self.is_peft_model = is_peft_available() and isinstance(model, PeftModel)
        self.ref_model = ref_model
        if self.ref_model is None and not self.is_peft_model:
            self.ref_model = create_reference_model(model)

        if peft_config is not None:
            if not is_peft_available():
                raise ValueError("PEFT is not installed but `peft_config` was provided.")
            if isinstance(model, PeftModel):
                raise ValueError("Received a PeftModel and `peft_config`. Pass either one, not both.")
            if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                supports_gc_kwargs = "gradient_checkpointing_kwargs" in inspect.signature(
                    prepare_model_for_kbit_training
                ).parameters
                prepare_model_kwargs = {"use_gradient_checkpointing": args.gradient_checkpointing}
                if supports_gc_kwargs:
                    prepare_model_kwargs["gradient_checkpointing_kwargs"] = args.gradient_checkpointing_kwargs
                model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)
            model = get_peft_model(model, peft_config)
            self.is_peft_model = True

        self.policy_model = model
        self.value_model = value_model
        if args.share_policy_and_value_backbone:
            self._share_policy_and_value_backbone()
        self.model = PolicyAndValueWrapper(self.policy_model, self.value_model)
        self.is_encoder_decoder = getattr(self.policy_model.config, "is_encoder_decoder", False)
        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        self._train_metrics_path = Path(args.output_dir) / "train_metrics.jsonl"

        if args.disable_dropout:
            disable_dropout_in_model(self.policy_model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)
            disable_dropout_in_model(self.value_model)

        if processing_class is None:
            raise ValueError("`processing_class` must be provided.")
        if data_collator is None:
            pad_token_id = processing_class.pad_token_id
            if pad_token_id is None:
                if processing_class.eos_token_id is not None:
                    pad_token_id = processing_class.eos_token_id
                    processing_class.pad_token = processing_class.eos_token
                else:
                    raise ValueError("`processing_class` must define `pad_token_id` (or `eos_token_id`).")
            data_collator = DataCollatorForDROV(pad_token_id=pad_token_id)

        if train_dataset is not None:
            train_dataset = self._prepare_dataset(train_dataset, processing_class, args, "train")
        if eval_dataset is not None:
            if isinstance(eval_dataset, dict):
                eval_dataset = {
                    key: self._prepare_dataset(dataset, processing_class, args, f"eval[{key}]")
                    for key, dataset in eval_dataset.items()
                }
            else:
                eval_dataset = self._prepare_dataset(eval_dataset, processing_class, args, "eval")

        super().__init__(
            model=self.model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # We compute custom loss, so we don't rely on model `loss` kwargs handling in parent trainer.
        self.model_accepts_loss_kwargs = False

        self.model.config = self.policy_model.config

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
            self.ref_model.eval()

        self.value_model.train()

        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

    def _share_policy_and_value_backbone(self) -> None:
        if not hasattr(self.policy_model, "base_model_prefix"):
            raise ValueError("Policy model must define `base_model_prefix` when sharing the value backbone.")
        if not hasattr(self.value_model, "base_model_prefix"):
            raise ValueError("Value model must define `base_model_prefix` when sharing the value backbone.")

        policy_backbone_name = self.policy_model.base_model_prefix
        value_backbone_name = self.value_model.base_model_prefix
        policy_backbone = getattr(self.policy_model, policy_backbone_name, None)
        value_backbone = getattr(self.value_model, value_backbone_name, None)
        if value_backbone is None:
            raise ValueError(f"Value model does not expose backbone attribute `{value_backbone_name}`.")

        if policy_backbone is not None:
            setattr(self.value_model, value_backbone_name, policy_backbone)
        elif all(hasattr(self.policy_model, attr) for attr in ("shared", "encoder", "decoder")) and all(
            hasattr(value_backbone, attr) for attr in ("shared", "encoder", "decoder")
        ):
            # T5-family fallback: share encoder/decoder stack when the policy does not expose `base_model_prefix`.
            value_backbone.shared = self.policy_model.shared
            value_backbone.encoder = self.policy_model.encoder
            value_backbone.decoder = self.policy_model.decoder
        else:
            raise ValueError(
                "Policy model does not expose a shareable backbone module for value sharing. "
                f"Tried `{policy_backbone_name}` and T5-style `(shared, encoder, decoder)` fallback."
            )

        logger.info(
            "Sharing policy backbone `%s` with value model backbone `%s`.",
            policy_backbone_name,
            value_backbone_name,
        )

    def _prepare_dataset(
        self,
        dataset: Dataset | IterableDataset,
        processing_class: PreTrainedTokenizerBase | BaseImageProcessor | FeatureExtractionMixin | ProcessorMixin,
        args: DROVConfig,
        dataset_name: str,
    ) -> Dataset | IterableDataset:
        required_columns = {"prompt_input_ids", "completion_input_ids"}
        if required_columns.issubset(set(dataset.column_names)) and (
            "reward" in dataset.column_names or "reward_z" in dataset.column_names or "label" in dataset.column_names
        ):
            return dataset

        if not isinstance(processing_class, PreTrainedTokenizerBase):
            raise ValueError("DROVTrainer requires a tokenizer-like `processing_class` for text preprocessing.")

        map_kwargs = {}
        if isinstance(dataset, Dataset):
            map_kwargs["num_proc"] = args.dataset_num_proc
            map_kwargs["writer_batch_size"] = 10

        with PartialState().main_process_first():
            if isinstance(dataset, Dataset):
                map_kwargs["desc"] = f"Extracting prompt in {dataset_name} dataset"
            dataset = dataset.map(maybe_extract_prompt, **map_kwargs)

            is_chat = is_conversational(next(iter(dataset)))
            if isinstance(dataset, Dataset):
                map_kwargs["desc"] = f"Applying chat template to {dataset_name} dataset"
            dataset = dataset.map(maybe_apply_chat_template, fn_kwargs={"tokenizer": processing_class}, **map_kwargs)

            if isinstance(dataset, Dataset):
                map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"
            dataset = dataset.map(
                self.tokenize_row,
                fn_kwargs={
                    "processing_class": processing_class,
                    "max_prompt_length": args.max_prompt_length,
                    "max_completion_length": args.max_completion_length,
                    "add_special_tokens": self.is_encoder_decoder,
                    "is_chat": is_chat,
                },
                **map_kwargs,
            )

        return dataset

    @staticmethod
    def tokenize_row(
        features: dict[str, Any],
        processing_class: PreTrainedTokenizerBase,
        max_prompt_length: int | None = None,
        max_completion_length: int | None = None,
        add_special_tokens: bool = True,
        is_chat: bool = False,
    ) -> dict[str, Any]:
        tokenizer = processing_class
        if "prompt" not in features:
            raise KeyError("DRO-V dataset row must contain `prompt`.")
        if "completion" not in features:
            raise KeyError("DRO-V dataset row must contain `completion`.")

        prompt_input_ids = tokenizer(features["prompt"], add_special_tokens=False)["input_ids"]
        completion_input_ids = tokenizer(features["completion"], add_special_tokens=False)["input_ids"]

        if add_special_tokens:
            if tokenizer.bos_token_id is not None:
                prompt_input_ids = [tokenizer.bos_token_id] + prompt_input_ids
            if tokenizer.eos_token_id is not None:
                prompt_input_ids = prompt_input_ids + [tokenizer.eos_token_id]

        if not is_chat and tokenizer.eos_token_id is not None:
            if len(completion_input_ids) == 0 or completion_input_ids[-1] != tokenizer.eos_token_id:
                completion_input_ids = completion_input_ids + [tokenizer.eos_token_id]

        if max_prompt_length is not None:
            prompt_input_ids = prompt_input_ids[-max_prompt_length:]
        if max_completion_length is not None:
            completion_input_ids = completion_input_ids[:max_completion_length]

        reward = features.get("reward", features.get("reward_z", features.get("label")))
        if reward is None:
            raise KeyError("DRO-V dataset row must contain one of: `reward`, `reward_z`, or `label`.")

        return {
            "prompt_input_ids": prompt_input_ids,
            "completion_input_ids": completion_input_ids,
            "reward": float(reward),
        }

    @contextmanager
    def null_ref_context(self):
        policy_model = self.accelerator.unwrap_model(self.model).policy
        with policy_model.disable_adapter() if self.is_peft_model else nullcontext():
            yield

    def create_optimizer(self) -> torch.optim.Optimizer:
        if self.optimizer is not None:
            return self.optimizer

        policy_params = [p for p in self.policy_model.parameters() if p.requires_grad]
        value_params = [p for p in self.value_model.parameters() if p.requires_grad]
        if not policy_params:
            raise ValueError("No trainable policy parameters found.")
        if not value_params:
            raise ValueError("No trainable value parameters found.")

        policy_param_ids = {id(param) for param in policy_params}
        shared_param_count = sum(id(param) in policy_param_ids for param in value_params)
        value_only_params = [param for param in value_params if id(param) not in policy_param_ids]
        if shared_param_count > 0:
            logger.warning(
                "Detected %d shared policy/value parameters; they will use policy learning rate.",
                shared_param_count,
            )

        optimizer_param_groups = [
            {
                "params": policy_params,
                "lr": self.args.policy_learning_rate / max(self.args.tau, 1e-12),
                "weight_decay": self.args.weight_decay,
            }
        ]
        if value_only_params:
            optimizer_param_groups.append(
                {
                    "params": value_only_params,
                    "lr": self.args.value_learning_rate,
                    "weight_decay": self.args.weight_decay,
                }
            )

        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, self.model)
        self.optimizer = optimizer_cls(optimizer_param_groups, **optimizer_kwargs)
        return self.optimizer

    def _extract_value_predictions(
        self,
        value_output: Any,
        prompt_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(value_output, torch.Tensor):
            values = value_output
        elif (
            isinstance(value_output, (tuple, list))
            and len(value_output) > 0
            and isinstance(value_output[0], torch.Tensor)
        ):
            values = value_output[0]
        elif hasattr(value_output, "logits") and isinstance(value_output.logits, torch.Tensor):
            values = value_output.logits
        elif hasattr(value_output, "value") and isinstance(value_output.value, torch.Tensor):
            values = value_output.value
        elif hasattr(value_output, "values") and isinstance(value_output.values, torch.Tensor):
            values = value_output.values
        else:
            raise ValueError(
                "Unsupported value model output. Expected tensor, or output with `values`/`value`/`logits`."
            )

        if values.ndim == 2 and values.shape[-1] == 1:
            values = values.squeeze(-1)
        elif values.ndim == 3 and values.shape[-1] == 1:
            values = values.squeeze(-1)
            mask = prompt_attention_mask.to(values.dtype)
            values = (values * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        elif values.ndim != 1:
            raise ValueError(f"Unsupported value tensor shape {tuple(values.shape)}. Expected [B] or [B,1].")

        return values.to(torch.float32)

    def _forward_value_model(
        self,
        value_model: nn.Module,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
    ) -> Any:
        try:
            return value_model(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
            )
        except TypeError as error:
            if not hasattr(value_model, "score"):
                raise
            if not hasattr(value_model, "base_model_prefix"):
                raise ValueError("Value model fallback requires both `score` and `base_model_prefix`.") from error
            critic_backbone = getattr(value_model, value_model.base_model_prefix)
            backbone_outputs = critic_backbone(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            hidden_states = backbone_outputs.hidden_states[-1]
            return value_model.score(hidden_states)

    def _forward_ref_encoder_decoder(
        self,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if self.ref_model is not None:
            with torch.no_grad():
                return self.ref_model(
                    input_ids=prompt_input_ids,
                    attention_mask=prompt_attention_mask,
                    labels=labels,
                    use_cache=False,
                ).logits
        with torch.no_grad(), self.null_ref_context():
            return self.model(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                labels=labels,
                use_cache=False,
            ).logits

    def _decoder_only_logps(
        self,
        model: PreTrainedModel | nn.Module,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        completion_input_ids: torch.Tensor,
        completion_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        input_ids = torch.cat([prompt_input_ids, completion_input_ids], dim=1)
        attention_mask = torch.cat([prompt_attention_mask, completion_attention_mask], dim=1)
        target_mask = torch.cat([torch.zeros_like(prompt_attention_mask), completion_attention_mask], dim=1)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        target_mask = target_mask[:, 1:].to(torch.float32)

        log_probs = torch.log_softmax(logits, dim=-1)
        token_logps = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        return (token_logps * target_mask).sum(dim=-1)

    def _reference_decoder_only_logps(
        self,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        completion_input_ids: torch.Tensor,
        completion_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.ref_model is not None:
            with torch.no_grad():
                return self._decoder_only_logps(
                    self.ref_model,
                    prompt_input_ids,
                    prompt_attention_mask,
                    completion_input_ids,
                    completion_attention_mask,
                )
        with torch.no_grad(), self.null_ref_context():
            return self._decoder_only_logps(
                self.model,
                prompt_input_ids,
                prompt_attention_mask,
                completion_input_ids,
                completion_attention_mask,
            )

    def get_batch_loss_metrics(
        self,
        model: PreTrainedModel | nn.Module,
        batch: dict[str, torch.Tensor | Any],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        batch = {
            key: (value.to(self.accelerator.device) if isinstance(value, torch.Tensor) else value)
            for key, value in batch.items()
        }
        prompt_input_ids = batch["prompt_input_ids"]
        prompt_attention_mask = batch["prompt_attention_mask"]
        completion_input_ids = batch["completion_input_ids"]
        completion_attention_mask = batch["completion_attention_mask"]
        rewards = batch["reward"].to(torch.float32)
        policy_model = model.policy if hasattr(model, "policy") else model
        value_model = model.value_model if hasattr(model, "value_model") else self.value_model

        if self.is_encoder_decoder:
            labels = completion_input_ids.masked_fill(
                completion_attention_mask == 0,
                self.args.label_pad_token_id,
            )
            policy_outputs = policy_model(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                labels=labels,
                use_cache=False,
            )
            policy_logps = _sequence_log_probs(policy_outputs.logits, labels, self.args.label_pad_token_id)
            ref_logits = self._forward_ref_encoder_decoder(prompt_input_ids, prompt_attention_mask, labels)
            ref_logps = _sequence_log_probs(ref_logits, labels, self.args.label_pad_token_id)
        else:
            policy_logps = self._decoder_only_logps(
                policy_model,
                prompt_input_ids,
                prompt_attention_mask,
                completion_input_ids,
                completion_attention_mask,
            )
            ref_logps = self._reference_decoder_only_logps(
                prompt_input_ids,
                prompt_attention_mask,
                completion_input_ids,
                completion_attention_mask,
            )

        value_outputs = self._forward_value_model(
            value_model,
            prompt_input_ids,
            prompt_attention_mask,
        )
        values = self._extract_value_predictions(value_outputs, prompt_attention_mask)

        loss, delta, log_ratio = compute_drov_residual_loss(
            rewards=rewards,
            values=values,
            policy_logps=policy_logps,
            ref_logps=ref_logps,
            tau=self.args.tau,
        )

        metrics = {
            "loss/drov": loss.detach().item(),
            "delta/mean": delta.detach().mean().item(),
            "delta/std": delta.detach().std(unbiased=False).item(),
            "value/mean": values.detach().mean().item(),
            "reward/mean": rewards.detach().mean().item(),
            "log_ratio/mean": log_ratio.detach().mean().item(),
            "log_ratio/std": log_ratio.detach().std(unbiased=False).item(),
        }
        return loss, metrics

    def compute_loss(
        self,
        model: PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        del num_items_in_batch
        loss, metrics = self.get_batch_loss_metrics(model, inputs)
        loss = loss.to(self.args.device)
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="train")
        if return_outputs:
            return loss, metrics
        return loss

    def prediction_step(
        self,
        model: PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor, None, None]:
        del ignore_keys
        with torch.no_grad():
            loss, metrics = self.get_batch_loss_metrics(model, inputs)
        self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return loss.detach(), None, None
        return loss.detach(), None, None

    def store_metrics(self, metrics: dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        train_eval = "train" if "loss" in logs else "eval"
        for key, metrics in self._stored_metrics[train_eval].items():
            if metrics:
                logs[key] = torch.tensor(metrics).mean().item()
        self._stored_metrics[train_eval].clear()

        if self.accelerator.is_main_process and train_eval == "train":
            self._train_metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with self._train_metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"step": self.state.global_step, **logs}) + "\n")

        return super().log(logs, start_time)

    def _save_value_model(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_value_model = (
            unwrapped_model.value_model if hasattr(unwrapped_model, "value_model") else self.value_model
        )
        if hasattr(unwrapped_value_model, "save_pretrained"):
            unwrapped_value_model.save_pretrained(output_dir)
        else:
            torch.save(unwrapped_value_model.state_dict(), output_dir / "pytorch_model.bin")

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False):
        backup_model = self.model
        policy_model = self.model.policy if hasattr(self.model, "policy") else self.model
        self.model = policy_model

        if self.is_deepspeed_enabled:
            backup_deepspeed = self.deepspeed
            self.deepspeed = self.model

        try:
            super().save_model(output_dir, _internal_call)
        finally:
            self.model = backup_model
            if self.is_deepspeed_enabled:
                self.deepspeed = backup_deepspeed

        save_dir = Path(output_dir) if output_dir is not None else Path(self.args.output_dir)
        self._save_value_model(save_dir / "value_model")

    def _save_checkpoint(self, model, trial) -> None:
        del model
        backup_model = self.model
        policy_model = self.model.policy if hasattr(self.model, "policy") else self.model
        self.model = policy_model
        try:
            super()._save_checkpoint(self.model, trial)
        finally:
            self.model = backup_model

        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        output_dir = Path(self.args.output_dir) / checkpoint_folder
        self._save_value_model(output_dir / "value_model")
