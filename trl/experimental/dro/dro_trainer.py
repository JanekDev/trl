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

import copy
import os
import textwrap
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import transformers
from accelerate import PartialState, logging
from accelerate.utils import tqdm
from datasets import Dataset
from packaging.version import Version
from torch import autocast
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from ...data_utils import maybe_apply_chat_template
from ...models.utils import prepare_deepspeed
from ...trainer.base_trainer import BaseTrainer
from ...trainer.utils import (
    create_model_from_path,
    disable_dropout_in_model,
    pad,
    selective_log_softmax,
)
from .dro_config import DROConfig


if is_peft_available():
    import peft


logger = logging.get_logger(__name__)


# ── Module-level helpers (picklable for multiprocessing .map()) ───────────────


def _create_reference_model(model: nn.Module) -> nn.Module:
    """Frozen deep copy of model for use as the reference policy."""
    if is_deepspeed_zero3_enabled():
        raise ValueError(
            "DeepSpeed ZeRO-3 is incompatible with `_create_reference_model()`. "
            "Instantiate your ref_model directly with `AutoModelForCausalLM.from_pretrained()`."
        )
    ref = copy.deepcopy(model)
    for param in ref.parameters():
        param.requires_grad = False
    return ref.eval()


def _tokenize(
    batch: dict[str, list[Any]],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, list[Any]]:
    """Tokenize a batch for DRO. Handles the subword merge artifact at the prompt/completion seam."""
    prompt_tokenized = tokenizer(batch["prompt"], add_special_tokens=False)
    prompt_input_ids = prompt_tokenized["input_ids"]
    prompt_attention_mask = prompt_tokenized["attention_mask"]

    prompt_and_completion = [
        prompt + completion for prompt, completion in zip(batch["prompt"], batch["completion"], strict=True)
    ]
    full_tokenized = tokenizer(prompt_and_completion, add_special_tokens=False)
    full_input_ids = full_tokenized["input_ids"]
    full_attention_mask = full_tokenized["attention_mask"]

    # Determine where the answer starts in the full sequence
    response_token_ids_start_idx = [len(p) for p in prompt_input_ids]

    # Handle subword merge artifact at the seam (same logic as KTO trainer lines 128–135):
    # On some tokenizers (e.g. Llama-2), the last prompt token may change when tokenized
    # jointly with the completion due to byte-pair merging.
    full_input_ids_np = [np.array(f) for f in full_input_ids]
    for idx, (p, f, r) in enumerate(
        zip(prompt_input_ids, full_input_ids_np, response_token_ids_start_idx, strict=True)
    ):
        if not np.array_equal(p, f[:r]):
            response_token_ids_start_idx[idx] -= 1

    prompt_input_ids = [f[:r].tolist() for f, r in zip(full_input_ids_np, response_token_ids_start_idx, strict=True)]
    prompt_attention_mask = [f[:r] for f, r in zip(full_attention_mask, response_token_ids_start_idx, strict=True)]
    answer_input_ids = [f[r:].tolist() for f, r in zip(full_input_ids_np, response_token_ids_start_idx, strict=True)]
    answer_attention_mask = [f[r:] for f, r in zip(full_attention_mask, response_token_ids_start_idx, strict=True)]

    return {
        "prompt_input_ids": prompt_input_ids,
        "prompt_attention_mask": prompt_attention_mask,
        "answer_input_ids": answer_input_ids,
        "answer_attention_mask": answer_attention_mask,
    }


def _process_tokens(
    example: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    max_prompt_length: int,
) -> dict[str, Any]:
    """
    Build the final token sequences for one example.

    Policy input  (completion_*): [BOS] + prompt + answer + [EOS]
        Completion truncated from the RIGHT when prompt+answer > max_length.

    Value model input (prompt_*): [BOS] + prompt (left-truncated when prompt > max_prompt_length).
        No EOS — the value model sees the prompt only.

    Labels (completion_labels): -100 for prompt/BOS positions; actual token IDs for answer + EOS.
    """
    prompt_ids = list(example["prompt_input_ids"])
    prompt_mask = list(example["prompt_attention_mask"])
    answer_ids = list(example["answer_input_ids"])
    answer_mask = list(example["answer_attention_mask"])

    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    # Decide whether BOS/EOS need to be prepended/appended
    needs_bos = bos_id is not None and (len(prompt_ids) == 0 or prompt_ids[0] != bos_id)
    needs_eos = eos_id is not None and (len(answer_ids) == 0 or answer_ids[-1] != eos_id)

    # Effective budget for prompt + answer tokens (after BOS/EOS are added)
    effective_max = max_length - (1 if needs_bos else 0) - (1 if needs_eos else 0)

    # Truncate answer from right if the sequence is too long
    if len(prompt_ids) + len(answer_ids) > effective_max:
        max_answer_len = effective_max - len(prompt_ids)
        if max_answer_len > 0:
            answer_ids = answer_ids[:max_answer_len]
            answer_mask = answer_mask[:max_answer_len]
        else:
            answer_ids = []
            answer_mask = []

    # ── Policy input ─────────────────────────────────────────────────────────
    completion_ids = prompt_ids + answer_ids
    completion_mask = prompt_mask + answer_mask

    if needs_bos:
        completion_ids = [bos_id] + completion_ids
        completion_mask = [1] + completion_mask
        prompt_prefix_len = len(prompt_ids) + 1  # first answer token position
    else:
        prompt_prefix_len = len(prompt_ids)

    if needs_eos:
        completion_ids = completion_ids + [eos_id]
        completion_mask = completion_mask + [1]

    # Labels: mask out prompt positions with -100
    completion_labels = list(completion_ids)
    for i in range(prompt_prefix_len):
        completion_labels[i] = -100

    # ── Value model input (left-truncated prompt) ─────────────────────────────
    effective_max_prompt = max_prompt_length - (1 if needs_bos else 0)
    if len(prompt_ids) > effective_max_prompt:
        vmodel_ids = prompt_ids[-effective_max_prompt:]
        vmodel_mask = prompt_mask[-effective_max_prompt:]
    else:
        vmodel_ids = list(prompt_ids)
        vmodel_mask = list(prompt_mask)

    if needs_bos:
        vmodel_ids = [bos_id] + vmodel_ids
        vmodel_mask = [1] + vmodel_mask

    return {
        "completion_input_ids": completion_ids,
        "completion_attention_mask": completion_mask,
        "completion_labels": completion_labels,
        "prompt_input_ids": vmodel_ids,
        "prompt_attention_mask": vmodel_mask,
        "reward": example["reward"],
        "prompt": example["prompt"],
        "completion": example["completion"],
    }


# ── Data Collator ─────────────────────────────────────────────────────────────


class DRODataCollator:
    """
    Collator for DRO datasets.

    - ``completion_*`` keys → right-padded int64 tensors (labels use -100 as pad)
    - ``prompt_*`` keys → **left-padded** int64 tensors (value model needs right-aligned end)
    - ``reward`` → float32 tensor of shape (B,)
    - ``reference_logps`` (optional) → float32 tensor of shape (B,)
    - ``prompt``, ``completion`` (str) → passed through as Python lists

    Custom collator rather than a DPODataCollatorWithPadding subclass to avoid
    the unknown-key ``reward`` raising an error in that collator.
    """

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        batch: dict[str, Any] = {}

        # completion_* → right-padded
        for key in ["completion_input_ids", "completion_attention_mask", "completion_labels"]:
            if key not in features[0]:
                continue
            pad_value = -100 if "labels" in key else (self.pad_token_id if "input_ids" in key else 0)
            tensors = [torch.tensor(f[key], dtype=torch.long) for f in features]
            batch[key] = pad(tensors, padding_value=pad_value, padding_side="right")

        # prompt_* → left-padded (value model needs the sequence right-aligned)
        for key in ["prompt_input_ids", "prompt_attention_mask"]:
            if key not in features[0]:
                continue
            pad_value = self.pad_token_id if "input_ids" in key else 0
            tensors = [torch.tensor(f[key], dtype=torch.long) for f in features]
            batch[key] = pad(tensors, padding_value=pad_value, padding_side="left")

        # Scalar fields
        if "reward" in features[0]:
            batch["reward"] = torch.tensor([f["reward"] for f in features], dtype=torch.float32)

        if "reference_logps" in features[0]:
            batch["reference_logps"] = torch.tensor(
                [f["reference_logps"] for f in features], dtype=torch.float32
            )

        # String fields (passed through for logging)
        for key in ["prompt", "completion"]:
            if key in features[0]:
                batch[key] = [f[key] for f in features]

        return batch


# ── PolicyValueWrapper ────────────────────────────────────────────────────────


class PolicyValueWrapper(nn.Module):
    """
    Wraps the policy (CausalLM) and value model (SequenceClassification) into a
    single ``nn.Module`` so that a **single** ``accelerator.prepare()`` call covers
    both parameter sets — required for correct DDP gradient all-reduce across GPUs.

    .. note::
        Always call ``model(...)`` (the wrapper) rather than
        ``unwrapped.policy(...)`` directly inside ``compute_loss``.
        Calling a submodel directly in DDP mode bypasses the grad all-reduce hook
        and silently produces wrong gradients on multi-GPU runs.
    """

    def __init__(self, policy: nn.Module, value_model: nn.Module) -> None:
        super().__init__()
        self.policy = policy
        self.value_model = value_model
        # Expose policy config for BaseTrainer.create_model_card and hub-push utilities
        self.config = policy.config
        self.is_gradient_checkpointing = getattr(policy, "is_gradient_checkpointing", False)

    @property
    def device(self):
        return next(self.policy.parameters()).device

    def generate(self, *args, **kwargs):
        return self.policy.generate(*args, **kwargs)

    def forward(
        self,
        policy_input_ids: torch.Tensor,
        policy_attention_mask: torch.Tensor,
        value_input_ids: torch.Tensor,
        value_attention_mask: torch.Tensor,
    ):
        """Forward through both sub-models in a single DDP-wrapped call."""
        policy_out = self.policy(
            input_ids=policy_input_ids,
            attention_mask=policy_attention_mask,
            use_cache=False,
        )
        value_out = self.value_model(
            input_ids=value_input_ids,
            attention_mask=value_attention_mask,
            use_cache=False,
        )
        return policy_out, value_out

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        self.is_gradient_checkpointing = True
        try:
            self.value_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        except AttributeError:
            pass

    def gradient_checkpointing_disable(self):
        self.policy.gradient_checkpointing_disable()
        self.is_gradient_checkpointing = False
        try:
            self.value_model.gradient_checkpointing_disable()
        except AttributeError:
            pass


# ── DROTrainer ────────────────────────────────────────────────────────────────


class DROTrainer(BaseTrainer):
    r"""
    Trainer for **DRO-V** (offline regularised RL with a learned value model).

    Implements Algorithm 1 from *"Offline Regularised Reinforcement Learning for Large Language
    Models Alignment"* (NeurIPS 2024). Trains on ``(prompt, completion, reward)`` triples
    without pairwise preference annotations via the joint quadratic loss:

    .. math::

        \mathcal{L}(\theta, \phi) =
            \tfrac{1}{2}\,\mathbb{E}\!\left[
                \bigl(r - V_\phi(x) - \tau \log\tfrac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}\bigr)^2
            \right]

    The policy and value model receive independent gradient signals (via ``detach()``).

    Args:
        model ([`~transformers.PreTrainedModel`] or `str`):
            Policy (CausalLM). If a string, loaded via
            ``AutoModelForCausalLM.from_pretrained``.
        value_model ([`~transformers.PreTrainedModel`] or `str`):
            Value model (SequenceClassification with ``num_labels=1``). If a string,
            loaded via ``AutoModelForSequenceClassification.from_pretrained``.
        args ([`experimental.dro.DROConfig`]):
            Training configuration.
        train_dataset ([`~datasets.Dataset`]):
            Dataset with ``prompt``, ``completion``, and ``reward`` columns.
        eval_dataset ([`~datasets.Dataset`] or `dict`, *optional*):
            Evaluation dataset.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*):
            Tokenizer / processor for the models.
        ref_model ([`~transformers.PreTrainedModel`], *optional*):
            Explicit reference model. If ``None`` and PEFT is not used, a frozen deep copy
            of the policy is created automatically. If ``None`` and PEFT is used, the
            reference distribution is obtained by disabling the adapters.
        data_collator ([`~transformers.DataCollator`], *optional*):
            Custom data collator. Defaults to ``DRODataCollator``.
        peft_config ([`~peft.PeftConfig`], *optional*):
            If provided, wraps the **policy** with the specified PEFT adapter.
        callbacks (`list[~transformers.TrainerCallback]`, *optional*):
            Additional training callbacks.
        optimizers (`tuple`, *optional*):
            ``(optimizer, scheduler)`` pair. Defaults to AdamW + linear schedule.
        compute_metrics (`Callable`, *optional*):
            Custom metrics function for evaluation.
    """

    _tag_names = ["trl", "dro"]
    _name = "DRO"
    _paper = {
        "title": "Offline Regularised Reinforcement Learning for Large Language Models Alignment",
        "id": "2405.19107",
        # docstyle-ignore
        "citation": textwrap.dedent("""\
            @inproceedings{richemond2024offline,
                title        = {{Offline Regularised Reinforcement Learning for Large Language Models Alignment}},
                author       = {Pierre Harvey Richemond and Shangmin Guo and Caglar Gulcehre and Daniele Calandriello and
                                Corrado Anselmi and Nikola Momchev and Olivier Bachem and Daniel Toyama and Zoe Stepleton and
                                Thomas Baines and Bilal Piot and Francesco Visin and Doina Precup and Rémi Munos},
                booktitle    = {Advances in Neural Information Processing Systems},
                year         = 2024,
                eprint       = {arXiv:2405.19107},
            }"""),
    }

    def __init__(
        self,
        model: PreTrainedModel | nn.Module | str = None,
        value_model: PreTrainedModel | nn.Module | str = None,
        args: DROConfig = None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        processing_class: PreTrainedTokenizerBase
        | BaseImageProcessor
        | FeatureExtractionMixin
        | ProcessorMixin
        | None = None,
        ref_model: PreTrainedModel | nn.Module | None = None,
        data_collator: DataCollator | None = None,
        peft_config: "peft.PeftConfig | None" = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        compute_metrics=None,
    ):
        if not isinstance(args, DROConfig):
            raise ValueError(
                f"`args` must be a `DROConfig` instance, got {type(args)}. "
                "Please pass a `DROConfig` instead of `TrainingArguments`."
            )

        # ── 1. Load policy from string ─────────────────────────────────────
        if isinstance(model, str):
            model_init_kwargs = args.model_init_kwargs or {}
            if args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
                model_init_kwargs["device_map"] = None
            model = create_model_from_path(model, **model_init_kwargs)
        elif args.model_init_kwargs is not None:
            logger.warning(
                "You passed `model_init_kwargs` to DROConfig but the model is already instantiated. "
                "`model_init_kwargs` will be ignored."
            )

        # ── 2. Load value model from string ───────────────────────────────
        if isinstance(value_model, str):
            value_model_init_kwargs = {"num_labels": 1, **(args.value_model_init_kwargs or {})}
            if args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
                value_model_init_kwargs["device_map"] = None
            value_model = create_model_from_path(
                value_model,
                architecture=AutoModelForSequenceClassification,
                **value_model_init_kwargs,
            )
        elif args.value_model_init_kwargs is not None:
            logger.warning(
                "You passed `value_model_init_kwargs` to DROConfig but value_model is already instantiated. "
                "`value_model_init_kwargs` will be ignored."
            )

        # ── 3. Validate no parameter sharing ──────────────────────────────
        if model is value_model:
            raise ValueError(
                "`model` and `value_model` cannot be the same object. Parameter sharing between the "
                "policy and value model is detrimental per DRO paper §4. Pass separate model instances."
            )

        # ── 4. Apply PEFT to policy and value model ────────────────────────
        self._peft_has_been_casted_to_bf16 = False

        if peft_config is not None:
            if not is_peft_available():
                raise ValueError(
                    "PEFT is not installed. Pass `pip install peft` to use a PEFT configuration."
                )
            if isinstance(model, peft.PeftModel):
                raise ValueError(
                    "You passed a `PeftModel` together with a `peft_config`. Please merge and unload "
                    "the existing adapter first, then pass the base model with the new `peft_config`."
                )

            if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                model = peft.prepare_model_for_kbit_training(
                    model, use_gradient_checkpointing=args.gradient_checkpointing
                )
            elif args.gradient_checkpointing:
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                else:

                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)

                    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

            model = peft.get_peft_model(model, peft_config)

            if args.bf16 and getattr(model, "is_loaded_in_4bit", False):
                # Cast LoRA layers to bf16 for stable 4-bit training
                for name, module in model.named_modules():
                    if "norm" in name:
                        module.to(torch.float32)
                    elif any(x in name for x in ["lm_head", "embed_tokens", "embed_in", "embed_out"]):
                        if hasattr(module, "weight") and module.weight.dtype == torch.float32:
                            module.to(torch.bfloat16)
                self._peft_has_been_casted_to_bf16 = True

            # Apply the same PEFT config to the value model (SEQ_CLS task type)
            value_peft_config = copy.copy(peft_config)
            value_peft_config.task_type = peft.TaskType.SEQ_CLS

            if getattr(value_model, "is_loaded_in_8bit", False) or getattr(value_model, "is_loaded_in_4bit", False):
                value_model = peft.prepare_model_for_kbit_training(
                    value_model, use_gradient_checkpointing=args.gradient_checkpointing
                )
            elif args.gradient_checkpointing:
                if hasattr(value_model, "enable_input_require_grads"):
                    value_model.enable_input_require_grads()
                else:

                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)

                    value_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

            value_model = peft.get_peft_model(value_model, value_peft_config)

            if args.bf16 and getattr(value_model, "is_loaded_in_4bit", False):
                for name, module in value_model.named_modules():
                    if "norm" in name:
                        module.to(torch.float32)
                    elif any(x in name for x in ["score", "classifier", "embed_tokens", "embed_in", "embed_out"]):
                        if hasattr(module, "weight") and module.weight.dtype == torch.float32:
                            module.to(torch.bfloat16)

        self.is_peft_model = is_peft_available() and isinstance(model, peft.PeftModel)
        self.is_value_peft_model = is_peft_available() and isinstance(value_model, peft.PeftModel)

        # ── 5. Set up reference model ──────────────────────────────────────
        if ref_model is not None:
            self.ref_model = ref_model
        elif self.is_peft_model or args.precompute_ref_log_probs:
            # The policy with adapters disabled serves as the reference
            self.ref_model = None
        else:
            self.ref_model = _create_reference_model(model)

        # ── 6. Sync pad_token_id into value model config ──────────────────
        # SequenceClassification models use pad_token_id to find the last real token
        # for pooling. Without it, batches with varying lengths raise a ValueError.
        if processing_class is not None and processing_class.pad_token_id is not None:
            if value_model.config.pad_token_id is None:
                value_model.config.pad_token_id = processing_class.pad_token_id

        # ── 7. Disable dropout ─────────────────────────────────────────────
        if args.disable_dropout:
            disable_dropout_in_model(model)
            disable_dropout_in_model(value_model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        # ── 7. Reward z-normalisation (train split only) ───────────────────
        self.reward_mean = None
        self.reward_std = None
        if args.normalize_rewards:
            rewards_np = np.array(train_dataset["reward"], dtype=np.float64)
            self.reward_mean = float(rewards_np.mean())
            self.reward_std = float(rewards_np.std()) + 1e-8
            logger.info(
                f"Reward normalisation: mean={self.reward_mean:.4f}, std={self.reward_std:.4f}, "
                f"range=[{rewards_np.min():.2f}, {rewards_np.max():.2f}]"
            )
            _mean, _std = self.reward_mean, self.reward_std

            def _normalize_reward(ex):
                return {"reward": (ex["reward"] - _mean) / _std}

            train_dataset = train_dataset.map(_normalize_reward, desc="Z-normalising train rewards")

        self.processing_class = processing_class

        # ── 8. Dataset preprocessing ───────────────────────────────────────
        with PartialState().main_process_first():
            # Apply chat template if the data is in conversational format
            train_dataset = train_dataset.map(
                maybe_apply_chat_template,
                fn_kwargs={"tokenizer": processing_class},
                num_proc=args.dataset_num_proc,
                desc="Applying chat template to train dataset",
            )
            if eval_dataset is not None:
                eval_dataset = eval_dataset.map(
                    maybe_apply_chat_template,
                    fn_kwargs={"tokenizer": processing_class},
                    num_proc=args.dataset_num_proc,
                    desc="Applying chat template to eval dataset",
                )

            # Tokenise (batched) then build final sequences (per-example)
            train_dataset = train_dataset.map(
                _tokenize,
                batched=True,
                fn_kwargs={"tokenizer": processing_class},
                num_proc=args.dataset_num_proc,
                desc="Tokenising train dataset",
            )
            train_dataset = train_dataset.map(
                _process_tokens,
                fn_kwargs={
                    "tokenizer": processing_class,
                    "max_length": args.max_length or 1024,
                    "max_prompt_length": args.max_prompt_length or 512,
                },
                num_proc=args.dataset_num_proc,
                desc="Processing train dataset tokens",
            )

            if eval_dataset is not None:
                eval_dataset = eval_dataset.map(
                    _tokenize,
                    batched=True,
                    fn_kwargs={"tokenizer": processing_class},
                    num_proc=args.dataset_num_proc,
                    desc="Tokenising eval dataset",
                )
                eval_dataset = eval_dataset.map(
                    _process_tokens,
                    fn_kwargs={
                        "tokenizer": processing_class,
                        "max_length": args.max_length or 1024,
                        "max_prompt_length": args.max_prompt_length or 512,
                    },
                    num_proc=args.dataset_num_proc,
                    desc="Processing eval dataset tokens",
                )

        # ── 9. Data collator ───────────────────────────────────────────────
        if data_collator is None:
            data_collator = DRODataCollator(pad_token_id=processing_class.pad_token_id)
            if args.remove_unused_columns:
                args.remove_unused_columns = False
                logger.warning(
                    "When using DRODataCollator, `remove_unused_columns` must be False. "
                    "It has been set to False automatically."
                )

        # ── 10. Wrap policy + value into a single module ───────────────────
        model_wrapper = PolicyValueWrapper(model, value_model)

        # ── 11. Store trainer-specific attributes ──────────────────────────
        self.tau = args.tau
        self.precompute_ref_log_probs = args.precompute_ref_log_probs
        self._stored_metrics: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False

        # Gradient-checkpointing compat (transformers <5.0.0 warning suppression)
        if args.gradient_checkpointing and Version(transformers.__version__) < Version("5.0.0"):
            args.gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
            args.gradient_checkpointing_kwargs.setdefault("use_reentrant", False)

        # ── 12. Call parent Trainer (creates accelerator + optimizer) ──────
        super().__init__(
            model=model_wrapper,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Disable automatic loss scaling from model (we compute our own loss)
        self.model_accepts_loss_kwargs = False

        if not hasattr(self, "accelerator"):
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. "
                "Consider upgrading `transformers`."
            )

        # ── 13. Prepare reference model with accelerator ───────────────────
        if self.ref_model is None:
            if not (self.is_peft_model or self.precompute_ref_log_probs):
                raise ValueError(
                    "No reference model provided and the model is not a PEFT model. "
                    "Either pass a `ref_model`, use PEFT, or set `precompute_ref_log_probs=True`."
                )
        else:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

    # ── Context managers ──────────────────────────────────────────────────────

    @contextmanager
    def null_ref_context(self):
        """Context manager that disables PEFT adapters on the policy to get reference log-probs."""
        policy = self.accelerator.unwrap_model(self.model).policy
        with (policy.disable_adapter() if self.is_peft_model else nullcontext()):
            yield

    # ── Core algorithm ────────────────────────────────────────────────────────

    @staticmethod
    def _get_batch_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Sum of per-token log-probabilities for the completion tokens.

        Args:
            logits: ``(B, T, V)`` — raw model logits.
            labels: ``(B, T)`` — token IDs with ``-100`` masking prompt positions.

        Returns:
            ``(B,)`` tensor of summed completion log-probs.
        """
        # Causal-LM shift: logit[t] predicts label[t+1]
        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != -100
        labels = labels.clamp(min=0)  # avoid out-of-bounds gather on -100 positions
        per_token_logps = selective_log_softmax(logits, labels)
        return (per_token_logps * loss_mask).sum(-1)  # (B,)

    def _compute_ref_log_probs(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute reference log-probs for *one* batch (used when precomputing)."""
        with torch.no_grad():
            if self.ref_model is None:
                with self.null_ref_context():
                    ref_out = self.accelerator.unwrap_model(self.model).policy(
                        input_ids=batch["completion_input_ids"],
                        attention_mask=batch["completion_attention_mask"],
                        use_cache=False,
                    )
            else:
                ref_out = self.ref_model(
                    input_ids=batch["completion_input_ids"],
                    attention_mask=batch["completion_attention_mask"],
                    use_cache=False,
                )
        return self._get_batch_logps(ref_out.logits, batch["completion_labels"])

    def get_batch_loss_metrics(
        self,
        model: nn.Module,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute the DRO-V loss and diagnostics for one batch.

        Steps follow Algorithm 1 of the paper:
        1. Forward policy + value through the wrapper (single DDP call).
        2. Compute reference log-probs (cached or live).
        3. Compute log-ratio log(π_θ / π_ref).
        4. Policy loss: –log π · advantage + τ/2 · log_ratio².
        5. Value loss: ½ (V – target)² where target = r – τ·log_ratio.
        6. Combined loss = policy_loss + value_loss.
        """
        device = self.accelerator.device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # ── Step 1: Policy logprobs log π_θ(y|x) ─────────────────────────
        # Called through the wrapper so DDP all-reduce fires for both param sets
        policy_out, value_out = model(
            policy_input_ids=batch["completion_input_ids"],
            policy_attention_mask=batch["completion_attention_mask"],
            value_input_ids=batch["prompt_input_ids"],
            value_attention_mask=batch["prompt_attention_mask"],
        )
        log_pi = self._get_batch_logps(policy_out.logits, batch["completion_labels"])  # (B,)

        # ── Step 2: Reference logprobs log π_ref(y|x) [no gradient] ──────
        if "reference_logps" in batch:
            log_pi_ref = batch["reference_logps"]
        else:
            with torch.no_grad():
                if self.ref_model is None:
                    # PEFT path: disable adapters to recover reference distribution
                    with self.null_ref_context():
                        ref_out = self.accelerator.unwrap_model(model).policy(
                            input_ids=batch["completion_input_ids"],
                            attention_mask=batch["completion_attention_mask"],
                            use_cache=False,
                        )
                else:
                    ref_out = self.ref_model(
                        input_ids=batch["completion_input_ids"],
                        attention_mask=batch["completion_attention_mask"],
                        use_cache=False,
                    )
            log_pi_ref = self._get_batch_logps(ref_out.logits, batch["completion_labels"])

        # log-ratio
        log_ratio = log_pi - log_pi_ref

        # value estimate
        values = value_out.logits.squeeze(-1)  # (B, 1) → (B,)

        # rewards
        rewards = batch["reward"]  # (B,)

        advantage = rewards - values

        # correct implementastion, let's ignore Tau separate tau scaling for policy, because we optimize with Adam
        loss = (advantage - self.tau * log_ratio).pow(2).mean()


        # ── Step 9: Diagnostics ───────────────────────────────────────────
        with torch.no_grad():
            metrics = {
                "loss/dro": self.accelerator.gather_for_metrics(loss.detach()).mean().item(),
                "train/log_ratio": self.accelerator.gather_for_metrics(log_ratio).mean().item(),
                # Non-negative KL proxy; avoids negative "KL" logs from signed per-sample log-ratios.
                "train/kl_approx": (0.5 * self.accelerator.gather_for_metrics(log_ratio).pow(2)).mean().item(),
                "train/values": self.accelerator.gather_for_metrics(values).mean().item(),
                "train/rewards": self.accelerator.gather_for_metrics(rewards).mean().item(),
                "train/advantage": self.accelerator.gather_for_metrics(advantage).mean().item(),
            }

        return loss, metrics

    # ── HF Trainer interface ──────────────────────────────────────────────────

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        ctx = autocast(self.accelerator.device.type) if self._peft_has_been_casted_to_bf16 else nullcontext()
        with ctx:
            loss, metrics = self.get_batch_loss_metrics(model, inputs)

        # Move loss to the correct device (matches accumulator in parent Trainer)
        loss = loss.to(self.args.device)

        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="train")

        return (loss, metrics) if return_outputs else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ):
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        ctx = autocast(self.accelerator.device.type) if self._peft_has_been_casted_to_bf16 else nullcontext()
        with torch.no_grad(), ctx:
            loss, metrics = self.get_batch_loss_metrics(model, inputs)

        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)

        # Return minimal logits/labels so EvalLoopOutput is well-formed
        logits = torch.tensor(
            [metrics.get("loss/policy", 0.0), metrics.get("loss/value", 0.0)],
            device=self.accelerator.device,
        )
        labels = torch.zeros(logits.shape[0], device=self.accelerator.device)
        return (loss.detach(), logits, labels)

    # ── Metric accumulation ───────────────────────────────────────────────────

    def store_metrics(
        self,
        metrics: dict[str, float],
        train_eval: str = "train",
    ) -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        """Flush accumulated per-step metrics into the ``logs`` dict before delegating to parent.

        Three cases:
        - Per-step training log: ``"loss"`` key present → flush ``_stored_metrics["train"]``.
        - End-of-training summary: ``"train_loss"`` key present → also flush train metrics so they
          appear in ``log_history`` even when ``logging_steps > max_steps``.
        - Eval log: ``"eval_loss"`` key present → flush ``_stored_metrics["eval"]``.

        For each custom metric we log two explicit statistics over the current logging window:
        ``<metric>_mean`` and ``<metric>_std``.
        """
        if "loss" in logs or ("train_loss" in logs and "train" in self._stored_metrics):
            train_eval = "train"
        else:
            train_eval = "eval"
        prefix = "eval_" if train_eval == "eval" else ""
        if train_eval in self._stored_metrics:
            for key, values in self._stored_metrics[train_eval].items():
                values_t = torch.tensor(values, dtype=torch.float32)
                logs[f"{prefix}{key}_mean"] = values_t.mean().item()
                logs[f"{prefix}{key}_std"] = values_t.std(unbiased=False).item()
            del self._stored_metrics[train_eval]
        return super().log(logs, start_time)

    # ── Precompute reference log-probs ────────────────────────────────────────

    def get_train_dataloader(self) -> DataLoader:
        """Override to optionally precompute reference log-probs before the first epoch."""
        if self.precompute_ref_log_probs and not self._precomputed_train_ref_log_probs:
            dataloader_params = {
                "batch_size": self.args.per_device_train_batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
            }
            data_loader = self.accelerator.prepare(DataLoader(self.train_dataset, **dataloader_params))
            reference_logps = []
            for batch in tqdm(data_loader, desc="Precomputing train reference log-probs"):
                ref_logps = self._compute_ref_log_probs(batch)
                ref_logps = self.accelerator.gather_for_metrics(ref_logps)
                reference_logps.append(ref_logps.cpu())
            self.train_dataset = self.train_dataset.add_column(
                name="reference_logps",
                column=torch.cat(reference_logps).float().numpy(),
            )
            self._precomputed_train_ref_log_probs = True
        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset: Dataset | None = None) -> DataLoader:
        """Override to optionally precompute reference log-probs for the eval set."""
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if self.precompute_ref_log_probs and not self._precomputed_eval_ref_log_probs:
            dataloader_params = {
                "batch_size": self.args.per_device_eval_batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
            }
            data_loader = self.accelerator.prepare(DataLoader(eval_dataset, **dataloader_params))
            reference_logps = []
            for batch in tqdm(data_loader, desc="Precomputing eval reference log-probs"):
                ref_logps = self._compute_ref_log_probs(batch)
                ref_logps = self.accelerator.gather_for_metrics(ref_logps)
                reference_logps.append(ref_logps.cpu())
            eval_dataset = eval_dataset.add_column(
                name="reference_logps",
                column=torch.cat(reference_logps).float().numpy(),
            )
            if self.eval_dataset is not None:
                self.eval_dataset = eval_dataset
            self._precomputed_eval_ref_log_probs = True

        return super().get_eval_dataloader(eval_dataset=eval_dataset)

    # ── Checkpointing & saving ────────────────────────────────────────────────

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False):
        """Save the policy for inference; save the value model in a ``value_model/`` subdirectory."""
        backup = self.model
        # Temporarily swap model pointer so the parent saves only the policy
        self.model = self.accelerator.unwrap_model(self.model).policy
        if self.is_deepspeed_enabled:
            backup_deepspeed = self.deepspeed
            self.deepspeed = self.model
        super().save_model(output_dir, _internal_call)
        self.model = backup
        if self.is_deepspeed_enabled:
            self.deepspeed = backup_deepspeed

        # Save value model alongside the policy (adapter-only if PEFT was applied)
        value_dir = os.path.join(output_dir or self.args.output_dir, "value_model")
        self.accelerator.unwrap_model(backup).value_model.save_pretrained(value_dir)

    def _save_checkpoint(self, model, trial):
        if self.args.hub_model_id is None:
            model_name = Path(self.args.output_dir).name
        else:
            model_name = self.args.hub_model_id.split("/")[-1]
        self.create_model_card(model_name=model_name)
        super()._save_checkpoint(model, trial)
