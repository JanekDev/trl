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

from contextlib import nullcontext
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from trl.experimental.dro import DROTrainer
from trl.trainer.base_trainer import BaseTrainer


class _DummyAccelerator:
    def __init__(self):
        self.device = torch.device("cpu")

    def gather_for_metrics(self, tensor):
        return tensor

    def unwrap_model(self, model):
        return model


class _ToyPolicyValueModel(nn.Module):
    def __init__(self, vocab_size: int = 13):
        super().__init__()
        self.vocab_size = vocab_size
        self.policy_scale = nn.Parameter(torch.tensor(1.7))
        self.policy_bias = nn.Parameter(torch.tensor(-0.3))
        self.value_scale = nn.Parameter(torch.tensor(0.2))
        self.value_bias = nn.Parameter(torch.tensor(0.1))

    def forward(self, policy_input_ids, policy_attention_mask, value_input_ids, value_attention_mask):
        del policy_attention_mask, value_attention_mask
        one_hot = F.one_hot(policy_input_ids % self.vocab_size, num_classes=self.vocab_size).float()
        policy_logits = one_hot * self.policy_scale + self.policy_bias
        value_logits = value_input_ids.float().mean(dim=1, keepdim=True) * self.value_scale + self.value_bias
        return SimpleNamespace(logits=policy_logits), SimpleNamespace(logits=value_logits)


class _ToyRefModel(nn.Module):
    def __init__(self, vocab_size: int = 13, scale: float = 0.4, bias: float = 0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.scale = scale
        self.bias = bias

    def forward(self, input_ids, attention_mask, use_cache=False):
        del attention_mask, use_cache
        one_hot = F.one_hot(input_ids % self.vocab_size, num_classes=self.vocab_size).float()
        return SimpleNamespace(logits=one_hot * self.scale + self.bias)


def _build_trainer(tau: float = 1.3) -> DROTrainer:
    trainer = object.__new__(DROTrainer)
    trainer.accelerator = _DummyAccelerator()
    trainer.tau = tau
    trainer.ref_model = _ToyRefModel()
    trainer.is_peft_model = False
    trainer.null_ref_context = lambda: nullcontext()
    return trainer


def _build_batch():
    return {
        "completion_input_ids": torch.tensor(
            [
                [1, 4, 2, 7, 5],
                [2, 9, 1, 3, 6],
            ],
            dtype=torch.long,
        ),
        "completion_attention_mask": torch.ones(2, 5, dtype=torch.long),
        "completion_labels": torch.tensor(
            [
                [-100, 4, 2, 7, 5],
                [-100, 9, 1, 3, 6],
            ],
            dtype=torch.long,
        ),
        "prompt_input_ids": torch.tensor(
            [
                [11, 2, 8],
                [3, 6, 1],
            ],
            dtype=torch.long,
        ),
        "prompt_attention_mask": torch.ones(2, 3, dtype=torch.long),
        "reward": torch.tensor([1.25, -0.4], dtype=torch.float32),
    }


def _manual_losses_and_metrics(trainer: DROTrainer, model: nn.Module, batch: dict[str, torch.Tensor]):
    policy_out, value_out = model(
        policy_input_ids=batch["completion_input_ids"],
        policy_attention_mask=batch["completion_attention_mask"],
        value_input_ids=batch["prompt_input_ids"],
        value_attention_mask=batch["prompt_attention_mask"],
    )
    log_pi = DROTrainer._get_batch_logps(policy_out.logits, batch["completion_labels"])

    with torch.no_grad():
        ref_out = trainer.ref_model(
            input_ids=batch["completion_input_ids"],
            attention_mask=batch["completion_attention_mask"],
            use_cache=False,
        )
    log_pi_ref = DROTrainer._get_batch_logps(ref_out.logits, batch["completion_labels"])
    log_ratio = log_pi - log_pi_ref

    values = value_out.logits.squeeze(-1)
    rewards = batch["reward"]
    advantage = (rewards - values).detach()
    policy_loss = (-log_pi * advantage + 0.5 * log_ratio.pow(2)).mean() / trainer.tau

    value_target = (rewards - trainer.tau * log_ratio).detach()
    value_loss = 0.5 * (values - value_target).pow(2).mean()
    loss = policy_loss + value_loss

    metrics = {
        "loss/policy": policy_loss.detach(),
        "loss/value": value_loss.detach(),
        "train/log_ratio": log_ratio.detach().mean(),
        "train/kl_approx": (0.5 * log_ratio.detach().pow(2)).mean(),
        "train/values": values.detach().mean(),
        "train/rewards": rewards.detach().mean(),
        "train/advantage": advantage.detach().mean(),
    }
    return loss, policy_loss, value_loss, metrics


def test_dro_loss_matches_manual_formula():
    trainer = _build_trainer(tau=0.9)
    model = _ToyPolicyValueModel()
    batch = _build_batch()

    loss, metrics = trainer.get_batch_loss_metrics(model, batch)
    expected_loss, _, _, expected_metrics = _manual_losses_and_metrics(trainer, model, batch)

    torch.testing.assert_close(loss, expected_loss, atol=1e-6, rtol=1e-6)
    for key in expected_metrics:
        torch.testing.assert_close(torch.tensor(metrics[key]), expected_metrics[key], atol=1e-6, rtol=1e-6)


def test_dro_gradients_match_detached_policy_value_split():
    trainer = _build_trainer(tau=1.2)
    model = _ToyPolicyValueModel()
    batch = _build_batch()

    loss, _ = trainer.get_batch_loss_metrics(model, batch)
    grad_policy, grad_value = torch.autograd.grad(loss, [model.policy_scale, model.value_scale], retain_graph=True)

    _, expected_policy_loss, expected_value_loss, _ = _manual_losses_and_metrics(trainer, model, batch)
    expected_grad_policy = torch.autograd.grad(expected_policy_loss, model.policy_scale, retain_graph=True)[0]
    expected_grad_value = torch.autograd.grad(expected_value_loss, model.value_scale, retain_graph=True)[0]
    cross_policy_to_value = torch.autograd.grad(expected_policy_loss, model.value_scale, allow_unused=True)[0]
    cross_value_to_policy = torch.autograd.grad(expected_value_loss, model.policy_scale, allow_unused=True)[0]

    torch.testing.assert_close(grad_policy, expected_grad_policy, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(grad_value, expected_grad_value, atol=1e-6, rtol=1e-6)
    assert cross_policy_to_value is None
    assert cross_value_to_policy is None


def test_dro_cached_reference_log_probs_match_live_reference_path():
    trainer = _build_trainer(tau=1.0)
    model = _ToyPolicyValueModel()
    batch = _build_batch()

    live_loss, live_metrics = trainer.get_batch_loss_metrics(model, batch)
    cached_ref = trainer._compute_ref_log_probs(batch)
    cached_batch = {**batch, "reference_logps": cached_ref}
    cached_loss, cached_metrics = trainer.get_batch_loss_metrics(model, cached_batch)

    torch.testing.assert_close(live_loss, cached_loss, atol=1e-6, rtol=1e-6)
    for key in live_metrics:
        torch.testing.assert_close(torch.tensor(live_metrics[key]), torch.tensor(cached_metrics[key]), atol=1e-6, rtol=1e-6)


def test_dro_kl_proxy_is_non_negative():
    trainer = _build_trainer()
    model = _ToyPolicyValueModel()
    batch = _build_batch()

    _, metrics = trainer.get_batch_loss_metrics(model, batch)
    assert metrics["train/kl_approx"] >= 0.0


def test_dro_log_emits_explicit_mean_and_std_keys():
    trainer = object.__new__(DROTrainer)
    trainer._stored_metrics = {
        "train": {
            "loss/policy": [1.0, 3.0],
            "train/advantage": [2.0, 2.0],
        }
    }

    original_base_log = BaseTrainer.log
    BaseTrainer.log = lambda self, logs, start_time=None: logs
    try:
        logs = {"loss": 0.5}
        out = trainer.log(logs)
    finally:
        BaseTrainer.log = original_base_log

    assert out["loss/policy_mean"] == 2.0
    assert out["loss/policy_std"] == 1.0
    assert out["train/advantage_mean"] == 2.0
    assert out["train/advantage_std"] == 0.0
    assert "loss/policy" not in out
    assert "train/advantage" not in out
