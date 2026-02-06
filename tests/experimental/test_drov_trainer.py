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

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from datasets import Dataset
from torch import nn
from transformers import T5Config, T5ForConditionalGeneration

from trl.experimental.drov import DROVConfig, DROVTrainer, DataCollatorForDROV
from trl.experimental.drov.drov_trainer import compute_drov_residual_loss

from ..testing_utils import TrlTestCase


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 2
    pad_token = "<pad>"
    eos_token = "</s>"

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        # deterministic toy tokenizer
        ids = [3 + (ord(char) % 17) for char in str(text)]
        return {"input_ids": ids}

    def save_pretrained(self, path: str | Path) -> None:
        output = Path(path)
        output.mkdir(parents=True, exist_ok=True)
        (output / "tokenizer.json").write_text('{"dummy": true}', encoding="utf-8")


class TinyValueModel(nn.Module):
    def __init__(self, hidden_size: int = 24) -> None:
        super().__init__()
        self.embedding = nn.Embedding(128, hidden_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        hidden = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        values = self.value_head(pooled).squeeze(-1)
        return SimpleNamespace(values=values)


def _build_policy_model() -> T5ForConditionalGeneration:
    config = T5Config(
        vocab_size=128,
        d_model=24,
        d_kv=12,
        d_ff=48,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        dropout_rate=0.0,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
    )
    return T5ForConditionalGeneration(config)


def _build_dataset() -> Dataset:
    rows = [
        {
            "prompt_input_ids": [5, 6, 7, 8],
            "prompt_attention_mask": [1, 1, 1, 1],
            "completion_input_ids": [9, 10, 1],
            "completion_attention_mask": [1, 1, 1],
            "reward": 0.25,
        },
        {
            "prompt_input_ids": [11, 12, 13],
            "prompt_attention_mask": [1, 1, 1],
            "completion_input_ids": [14, 1],
            "completion_attention_mask": [1, 1],
            "reward": -0.5,
        },
        {
            "prompt_input_ids": [15, 16, 17, 18, 19],
            "prompt_attention_mask": [1, 1, 1, 1, 1],
            "completion_input_ids": [20, 21, 22, 1],
            "completion_attention_mask": [1, 1, 1, 1],
            "reward": 0.1,
        },
    ]
    return Dataset.from_list(rows)


class TestDROVTrainer(TrlTestCase):
    def test_drov_loss_matches_manual(self):
        rewards = torch.tensor([0.4, -0.2], dtype=torch.float32)
        values = torch.tensor([0.1, -0.1], dtype=torch.float32)
        policy_logps = torch.tensor([-3.0, -2.0], dtype=torch.float32)
        ref_logps = torch.tensor([-2.8, -2.4], dtype=torch.float32)

        loss, delta, log_ratio = compute_drov_residual_loss(
            rewards=rewards,
            values=values,
            policy_logps=policy_logps,
            ref_logps=ref_logps,
            tau=1.0,
        )

        expected_log_ratio = policy_logps - ref_logps
        expected_delta = rewards - values - expected_log_ratio
        expected_loss = 0.5 * expected_delta.pow(2).mean()

        torch.testing.assert_close(log_ratio, expected_log_ratio)
        torch.testing.assert_close(delta, expected_delta)
        torch.testing.assert_close(loss, expected_loss)

    def test_tokenize_row(self):
        tokenizer = DummyTokenizer()
        result = DROVTrainer.tokenize_row(
            {"prompt": "abc", "completion": "xy", "reward": 1.5},
            processing_class=tokenizer,
            max_prompt_length=4,
            max_completion_length=3,
            add_special_tokens=True,
            is_chat=False,
        )
        assert "prompt_input_ids" in result
        assert "completion_input_ids" in result
        assert result["reward"] == 1.5
        assert len(result["prompt_input_ids"]) <= 4
        assert len(result["completion_input_ids"]) <= 3

    def test_data_collator(self):
        collator = DataCollatorForDROV(pad_token_id=0)
        batch = collator(
            [
                {
                    "prompt_input_ids": [1, 2],
                    "completion_input_ids": [3],
                    "reward": 0.1,
                },
                {
                    "prompt_input_ids": [4],
                    "completion_input_ids": [5, 6],
                    "reward": -0.2,
                },
            ]
        )
        assert batch["prompt_input_ids"].shape == (2, 2)
        assert batch["completion_input_ids"].shape == (2, 2)
        assert batch["reward"].shape == (2,)

    def test_training_step_updates_policy_and_value(self):
        policy_model = _build_policy_model()
        ref_model = _build_policy_model()
        for param in ref_model.parameters():
            param.requires_grad_(False)

        dataset = _build_dataset()
        args = DROVConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=1,
            gradient_accumulation_steps=1,
            remove_unused_columns=False,
            report_to="none",
            bf16=False,
            fp16=False,
            disable_tqdm=True,
            save_strategy="no",
            eval_strategy="no",
        )
        trainer = DROVTrainer(
            model=policy_model,
            ref_model=ref_model,
            value_model=TinyValueModel(),
            args=args,
            processing_class=DummyTokenizer(),
            train_dataset=dataset,
        )

        batch = trainer.data_collator([dataset[0], dataset[1]])
        loss, metrics = trainer.compute_loss(trainer.model, batch, return_outputs=True)
        loss.backward()

        assert "loss/drov" in metrics
        assert any(
            param.grad is not None and torch.count_nonzero(param.grad).item() > 0
            for param in trainer.model.parameters()
            if param.requires_grad
        )
        assert any(
            param.grad is not None and torch.count_nonzero(param.grad).item() > 0
            for param in trainer.value_model.parameters()
            if param.requires_grad
        )
        assert all(param.grad is None for param in trainer.ref_model.parameters())

    def test_training_saves_value_model_checkpoint(self):
        policy_model = _build_policy_model()
        ref_model = _build_policy_model()
        dataset = _build_dataset()
        args = DROVConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=1,
            gradient_accumulation_steps=1,
            save_strategy="steps",
            save_steps=1,
            remove_unused_columns=False,
            report_to="none",
            bf16=False,
            fp16=False,
            disable_tqdm=True,
            eval_strategy="no",
        )
        trainer = DROVTrainer(
            model=policy_model,
            ref_model=ref_model,
            value_model=TinyValueModel(),
            args=args,
            processing_class=DummyTokenizer(),
            train_dataset=dataset,
        )
        trainer.train()

        checkpoints = list(Path(self.tmp_dir).glob("checkpoint-*"))
        assert checkpoints
        assert (checkpoints[0] / "value_model" / "pytorch_model.bin").exists()
