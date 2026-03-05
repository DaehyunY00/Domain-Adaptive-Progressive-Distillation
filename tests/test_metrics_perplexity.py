from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from datasets import Dataset
from torch import nn

from dapd.metrics.core import compute_perplexity


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    pad_token = "<pad>"
    eos_token = "<eos>"

    def pad(
        self,
        features: list[dict[str, list[int]]],
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        max_len = max(len(f["input_ids"]) for f in features)

        def _pad(key: str, value: int) -> torch.Tensor:
            rows = []
            for feat in features:
                row = list(feat[key])
                if len(row) < max_len:
                    row = row + [value] * (max_len - len(row))
                rows.append(row)
            return torch.tensor(rows, dtype=torch.long)

        return {
            "input_ids": _pad("input_ids", 0),
            "attention_mask": _pad("attention_mask", 0),
            "labels": _pad("labels", -100),
        }


class UniformLM(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> Any:
        del attention_mask, labels
        batch, seq_len = input_ids.shape
        logits = torch.zeros(batch, seq_len, self.vocab_size, device=input_ids.device)
        return SimpleNamespace(logits=logits)


def test_compute_perplexity_uses_token_level_sum_loss_with_padding() -> None:
    vocab_size = 7
    model = UniformLM(vocab_size=vocab_size)
    tokenizer = DummyTokenizer()
    dataset = Dataset.from_list(
        [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, 2, 3]},
            {"input_ids": [4, 5], "attention_mask": [1, 1], "labels": [-100, 5]},
        ]
    )

    ppl = compute_perplexity(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        batch_size=2,
        device=torch.device("cpu"),
    )

    # Uniform logits => per-token CE is log(vocab_size), so perplexity is vocab_size.
    assert math.isfinite(ppl)
    assert ppl == pytest.approx(float(vocab_size), rel=1e-5, abs=1e-5)
