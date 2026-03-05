from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from dapd.distillation import (
    _compute_masked_kl_loss,
    _resolve_kl_usage,
    _scheduled_temperature,
)


class DummyTokenizer:
    def __init__(self, vocab: dict[str, int], special_tokens_map: dict[str, str] | None = None) -> None:
        self._vocab = vocab
        self.vocab_size = len(vocab)
        self.special_tokens_map = special_tokens_map or {}

    def get_vocab(self) -> dict[str, int]:
        return self._vocab


def test_kl_uses_shifted_labels_for_causal_lm() -> None:
    # Only position 1 differs. For causal LM, this is supervised by labels[2].
    student_logits = torch.zeros((1, 4, 2), dtype=torch.float32)
    teacher_logits = torch.zeros((1, 4, 2), dtype=torch.float32)
    teacher_logits[0, 1, 0] = 6.0
    teacher_logits[0, 1, 1] = -6.0

    labels = torch.tensor([[-100, -100, 1, 1]], dtype=torch.long)

    loss = _compute_masked_kl_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        labels=labels,
        attention_mask=None,
        temperature=1.0,
    )

    assert loss.item() > 0.01


def test_kl_direction_and_temperature_scaling_match_formula() -> None:
    student_logits = torch.tensor(
        [[[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]]],
        dtype=torch.float32,
    )
    teacher_logits = torch.tensor(
        [[[0.0, 2.0], [2.0, 0.0], [1.0, 1.0]]],
        dtype=torch.float32,
    )
    labels = torch.tensor([[-100, 1, 1]], dtype=torch.long)

    temperature = 2.0
    loss = _compute_masked_kl_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        labels=labels,
        attention_mask=None,
        temperature=temperature,
    )

    s = student_logits[:, :-1, :] / temperature
    t = teacher_logits[:, :-1, :] / temperature
    expected = (
        F.kl_div(F.log_softmax(s, dim=-1), F.softmax(t, dim=-1), reduction="none").sum(-1).mean()
        * (temperature**2)
    )
    assert torch.isclose(loss, expected, atol=1e-6)


def test_padding_tokens_are_masked_out() -> None:
    # Difference only where shifted label is padding -> KD should be zero.
    student_logits = torch.zeros((1, 4, 2), dtype=torch.float32)
    teacher_logits = torch.zeros((1, 4, 2), dtype=torch.float32)
    teacher_logits[0, 0, :] = torch.tensor([8.0, -8.0])

    # Shifted labels correspond to positions [1,2,3]. Make all ignored.
    labels = torch.tensor([[-100, -100, -100, -100]], dtype=torch.long)

    loss = _compute_masked_kl_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        labels=labels,
        attention_mask=None,
        temperature=1.0,
    )

    assert loss.item() == pytest.approx(0.0)


def test_attention_mask_used_when_labels_missing() -> None:
    # Difference exists only where shifted attention_mask is zero.
    student_logits = torch.zeros((1, 4, 2), dtype=torch.float32)
    teacher_logits = torch.zeros((1, 4, 2), dtype=torch.float32)
    teacher_logits[0, 1, :] = torch.tensor([10.0, -10.0])

    # Shifted mask over positions [1,2,3] => [1,0,0], so position 1 contribution is dropped.
    attention_mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.long)

    loss = _compute_masked_kl_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        labels=None,
        attention_mask=attention_mask,
        temperature=1.0,
    )

    assert loss.item() == pytest.approx(0.0)


def test_tokenizer_mismatch_raises_without_explicit_fallback() -> None:
    teacher = DummyTokenizer({"a": 0, "b": 1})
    student = DummyTokenizer({"a": 0, "c": 1})

    with pytest.raises(ValueError):
        _resolve_kl_usage(
            teacher_tokenizer=teacher,
            student_tokenizer=student,
            allow_kl_fallback_to_ce=False,
        )


def test_tokenizer_mismatch_can_fallback_when_enabled() -> None:
    teacher = DummyTokenizer({"a": 0, "b": 1})
    student = DummyTokenizer({"a": 0, "c": 1})

    with pytest.warns(UserWarning):
        use_kl = _resolve_kl_usage(
            teacher_tokenizer=teacher,
            student_tokenizer=student,
            allow_kl_fallback_to_ce=True,
        )

    assert use_kl is False


def test_temperature_schedule_linear_and_cosine() -> None:
    t_linear_start = _scheduled_temperature(0, 10, base_temperature=2.0, min_temperature=1.0, schedule="linear")
    t_linear_end = _scheduled_temperature(9, 10, base_temperature=2.0, min_temperature=1.0, schedule="linear")
    assert t_linear_start == pytest.approx(2.0)
    assert t_linear_end == pytest.approx(1.0)

    t_cos_start = _scheduled_temperature(0, 10, base_temperature=2.0, min_temperature=1.0, schedule="cosine")
    t_cos_end = _scheduled_temperature(9, 10, base_temperature=2.0, min_temperature=1.0, schedule="cosine")
    assert t_cos_start == pytest.approx(2.0)
    assert t_cos_end == pytest.approx(1.0)
