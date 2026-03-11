from __future__ import annotations

from dapd.utils import resolve_training_strategy_kwargs


def test_resolve_training_strategy_kwargs_returns_supported_keys() -> None:
    out = resolve_training_strategy_kwargs(
        evaluation_strategy="steps",
        save_strategy="steps",
    )
    assert ("eval_strategy" in out) or ("evaluation_strategy" in out)
    assert ("save_strategy" in out) or ("checkpoint_strategy" in out)
