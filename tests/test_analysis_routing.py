from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from dapd.analysis import analyze_teacher_distributions


def test_analyze_teacher_distributions_rejects_mixed_forward_and_legacy_kwargs() -> None:
    with pytest.raises(TypeError):
        analyze_teacher_distributions(
            general_teacher_path="dummy/general",
            domain_teacher_path="dummy/domain",
            dataset=[],
            lm_dataset=[],
            device=torch.device("cpu"),
            runtime=SimpleNamespace(log_level="INFO"),
        )
