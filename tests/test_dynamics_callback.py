from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from dapd.analysis import create_dynamics_callback


def test_dynamics_callback_on_log_and_train_end(tmp_path: Path) -> None:
    out_path = tmp_path / "distill_dynamics.json"
    callback = create_dynamics_callback(str(out_path))

    state = SimpleNamespace(global_step=7, max_steps=100)
    control = SimpleNamespace()

    returned = callback.on_log(
        args=SimpleNamespace(),
        state=state,
        control=control,
        logs={"ce_loss": 1.2, "kd_loss": 0.8, "learning_rate": 1e-4},
    )
    assert returned is control

    callback.on_train_end(
        args=SimpleNamespace(),
        state=state,
        control=control,
    )

    assert out_path.exists()
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert "log_steps" in data
    assert len(data["log_steps"]) == 1
    assert data["log_steps"][0]["global_step"] == 7
