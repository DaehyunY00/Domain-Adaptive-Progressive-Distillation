from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_run_pipeline_help_smoke() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "run_pipeline.py"

    proc = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0
    assert "Run DAPD full training pipeline" in proc.stdout
