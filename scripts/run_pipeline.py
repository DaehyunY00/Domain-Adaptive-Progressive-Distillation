#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _ensure_local_src_on_path() -> None:
    """Allow running this script without editable install.

    Keeps CLI compatibility for:
      python scripts/run_pipeline.py --config ...
    """
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if src_dir.is_dir():
        src_path = str(src_dir)
        if src_path not in sys.path:
            sys.path.insert(0, src_path)


_ensure_local_src_on_path()

from dapd.data import bioasq_proxy_fallback_used, reset_bioasq_proxy_fallback_flag
from dapd.pipeline import DAPDPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DAPD full training pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    reset_bioasq_proxy_fallback_flag()
    pipeline = DAPDPipeline.from_yaml(args.config)
    summary = pipeline.run()
    if bioasq_proxy_fallback_used():
        raise RuntimeError(
            "BioASQ fallback to pubmed_qa proxy was used. "
            "OOD evaluation is not valid for reporting. Please provide BioASQ explicitly."
        )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nPipeline completed. Summary saved to: {Path(summary['final_model_path']).resolve().parent}")


if __name__ == "__main__":
    main()
