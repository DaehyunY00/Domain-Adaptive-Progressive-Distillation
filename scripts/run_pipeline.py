#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dapd.pipeline import DAPDPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DAPD full training pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    pipeline = DAPDPipeline.from_yaml(args.config)
    summary = pipeline.run()

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nPipeline completed. Summary saved to: {Path(summary['final_model_path']).resolve().parent}")


if __name__ == "__main__":
    main()
