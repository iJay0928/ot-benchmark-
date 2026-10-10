"""Huawei Cloud ModelArts training job launcher.

Generates the necessary configuration to run OT benchmarks as
ModelArts training jobs, enabling distributed multi-GPU execution.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from core.logging_config import get_logger

logger = get_logger("cloud.modelarts")


def generate_modelarts_config(
    output_dir: str | Path,
    obs_input: str = "",
    obs_output: str = "",
    instance_type: str = "modelarts.vm.gpu.v100",
    instance_count: int = 1,
) -> Path:
    """Generate a ModelArts training job configuration.

    Args:
        output_dir: Local directory for config files.
        obs_input: OBS input path for datasets.
        obs_output: OBS output path for results.
        instance_type: ModelArts instance flavor.
        instance_count: Number of instances (for distributed).

    Returns:
        Path to the generated config file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate training entry point
    train_py = output_dir / "train.py"
    train_py.write_text("""#!/usr/bin/env python3
\"\"\"ModelArts training entry point for OT Benchmark.\"\"\"
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Install dependencies if needed
os.system("pip install pot numpy scipy matplotlib tqdm pyyaml psutil -q")

from ot_benchmark.main import main as run_benchmark

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-url", type=str, default="")
    parser.add_argument("--train-url", type=str, default="")
    args, unknown = parser.parse_known_args()

    if args.data_url:
        os.makedirs("/cache/data", exist_ok=True)
        os.system(f"cp -r {args.data_url}/* /cache/data/ 2>/dev/null")

    sys.argv = [sys.argv[0], "--all", "--no-ui"]
    if args.train_url:
        sys.argv.extend(["--output-dir", args.train_url])
    run_benchmark()
""", encoding="utf-8")

    # Generate run script
    run_sh = output_dir / "run.sh"
    run_sh.write_text("""#!/bin/bash
set -e
pip install pot numpy scipy matplotlib tqdm pyyaml psutil -q
python train.py
""", encoding="utf-8")
    os.chmod(run_sh, 0o755)

    # Config JSON for ModelArts
    config = {
        "job_name": "ot-benchmark-job",
        "algorithm": {
            "code_dir": str(output_dir),
            "boot_file": "train.py",
            "parameters": [],
        },
        "resource": {
            "instance_type": instance_type,
            "instance_count": instance_count,
        },
        "obs": {
            "input": obs_input,
            "output": obs_output,
        },
    }

    config_path = output_dir / "modelarts_config.json"
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("ModelArts config generated: %s", config_path)
    logger.info("  Instance: %s x %d", instance_type, instance_count)
    logger.info("  OBS input: %s", obs_input or "(none)")
    logger.info("  OBS output: %s", obs_output or "(none)")

    return config_path
