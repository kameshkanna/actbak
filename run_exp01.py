"""Run experiment 01 (norm profiling) for all models sequentially on one GPU.

Usage:
    python run_exp01.py
    python run_exp01.py --models llama-3.1-8b-instruct mistral-7b-instruct
"""

from __future__ import annotations

import argparse
import subprocess
import sys

import yaml


def get_models(config_path: str = "config/models.yml") -> list[str]:
    with open(config_path) as f:
        return list(yaml.safe_load(f).get("models", {}).keys())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*")
    args = parser.parse_args()

    models = args.models or get_models()
    print(f"Norm profiling: {len(models)} models on cuda:0", flush=True)

    failed = []
    for model in models:
        print(f"\n  → {model}", flush=True)
        rc = subprocess.run(
            [sys.executable, "experiments/01_norm_profiling.py", "--models", model]
        ).returncode
        if rc != 0:
            print(f"  FAILED: {model}", flush=True)
            failed.append(model)

    print("\n=== Done ===")
    if failed:
        print(f"Failed: {failed}")
    else:
        print("All models profiled successfully.")


if __name__ == "__main__":
    main()
