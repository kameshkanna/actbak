"""Run experiment 02 (direction extraction) for all models sequentially on one GPU.

Each model is loaded once and all behaviors are extracted while it is loaded.

Usage:
    python run_exp02.py
    python run_exp02.py --models llama-3.1-8b-instruct
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
    print(f"Direction extraction: {len(models)} models on cuda:0", flush=True)

    failed = []
    for model in models:
        print(f"\n  → {model}", flush=True)
        rc = subprocess.run(
            [sys.executable, "experiments/02_direction_extraction.py", "--models", model]
        ).returncode
        if rc != 0:
            print(f"  FAILED: {model}", flush=True)
            failed.append(model)

    print("\n=== Done ===")
    if failed:
        print(f"Failed: {failed}")
    else:
        print("All models extracted successfully.")


if __name__ == "__main__":
    main()
