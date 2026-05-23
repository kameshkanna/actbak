"""Run experiment 02 (direction extraction) for all models in parallel.

Each model is automatically assigned to one GPU. All behaviors are extracted
for each model while it is loaded — one model load per GPU.

Usage:
    python run_exp02.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import yaml


def get_models(config_path: str = "config/models.yml") -> list[str]:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return list(cfg.get("models", {}).keys())


def run_model(gpu_id: int, model: str) -> tuple[str, int]:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
    result = subprocess.run(
        [sys.executable, "experiments/02_direction_extraction.py", "--models", model],
        env=env,
    )
    return model, result.returncode


def main() -> None:
    models = get_models()
    n_gpus = len(models)

    print(f"Direction extraction: {len(models)} models across {n_gpus} GPUs", flush=True)
    for i, m in enumerate(models):
        print(f"  GPU {i} → {m}", flush=True)
    print(flush=True)

    failed: list[str] = []
    with ProcessPoolExecutor(max_workers=n_gpus) as pool:
        futures = {
            pool.submit(run_model, i, model): model
            for i, model in enumerate(models)
        }
        for future in as_completed(futures):
            model, rc = future.result()
            status = "OK" if rc == 0 else f"FAILED (rc={rc})"
            print(f"  Finished: {model} [{status}]", flush=True)
            if rc != 0:
                failed.append(model)

    print("\n=== Done ===")
    if failed:
        print(f"Failed: {failed}")
    else:
        print("All models extracted successfully.")


if __name__ == "__main__":
    main()
