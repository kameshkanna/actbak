"""Run experiment 03 across all models × behaviors × k-scales in parallel.

Strategy: group by (model, behavior) — 12 pairs. Each GPU worker runs all 7
k-scales for one pair sequentially, while 8 pairs execute in parallel across
8 GPUs. 12 pairs / 8 GPUs = 2 rounds instead of 84 sequential runs.

Usage:
    python run_exp03.py              # 8 GPUs (default)
    python run_exp03.py --n-gpus 4  # 4 GPUs
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

MODELS = [
    "llama-3.1-8b-instruct",
    "mistral-7b-instruct",
    "qwen2.5-7b-instruct",
    "gemma-2-9b-it",
]
BEHAVIORS = ["safety", "refusal", "sycophancy"]
K_SCALES   = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0]


def run_pair(gpu_id: int, model: str, behavior: str) -> tuple[str, str, list[float]]:
    """Run all k-scales for one (model, behavior) pair on the given GPU."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
    failures: list[float] = []
    for scale in K_SCALES:
        cmd = [
            sys.executable, "experiments/03_ramp_steering_eval.py",
            "--model", model,
            "--behavior", behavior,
            "--k-scale", str(scale),
        ]
        print(f"  [GPU {gpu_id}] {model} | {behavior} | k={scale}", flush=True)
        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            print(f"  [GPU {gpu_id}] FAILED: {model} | {behavior} | k={scale}", flush=True)
            failures.append(scale)
    return model, behavior, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-gpus", type=int, default=8, help="Number of GPUs available.")
    args = parser.parse_args()

    pairs = list(product(MODELS, BEHAVIORS))  # 12 pairs
    print(
        f"Launching {len(pairs)} (model, behavior) pairs across {args.n_gpus} GPUs "
        f"({len(K_SCALES)} k-scales each = {len(pairs) * len(K_SCALES)} total runs).",
        flush=True,
    )

    failed: list[tuple[str, str, float]] = []
    with ProcessPoolExecutor(max_workers=args.n_gpus) as pool:
        future_to_pair = {
            pool.submit(run_pair, i % args.n_gpus, model, behavior): (model, behavior)
            for i, (model, behavior) in enumerate(pairs)
        }
        for future in as_completed(future_to_pair):
            model, behavior, pair_failures = future.result()
            status = f"FAILED k={pair_failures}" if pair_failures else "OK"
            print(f"  Finished: {model} | {behavior} [{status}]", flush=True)
            failed.extend((model, behavior, s) for s in pair_failures)

    print("\n=== Done ===")
    if failed:
        print(f"{len(failed)} failed run(s):")
        for model, behavior, scale in failed:
            print(f"  {model} | {behavior} | k={scale}")
    else:
        print("All runs succeeded.")


if __name__ == "__main__":
    main()
