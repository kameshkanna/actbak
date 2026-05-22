"""Run experiment 03 for all models × behaviors × k-scales."""

from __future__ import annotations

import subprocess
import sys
from itertools import product

MODELS = [
    "llama-3.1-8b-instruct",
    "mistral-7b-instruct",
    "qwen2.5-7b-instruct",
    "gemma-2-9b-it",
]
BEHAVIORS = ["safety", "refusal", "sycophancy"]
K_SCALES   = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0]

combos = list(product(MODELS, BEHAVIORS, K_SCALES))
total  = len(combos)

for i, (model, behavior, scale) in enumerate(combos, 1):
    cmd = [
        sys.executable, "experiments/03_ramp_steering_eval.py",
        "--model", model,
        "--behavior", behavior,
        "--k-scale", str(scale),
    ]
    print(f"\n[{i}/{total}] {model} | {behavior} | k={scale}", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"FAILED: {model} | {behavior} | k={scale} — continuing.", flush=True)

print("\nDone.")
