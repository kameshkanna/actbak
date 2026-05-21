#!/usr/bin/env bash
# setup.sh — one-shot environment setup for activation-baking
# Requires: Python >= 3.10, CUDA GPU (bfloat16 inference)
set -euo pipefail

PYTHON=${PYTHON:-python3}

echo "=== Creating virtual environment ==="
"$PYTHON" -m venv .venv
source .venv/bin/activate

echo "=== Installing dependencies ==="
pip install --upgrade pip --quiet
pip install -e . --quiet
pip install -r requirements.txt --quiet

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  source .venv/bin/activate"
echo ""
echo "  # Download all required datasets (reads sample counts from config/experiment.yml):"
echo "  python data/download_datasets.py"
echo ""
echo "  # Or selectively (local JSONL files only, skip HF cache):"
echo "  python data/download_datasets.py --only mtbench harmbench_eval clearharm_eval"
echo ""
echo "  # Then run experiments in order:"
echo "  python experiments/01_norm_profiling.py"
echo "  python experiments/02_direction_extraction.py --models llama-3.1-8b-instruct"
echo "  python experiments/03_ramp_steering_eval.py --model llama-3.1-8b-instruct --behavior safety --k-scale 1.0"
echo "  python experiments/04_baking_eval.py --model llama-3.1-8b-instruct --behavior safety"
