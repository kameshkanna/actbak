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
echo "  # Download all datasets:"
echo "  python data/download_datasets.py"
echo ""
echo "  # Run experiments in order:"
echo "  python experiments/01_norm_profiling.py   # norm profiling"
echo "  python experiments/02_direction_extraction.py   # direction extraction"
echo "  python experiments/03_ramp_steering_eval.py   # ramp steering eval"
