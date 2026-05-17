#!/usr/bin/env bash
set -e

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip --quiet
pip install -e . --quiet
pip install -r requirements.txt --quiet
pip install scikit-learn --quiet

echo ""
echo "Setup complete. Run: source .venv/bin/activate"
