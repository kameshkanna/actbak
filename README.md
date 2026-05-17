# Activation Baking

Persistent behavioral biases via activation vectors written into transformer MLP weights.

## Setup

```bash
bash setup.sh
source .venv/bin/activate
```

## Experiments

### 01 — Norm Profiling

```bash
python data/download_mtbench.py
python experiments/01_norm_profiling.py
```

Outputs per-layer norm profiles and K_ℓ = μ̄_ℓ / √d values for all four models.
Results saved to `results/norm_profiles/`. Figures saved to `figures/`.

To run a single model:
```bash
python experiments/01_norm_profiling.py --models llama-3.1-8b-instruct
```

For machines with limited VRAM:
```bash
python experiments/01_norm_profiling.py --load-in-4bit
```

## Models

| Model | Norm | Hidden | Layers |
|---|---|---|---|
| Llama 3.1 8B Instruct | Pre | 4096 | 32 |
| Qwen3 8B Instruct | Pre | 4096 | 36 |
| Gemma 2 9B IT | Pre+Post | 3584 | 42 |
| Phi-4 14B | Pre | 5120 | 40 |
