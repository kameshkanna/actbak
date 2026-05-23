# Activation Baking

**Ramp-calibrated behavioral steering that fights growing residual stream norms — and a weight-space formulation that makes it permanent.**

---

## The Problem with Flat-K Steering

Activation steering adds a vector to the residual stream at layer ℓ:

```
h_ℓ  →  h_ℓ + K · ĉ
```

Prior work uses a fixed K across all layers. This breaks down because residual stream norms grow monotonically with depth — by ×84 in Llama-3.1-8B. A flat K calibrated for shallow layers gets drowned out at deep layers where ‖h_ℓ‖ is large.

```
relative influence  =  K / ‖h_ℓ‖   →  0   as layer depth increases
```

---

## The Fix: K_ℓ = μ̄_ℓ / √d

```
K_ℓ  =  μ̄_ℓ / √d
```

`μ̄_ℓ` is the mean L2 norm of the residual stream at layer ℓ, measured on a representative corpus. `d` is the hidden dimension. This formula ensures the steering vector maintains **constant relative influence** across all layers — shallow layers get small K, deep layers get large K, naturally producing a **ramped schedule**.

---

## Models

| Model | Type |
|---|---|
| `meta-llama/Meta-Llama-3.1-8B-Instruct` | Instruct |
| `mistralai/Mistral-7B-Instruct-v0.2` | Instruct |
| `Qwen/Qwen2.5-7B-Instruct` | Instruct |
| `google/gemma-2-9b-it` | Instruct |
| `meta-llama/Meta-Llama-3.1-8B` | Base |
| `mistralai/Mistral-7B-v0.1` | Base |
| `Qwen/Qwen2.5-7B` | Base |
| `google/gemma-2-9b` | Base |

---

## Installation

```bash
git clone https://github.com/kameshkanna/actbak
cd actbak
bash setup.sh
source .venv/bin/activate
```

Requires Python ≥ 3.10 and a CUDA GPU. Each 8B model fits on a single 40 GB A100 in bfloat16.

---

## Datasets

```bash
# Download all datasets
python data/download_datasets.py
```

| Dataset | Used by |
|---|---|
| MT-Bench (80 prompts) | Exp 01 — norm profiling |
| HarmBench behaviors | Exp 03 — steering eval |
| ClearHarm | Exp 03 — steering eval |
| GSM8K / MMLU / TruthfulQA | Capability benchmarks (cached) |

---

## Pipeline

### Step 1 — Norm Profiling

Measures per-layer residual stream L2 norms and derives the K_ℓ = μ̄_ℓ / √d schedule.

```bash
python run_exp01.py
```

**Outputs:** `results/norm_profiles/{model}.csv`, `figures/norm_profiles/`

---

### Step 2 — Direction Extraction

Extracts per-layer behavioral directions via CAA (contrastive activation addition) using completion-only pooling.

```bash
python run_exp02.py
```

**Outputs:** `results/directions/{model}/{behavior}.npz`

---

### Step 3 — Ramp Steering Evaluation

Evaluates ramp-K steering across all model families and behaviors. Three conditions: `baseline`, `ramp_pos`, `ramp_neg`. Sweeps K scales to locate the lobotomy cliff.

```bash
python run_exp03.py
```

Flags:

```bash
python run_exp03.py --batch-size 16                          # tune to GPU VRAM
python run_exp03.py --models llama-3.1-8b-instruct           # single model
python run_exp03.py --behaviors safety --k-scales 0.5 1.0    # subset
```

**Outputs:** `results/ramp_eval/{model}/{behavior}/k{scale}/scored_results.csv`, `figures/`

---

### Full pipeline

```bash
bash setup.sh && source .venv/bin/activate

python data/download_datasets.py

python run_exp01.py
python run_exp02.py
python run_exp03.py
```

---

## Project Structure

```
activation_baking/
  config.py              ModelConfig, ExperimentConfig dataclasses
  registry.py            ModelRegistry — load once, reuse across steps
  model_utils.py         Model loading, prompt formatting, generation
  norm_profiler.py       Hook-based residual stream norm measurement
  direction_extractor.py Contrastive activation direction extraction (CAA)
  steerer.py             Inference-time ramp steering
  baker.py               Weight-space baking — writes K_ℓ·ĉ into W_down.bias
  judges.py              SmallModelJudge (SAFE/UNSAFE/GIBBERISH)

config/
  models.yml             8 model variants (4 instruct + 4 base)
  experiment.yml         Hyperparameters and sample counts

data/
  contrastive_pairs/     CAA minimal pairs (safety, refusal, sycophancy)
  download_datasets.py   Downloads all required datasets

experiments/
  01_norm_profiling.py       Measure norm growth, derive K_ℓ schedule
  02_direction_extraction.py Extract behavioral directions via CAA
  03_ramp_steering_eval.py   Ramp steering — all models, behaviors, scales

paper/
  formula_derivation.md  K_ℓ = μ̄_ℓ / √d derivation

run_exp01.py   Parallel norm profiling — one model per GPU
run_exp02.py   Parallel direction extraction — one model per GPU
run_exp03.py   Batched ramp eval — one model load, all scales
```

---

## Citation

```bibtex
@article{activationbaking2026,
  title  = {Activation Baking: Ramp-Calibrated Behavioral Steering via Weight-Space Direction Injection},
  author = {Kamesh R},
  year   = {2026},
}
```
