# Activation Baking

**Ramp-calibrated behavioral steering that compensates for growing residual stream norms.**

---

## The Problem with Flat-K Steering

Activation steering adds a direction vector to the residual stream at layer ℓ:

```
h_ℓ  →  h_ℓ + K · ĉ
```

Prior work uses a fixed K across all layers. This fails because residual stream norms grow monotonically with depth — by ×84 in Llama-3.1-8B. A flat K calibrated for shallow layers is drowned out at deep layers:

```
relative influence  =  K / ‖h_ℓ‖   →  0   as depth increases
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

Requires Python ≥ 3.10 and a CUDA GPU. Each 8B model fits on a single 40 GB A100 in bfloat16; optimised for GH200 (96 GB) with `batch_size=64`.

---

## Datasets

```bash
python data/download_datasets.py
```

| Dataset | Behavior | Used by |
|---|---|---|
| MT-Bench (80 prompts) | — | Exp 01 — norm profiling |
| HarmBench behaviors | Safety | Exp 03, Ablation 01, Ablation 02 |
| ClearHarm | Safety | Exp 03 |
| Anthropic model-written-evals (sycophancy) | Sycophancy | Exp 03 |

---

## Pipeline

### Step 1 — Norm Profiling

Measures per-layer residual stream L2 norms and derives the K_ℓ = μ̄_ℓ / √d schedule.

```bash
python experiments/01_norm_profiling.py
```

**Outputs:** `results/norm_profiles/{model}.csv`, `figures/norm_profiles/`

---

### Step 2 — Direction Extraction

Extracts per-layer behavioral directions via CAA (contrastive activation addition) using completion-only pooling.

```bash
python experiments/02_direction_extraction.py
```

**Outputs:** `results/directions/{model}/{behavior}.npz`

---

### Step 3 — Ramp Steering Evaluation

Evaluates ramp-K steering across all model families and behaviors.

- **Safety** (HarmBench + ClearHarm): scored SAFE / UNSAFE / GIBBERISH
- **Sycophancy** (Anthropic evals): scored SYCOPHANTIC / CONSISTENT / GIBBERISH

Three conditions per (behavior, k_scale): `baseline` (once per behavior), `ramp_pos`, `ramp_neg`.

```bash
python experiments/03_ramp_steering_eval.py
```

Flags:

```bash
python experiments/03_ramp_steering_eval.py --batch-size 16          # tune to VRAM
python experiments/03_ramp_steering_eval.py --models llama-3.1-8b-instruct
python experiments/03_ramp_steering_eval.py --behaviors safety --k-scales 0.5 1.0
```

**Outputs:** `results/ramp_eval/{model}/{behavior}/{baseline,k{scale}}/scored_results.csv`

---

### Full Pipeline

```bash
bash setup.sh && source .venv/bin/activate

python data/download_datasets.py

python experiments/01_norm_profiling.py
python experiments/02_direction_extraction.py
python experiments/03_ramp_steering_eval.py
```

---

## Ablations

### Ablation 01 — Flat-K vs Ramp-K

Proves that ramp-K is necessary by showing flat-K (fixed mean K across all layers) fails. As residual stream norms grow monotonically with depth, a K calibrated at shallow layers has vanishing relative influence at deep layers (K / ‖h_ℓ‖ → 0). Ramp-K maintains constant relative influence at every layer.

| Condition | Description | Expected outcome |
|---|---|---|
| baseline | No steering | Unsafe responses |
| flat_K | All layers steered with K = mean(K_ℓ) | Weak deep-layer influence, partial safety |
| ramp_K | All layers steered with K_ℓ = μ̄_ℓ / √d | Full safety, coherent output |

```bash
python ablations/01_flat_k_vs_ramp.py
```

**Outputs:** `results/ablations/flat_k_vs_ramp/{model}/`, `figures/ablations/flat_k_vs_ramp/`

---

### Ablation 02 — Single-Layer Collapse

Steers a single layer at 10× K_ℓ — far above the operating point — at five depth targets (40%, 50%, 60%, 70%, 80%). Measures judge score and gibberish rate per layer to show that high-K single-layer injection collapses the residual stream into incoherent output, motivating distributed ramp steering.

```bash
python ablations/02_single_layer_collapse.py
```

**Outputs:** `results/ablations/single_layer_collapse/{model}/`, `figures/ablations/single_layer_collapse/`

---

## Project Structure

```
activation_baking/
  config.py              ModelConfig, ExperimentConfig dataclasses
  registry.py            ModelRegistry — load once, reuse across steps
  model_utils.py         Model loading, prompt formatting, generation
  norm_profiler.py       Hook-based residual stream norm measurement
  direction_extractor.py Contrastive activation direction extraction (CAA)
  steerer.py             Inference-time ramp steering + config builders
  judges.py              SmallModelJudge (behavior-aware: safety + sycophancy)

config/
  models.yml             8 model variants (4 instruct + 4 base)
  experiment.yml         Hyperparameters, sample counts, ablation settings

data/
  contrastive_pairs/     CAA minimal pairs (safety, sycophancy)
  download_datasets.py   Downloads all required datasets

experiments/
  01_norm_profiling.py       Measure norm growth, derive K_ℓ schedule
  02_direction_extraction.py Extract behavioral directions via CAA
  03_ramp_steering_eval.py   Ramp steering — all models, behaviors, K scales

ablations/
  01_flat_k_vs_ramp.py       Flat K vs ramp K — proves depth-adaptive K is necessary
  02_single_layer_collapse.py    Single high-K layer injection — shows residual collapse

paper/
  formula_derivation.md  K_ℓ = μ̄_ℓ / √d derivation
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
