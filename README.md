# Activation Baking

**Ramp-calibrated CAA steering with depth-adaptive K, evaluated across injection modes.**

---

## Core Idea

Activation steering adds a direction vector to the residual stream at layer ℓ:

```
h_ℓ  →  h_ℓ + K · ĉ
```

Prior work uses a fixed K across all layers. This fails because residual stream norms grow monotonically with depth — by ×84 in Llama-3.1-8B. A flat K calibrated for shallow layers is drowned out at deep layers:

```
relative influence  =  K / ‖h_ℓ‖   →  0   as depth increases
```

**The fix:** derive K per layer from the norm profile:

```
K_ℓ  =  μ̄_ℓ / √d
```

`μ̄_ℓ` = mean L2 norm of the residual stream at layer ℓ, `d` = hidden dimension. This gives constant relative influence across all layers — a principled ramp schedule, not a hand-tuned one.

---

## Method

### Direction Extraction

Directions are extracted via **CAA (Contrastive Activation Addition)** — mean difference of last-token hidden states between positive and negative prompt completions, at layers 40–90% of model depth:

```
ĉ_ℓ = normalise( mean(H_ℓ(positive)) − mean(H_ℓ(negative)) )
```

261 contrastive pairs covering: absolute refusal logic, anti-jailbreak, cyber, CBRN, fraud, hate, and drugs.

Both **mean-direction** (CAA) and **PCA direction** (leading PC of diff matrix) are extracted and saved per layer.

### Injection Modes

Two injection modes are compared:

| Mode | Mechanism | Equivalent to |
|---|---|---|
| `broadcast` | `h + K·ĉ` at all token positions | Baked `resid_bias` in weight-modified models |
| `last_token` | `h[:, -1, :] + K·ĉ` at last position per step | Classical CAA / RepE inference-time steering |

### Eval Conditions (Exp 03)

For each model × K-scale:

| Condition | Direction | Mode |
|---|---|---|
| `baseline` | — | — |
| `caa_broadcast_pos` | `+mean_dir` | broadcast |
| `caa_broadcast_neg` | `−mean_dir` | broadcast |
| `caa_last_token_pos` | `+mean_dir` | last token |
| `caa_last_token_neg` | `−mean_dir` | last token |

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

Requires Python ≥ 3.10 and CUDA. 8B/7B models fit on a single 40 GB A100 in bfloat16. Optimised for GH200 (96 GB) at `batch_size=64`.

---

## Datasets

```bash
python data/download_datasets.py
```

| Dataset | Behavior | Used by |
|---|---|---|
| MT-Bench (80 prompts) | — | Exp 01 — norm profiling |
| HarmBench + ClearHarm | Safety | Exp 03, Ablations 01 & 02 |

---

## Running the Pipeline

### Step 1 — Norm Profiling

Measures per-layer residual stream L2 norms and derives K_ℓ = μ̄_ℓ / √d.

```bash
python experiments/01_norm_profiling.py
```

Outputs: `results/norm_profiles/{model}.csv`, `figures/norm_profiles/`

---

### Step 2 — Direction Extraction

Extracts per-layer CAA directions (mean-diff + PCA) at last-token hidden states, layers 40–90% depth.

```bash
python experiments/02_direction_extraction.py --behaviors safety
```

Outputs: `results/directions/{model}/safety.npz`

---

### Step 3 — Ramp Steering Evaluation

Evaluates all 4 steering conditions across all K-scales. Generation runs in parallel (one model per GPU), judging runs after all generation completes using Qwen2.5-32B-Instruct across all GPUs.

```bash
# Single GPU
python experiments/03_ramp_steering_eval.py --behaviors safety

# 8× A100 — 4 models in parallel, 1 GPU each
python experiments/03_ramp_steering_eval.py --behaviors safety --batch-size 64

# 8× A100 — 4 models in parallel, 2 GPUs each (tensor parallel)
python experiments/03_ramp_steering_eval.py --behaviors safety --gpus-per-model 2
```

Generation checkpoints to `results/ramp_eval/{model}/raw_responses.csv` after each model — resumable on crash.

Outputs: `results/ramp_eval/{model}/{behavior}/{condition}/scored_results.csv`

---

### Full Pipeline

```bash
bash setup.sh && source .venv/bin/activate
python data/download_datasets.py
python experiments/01_norm_profiling.py
python experiments/02_direction_extraction.py --behaviors safety
python experiments/03_ramp_steering_eval.py --behaviors safety
```

---

## Ablations

### Ablation 01 — Flat-K vs Ramp-K

Compares flat K (mean K across all layers) against ramp K (formula K per layer) on safety. Proves depth-adaptive K is necessary — flat K loses relative influence as norms grow.

```bash
python ablations/01_flat_k_vs_ramp.py
```

| Condition | K schedule |
|---|---|
| `baseline` | No steering |
| `flat_K` | K = mean(K_ℓ) at all middle layers |
| `ramp_K` | K_ℓ = μ̄_ℓ / √d per layer |

---

### Ablation 02 — Single-Layer Collapse

Steers one layer at 10×K_ℓ across depths 40–80%. Shows that high-K single-layer injection collapses output into gibberish at shallow layers (where K/‖h_ℓ‖ is large), motivating distributed ramp steering.

```bash
python ablations/02_single_layer_collapse.py
```

---

## Project Structure

```
activation_baking/
  config.py              ModelConfig, ExperimentConfig dataclasses
  registry.py            ModelRegistry — load once, reuse across steps
  model_utils.py         Model loading, prompt formatting
  norm_profiler.py       Hook-based residual stream norm measurement
  direction_extractor.py Last-token CAA direction extraction, layers 40-90%
  steerer.py             Ramp steering — broadcast and last-token injection modes
  baker.py               Persistent weight-space direction baking (down_proj bias)
  judges.py              SmallModelJudge — Qwen2.5-32B-Instruct behavior scorer

config/
  models.yml             8 model variants (4 instruct + 4 base)
  experiment.yml         K-scales, generation budget, judge model, behaviors

data/
  contrastive_pairs/     261 safety CAA pairs (refusal, anti-jailbreak, cyber, CBRN, drugs)
  download_datasets.py   Downloads HarmBench, ClearHarm, MT-Bench

experiments/
  01_norm_profiling.py       Measure norm growth, derive K_ℓ schedule
  02_direction_extraction.py Extract behavioral directions via CAA (last-token)
  03_ramp_steering_eval.py   Parallel ramp eval — all models, K-scales, injection modes

ablations/
  01_flat_k_vs_ramp.py           Flat K vs ramp K — depth-adaptive K is necessary
  02_single_layer_collapse.py    High-K single-layer injection — residual collapse
```

---

## Citation

```bibtex
@article{activationbaking2026,
  title  = {Activation Baking: Ramp-Calibrated Behavioral Steering via Depth-Adaptive K},
  author = {Kamesh R},
  year   = {2026},
}
```
