# Activation Baking

**Ramp-calibrated behavioral steering that fights growing residual stream norms — and a weight-space formulation that makes it permanent.**

---

## The Problem with Flat-K Steering

Activation steering adds a vector to the residual stream at layer ℓ:

```
h_ℓ  →  h_ℓ + K · ĉ
```

Prior work uses a fixed K across all layers. This breaks down because residual stream norms grow monotonically with depth — by ×84 in Llama-3.1-8B. A flat K that is calibrated for shallow layers gets drowned out at deep layers where ‖h_ℓ‖ is large. The steering vector's **relative influence shrinks with depth**.

```
relative influence  =  K / ‖h_ℓ‖   →  0   as layer depth increases
```

Flat-K injection at matched budget consistently under-steers at the layers that matter most.

---

## The Fix: K_ℓ = μ̄_ℓ / √d

The correct magnitude at each layer is proportional to the residual stream norm at that layer:

```
K_ℓ  =  μ̄_ℓ / √d
```

`μ̄_ℓ` is the mean L2 norm of the residual stream at layer ℓ, measured on a representative corpus. `d` is the hidden dimension. This formula ensures the steering vector maintains **constant relative influence** across the full depth of the network — shallow layers get small K, deep layers get large K.

Because norms grow with depth, this naturally produces a **ramped schedule**. See [`paper/formula_derivation.md`](paper/formula_derivation.md) for the rank-1 perturbation derivation.

---

## Activation Baking (Engineering Result)

Once the ramp schedule is established, the same effect can be written permanently into model weights. For each layer ℓ, instead of a runtime hook:

```
W_down.bias  +=  K_ℓ · ĉ
```

A rank-1 perturbation equivalence guarantees these are identical in expectation. This is zero-cost at inference time — no hooks, no overhead. See [`paper/formula_derivation.md`](paper/formula_derivation.md).

---

## Models

| Model | Type | Base counterpart |
|---|---|---|
| `meta-llama/Meta-Llama-3.1-8B-Instruct` | Instruct | `Meta-Llama-3.1-8B` |
| `mistralai/Mistral-7B-Instruct-v0.2` | Instruct | `Mistral-7B-v0.1` |
| `Qwen/Qwen2.5-7B-Instruct` | Instruct | `Qwen2.5-7B` |
| `google/gemma-2-9b-it` | Instruct | `gemma-2-9b` |

All four model pairs are evaluated. Base models are included to isolate the effect of RLHF on steering behavior.

---

## Installation

```bash
git clone https://github.com/kameshkanna/actbak
cd actbak
bash setup.sh
source .venv/bin/activate
```

Requires Python ≥ 3.10 and a CUDA GPU. All four model pairs fit on a single 40 GB A100 in bfloat16.

---

## Datasets

All datasets are downloaded once before running experiments. Sample counts are read from `config/experiment.yml`.

| Dataset | Used by | Source |
|---|---|---|
| MT-Bench (80 prompts) | Exp 01 — norm profiling | FastChat GitHub |
| HarmBench behaviors | Exp 03, 04 — steering eval | HarmBench GitHub CSV |
| ClearHarm | Exp 03, 04 — steering eval | HuggingFace |
| GSM8K | Exp 05 — baking eval | HuggingFace (auto-cached) |
| MMLU | Exp 05 — baking eval | HuggingFace (auto-cached) |
| TruthfulQA | Exp 05 — baking eval | HuggingFace (auto-cached) |
| HarmBench (HF) | Exp 05 — baking eval | HuggingFace (auto-cached) |

```bash
# Download everything
python data/download_datasets.py

# Local JSONL files only (skip HuggingFace pre-caching)
python data/download_datasets.py --skip-hf-cache

# Override sample counts
python data/download_datasets.py --n-harmbench 100 --n-clearharm 100

# Specific datasets only
python data/download_datasets.py --only mtbench harmbench_eval clearharm_eval
```

---

## Pipeline

### Step 0 — Download datasets

```bash
python data/download_datasets.py
```

---

### Step 1 — Norm Profiling

Measures per-layer residual stream L2 norms on MT-Bench prompts and derives the K_ℓ = μ̄_ℓ / √d schedule. This is the empirical foundation for the ramp — it shows norms growing monotonically with depth and quantifies by how much.

```bash
# All models
python experiments/01_norm_profiling.py

# Subset
python experiments/01_norm_profiling.py --models llama-3.1-8b-instruct mistral-7b-instruct
```

**Outputs:**
- `results/norm_profiles/{model}.csv` — per-layer mean norm, std, K_ℓ
- `figures/norm_profiles/fig1_norm_profiles.pdf` — norm growth curves per model
- `figures/norm_profiles/fig2_k_profiles.pdf` — K_ℓ ramp schedule per model
- `figures/norm_profiles/fig3_norm_comparison.pdf` — normalised depth comparison across all models

**Requires:** `data/mtbench_questions.jsonl`

---

### Step 2 — Direction Extraction

Extracts per-layer behavioral directions via CAA (contrastive activation addition). Uses completion-only pooling — activations are pooled over the completion tokens only, not the shared context prefix, for a sharper directional signal.

```bash
# All behaviors, one model
python experiments/02_direction_extraction.py \
    --models llama-3.1-8b-instruct \
    --behaviors safety refusal sycophancy

# All models, all behaviors
python experiments/02_direction_extraction.py
```

**Outputs:**
- `results/directions/{model}/{behavior}.npz` — PCA direction, mean direction, K_ℓ values

**Requires:** Step 1 (norm profiles for the target models)

---

### Step 3 — Ramp Steering Evaluation

Evaluates ramp-K steering across all model families and behaviors. Three conditions: baseline (no steering), ramp_pos (steer toward behavior), ramp_neg (steer away). Sweeps K scales to locate the lobotomy cliff — the scale at which the ramp overcomes RLHF conditioning.

```bash
# One model, one behavior, formula K
python experiments/03_ramp_steering_eval.py \
    --model llama-3.1-8b-instruct \
    --behavior safety \
    --k-scale 1.0

# Scale sweep to find the lobotomy cliff
for scale in 0.5 1.0 2.0 3.0; do
    python experiments/03_ramp_steering_eval.py \
        --model llama-3.1-8b-instruct \
        --behavior safety \
        --k-scale $scale
done

# All models, one behavior
for model in llama-3.1-8b-instruct mistral-7b-instruct qwen2.5-7b-instruct gemma-2-9b-it; do
    python experiments/03_ramp_steering_eval.py \
        --model $model \
        --behavior safety \
        --k-scale 1.0
done
```

**Outputs** (under `results/ramp_eval/{model}/{behavior}/k{scale}/`):
- `raw_generations.csv`
- `scored_results.csv` — SAFE / UNSAFE / GIBBERISH judge scores
- `summary.csv` — mean and std per condition
- `figures/ramp_eval/.../condition_comparison.pdf`
- `figures/ramp_eval/.../per_prompt_heatmap.pdf`

**Requires:** Steps 1 and 2; `data/eval_prompts/`

---

### Step 4 — Flat-K vs Ramp-K Comparison

Direct empirical validation of the main claim. Compares ramp-K against two flat-K baselines at matched and generous budget. Shows that flat injection under-steers at deep layers regardless of budget — the ramp is not just incrementally better, it is structurally correct.

```bash
# One model
python experiments/04_flat_k_comparison.py \
    --model llama-3.1-8b-instruct \
    --behavior safety

# All instruct models (default when --model is omitted)
python experiments/04_flat_k_comparison.py --behavior safety

# All behaviors
for behavior in safety refusal sycophancy; do
    python experiments/04_flat_k_comparison.py --behavior $behavior
done
```

**Conditions:**

| Condition | K at each layer | Budget |
|---|---|---|
| `baseline` | 0 | — |
| `ramp_pos` | K_ℓ = μ̄_ℓ / √d | formula |
| `flat_pos` | mean(K_ℓ) uniform | matched to ramp |
| `flat_pos_max` | max(K_ℓ) uniform | generous (more than ramp) |

**Outputs** (under `results/flat_k_comparison/{model}/{behavior}/`):
- `scored_results.csv`
- `summary.csv`
- `figures/flat_k_comparison/.../ramp_vs_flat.pdf`
- `figures/flat_k_comparison/.../k_schedule.pdf` — ramp vs flat K visualisation
- `figures/flat_k_comparison/.../per_prompt_heatmap.pdf`

**Requires:** Steps 1 and 2; `data/eval_prompts/`

---

### Step 5 — Baking Evaluation (Engineering)

Validates the weight-space equivalence: baking K_ℓ · ĉ into `W_down.bias` produces the same behavioral effect as the runtime ramp hook, with zero inference overhead. Runs the full capability benchmark suite to confirm baking does not degrade general performance.

```bash
python experiments/05_baking_eval.py \
    --model llama-3.1-8b-instruct \
    --behavior safety

# Save baked model weights
python experiments/05_baking_eval.py \
    --model llama-3.1-8b-instruct \
    --behavior safety \
    --save-baked-model
```

**Benchmarks:**

| Benchmark | Metric |
|---|---|
| HarmBench | safe_rate |
| GSM8K | exact_match |
| MMLU | accuracy |
| TruthfulQA | mc1_accuracy |

**Outputs** (under `results/baking_eval/{model}/{behavior}/k{scale}/`):
- `results.csv`, `per_sample/`
- `figures/baking_eval/.../benchmark_comparison.pdf`
- `figures/baking_eval/.../delta_heatmap.pdf`

**Requires:** Steps 1 and 2; HuggingFace datasets cached (Step 0)

---

### Full pipeline — one model

```bash
bash setup.sh && source .venv/bin/activate

python data/download_datasets.py

python experiments/01_norm_profiling.py --models llama-3.1-8b-instruct

python experiments/02_direction_extraction.py \
    --models llama-3.1-8b-instruct \
    --behaviors safety refusal sycophancy

for scale in 0.5 1.0 2.0 3.0; do
    python experiments/03_ramp_steering_eval.py \
        --model llama-3.1-8b-instruct --behavior safety --k-scale $scale
done

python experiments/04_flat_k_comparison.py \
    --model llama-3.1-8b-instruct --behavior safety

python experiments/05_baking_eval.py \
    --model llama-3.1-8b-instruct --behavior safety
```

---

## Key Results (Llama-3.1-8B-Instruct, safety)

**Ramp vs baseline:**

| K scale | ramp_pos (safe %) | ramp_neg (safe %) | Baseline |
|---|---|---|---|
| 1.0 × K_ℓ | **70%** | 45% | 55% |
| 2.0 × K_ℓ | 55% | 15% | 55% |
| 3.0 × K_ℓ | 0% *(gibberish)* | 5% | 55% |

Formula K (1×) gives clean bidirectional control. At 3×K_ℓ the ramp overcomes RLHF conditioning and the model degrades — confirming the lobotomy ceiling.

---

## Project Structure

```
activation_baking/
  config.py              ModelConfig, ExperimentConfig dataclasses
  registry.py            ModelRegistry — load once, reuse across all steps
  model_utils.py         Model loading, prompt formatting, response generation
  norm_profiler.py       Hook-based residual stream norm measurement
  direction_extractor.py Contrastive activation direction extraction (CAA)
  steerer.py             Inference-time ramp steering (no weight changes)
  baker.py               Persistent baking — writes K_ℓ·ĉ into W_down.bias
  judges.py              ActivationJudge + SmallModelJudge (SAFE/UNSAFE/GIBBERISH)
  evaluators.py          GSM8K, MMLU, TruthfulQA, HarmBench runners

config/
  models.yml             8 model variants (4 instruct + 4 base pairs)
  experiment.yml         Shared hyperparameters and sample counts

data/
  contrastive_pairs/     CAA minimal pairs (safety, refusal, sycophancy)
  download_datasets.py   Downloads MT-Bench, HarmBench, ClearHarm; pre-caches HF datasets

experiments/
  01_norm_profiling.py       Measure norm growth, derive K_ℓ schedule
  02_direction_extraction.py Extract behavioral directions via CAA
  03_ramp_steering_eval.py   Ramp steering across all models, behaviors, scales
  04_flat_k_comparison.py    Flat-K vs Ramp-K at matched budget
  05_baking_eval.py          Weight-space baking + capability benchmarks

paper/
  formula_derivation.md  K_ℓ = μ̄_ℓ / √d derivation from rank-1 perturbation equivalence
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
