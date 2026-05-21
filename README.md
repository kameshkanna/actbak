# Activation Baking

**Persistent behavioral modification of LLMs by writing activation vectors as biases into transformer MLP weights.**

Activation baking extends inference-time steering (CAA) to a *permanent* operation: behavioral directions are extracted from contrastive pairs, calibrated via the K formula, and written directly into `W_down` weight biases. No inference-time hooks, no prompt engineering — the behavior is baked in.

---

## The Core Idea

Standard activation steering adds a direction vector to the residual stream at runtime:

```
h_ℓ  →  h_ℓ + K · ĉ
```

Activation baking achieves the same effect permanently by modifying the MLP output projection bias:

```
W_down.bias  +=  K_ℓ · ĉ
```

A rank-1 perturbation equivalence guarantees these are identical in expectation. See [`paper/formula_derivation.md`](paper/formula_derivation.md) for the full derivation.

### The K Formula

```
K_ℓ  =  μ̄_ℓ / √d
```

`μ̄_ℓ` is the mean residual stream norm at layer ℓ measured on a representative corpus; `d` is the hidden dimension. Because residual norms grow monotonically with depth, this formula naturally produces a **ramped schedule**. Flat-K injection is structurally inadequate.

---

## Models

| Model | Type | Base counterpart |
|---|---|---|
| `meta-llama/Meta-Llama-3.1-8B-Instruct` | Instruct | `Meta-Llama-3.1-8B` |
| `mistralai/Mistral-7B-Instruct-v0.2` | Instruct | `Mistral-7B-v0.1` |
| `Qwen/Qwen2.5-7B-Instruct` | Instruct | `Qwen2.5-7B` |
| `google/gemma-2-9b-it` | Instruct | `gemma-2-9b` |

Base models expose the **RLHF asymmetry**: instruction-tuned models degrade asymmetrically under positive vs negative directional steering; base models degrade symmetrically in both directions.

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

All datasets are downloaded once before running experiments. Sample counts are read from `config/experiment.yml` by default.

### What is needed

| Dataset | Used by | Where it lives |
|---|---|---|
| MT-Bench (80 prompts) | Exp 01 — norm profiling | `data/mtbench_questions.jsonl` |
| HarmBench behaviors | Exp 03 — ramp steering eval | `data/eval_prompts/harmbench.jsonl` |
| ClearHarm | Exp 03 — ramp steering eval | `data/eval_prompts/clearharm.jsonl` |
| GSM8K | Exp 04 — baking eval (capability) | HuggingFace cache |
| MMLU | Exp 04 — baking eval (capability) | HuggingFace cache |
| TruthfulQA | Exp 04 — baking eval (capability) | HuggingFace cache |
| HarmBench (HF) | Exp 04 — baking eval (safety) | HuggingFace cache |

The local JSONL files (MT-Bench, HarmBench eval, ClearHarm) are downloaded from their respective sources. The HuggingFace datasets (GSM8K, MMLU, TruthfulQA, HarmBench) are auto-cached the first time an evaluator runs; calling `download_datasets.py` pre-caches them ahead of time.

### Download commands

```bash
# Download everything — reads n_harmbench, n_clearharm, seed from config/experiment.yml
python data/download_datasets.py

# Skip HuggingFace pre-caching (download only local JSONL files)
python data/download_datasets.py --skip-hf-cache

# Override sample counts
python data/download_datasets.py --n-harmbench 100 --n-clearharm 100

# Download specific datasets only
python data/download_datasets.py --only mtbench
python data/download_datasets.py --only harmbench_eval clearharm_eval
python data/download_datasets.py --only gsm8k mmlu truthfulqa harmbench_hf

# Re-download even if files already exist
python data/download_datasets.py --force
```

---

## Running the Pipeline

Each experiment script loads models **once** via `ModelRegistry` and releases them on exit — no redundant loads across steps.

### Step 0 — Download datasets

```bash
python data/download_datasets.py
```

### Step 1 — Norm Profiling

Measures per-layer residual stream L2 norms across MT-Bench prompts. Derives the calibrated K_ℓ = μ̄_ℓ / √d schedule used by all downstream experiments.

```bash
# All models
python experiments/01_norm_profiling.py

# Subset of models
python experiments/01_norm_profiling.py --models llama-3.1-8b-instruct mistral-7b-instruct

# 4-bit quantisation (lower VRAM, slightly less accurate norms)
python experiments/01_norm_profiling.py --load-in-4bit
```

**Outputs:**
- `results/norm_profiles/{model}.csv` — per-layer mean norm, std, and K_ℓ value
- `figures/norm_profiles/fig1_norm_profiles.pdf` — norm growth curves
- `figures/norm_profiles/fig2_k_profiles.pdf` — K_ℓ schedule per model
- `figures/norm_profiles/fig3_norm_comparison.pdf` — normalised comparison across all models

**Prerequisite:** `data/mtbench_questions.jsonl` (downloaded in Step 0)

---

### Step 2 — Direction Extraction

Extracts per-layer behavioral directions from contrastive prompt pairs using CAA. Activation differences are pooled over completion tokens only (not the shared context prefix) for a sharper signal.

```bash
# All behaviors for one model
python experiments/02_direction_extraction.py \
    --models llama-3.1-8b-instruct \
    --behaviors safety refusal sycophancy

# All models, all behaviors
python experiments/02_direction_extraction.py

# Single behavior
python experiments/02_direction_extraction.py \
    --models llama-3.1-8b-instruct \
    --behaviors safety
```

**Outputs:**
- `results/directions/{model}/{behavior}.npz` — PCA direction, mean direction, K values per layer

**Prerequisite:** Step 1 complete (norm profiles for the target models)

---

### Step 3 — Ramp Steering Evaluation

Evaluates three conditions on a single model × behavior pair: baseline (no steering), ramp_pos (steer toward behavior at +K_ℓ × scale), ramp_neg (steer away at −K_ℓ × scale). Tests on HarmBench + ClearHarm prompts. Used to locate the lobotomy cliff and validate the formula K.

```bash
# Single scale
python experiments/03_ramp_steering_eval.py \
    --model llama-3.1-8b-instruct \
    --behavior safety \
    --k-scale 1.0

# Sweep scales to map the lobotomy cliff
for scale in 0.5 1.0 2.0 3.0; do
    python experiments/03_ramp_steering_eval.py \
        --model llama-3.1-8b-instruct \
        --behavior safety \
        --k-scale $scale
done

# Different model or behavior
python experiments/03_ramp_steering_eval.py \
    --model mistral-7b-instruct \
    --behavior refusal \
    --k-scale 1.0
```

**Outputs** (under `results/ramp_eval/{model}/{behavior}/k{scale}/`):
- `raw_generations.csv` — all generated responses across three conditions
- `scored_results.csv` — responses with SAFE / UNSAFE / GIBBERISH judge scores
- `summary.csv` — mean, std, count per condition
- `figures/ramp_eval/.../condition_comparison.pdf`
- `figures/ramp_eval/.../by_source.pdf`
- `figures/ramp_eval/.../per_prompt_heatmap.pdf`

**Prerequisites:** Steps 1 and 2 complete; `data/eval_prompts/harmbench.jsonl` and `data/eval_prompts/clearharm.jsonl` (downloaded in Step 0)

---

### Step 4 — Baking Evaluation

Bakes behavioral directions into model weights in-place, runs the full benchmark suite on baseline, baked_pos, and baked_neg conditions, then unbakes to restore the original weights. No deepcopy — one model copy in VRAM at all times.

```bash
# Baking eval with formula K (scale=1.0)
python experiments/04_baking_eval.py \
    --model llama-3.1-8b-instruct \
    --behavior safety

# Different scale
python experiments/04_baking_eval.py \
    --model llama-3.1-8b-instruct \
    --behavior safety \
    --k-scale 2.0

# Save the baked model weights to disk
python experiments/04_baking_eval.py \
    --model llama-3.1-8b-instruct \
    --behavior safety \
    --save-baked-model
```

**Benchmarks run:**

| Benchmark | Metric | Split |
|---|---|---|
| HarmBench | safe_rate (SAFE / total) | `cais/harmbench` standard/test |
| GSM8K | exact_match | `gsm8k` main/test |
| MMLU | accuracy | `cais/mmlu` all/test |
| TruthfulQA | mc1_accuracy | `truthful_qa` multiple_choice/validation |

**Outputs** (under `results/baking_eval/{model}/{behavior}/k{scale}/`):
- `results.csv` — scores for all benchmarks × conditions
- `per_sample/{condition}_{benchmark}.csv` — per-sample predictions
- `figures/baking_eval/.../benchmark_comparison.pdf`
- `figures/baking_eval/.../delta_heatmap.pdf` — score delta (baked − baseline)

**Prerequisites:** Steps 1 and 2 complete; HuggingFace datasets cached (Step 0)

---

### Full pipeline — one model

```bash
# 0. Environment + datasets
bash setup.sh && source .venv/bin/activate
python data/download_datasets.py

# 1. Norm profiling
python experiments/01_norm_profiling.py --models llama-3.1-8b-instruct

# 2. Direction extraction
python experiments/02_direction_extraction.py \
    --models llama-3.1-8b-instruct \
    --behaviors safety refusal sycophancy

# 3. Ramp steering sweep
for scale in 0.5 1.0 2.0 3.0; do
    python experiments/03_ramp_steering_eval.py \
        --model llama-3.1-8b-instruct \
        --behavior safety \
        --k-scale $scale
done

# 4. Baking eval
python experiments/04_baking_eval.py \
    --model llama-3.1-8b-instruct \
    --behavior safety
```

---

## Key Results (Llama-3.1-8B-Instruct, safety)

| K scale | ramp_pos (safe %) | ramp_neg (safe %) | Baseline |
|---|---|---|---|
| 1.0 × K_ℓ | **70%** | 45% | 55% |
| 2.0 × K_ℓ | 55% | 15% | 55% |
| 3.0 × K_ℓ | 0% *(gibberish)* | 5% | 55% |

Formula K (1×) gives clean bidirectional control without degradation. At 3×K_ℓ ramp_pos collapses — confirming the lobotomy ceiling.

---

## Project Structure

```
activation_baking/
  config.py              ModelConfig, ExperimentConfig dataclasses
  registry.py            ModelRegistry — load once, reuse across all steps
  model_utils.py         Model loading, prompt formatting, response generation
  norm_profiler.py       Hook-based residual stream norm measurement
  direction_extractor.py Contrastive activation direction extraction (CAA)
  steerer.py             Inference-time hook-based steering (no weight changes)
  baker.py               Persistent baking — writes K·ĉ into W_down.bias
  judges.py              ActivationJudge + SmallModelJudge (SAFE/UNSAFE/GIBBERISH)
  evaluators.py          GSM8K, MMLU, TruthfulQA, HarmBench runners

config/
  models.yml             All 8 model variants (4 instruct + 4 base pairs)
  experiment.yml         Shared hyperparameters and sample counts

data/
  contrastive_pairs/     CAA minimal pairs for direction extraction (safety, refusal, sycophancy)
  download_datasets.py   Downloads MT-Bench, HarmBench, ClearHarm; pre-caches HF datasets

experiments/
  01_norm_profiling.py
  02_direction_extraction.py
  03_ramp_steering_eval.py
  04_baking_eval.py

paper/
  formula_derivation.md  Full K_ℓ = μ̄_ℓ / √d derivation
```

---

## Citation

```bibtex
@article{activationbaking2026,
  title  = {Activation Baking: Persistent Behavioral Modification via Weight-Space Direction Injection},
  author = {Kamesh R},
  year   = {2026},
}
```
