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

### Dataset download

```bash
# Download everything (reads sample counts from config/experiment.yml)
python data/download_datasets.py

# Or selectively
python data/download_datasets.py --only mtbench harmbench_eval clearharm_eval
python data/download_datasets.py --only gsm8k mmlu truthfulqa harmbench_hf

# Override sample counts
python data/download_datasets.py --n-harmbench 100 --n-clearharm 100 --seed 0
```

| Dataset | Used by | Source |
|---|---|---|
| MT-Bench 80 prompts | Experiment 01 (norm profiling) | FastChat GitHub |
| HarmBench eval JSONL | Experiment 03 (ramp steering) | HarmBench GitHub CSV |
| ClearHarm eval JSONL | Experiment 03 (ramp steering) | HuggingFace |
| GSM8K, MMLU, TruthfulQA, HarmBench | Experiment 04 (baking eval) | HuggingFace (auto-cached) |

---

## Pipeline

Each script loads a model **once** via `ModelRegistry` — no redundant loads across steps.

### 1 — Norm Profiling

```bash
# Profile all models
python experiments/01_norm_profiling.py

# Or single model
python experiments/01_norm_profiling.py --models llama-3.1-8b-instruct
```

Outputs: `results/norm_profiles/{model}.csv`, figures in `figures/norm_profiles/`

### 2 — Direction Extraction

```bash
python experiments/02_direction_extraction.py \
    --model llama-3.1-8b-instruct \
    --behaviors safety refusal sycophancy
```

Outputs: `results/directions/{model}/{behavior}.npz`

### 3 — Ramp Steering Evaluation

```bash
# Sweep K scales to find the formula ceiling and lobotomy cliff
for scale in 1.0 2.0 3.0; do
    python experiments/03_ramp_steering_eval.py \
        --model llama-3.1-8b-instruct \
        --behavior safety \
        --k-scale $scale
done
```

Outputs: `results/ramp_eval/{model}/{behavior}/k{scale}/`

### 4 — Baking Evaluation

```bash
python experiments/04_baking_eval.py \
    --model llama-3.1-8b-instruct \
    --behavior safety
```

Bakes directions into weights, runs full benchmark suite (GSM8K, MMLU, TruthfulQA, HarmBench).
Outputs: `results/baking_eval/{model}/{behavior}/`

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
  judges.py              SAFE / UNSAFE / GIBBERISH judge
  evaluators.py          GSM8K, MMLU, TruthfulQA, HarmBench runners

config/
  models.yml             All model variants with instruct/base pairs
  experiment.yml         Shared hyperparameters

data/
  contrastive_pairs/     CAA minimal pairs for direction extraction
  test_prompts/          Held-out evaluation prompts

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
