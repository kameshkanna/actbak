"""Experiment 01 — Per-layer residual stream norm profiling.

For each configured model:
  1. Load MT-Bench prompts (80 first-turn questions).
  2. Format with the model's chat template.
  3. Run NormProfiler to compute μ̄_ℓ and K_ℓ = μ̄_ℓ / √d per layer.
  4. Save results to results/norm_profiles/<model_name>.csv.
  5. Generate norm-growth and K_ℓ figures in figures/.

Usage:
    python experiments/01_norm_profiling.py \
        --config config/models.yml \
        --prompts data/mtbench_questions.jsonl \
        --output-dir results/norm_profiles \
        --figures-dir figures \
        [--load-in-4bit]
"""

import argparse
import gc
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from tqdm import tqdm

from activation_baking.model_utils import format_prompt, load_model_and_tokenizer
from activation_baking.norm_profiler import LayerNormStats, NormProfiler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

NORM_GROWTH_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def load_prompts(path: Path) -> list[str]:
    """Load raw user messages from a JSONL file with a 'prompt' field."""
    prompts = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            prompts.append(obj["prompt"])
    logger.info("Loaded %d prompts from %s", len(prompts), path)
    return prompts


def run_model(
    model_cfg: dict,
    raw_prompts: list[str],
    load_in_4bit: bool,
) -> list[LayerNormStats]:
    """Load one model, profile norms, unload, and return stats."""
    model, tokenizer = load_model_and_tokenizer(
        hf_id=model_cfg["hf_id"],
        dtype=model_cfg.get("dtype", "bfloat16"),
        load_in_4bit=load_in_4bit,
    )

    extra_cfg = model_cfg.get("extra", {})
    formatted = [format_prompt(tokenizer, p, extra_cfg) for p in raw_prompts]

    profiler = NormProfiler(
        model=model,
        tokenizer=tokenizer,
        hidden_size=model_cfg["hidden_size"],
        num_layers=model_cfg["num_layers"],
    )
    stats = profiler.profile(formatted)

    del model, tokenizer, profiler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return stats


def save_results(
    stats: list[LayerNormStats],
    model_name: str,
    output_dir: Path,
) -> Path:
    """Persist per-layer stats to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [s.to_dict() for s in stats]
    df = pd.DataFrame(records)
    df["model"] = model_name
    out_path = output_dir / f"{model_name}.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved %s", out_path)
    return out_path


def plot_norm_profiles(
    all_results: dict[str, list[LayerNormStats]],
    figures_dir: Path,
) -> None:
    """Generate norm-growth and K_ℓ figures across all models."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    model_names = list(all_results.keys())
    n = len(model_names)
    colors = NORM_GROWTH_PALETTE[:n]

    # --- Figure 1: raw norm growth per model ---
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, name, color in zip(axes, model_names, colors):
        stats = all_results[name]
        layers = [s.layer_idx for s in stats]
        means = np.array([s.mean_norm for s in stats])
        stds = np.array([s.std_norm for s in stats])

        ax.plot(layers, means, color=color, linewidth=2, label=name)
        ax.fill_between(layers, means - stds, means + stds, alpha=0.2, color=color)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean L2 norm (μ̄_ℓ)")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Residual stream norm growth across layers", fontsize=12)
    fig.tight_layout()
    out = figures_dir / "fig1_norm_profiles.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out)

    # --- Figure 2: K_ℓ = μ̄_ℓ / √d per model ---
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, name, color in zip(axes, model_names, colors):
        stats = all_results[name]
        layers = [s.layer_idx for s in stats]
        k_values = [s.k_value for s in stats]

        ax.plot(layers, k_values, color=color, linewidth=2)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Layer")
        ax.set_ylabel("K_ℓ = μ̄_ℓ / √d")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Calibrated K values per layer (K_ℓ = μ̄_ℓ / √d)", fontsize=12)
    fig.tight_layout()
    out = figures_dir / "fig2_k_profiles.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out)

    # --- Figure 3: normalised comparison (all models, layer depth 0–1) ---
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, color in zip(model_names, colors):
        stats = all_results[name]
        n_layers = len(stats)
        depth = np.array([s.layer_idx / (n_layers - 1) for s in stats])
        means = np.array([s.mean_norm for s in stats])
        means_norm = means / means[0]
        ax.plot(depth, means_norm, color=color, linewidth=2, label=name)

    ax.set_xlabel("Relative layer depth")
    ax.set_ylabel("Norm (relative to layer 0)")
    ax.set_title("Normalised norm growth — all models")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = figures_dir / "fig3_norm_comparison.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 01 — norm profiling")
    parser.add_argument("--config", default="config/models.yml")
    parser.add_argument("--prompts", default="data/mtbench_questions.jsonl")
    parser.add_argument("--output-dir", default="results/norm_profiles")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--models", nargs="*", help="Subset of model names to run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfgs: list[dict] = cfg["models"]
    if args.models:
        model_cfgs = [m for m in model_cfgs if m["name"] in args.models]
        if not model_cfgs:
            logger.error("No models matched: %s", args.models)
            sys.exit(1)

    raw_prompts = load_prompts(Path(args.prompts))
    output_dir = Path(args.output_dir)
    figures_dir = Path(args.figures_dir)

    all_results: dict[str, list[LayerNormStats]] = {}

    for model_cfg in model_cfgs:
        name = model_cfg["name"]
        logger.info("=== %s ===", name)
        stats = run_model(model_cfg, raw_prompts, args.load_in_4bit)
        save_results(stats, name, output_dir)
        all_results[name] = stats

        logger.info(
            "%s — layer 0 norm: %.2f | final layer norm: %.2f | K range: %.4f–%.4f",
            name,
            stats[0].mean_norm,
            stats[-1].mean_norm,
            stats[0].k_value,
            stats[-1].k_value,
        )

    plot_norm_profiles(all_results, figures_dir)
    logger.info("Experiment 01 complete.")


if __name__ == "__main__":
    main()
