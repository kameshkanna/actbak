"""Experiment 01 — Per-layer residual stream norm profiling.

For each configured model:
  1. Load MT-Bench prompts (80 first-turn questions).
  2. Format with the model's chat template.
  3. Run NormProfiler to compute μ̄_ℓ and K_ℓ = μ̄_ℓ / √d per layer.
  4. Save results to results/norm_profiles/<model_name>.csv.
  5. Generate norm-growth and K_ℓ figures in figures/norm_profiles/.

Usage:
    # Profile all models
    python experiments/01_norm_profiling.py --config config/models.yml

    # Profile a subset
    python experiments/01_norm_profiling.py --models llama-3.1-8b-instruct mistral-7b-instruct
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from activation_baking.config import ModelConfig, load_model_configs
from activation_baking.model_utils import format_prompt
from activation_baking.norm_profiler import LayerNormStats, NormProfiler, save_profile
from activation_baking.registry import ModelRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860", "#DA8BC3", "#8C8C8C"]


def load_prompts(path: Path) -> list[str]:
    """Load raw user messages from a JSONL file with a 'prompt' field."""
    prompts: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            prompts.append(obj["prompt"])
    logger.info("Loaded %d prompts from %s", len(prompts), path)
    return prompts


def run_model(
    model_cfg: ModelConfig,
    raw_prompts: list[str],
    registry: ModelRegistry,
) -> list[LayerNormStats]:
    """Profile a single model via the registry and return per-layer stats."""
    with registry.loaded(model_cfg) as (model, tokenizer):
        formatted = [format_prompt(tokenizer, p, model_cfg.extra) for p in raw_prompts]
        profiler = NormProfiler(model=model, tokenizer=tokenizer, model_cfg=model_cfg)
        return profiler.profile(formatted)


def plot_norm_profiles(
    all_results: dict[str, list[LayerNormStats]],
    figures_dir: Path,
) -> None:
    """Generate norm-growth and K_ℓ figures across all models."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    names = list(all_results.keys())
    n = len(names)
    colors = _PALETTE[:n]

    # Figure 1: raw norm growth per model
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, name, color in zip(axes, names, colors):
        stats = all_results[name]
        layers = [s.layer_idx for s in stats]
        means = np.array([s.mean_norm for s in stats])
        stds = np.array([s.std_norm for s in stats])
        ax.plot(layers, means, color=color, linewidth=2)
        ax.fill_between(layers, means - stds, means + stds, alpha=0.2, color=color)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean L2 norm (μ̄_ℓ)")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Residual stream norm growth", fontsize=12)
    fig.tight_layout()
    _savefig(fig, figures_dir / "fig1_norm_profiles.pdf")

    # Figure 2: K_ℓ per model
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, name, color in zip(axes, names, colors):
        stats = all_results[name]
        ax.plot([s.layer_idx for s in stats], [s.k_value for s in stats],
                color=color, linewidth=2)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Layer")
        ax.set_ylabel("K_ℓ = μ̄_ℓ / √d")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Calibrated K per layer (K_ℓ = μ̄_ℓ / √d)", fontsize=12)
    fig.tight_layout()
    _savefig(fig, figures_dir / "fig2_k_profiles.pdf")

    # Figure 3: normalised comparison across all models
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, color in zip(names, colors):
        stats = all_results[name]
        n_layers = len(stats)
        depth = np.array([s.layer_idx / max(n_layers - 1, 1) for s in stats])
        means = np.array([s.mean_norm for s in stats])
        ax.plot(depth, means / means[0], color=color, linewidth=2, label=name)
    ax.set_xlabel("Relative layer depth")
    ax.set_ylabel("Norm (relative to layer 0)")
    ax.set_title("Normalised norm growth — all models")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _savefig(fig, figures_dir / "fig3_norm_comparison.pdf")


def _savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 01 — norm profiling")
    parser.add_argument("--config", default="config/models.yml")
    parser.add_argument("--prompts", default="data/mtbench_questions.jsonl")
    parser.add_argument("--output-dir", default="results/norm_profiles")
    parser.add_argument("--figures-dir", default="figures/norm_profiles")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument(
        "--device", default="cuda:0",
        help="Single GPU to use for norm profiling (default: cuda:0).",
    )
    parser.add_argument(
        "--models", nargs="*",
        help="Subset of model names to run (default: all in config)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    all_configs: dict[str, ModelConfig] = load_model_configs(args.config)
    if args.models:
        missing = set(args.models) - set(all_configs)
        if missing:
            logger.error("Unknown model name(s): %s. Available: %s", missing, list(all_configs))
            sys.exit(1)
        selected = {n: all_configs[n] for n in args.models}
    else:
        selected = all_configs

    prompts_path = Path(args.prompts)
    if not prompts_path.exists():
        logger.error(
            "%s not found — run `python data/download_mtbench.py` first.", prompts_path
        )
        sys.exit(1)

    raw_prompts = load_prompts(prompts_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    registry = ModelRegistry(load_in_4bit=args.load_in_4bit, device_map=args.device)
    all_results: dict[str, list[LayerNormStats]] = {}

    for name, model_cfg in selected.items():
        logger.info("=== Profiling: %s ===", name)
        stats = run_model(model_cfg, raw_prompts, registry)

        out_path = output_dir / f"{name}.csv"
        save_profile(stats, out_path)
        all_results[name] = stats

        logger.info(
            "%s — layers=%d | norm: layer0=%.2f → final=%.2f | K: [%.4f, %.4f]",
            name, len(stats),
            stats[0].mean_norm, stats[-1].mean_norm,
            stats[0].k_value, stats[-1].k_value,
        )

    plot_norm_profiles(all_results, Path(args.figures_dir))
    logger.info("Experiment 01 complete.")


if __name__ == "__main__":
    main()
