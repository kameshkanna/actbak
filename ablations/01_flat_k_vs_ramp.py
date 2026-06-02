"""Ablation 01 — Flat-K steering vs ramp-K steering across all middle layers.

Proves that ramp-K is necessary by showing flat-K fails:

  baseline  — no steering
  flat_K    — all middle layers steered with fixed K = mean(K_ℓ)
  ramp_K    — all middle layers steered with K_ℓ = μ̄_ℓ / √d  (our formula)

Flat-K loses because residual stream norms grow monotonically with depth.
A K calibrated for shallow layers gets drowned at deep layers where ‖h_ℓ‖
is large — relative influence K/‖h_ℓ‖ → 0. Ramp-K maintains constant
relative influence at every layer.

Outputs:
  results/ablations/flat_k_vs_ramp/{model}/scored_results.csv
  results/ablations/flat_k_vs_ramp/{model}/summary.csv
  figures/ablations/flat_k_vs_ramp/{model}/condition_comparison.pdf
  figures/ablations/flat_k_vs_ramp/all_models_summary.pdf

Usage:
    python ablations/01_flat_k_vs_ramp.py
    python ablations/01_flat_k_vs_ramp.py --models llama-3.1-8b-instruct
    python ablations/01_flat_k_vs_ramp.py --n-prompts 25
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from activation_baking.config import ExperimentConfig, ModelConfig, load_experiment_config, load_model_configs
from activation_baking.direction_extractor import BehavioralDirection, load_directions
from activation_baking.judges import SmallModelJudge
from activation_baking.model_utils import format_prompt
from activation_baking.norm_profiler import LayerNormStats, load_profile
from activation_baking.registry import ModelRegistry
from activation_baking.steerer import ActivationSteerer

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CONDITIONS       = ["baseline", "flat_K", "ramp_K"]
CONDITION_COLORS = {"baseline": "#6b7280", "flat_K": "#f59e0b", "ramp_K": "#2563eb"}
CONDITION_LABELS = {"baseline": "Baseline", "flat_K": "Flat-K (mean)", "ramp_K": "Ramp-K (ours)"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_harmbench_prompts(eval_dir: Path, n: int) -> list[dict]:
    """Load the first ``n`` HarmBench prompts from the eval JSONL.

    Args:
        eval_dir: Directory containing ``harmbench.jsonl``.
        n:        Maximum number of prompts to load.

    Returns:
        List of dicts with ``source``, ``id``, ``prompt`` keys.

    Raises:
        SystemExit: If the JSONL file is missing.
    """
    path = eval_dir / "harmbench.jsonl"
    if not path.exists():
        logger.error(
            "harmbench.jsonl not found at %s — run `python data/download_datasets.py` first.", path
        )
        sys.exit(1)
    records: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
            if len(records) >= n:
                break
    logger.info("Loaded %d HarmBench prompts.", len(records))
    return records


def load_norm_profile(norm_dir: Path, model_name: str) -> dict[int, LayerNormStats]:
    """Load norm profile CSV and index by layer_idx.

    Args:
        norm_dir:   Root directory for per-model CSV norm profiles.
        model_name: Short model slug.

    Returns:
        Mapping of ``layer_idx → LayerNormStats``.

    Raises:
        SystemExit: If the CSV is missing.
    """
    path = norm_dir / f"{model_name}.csv"
    if not path.exists():
        logger.error("Norm profile missing: %s — run experiment 01 first.", path)
        sys.exit(1)
    return {s.layer_idx: s for s in load_profile(path)}


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_batched(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    steerer: ActivationSteerer,
    layer_config: dict[int, tuple[np.ndarray, float]],
    prompts: list[str],
    max_new_tokens: int,
    batch_size: int,
    extra_cfg: dict[str, Any],
) -> list[str]:
    """Generate responses in batches under an optional steering config.

    Args:
        model:          Loaded causal LM.
        tokenizer:      Corresponding tokenizer.
        steerer:        ActivationSteerer bound to the model.
        layer_config:   Steering config; empty dict = no steering.
        prompts:        Raw user prompt strings.
        max_new_tokens: Token budget per prompt.
        batch_size:     Prompts per forward pass.
        extra_cfg:      Per-model extras forwarded to ``format_prompt``.

    Returns:
        Decoded response strings in the same order as ``prompts``.
    """
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    device = next(model.parameters()).device
    responses: list[str] = []

    for start in tqdm(
        range(0, len(prompts), batch_size), desc="  batches", leave=False, dynamic_ncols=True
    ):
        batch = [
            format_prompt(tokenizer, p, extra_cfg)
            for p in prompts[start : start + batch_size]
        ]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=1024
        ).to(device)
        prompt_len = inputs["input_ids"].shape[1]

        with steerer.steer(layer_config):
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        for output_ids in out:
            responses.append(
                tokenizer.decode(output_ids[prompt_len:], skip_special_tokens=True).strip()
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return responses


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_condition_comparison(
    summary: pd.Series,
    model_name: str,
    fig_dir: Path,
) -> None:
    """Bar chart: baseline vs flat_K vs ramp_K mean judge scores.

    Args:
        summary:    Series indexed by condition with mean judge scores.
        model_name: Short model slug (used in title).
        fig_dir:    Output directory.
    """
    vals = summary.reindex(CONDITIONS).fillna(0)
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        range(3), vals.values,
        color=[CONDITION_COLORS[c] for c in CONDITIONS],
        width=0.5,
    )
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=10)
    ax.set_xticks(range(3))
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITIONS], fontsize=10)
    ax.set_ylabel("Mean judge score (safety)")
    ax.set_ylim(0, 1.15)
    ax.set_title(f"{model_name} — Flat-K vs Ramp-K")
    ax.grid(True, axis="y", alpha=0.3)
    _savefig(fig, fig_dir / "condition_comparison.pdf")


def plot_norm_growth_with_k(
    profile: dict[int, LayerNormStats],
    directions: list[BehavioralDirection],
    model_name: str,
    fig_dir: Path,
) -> None:
    """Plot ‖h_ℓ‖, flat K, and ramp K across all layers.

    Illustrates why flat-K loses relative influence at depth while ramp-K
    tracks the norm curve.

    Args:
        profile:    Full norm profile indexed by layer_idx.
        directions: Extracted directions (provides ramp K_ℓ per layer).
        model_name: Short model slug.
        fig_dir:    Output directory.
    """
    sorted_layers = sorted(profile)
    norms    = np.array([profile[l].mean_norm for l in sorted_layers])
    ramp_ks  = np.array([profile[l].k_value   for l in sorted_layers])

    dir_map  = {d.layer_idx: d for d in directions}
    middle   = sorted(dir_map)
    flat_k   = float(np.mean([dir_map[l].k_value for l in middle]))

    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax2 = ax1.twinx()

    ax1.fill_between(sorted_layers, norms, alpha=0.12, color="#6b7280")
    ax1.plot(sorted_layers, norms, color="#6b7280", linewidth=1.5, label="‖h_ℓ‖ mean norm")

    ax2.plot(sorted_layers, ramp_ks,              color="#2563eb", linewidth=2,   label="Ramp-K  K_ℓ = μ̄_ℓ/√d")
    ax2.axhline(flat_k, color="#f59e0b", linewidth=2, linestyle="--", label=f"Flat-K  K={flat_k:.3f}")

    ax1.set_xlabel("Layer index")
    ax1.set_ylabel("Mean ‖h_ℓ‖", color="#6b7280")
    ax2.set_ylabel("Steering magnitude K", color="#2563eb")
    ax1.set_title(f"{model_name} — Relative influence: K / ‖h_ℓ‖")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.2)
    _savefig(fig, fig_dir / "norm_and_k_growth.pdf")


def plot_all_models_summary(
    all_summaries: dict[str, pd.Series],
    fig_dir: Path,
) -> None:
    """Grouped bar chart comparing all models side by side.

    Args:
        all_summaries: Mapping of ``model_name → Series(condition → mean_score)``.
        fig_dir:       Output directory.
    """
    model_names = list(all_summaries)
    x = np.arange(len(model_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(7, len(model_names) * 2.5), 5))
    for i, condition in enumerate(CONDITIONS):
        scores = [all_summaries[m].get(condition, 0.0) for m in model_names]
        bars = ax.bar(
            x + (i - 1) * width, scores, width,
            label=CONDITION_LABELS[condition],
            color=CONDITION_COLORS[condition],
        )
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("-instruct", "") for m in model_names], fontsize=9)
    ax.set_ylabel("Mean judge score (safety)")
    ax.set_ylim(0, 1.2)
    ax.set_title("Flat-K vs Ramp-K — all models")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    _savefig(fig, fig_dir / "all_models_summary.pdf")


# ---------------------------------------------------------------------------
# Per-model runner
# ---------------------------------------------------------------------------

def run_model(
    model_cfg: ModelConfig,
    eval_prompts: list[dict],
    profile: dict[int, LayerNormStats],
    directions: list[BehavioralDirection],
    output_root: Path,
    figures_root: Path,
    exp_cfg: ExperimentConfig,
    registry: ModelRegistry,
    batch_size: int,
) -> pd.Series:
    """Run flat-K vs ramp-K ablation for one model.

    Returns a Series of mean judge scores indexed by condition, so the
    caller can aggregate a cross-model summary plot.

    Args:
        model_cfg:    Config for the model.
        eval_prompts: HarmBench prompt records.
        profile:      Norm profile indexed by layer_idx.
        directions:   Extracted safety behavioral directions.
        output_root:  CSV output root.
        figures_root: Figure output root.
        exp_cfg:      Loaded ExperimentConfig.
        registry:     ModelRegistry instance.
        batch_size:   Generation batch size.

    Returns:
        ``pd.Series`` of mean judge scores keyed by condition name.
    """
    logger.info("=== %s ===", model_cfg.name)

    flat_k_cfg = steerer_ref = None  # filled inside context
    ramp_cfg   = {d.layer_idx: (d.mean_direction, d.k_value) for d in directions}
    flat_k_val = float(np.mean([d.k_value for d in directions]))
    flat_cfg   = {d.layer_idx: (d.mean_direction, flat_k_val) for d in directions}

    prompts = [item["prompt"] for item in eval_prompts]
    judge   = SmallModelJudge(model_id=exp_cfg.judge_model)
    records: list[dict] = []

    with registry.loaded(model_cfg) as (model, tokenizer):
        steerer = ActivationSteerer(model)

        for condition, layer_config in [
            ("baseline", {}),
            ("flat_K",   flat_cfg),
            ("ramp_K",   ramp_cfg),
        ]:
            logger.info("  [%s | %s]", model_cfg.name, condition)
            resps = generate_batched(
                model, tokenizer, steerer, layer_config,
                prompts, exp_cfg.max_new_tokens, batch_size, model_cfg.extra,
            )
            for item, resp in zip(eval_prompts, resps):
                records.append({
                    "model_name": model_cfg.name,
                    "condition":  condition,
                    "behavior":   "safety",
                    "source":     item["source"],
                    "prompt_id":  item["id"],
                    "user_input": item["prompt"],
                    "response":   resp,
                })

    scores = judge.score_records(records, batch_size=exp_cfg.judge_batch_size)
    judge.unload()
    for rec, score in zip(records, scores):
        rec["judge_score"] = score

    df = pd.DataFrame(records)
    out_dir = output_root / model_cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "scored_results.csv", index=False)

    summary = df.groupby("condition")["judge_score"].agg(["mean", "std", "count"])
    summary.columns = ["mean_score", "std_score", "n"]
    summary.to_csv(out_dir / "summary.csv")
    logger.info("=== %s ===\n%s", model_cfg.name, summary.to_string())

    fig_dir = figures_root / model_cfg.name
    plot_condition_comparison(summary["mean_score"], model_cfg.name, fig_dir)
    plot_norm_growth_with_k(profile, directions, model_cfg.name, fig_dir)

    return summary["mean_score"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablation 01 — Flat-K vs Ramp-K")
    parser.add_argument("--config",            default="config/models.yml")
    parser.add_argument("--experiment-config", default="config/experiment.yml")
    parser.add_argument("--norm-profiles",     default="results/norm_profiles")
    parser.add_argument("--directions",        default="results/directions")
    parser.add_argument("--eval-prompts",      default="data/eval_prompts")
    parser.add_argument("--output-dir",        default="results/ablations/flat_k_vs_ramp")
    parser.add_argument("--figures-dir",       default="figures/ablations/flat_k_vs_ramp")
    parser.add_argument("--batch-size",  type=int, default=64)
    parser.add_argument("--n-prompts",   type=int, default=None, help="Override ablation_n_prompts.")
    parser.add_argument("--models",    nargs="*",               help="Default: all instruct models.")
    parser.add_argument("--load-in-4bit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args     = parse_args()
    all_configs = load_model_configs(args.config)
    exp_cfg     = load_experiment_config(args.experiment_config)
    n_prompts   = args.n_prompts if args.n_prompts is not None else exp_cfg.ablation_n_prompts

    if args.models:
        missing = set(args.models) - set(all_configs)
        if missing:
            logger.error("Unknown models: %s", missing)
            sys.exit(1)
        target_names = args.models
    else:
        target_names = [n for n, c in all_configs.items() if c.is_instruct]

    eval_prompts = load_harmbench_prompts(Path(args.eval_prompts), n_prompts)
    registry     = ModelRegistry(load_in_4bit=args.load_in_4bit, device_map="cuda:0")

    logger.info(
        "%d models | %d prompts | conditions: %s",
        len(target_names), n_prompts, CONDITIONS,
    )

    all_summaries: dict[str, pd.Series] = {}
    for name in target_names:
        profile         = load_norm_profile(Path(args.norm_profiles), name)
        directions_path = Path(args.directions) / name / "safety.npz"
        if not directions_path.exists():
            logger.error("Directions missing: %s — run experiment 02 first.", directions_path)
            continue
        directions = load_directions(str(directions_path))

        mean_scores = run_model(
            model_cfg=all_configs[name],
            eval_prompts=eval_prompts,
            profile=profile,
            directions=directions,
            output_root=Path(args.output_dir),
            figures_root=Path(args.figures_dir),
            exp_cfg=exp_cfg,
            registry=registry,
            batch_size=args.batch_size,
        )
        all_summaries[name] = mean_scores

    if len(all_summaries) > 1:
        plot_all_models_summary(
            all_summaries,
            Path(args.figures_dir),
        )
        logger.info("Cross-model summary plot saved to %s", args.figures_dir)

    logger.info("Ablation 01 complete.")


if __name__ == "__main__":
    main()
