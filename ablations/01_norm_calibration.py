"""Ablation 01 — Single-layer steering with K values relative to ‖h_ℓ‖.

Demonstrates that K_ℓ = μ̄_ℓ / √d is the correct operating point for each layer.

For each model × layer in the 40–80% depth range:
  K_low   = 0.1 × K_ℓ  →  K << ‖h_ℓ‖, relative influence ≈ 0, near-baseline output
  K_ramp  = 1.0 × K_ℓ  →  formula value, constant relative influence, effective steering
  K_high  = 10.0 × K_ℓ →  K >> ‖h_ℓ‖, residual stream blown out, incoherent output

Norm growth data from exp01 is overlaid on the judge score plots to make the
mechanistic argument visually explicit: score degrades exactly where K/‖h_ℓ‖ goes wrong.

Outputs:
  results/ablations/norm_calibration/{model}/results.csv
  figures/ablations/norm_calibration/{model}/score_heatmap.pdf
  figures/ablations/norm_calibration/{model}/score_vs_kscale.pdf
  figures/ablations/norm_calibration/{model}/norm_growth.pdf

Usage:
    python ablations/01_norm_calibration.py
    python ablations/01_norm_calibration.py --models llama-3.1-8b-instruct
    python ablations/01_norm_calibration.py --n-prompts 25 --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from activation_baking.config import ExperimentConfig, ModelConfig, load_experiment_config, load_model_configs
from typing import Any

from activation_baking.direction_extractor import load_directions
from activation_baking.judges import SmallModelJudge
from activation_baking.model_utils import format_prompt
from activation_baking.norm_profiler import load_profile, LayerNormStats
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

K_SCALE_LABELS = {0.1: "K_low (0.1×)", 1.0: "K_ramp (1.0×)", 10.0: "K_high (10.0×)"}
K_SCALE_COLORS = {0.1: "#f59e0b", 1.0: "#2563eb", 10.0: "#dc2626"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_harmbench_prompts(eval_dir: Path, n: int) -> list[dict]:
    """Load the first ``n`` HarmBench prompts from the eval JSONL.

    Args:
        eval_dir: Directory containing ``harmbench.jsonl``.
        n:        Maximum number of prompts to load.

    Returns:
        List of prompt dicts (``source``, ``id``, ``prompt``).

    Raises:
        SystemExit: If the JSONL file is missing.
    """
    path = eval_dir / "harmbench.jsonl"
    if not path.exists():
        logger.error("harmbench.jsonl not found at %s — run `python data/download_datasets.py`.", path)
        sys.exit(1)
    records: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
            if len(records) >= n:
                break
    logger.info("Loaded %d HarmBench prompts for ablation.", len(records))
    return records


def load_norm_profile(
    norm_dir: Path,
    model_name: str,
) -> dict[int, LayerNormStats]:
    """Load norm profile CSV and index by layer_idx.

    Args:
        norm_dir:   Root directory containing per-model CSV files.
        model_name: Short model slug.

    Returns:
        Mapping of ``layer_idx → LayerNormStats``.

    Raises:
        SystemExit: If the CSV is missing (exp01 has not been run).
    """
    path = norm_dir / f"{model_name}.csv"
    if not path.exists():
        logger.error("Norm profile missing: %s — run experiment 01 first.", path)
        sys.exit(1)
    stats = load_profile(path)
    return {s.layer_idx: s for s in stats}


def resolve_ablation_layers(num_layers: int, fracs: list[float]) -> list[int]:
    """Convert fractional depth targets to concrete layer indices.

    Args:
        num_layers: Total transformer layers in the model.
        fracs:      Depth fractions, e.g. ``[0.4, 0.5, 0.6, 0.7, 0.8]``.

    Returns:
        Unique, sorted list of layer indices.
    """
    return sorted(set(int(f * num_layers) for f in fracs))


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
        layer_config:   Steering config; empty dict = baseline.
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

    for start in tqdm(range(0, len(prompts), batch_size), desc="  batches", leave=False, dynamic_ncols=True):
        batch = [format_prompt(tokenizer, p, extra_cfg) for p in prompts[start : start + batch_size]]
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


def plot_norm_growth(
    profile: dict[int, LayerNormStats],
    model_name: str,
    ablation_layers: list[int],
    fig_dir: Path,
) -> None:
    """Plot mean residual stream norm and K_ℓ across all layers.

    Ablation target layers are highlighted with vertical dashed lines.

    Args:
        profile:         Full norm profile indexed by layer_idx.
        model_name:      Short model slug (used in title).
        ablation_layers: Layer indices used in the ablation.
        fig_dir:         Output directory.
    """
    sorted_layers = sorted(profile)
    norms    = [profile[l].mean_norm for l in sorted_layers]
    k_values = [profile[l].k_value   for l in sorted_layers]

    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax2 = ax1.twinx()

    ax1.plot(sorted_layers, norms,    color="#6b7280", linewidth=1.5, label="‖h_ℓ‖ (mean norm)")
    ax2.plot(sorted_layers, k_values, color="#2563eb", linewidth=1.5, linestyle="--", label="K_ℓ = μ̄_ℓ/√d")

    for l in ablation_layers:
        ax1.axvline(l, color="#dc2626", linestyle=":", alpha=0.7, linewidth=1.2)

    ax1.set_xlabel("Layer index")
    ax1.set_ylabel("Mean ‖h_ℓ‖", color="#6b7280")
    ax2.set_ylabel("K_ℓ", color="#2563eb")
    ax1.set_title(f"{model_name} — Residual stream norm growth")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.25)
    _savefig(fig, fig_dir / "norm_growth.pdf")


def plot_score_heatmap(
    df: pd.DataFrame,
    model_name: str,
    ablation_layers: list[int],
    k_scales: list[float],
    fig_dir: Path,
) -> None:
    """Heatmap of mean judge score at each (layer, k_scale) cell.

    Args:
        df:              Results DataFrame with columns ``layer_idx``, ``k_scale``, ``judge_score``.
        model_name:      Short model slug.
        ablation_layers: Ordered layer indices (x-axis).
        k_scales:        Ordered K scale values (y-axis).
        fig_dir:         Output directory.
    """
    pivot = (
        df.groupby(["k_scale", "layer_idx"])["judge_score"]
        .mean()
        .unstack(level="layer_idx")
        .reindex(index=k_scales, columns=ablation_layers)
        .fillna(0.0)
    )
    layer_pcts = [f"{int(l / df['num_layers'].iloc[0] * 100)}%" for l in ablation_layers]

    fig, ax = plt.subplots(figsize=(max(5, len(ablation_layers) * 1.2), 3.5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(ablation_layers)))
    ax.set_xticklabels(layer_pcts, fontsize=9)
    ax.set_yticks(range(len(k_scales)))
    ax.set_yticklabels([K_SCALE_LABELS.get(k, str(k)) for k in k_scales], fontsize=9)
    ax.set_xlabel("Layer depth")
    ax.set_ylabel("K scale")
    ax.set_title(f"{model_name} — Ramp+ judge score | single-layer steering")

    for r in range(len(k_scales)):
        for c in range(len(ablation_layers)):
            val = pivot.values[r, c]
            ax.text(c, r, f"{val:.2f}", ha="center", va="center", fontsize=8,
                    color="black" if 0.3 < val < 0.7 else "white")

    plt.colorbar(im, ax=ax, label="Mean judge score")
    _savefig(fig, fig_dir / "score_heatmap.pdf")


def plot_score_vs_kscale(
    df: pd.DataFrame,
    model_name: str,
    ablation_layers: list[int],
    k_scales: list[float],
    profile: dict[int, LayerNormStats],
    fig_dir: Path,
) -> None:
    """Line plot: judge score vs K scale for each ablation layer, with norm annotation.

    Args:
        df:              Results DataFrame.
        model_name:      Short model slug.
        ablation_layers: Layer indices to plot.
        k_scales:        K scale values (x-axis).
        profile:         Norm profile for norm annotation.
        fig_dir:         Output directory.
    """
    num_layers = df["num_layers"].iloc[0]
    fig, axes = plt.subplots(1, len(ablation_layers), figsize=(3.5 * len(ablation_layers), 4), sharey=True)
    if len(ablation_layers) == 1:
        axes = [axes]

    for ax, layer in zip(axes, ablation_layers):
        sub = df[df["layer_idx"] == layer]
        scores = sub.groupby("k_scale")["judge_score"].mean().reindex(k_scales)
        baseline_mean = df[df["k_scale"].isna()]["judge_score"].mean() if any(df["k_scale"].isna()) else None

        ax.plot(
            range(len(k_scales)), scores.values,
            marker="o", color="#2563eb", linewidth=2, markersize=7,
        )
        if baseline_mean is not None:
            ax.axhline(baseline_mean, color="#6b7280", linestyle="--", linewidth=1.2, label="baseline")

        ax.set_xticks(range(len(k_scales)))
        ax.set_xticklabels([K_SCALE_LABELS.get(k, str(k)) for k in k_scales], fontsize=7, rotation=20)
        ax.set_ylim(-0.1, 1.15)
        pct = int(layer / num_layers * 100)
        mean_norm = profile[layer].mean_norm if layer in profile else float("nan")
        ax.set_title(f"Layer {layer} ({pct}%)\n‖h_ℓ‖={mean_norm:.1f}", fontsize=9)
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("Mean judge score (ramp+)")
    fig.suptitle(f"{model_name} — Score vs K scale at ablation layers", fontsize=11)
    plt.tight_layout()
    _savefig(fig, fig_dir / "score_vs_kscale.pdf")


# ---------------------------------------------------------------------------
# Per-model ablation runner
# ---------------------------------------------------------------------------

def run_model(
    model_cfg: ModelConfig,
    ablation_prompts: list[dict],
    profile: dict[int, LayerNormStats],
    directions: list,
    k_scales: list[float],
    output_root: Path,
    figures_root: Path,
    exp_cfg: ExperimentConfig,
    registry: ModelRegistry,
    batch_size: int,
) -> None:
    """Run norm calibration ablation for one model.

    Evaluates single-layer ramp+ steering at each ablation layer × K scale.
    Baseline (no steering) is run once and appended to the results.

    Args:
        model_cfg:       Config for the model.
        ablation_prompts: HarmBench prompts for this ablation.
        profile:         Norm profile indexed by layer_idx.
        directions:      Extracted behavioral directions for safety.
        k_scales:        K scale multipliers.
        output_root:     CSV output root.
        figures_root:    Figure output root.
        exp_cfg:         Loaded ExperimentConfig.
        registry:        ModelRegistry instance.
        batch_size:      Generation batch size.
    """
    ablation_layers = resolve_ablation_layers(
        model_cfg.num_layers, exp_cfg.ablation_layer_fracs
    )
    logger.info(
        "=== %s | ablation layers: %s ===", model_cfg.name, ablation_layers
    )

    # Validate that all ablation layers have extracted directions
    dir_map = {d.layer_idx: d for d in directions}
    missing = [l for l in ablation_layers if l not in dir_map]
    if missing:
        logger.warning(
            "Layers %s not in extracted directions for %s — they will be skipped.",
            missing, model_cfg.name,
        )
        ablation_layers = [l for l in ablation_layers if l in dir_map]
    if not ablation_layers:
        logger.error("No valid ablation layers for %s — skipping.", model_cfg.name)
        return

    prompts = [item["prompt"] for item in ablation_prompts]
    judge = SmallModelJudge(model_id=exp_cfg.judge_model)
    records: list[dict] = []

    with registry.loaded(model_cfg) as (model, tokenizer):
        steerer = ActivationSteerer(model)

        # Baseline — no steering
        logger.info("  [%s | baseline]", model_cfg.name)
        baseline_resps = generate_batched(
            model, tokenizer, steerer, {},
            prompts, exp_cfg.max_new_tokens, batch_size, model_cfg.extra,
        )
        for item, resp in zip(ablation_prompts, baseline_resps):
            records.append({
                "model_name": model_cfg.name,
                "num_layers": model_cfg.num_layers,
                "layer_idx":  None,
                "layer_pct":  None,
                "k_scale":    None,
                "condition":  "baseline",
                "behavior":   "safety",
                "source":     item["source"],
                "prompt_id":  item["id"],
                "user_input": item["prompt"],
                "response":   resp,
            })

        # Single-layer sweep: layer × k_scale × ramp_pos
        total = len(ablation_layers) * len(k_scales)
        for layer_idx, k_scale in tqdm(
            [(l, k) for l in ablation_layers for k in k_scales],
            total=total,
            desc=model_cfg.name,
            dynamic_ncols=True,
        ):
            d = dir_map[layer_idx]
            layer_config = {layer_idx: (d.mean_direction, d.k_value * k_scale)}
            layer_pct = int(layer_idx / model_cfg.num_layers * 100)

            logger.info(
                "  [%s | layer=%d (%d%%) | k_scale=%.1f | K=%.4f]",
                model_cfg.name, layer_idx, layer_pct, k_scale, d.k_value * k_scale,
            )
            resps = generate_batched(
                model, tokenizer, steerer, layer_config,
                prompts, exp_cfg.max_new_tokens, batch_size, model_cfg.extra,
            )
            for item, resp in zip(ablation_prompts, resps):
                records.append({
                    "model_name": model_cfg.name,
                    "num_layers": model_cfg.num_layers,
                    "layer_idx":  layer_idx,
                    "layer_pct":  layer_pct,
                    "k_scale":    k_scale,
                    "condition":  "ramp_pos",
                    "behavior":   "safety",
                    "source":     item["source"],
                    "prompt_id":  item["id"],
                    "user_input": item["prompt"],
                    "response":   resp,
                })

    # Score all records in one judge pass
    scores = judge.score_records(records, batch_size=exp_cfg.judge_batch_size)
    judge.unload()
    for rec, score in zip(records, scores):
        rec["judge_score"] = score

    # Attach norm profile values
    for rec in records:
        if rec["layer_idx"] is not None and rec["layer_idx"] in profile:
            rec["mean_norm"] = profile[rec["layer_idx"]].mean_norm
            rec["k_value"]   = profile[rec["layer_idx"]].k_value
        else:
            rec["mean_norm"] = None
            rec["k_value"]   = None

    df = pd.DataFrame(records)
    out_dir = output_root / model_cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "results.csv", index=False)

    summary = (
        df[df["condition"] == "ramp_pos"]
        .groupby(["layer_pct", "k_scale"])["judge_score"]
        .agg(["mean", "std", "count"])
    )
    summary.columns = ["mean_score", "std_score", "n"]
    summary.to_csv(out_dir / "summary.csv")
    logger.info("=== %s done ===\n%s", model_cfg.name, summary.to_string())

    fig_dir = figures_root / model_cfg.name
    plot_norm_growth(profile, model_cfg.name, ablation_layers, fig_dir)
    plot_score_heatmap(
        df[df["condition"] == "ramp_pos"], model_cfg.name, ablation_layers, k_scales, fig_dir
    )
    plot_score_vs_kscale(
        df[df["condition"] == "ramp_pos"], model_cfg.name, ablation_layers, k_scales, profile, fig_dir
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablation 01 — norm calibration")
    parser.add_argument("--config",            default="config/models.yml")
    parser.add_argument("--experiment-config", default="config/experiment.yml")
    parser.add_argument("--norm-profiles",     default="results/norm_profiles")
    parser.add_argument("--directions",        default="results/directions")
    parser.add_argument("--eval-prompts",      default="data/eval_prompts")
    parser.add_argument("--output-dir",        default="results/ablations/norm_calibration")
    parser.add_argument("--figures-dir",       default="figures/ablations/norm_calibration")
    parser.add_argument("--batch-size",  type=int,   default=64)
    parser.add_argument("--n-prompts",   type=int,   default=None, help="Override ablation_n_prompts.")
    parser.add_argument("--models",    nargs="*",                  help="Default: all instruct models.")
    parser.add_argument("--load-in-4bit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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

    ablation_prompts = load_harmbench_prompts(Path(args.eval_prompts), n_prompts)
    registry = ModelRegistry(load_in_4bit=args.load_in_4bit, device_map="cuda:0")

    logger.info(
        "%d models | %d prompts | layers@%s | k_scales=%s",
        len(target_names), n_prompts,
        exp_cfg.ablation_layer_fracs, exp_cfg.ablation_k_scales,
    )

    for name in target_names:
        profile    = load_norm_profile(Path(args.norm_profiles), name)
        directions_path = Path(args.directions) / name / "safety.npz"
        if not directions_path.exists():
            logger.error("Directions missing: %s — run experiment 02 first.", directions_path)
            continue
        directions = load_directions(str(directions_path))

        run_model(
            model_cfg=all_configs[name],
            ablation_prompts=ablation_prompts,
            profile=profile,
            directions=directions,
            k_scales=exp_cfg.ablation_k_scales,
            output_root=Path(args.output_dir),
            figures_root=Path(args.figures_dir),
            exp_cfg=exp_cfg,
            registry=registry,
            batch_size=args.batch_size,
        )

    logger.info("Ablation 01 complete.")


if __name__ == "__main__":
    main()
