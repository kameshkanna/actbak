"""Ablation 02 — Single-layer high-K steering causes output collapse.

Proves the complementary failure mode to flat-K: when steering is
concentrated at a single layer with K >> ‖h_ℓ‖, the residual stream is
catastrophically perturbed and the model produces incoherent output.

Method:
  For each model, sweep layers at 40–80% depth.
  At each layer steer with K = collapse_k_scale × K_ℓ  (default 10×).
  Measure two signals:
    judge_score   — how close to the target behaviour (lower = less effective)
    gibberish_rate — fraction of responses scored GIBBERISH by the judge

  Plot both against layer index with ‖h_ℓ‖ overlaid.

Expected pattern:
  - Shallow-middle layers (40–50%): K/‖h_ℓ‖ is large → residual stream
    blown out → high gibberish rate, judge score collapses to -0.5.
  - Deep-middle layers (70–80%): K/‖h_ℓ‖ is smaller → output survives but
    steering is still ineffective because one layer cannot carry the full
    behavioural load.

Outputs:
  results/ablations/single_layer_collapse/{model}/scored_results.csv
  results/ablations/single_layer_collapse/{model}/summary.csv
  figures/ablations/single_layer_collapse/{model}/collapse_by_layer.pdf
  figures/ablations/single_layer_collapse/all_models_collapse.pdf

Usage:
    python ablations/02_single_layer_collapse.py
    python ablations/02_single_layer_collapse.py --models llama-3.1-8b-instruct
    python ablations/02_single_layer_collapse.py --collapse-k-scale 5.0 --n-prompts 25
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

# Layers to probe expressed as fractions of total model depth
LAYER_FRACS = [0.4, 0.5, 0.6, 0.7, 0.8]


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


def resolve_probe_layers(num_layers: int) -> list[int]:
    """Convert LAYER_FRACS to concrete, deduplicated layer indices.

    Args:
        num_layers: Total transformer layers in the model.

    Returns:
        Sorted unique layer indices.
    """
    return sorted(set(int(f * num_layers) for f in LAYER_FRACS))


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


def plot_collapse_by_layer(
    summary_df: pd.DataFrame,
    profile: dict[int, LayerNormStats],
    model_name: str,
    baseline_score: float,
    collapse_k_scale: float,
    fig_dir: Path,
) -> None:
    """Dual-axis plot: judge score and gibberish rate vs layer, with norm overlay.

    The norm overlay makes the mechanistic argument visual: the gibberish
    rate mirrors the relative influence K / ‖h_ℓ‖, which is highest where
    norms are smallest.

    Args:
        summary_df:       DataFrame with columns ``layer_idx``, ``layer_pct``,
                          ``mean_judge_score``, ``gibberish_rate``, ``mean_norm``.
        profile:          Full norm profile indexed by layer_idx.
        model_name:       Short model slug (used in title).
        baseline_score:   Mean judge score under no-steering baseline.
        collapse_k_scale: K multiplier used (for title annotation).
        fig_dir:          Output directory.
    """
    layers    = summary_df["layer_idx"].tolist()
    pcts      = [f"{p}%" for p in summary_df["layer_pct"].tolist()]
    scores    = summary_df["mean_judge_score"].tolist()
    gibberish = summary_df["gibberish_rate"].tolist()
    norms     = summary_df["mean_norm"].tolist()

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 60))

    x = np.arange(len(layers))

    ax1.plot(x, scores,    color="#2563eb", marker="o", linewidth=2, markersize=7, label="Judge score")
    ax1.axhline(baseline_score, color="#6b7280", linestyle="--", linewidth=1.2, label=f"Baseline ({baseline_score:.3f})")
    ax2.bar(x, gibberish, color="#dc2626", alpha=0.35, width=0.5, label="Gibberish rate")
    ax3.plot(x, norms,     color="#059669", marker="s", linewidth=1.5, markersize=5, linestyle=":", label="‖h_ℓ‖ mean norm")

    ax1.set_xticks(x)
    ax1.set_xticklabels(pcts, fontsize=10)
    ax1.set_xlabel("Layer depth")
    ax1.set_ylabel("Mean judge score", color="#2563eb")
    ax2.set_ylabel("Gibberish rate", color="#dc2626")
    ax3.set_ylabel("Mean ‖h_ℓ‖", color="#059669")
    ax1.set_ylim(-0.6, 1.15)
    ax2.set_ylim(0, 1.05)

    ax1.set_title(
        f"{model_name} — Single-layer steering at K = {collapse_k_scale:.0f}×K_ℓ\n"
        "High K at shallow layers collapses output; deep layers remain unsteered"
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc="lower right", fontsize=8)
    ax1.grid(True, alpha=0.2)
    _savefig(fig, fig_dir / "collapse_by_layer.pdf")


def plot_all_models_collapse(
    all_data: dict[str, pd.DataFrame],
    fig_dir: Path,
    collapse_k_scale: float,
) -> None:
    """Grid of gibberish-rate bars per model for the cross-model view.

    Args:
        all_data:         Mapping of ``model_name → summary DataFrame``.
        fig_dir:          Output directory.
        collapse_k_scale: K multiplier used (for title).
    """
    n_models = len(all_data)
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, (model_name, df) in zip(axes, all_data.items()):
        pcts = [f"{p}%" for p in df["layer_pct"].tolist()]
        ax.bar(range(len(pcts)), df["gibberish_rate"].tolist(), color="#dc2626", alpha=0.75, width=0.6)
        ax.set_xticks(range(len(pcts)))
        ax.set_xticklabels(pcts, fontsize=8)
        ax.set_title(model_name.replace("-instruct", ""), fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.25)

    axes[0].set_ylabel("Gibberish rate")
    fig.suptitle(
        f"Single-layer collapse (K = {collapse_k_scale:.0f}×K_ℓ) — all models",
        fontsize=11,
    )
    plt.tight_layout()
    _savefig(fig, fig_dir / "all_models_collapse.pdf")


# ---------------------------------------------------------------------------
# Per-model runner
# ---------------------------------------------------------------------------

def run_model(
    model_cfg: ModelConfig,
    eval_prompts: list[dict],
    profile: dict[int, LayerNormStats],
    directions: list[BehavioralDirection],
    collapse_k_scale: float,
    output_root: Path,
    figures_root: Path,
    exp_cfg: ExperimentConfig,
    registry: ModelRegistry,
    batch_size: int,
) -> pd.DataFrame:
    """Run single-layer collapse ablation for one model.

    Steers each probe layer individually at ``collapse_k_scale × K_ℓ`` and
    records judge score and gibberish rate per layer.

    Args:
        model_cfg:        Config for the model.
        eval_prompts:     HarmBench prompt records.
        profile:          Norm profile indexed by layer_idx.
        directions:       Extracted safety behavioral directions.
        collapse_k_scale: K multiplier for the high-K steering (e.g. 10.0).
        output_root:      CSV output root.
        figures_root:     Figure output root.
        exp_cfg:          Loaded ExperimentConfig.
        registry:         ModelRegistry instance.
        batch_size:       Generation batch size.

    Returns:
        Summary DataFrame with one row per probe layer.
    """
    probe_layers = resolve_probe_layers(model_cfg.num_layers)
    dir_map      = {d.layer_idx: d for d in directions}

    # Keep only layers that have extracted directions
    valid_layers = [l for l in probe_layers if l in dir_map]
    if not valid_layers:
        logger.error("No valid probe layers for %s — skipping.", model_cfg.name)
        return pd.DataFrame()

    logger.info(
        "=== %s | layers=%s | K=%.1f×K_ℓ ===",
        model_cfg.name, valid_layers, collapse_k_scale,
    )

    prompts = [item["prompt"] for item in eval_prompts]
    judge   = SmallModelJudge(model_id=exp_cfg.judge_model)
    records: list[dict] = []

    with registry.loaded(model_cfg) as (model, tokenizer):
        steerer = ActivationSteerer(model)

        # Baseline — no steering
        logger.info("  [%s | baseline]", model_cfg.name)
        baseline_resps = generate_batched(
            model, tokenizer, steerer, {},
            prompts, exp_cfg.max_new_tokens, batch_size, model_cfg.extra,
        )
        for item, resp in zip(eval_prompts, baseline_resps):
            records.append({
                "model_name": model_cfg.name,
                "layer_idx":  None,
                "layer_pct":  None,
                "condition":  "baseline",
                "behavior":   "safety",
                "source":     item["source"],
                "prompt_id":  item["id"],
                "user_input": item["prompt"],
                "response":   resp,
            })

        # Single-layer sweep
        for layer_idx in tqdm(valid_layers, desc=model_cfg.name, dynamic_ncols=True):
            d          = dir_map[layer_idx]
            k_applied  = d.k_value * collapse_k_scale
            layer_pct  = int(layer_idx / model_cfg.num_layers * 100)
            layer_config = {layer_idx: (d.mean_direction, k_applied)}

            logger.info(
                "  [%s | layer=%d (%d%%) | K_ℓ=%.4f | K_applied=%.4f]",
                model_cfg.name, layer_idx, layer_pct, d.k_value, k_applied,
            )
            resps = generate_batched(
                model, tokenizer, steerer, layer_config,
                prompts, exp_cfg.max_new_tokens, batch_size, model_cfg.extra,
            )
            for item, resp in zip(eval_prompts, resps):
                records.append({
                    "model_name": model_cfg.name,
                    "layer_idx":  layer_idx,
                    "layer_pct":  layer_pct,
                    "condition":  "single_layer",
                    "behavior":   "safety",
                    "source":     item["source"],
                    "prompt_id":  item["id"],
                    "user_input": item["prompt"],
                    "response":   resp,
                })

    # Score all in one judge pass
    scores = judge.score_records(records, batch_size=exp_cfg.judge_batch_size)
    judge.unload()
    for rec, score in zip(records, scores):
        rec["judge_score"] = score

    # Attach norm profile values
    for rec in records:
        li = rec["layer_idx"]
        rec["mean_norm"] = profile[li].mean_norm if li is not None and li in profile else None
        rec["k_value"]   = profile[li].k_value   if li is not None and li in profile else None

    df = pd.DataFrame(records)
    out_dir = output_root / model_cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "scored_results.csv", index=False)

    # Per-layer summary
    baseline_score = float(df[df["condition"] == "baseline"]["judge_score"].mean())
    layer_df = df[df["condition"] == "single_layer"].copy()
    summary_rows = []
    for layer_idx in valid_layers:
        sub = layer_df[layer_df["layer_idx"] == layer_idx]
        gibberish_rate = (sub["judge_score"] == -0.5).mean()
        summary_rows.append({
            "layer_idx":        layer_idx,
            "layer_pct":        int(layer_idx / model_cfg.num_layers * 100),
            "mean_judge_score": float(sub["judge_score"].mean()),
            "gibberish_rate":   float(gibberish_rate),
            "std_judge_score":  float(sub["judge_score"].std()),
            "n":                len(sub),
            "mean_norm":        profile[layer_idx].mean_norm if layer_idx in profile else None,
            "k_value":          profile[layer_idx].k_value   if layer_idx in profile else None,
            "k_applied":        (profile[layer_idx].k_value * collapse_k_scale
                                 if layer_idx in profile else None),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    logger.info(
        "=== %s ===\nbaseline=%.3f\n%s",
        model_cfg.name, baseline_score,
        summary_df[["layer_pct", "mean_judge_score", "gibberish_rate"]].to_string(index=False),
    )

    fig_dir = figures_root / model_cfg.name
    plot_collapse_by_layer(summary_df, profile, model_cfg.name, baseline_score, collapse_k_scale, fig_dir)

    return summary_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablation 02 — Single-layer collapse")
    parser.add_argument("--config",            default="config/models.yml")
    parser.add_argument("--experiment-config", default="config/experiment.yml")
    parser.add_argument("--norm-profiles",     default="results/norm_profiles")
    parser.add_argument("--directions",        default="results/directions")
    parser.add_argument("--eval-prompts",      default="data/eval_prompts")
    parser.add_argument("--output-dir",        default="results/ablations/single_layer_collapse")
    parser.add_argument("--figures-dir",       default="figures/ablations/single_layer_collapse")
    parser.add_argument("--batch-size",       type=int,   default=64)
    parser.add_argument("--collapse-k-scale", type=float, default=None,
                        help="Override ablation_k_scales[2] (the high-K entry).")
    parser.add_argument("--n-prompts",        type=int,   default=None,
                        help="Override ablation_n_prompts.")
    parser.add_argument("--models",    nargs="*", help="Default: all instruct models.")
    parser.add_argument("--load-in-4bit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args     = parse_args()
    all_configs = load_model_configs(args.config)
    exp_cfg     = load_experiment_config(args.experiment_config)

    # Use the highest k_scale from ablation_k_scales as the collapse magnitude
    collapse_k_scale = (
        args.collapse_k_scale
        if args.collapse_k_scale is not None
        else max(exp_cfg.ablation_k_scales)
    )
    n_prompts = args.n_prompts if args.n_prompts is not None else exp_cfg.ablation_n_prompts

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
        "%d models | %d prompts | layers@%s | K=%.1f×K_ℓ",
        len(target_names), n_prompts, LAYER_FRACS, collapse_k_scale,
    )

    all_summaries: dict[str, pd.DataFrame] = {}
    for name in target_names:
        profile         = load_norm_profile(Path(args.norm_profiles), name)
        directions_path = Path(args.directions) / name / "safety.npz"
        if not directions_path.exists():
            logger.error("Directions missing: %s — run experiment 02 first.", directions_path)
            continue
        directions = load_directions(str(directions_path))

        summary_df = run_model(
            model_cfg=all_configs[name],
            eval_prompts=eval_prompts,
            profile=profile,
            directions=directions,
            collapse_k_scale=collapse_k_scale,
            output_root=Path(args.output_dir),
            figures_root=Path(args.figures_dir),
            exp_cfg=exp_cfg,
            registry=registry,
            batch_size=args.batch_size,
        )
        if not summary_df.empty:
            all_summaries[name] = summary_df

    if len(all_summaries) > 1:
        plot_all_models_collapse(
            all_summaries,
            Path(args.figures_dir),
            collapse_k_scale,
        )
        logger.info("Cross-model collapse plot saved to %s", args.figures_dir)

    logger.info("Ablation 02 complete.")


if __name__ == "__main__":
    main()
