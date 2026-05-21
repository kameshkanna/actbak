"""Experiment 04 — Baking evaluation: persistent weight injection vs baseline.

Bakes a pre-extracted behavioral direction into the model's MLP biases and
measures the impact on both safety benchmarks and general capability:

  baseline   — unmodified model
  baked_pos  — direction baked in at +scale × K_ℓ (elicits behavior)
  baked_neg  — direction baked in at −scale × K_ℓ (suppresses behavior)

Safety benchmarks:  HarmBench (safe_rate via SmallModelJudge)
Capability benchmarks: GSM8K (exact_match), MMLU (accuracy), TruthfulQA (mc1_accuracy)

Baking is performed in-place; the original weights are restored via ``unbake``
after each condition so only one model copy lives in VRAM at a time.

Usage:
    python experiments/04_baking_eval.py \\
        --model llama-3.1-8b-instruct \\
        --behavior safety \\
        --k-scale 1.0
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
import pandas as pd

from activation_baking.baker import ModelBaker
from activation_baking.config import ModelConfig, load_experiment_config, load_model_configs
from activation_baking.direction_extractor import BehavioralDirection, load_directions
from activation_baking.evaluators import (
    EvalResult,
    GSM8KEvaluator,
    HarmBenchEvaluator,
    MMLUEvaluator,
    TruthfulQAEvaluator,
)
from activation_baking.judges import SmallModelJudge
from activation_baking.registry import ModelRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CONDITIONS = ["baseline", "baked_pos", "baked_neg"]

BENCHMARKS = ["harmbench", "gsm8k", "mmlu", "truthful_qa"]
METRICS = {
    "harmbench":   "safe_rate",
    "gsm8k":       "exact_match",
    "mmlu":        "accuracy",
    "truthful_qa": "mc1_accuracy",
}


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_benchmarks(
    model,
    tokenizer,
    judge: SmallModelJudge,
    exp_cfg,
    condition_label: str,
) -> dict[str, EvalResult]:
    """Run all four benchmarks on the current model state."""
    logger.info("=== Running benchmarks: %s ===", condition_label)
    results: dict[str, EvalResult] = {}

    results["harmbench"] = HarmBenchEvaluator(judge).evaluate(
        model, tokenizer, n_samples=exp_cfg.n_harmbench, seed=exp_cfg.seed
    )
    results["gsm8k"] = GSM8KEvaluator().evaluate(
        model, tokenizer, n_samples=exp_cfg.n_gsm8k, seed=exp_cfg.seed
    )
    results["mmlu"] = MMLUEvaluator().evaluate(
        model, tokenizer, n_samples=exp_cfg.n_mmlu, seed=exp_cfg.seed
    )
    results["truthful_qa"] = TruthfulQAEvaluator().evaluate(
        model, tokenizer, n_samples=exp_cfg.n_truthfulqa, seed=exp_cfg.seed
    )

    for name, res in results.items():
        logger.info("  %-14s %s=%.4f  (n=%d)", name, res.metric, res.score, res.n_samples)

    return results


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------


def build_layer_configs(
    directions: list[BehavioralDirection],
    scale: float,
) -> dict[int, tuple[np.ndarray, float]]:
    """Build layer_configs mapping for ModelBaker from extracted directions."""
    return {d.layer_idx: (d.mean_direction, d.k_value * scale) for d in directions}


def results_to_rows(
    all_results: dict[str, dict[str, EvalResult]],
    model_name: str,
    behavior: str,
    k_scale: float,
) -> list[dict]:
    """Flatten nested results into a list of flat dicts for DataFrame construction."""
    rows = []
    for condition, benchmark_results in all_results.items():
        for benchmark, res in benchmark_results.items():
            rows.append({
                "model":     model_name,
                "behavior":  behavior,
                "k_scale":   k_scale,
                "condition": condition,
                "benchmark": benchmark,
                "metric":    res.metric,
                "score":     res.score,
                "n_samples": res.n_samples,
            })
    return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", path)


def plot_benchmark_comparison(
    df: pd.DataFrame,
    model_name: str,
    behavior: str,
    fig_dir: Path,
) -> None:
    """Grouped bar chart: score per benchmark, grouped by condition."""
    fig, axes = plt.subplots(1, len(BENCHMARKS), figsize=(4 * len(BENCHMARKS), 4), sharey=False)
    condition_colors = {
        "baseline":  "#6b7280",
        "baked_pos": "#2563eb",
        "baked_neg": "#dc2626",
    }

    for ax, benchmark in zip(axes, BENCHMARKS):
        sub = df[df["benchmark"] == benchmark].set_index("condition")["score"]
        sub = sub.reindex(CONDITIONS).fillna(0)
        bars = ax.bar(
            sub.index,
            sub.values,
            color=[condition_colors.get(c, "gray") for c in sub.index],
            width=0.55,
            edgecolor="white",
        )
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        ax.set_title(f"{benchmark}\n({METRICS[benchmark]})", fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.tick_params(axis="x", labelsize=8, rotation=10)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"{model_name} | {behavior} — baking evaluation", fontsize=11)
    fig.tight_layout()
    _savefig(fig, fig_dir / "benchmark_comparison.pdf")


def plot_delta_heatmap(
    df: pd.DataFrame,
    model_name: str,
    behavior: str,
    fig_dir: Path,
) -> None:
    """Heatmap showing score delta (baked − baseline) for each condition × benchmark."""
    baseline = df[df["condition"] == "baseline"].set_index("benchmark")["score"]

    rows_list = []
    for condition in ["baked_pos", "baked_neg"]:
        cond_scores = df[df["condition"] == condition].set_index("benchmark")["score"]
        row = {b: float(cond_scores.get(b, 0) - baseline.get(b, 0)) for b in BENCHMARKS}
        row["condition"] = condition
        rows_list.append(row)

    delta_df = pd.DataFrame(rows_list).set_index("condition")[BENCHMARKS]

    fig, ax = plt.subplots(figsize=(len(BENCHMARKS) * 1.5 + 1, 2.5))
    im = ax.imshow(delta_df.values, aspect="auto", cmap="RdYlGn", vmin=-0.3, vmax=0.3)
    ax.set_xticks(range(len(BENCHMARKS)))
    ax.set_xticklabels(BENCHMARKS, fontsize=9)
    ax.set_yticks(range(len(delta_df)))
    ax.set_yticklabels(delta_df.index, fontsize=9)

    for i in range(len(delta_df)):
        for j in range(len(BENCHMARKS)):
            val = delta_df.values[i, j]
            ax.text(j, i, f"{val:+.3f}", ha="center", va="center", fontsize=8,
                    color="black" if abs(val) < 0.2 else "white")

    plt.colorbar(im, ax=ax, label="Score delta (baked − baseline)")
    ax.set_title(f"{model_name} | {behavior} — baking impact")
    fig.tight_layout()
    _savefig(fig, fig_dir / "delta_heatmap.pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 04 — baking evaluation")
    parser.add_argument("--config", default="config/models.yml")
    parser.add_argument("--experiment-config", default="config/experiment.yml")
    parser.add_argument("--directions", default="results/directions")
    parser.add_argument("--output-dir", default="results/baking_eval")
    parser.add_argument("--model", required=True, help="Model name from config/models.yml")
    parser.add_argument(
        "--behavior", required=True, choices=["sycophancy", "safety", "refusal"]
    )
    parser.add_argument(
        "--k-scale", type=float, default=1.0,
        help="Multiply all K_ℓ values by this factor before baking.",
    )
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument(
        "--save-baked-model", action="store_true",
        help="Persist the baked_pos model to disk via save_pretrained.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    all_configs: dict[str, ModelConfig] = load_model_configs(args.config)
    if args.model not in all_configs:
        logger.error("Model '%s' not found. Available: %s", args.model, list(all_configs))
        sys.exit(1)
    model_cfg: ModelConfig = all_configs[args.model]

    exp_cfg = load_experiment_config(args.experiment_config)

    scale_tag = f"k{args.k_scale:.2f}".replace(".", "p")
    out_dir = Path(args.output_dir) / args.model / args.behavior / scale_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(exp_cfg.figures_dir) / "baking_eval" / args.model / args.behavior / scale_tag

    # ── Directions ────────────────────────────────────────────────────────────
    dir_path = Path(args.directions) / args.model / f"{args.behavior}.npz"
    if not dir_path.exists():
        logger.error(
            "Directions not found at %s — run experiment 02 first.", dir_path
        )
        sys.exit(1)
    directions = load_directions(str(dir_path))
    logger.info(
        "Loaded %d direction vectors for %s | %s.",
        len(directions), args.model, args.behavior,
    )

    layer_configs_pos = build_layer_configs(directions, scale=args.k_scale)
    layer_configs_neg = build_layer_configs(directions, scale=-args.k_scale)

    # ── Judge (loaded once, shared across all conditions) ─────────────────────
    logger.info("Loading judge: %s", exp_cfg.judge_model)
    judge = SmallModelJudge(model_id=exp_cfg.judge_model)

    # ── Model — single load, in-place bake/unbake cycle ──────────────────────
    registry = ModelRegistry(load_in_4bit=args.load_in_4bit)
    all_results: dict[str, dict[str, EvalResult]] = {}

    with registry.loaded(model_cfg) as (model, tokenizer):
        baker = ModelBaker(model)

        # --- Baseline ---
        all_results["baseline"] = run_benchmarks(
            model, tokenizer, judge, exp_cfg, "baseline"
        )

        # --- Baked positive ---
        logger.info("Baking +direction (scale=%.2f) into model weights.", args.k_scale)
        baker.bake(layer_configs_pos, inplace=True)

        if args.save_baked_model:
            baked_model_path = str(out_dir / "baked_pos_model")
            baker.save_baked_model(model, baked_model_path)
            logger.info("Baked model saved to %s", baked_model_path)

        all_results["baked_pos"] = run_benchmarks(
            model, tokenizer, judge, exp_cfg, "baked_pos"
        )

        logger.info("Unbaking +direction — restoring original weights.")
        baker.unbake(layer_configs_pos)

        # --- Baked negative ---
        logger.info("Baking −direction (scale=%.2f) into model weights.", args.k_scale)
        baker.bake(layer_configs_neg, inplace=True)

        all_results["baked_neg"] = run_benchmarks(
            model, tokenizer, judge, exp_cfg, "baked_neg"
        )

        logger.info("Unbaking −direction — restoring original weights.")
        baker.unbake(layer_configs_neg)

    judge.unload()

    # ── Persist results ───────────────────────────────────────────────────────
    rows = results_to_rows(all_results, args.model, args.behavior, args.k_scale)
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / "results.csv", index=False)

    per_sample_path = out_dir / "per_sample"
    per_sample_path.mkdir(exist_ok=True)
    for condition, benchmark_results in all_results.items():
        for benchmark, res in benchmark_results.items():
            if "per_sample" in res.details:
                pd.DataFrame(res.details["per_sample"]).to_csv(
                    per_sample_path / f"{condition}_{benchmark}.csv", index=False
                )

    # ── Summary table ─────────────────────────────────────────────────────────
    pivot = summary_df.pivot_table(
        index="benchmark", columns="condition", values="score"
    ).reindex(index=BENCHMARKS, columns=CONDITIONS)
    logger.info("\n%s", pivot.to_string())

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_benchmark_comparison(summary_df, args.model, args.behavior, fig_dir)
    plot_delta_heatmap(summary_df, args.model, args.behavior, fig_dir)

    logger.info("Experiment 04 complete. Results → %s", out_dir)


if __name__ == "__main__":
    main()
