"""Experiment 04 — Flat-K vs Ramp-K comparison at matched budget.

Core claim validation: K_ℓ = μ̄_ℓ / √d is the correct calibration because residual
stream norms grow with depth. A flat-K schedule at the same total budget spreads
injection evenly across layers, under-injecting at deep layers (where norms are large
and the signal is drowned out) and over-injecting at shallow layers. The ramp
concentrates magnitude proportionally to norm — maintaining constant relative
influence at every layer.

Conditions
----------
  baseline   — no steering
  ramp_pos   — K_ℓ = μ̄_ℓ / √d at each layer (formula schedule)
  flat_pos   — K = mean(K_ℓ) at every layer (same total budget as ramp_pos)
  flat_pos_max — K = max(K_ℓ) at every layer (generous flat budget — shows ceiling)

The matched-budget flat_pos is the primary comparison. flat_pos_max shows that even
with more budget, flat injection either under-steers or degrades before ramp reaches
its ceiling.

Usage:
    python experiments/04_flat_k_comparison.py \\
        --model llama-3.1-8b-instruct \\
        --behavior safety

    # All models
    python experiments/04_flat_k_comparison.py --behavior safety
"""

from __future__ import annotations

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
import torch
from tqdm import tqdm

from activation_baking.config import ModelConfig, load_experiment_config, load_model_configs
from activation_baking.direction_extractor import load_directions
from activation_baking.judges import SmallModelJudge
from activation_baking.model_utils import format_prompt
from activation_baking.registry import ModelRegistry
from activation_baking.steerer import ActivationSteerer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CONDITIONS = ["baseline", "ramp_pos", "flat_pos", "flat_pos_max"]
CONDITION_LABELS = {
    "baseline":     "Baseline",
    "ramp_pos":     "Ramp-K (μ̄_ℓ/√d)",
    "flat_pos":     "Flat-K (mean K, matched budget)",
    "flat_pos_max": "Flat-K (max K, generous budget)",
}
CONDITION_COLORS = {
    "baseline":     "#6b7280",
    "ramp_pos":     "#2563eb",
    "flat_pos":     "#f59e0b",
    "flat_pos_max": "#ef4444",
}


# ---------------------------------------------------------------------------
# Steering config builders
# ---------------------------------------------------------------------------


def build_flat_config(
    directions: list,
    k_override: float,
) -> dict[int, tuple[np.ndarray, float]]:
    """Build a flat-K layer config with a fixed magnitude at every middle layer."""
    return {d.layer_idx: (d.mean_direction, k_override) for d in directions}


def build_ramp_config(directions: list) -> dict[int, tuple[np.ndarray, float]]:
    """Build a ramp-K layer config using the formula K_ℓ = μ̄_ℓ / √d."""
    return {d.layer_idx: (d.mean_direction, d.k_value) for d in directions}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def load_eval_prompts(eval_dir: Path, n_harmbench: int, n_clearharm: int) -> list[dict]:
    records: list[dict] = []
    for fname, key, cap in (
        ("harmbench.jsonl", "harmbench", n_harmbench),
        ("clearharm.jsonl", "clearharm", n_clearharm),
    ):
        path = eval_dir / fname
        if not path.exists():
            logger.error(
                "%s not found — run `python data/download_datasets.py` first.", path
            )
            sys.exit(1)
        batch: list[dict] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                if len(batch) >= cap:
                    break
                batch.append(json.loads(line))
        records.extend(batch)
    logger.info(
        "Loaded %d eval prompts (%d harmbench + %d clearharm).",
        len(records), n_harmbench, n_clearharm,
    )
    return records


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


@torch.no_grad()
def _generate_one(model, tokenizer, prompt: str, max_new_tokens: int, extra_cfg: dict) -> str:
    formatted = format_prompt(tokenizer, prompt, extra_cfg)
    device = next(model.parameters()).device
    inputs = tokenizer(
        formatted, return_tensors="pt", truncation=True, max_length=1024, padding=False,
    ).to(device)
    prompt_len = inputs["input_ids"].shape[1]
    out = model.generate(
        **inputs, max_new_tokens=max_new_tokens,
        do_sample=False, pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()


def generate_condition(
    model, tokenizer, steerer: ActivationSteerer, layer_config: dict,
    eval_prompts: list[dict], behavior: str, max_new_tokens: int,
    extra_cfg: dict, condition_name: str,
) -> list[dict]:
    records: list[dict] = []
    with steerer.steer(layer_config):
        for item in tqdm(
            eval_prompts, desc=f"  [{condition_name}]", dynamic_ncols=True, leave=False,
        ):
            resp = _generate_one(model, tokenizer, item["prompt"], max_new_tokens, extra_cfg)
            records.append({
                "condition":  condition_name,
                "behavior":   behavior,
                "source":     item["source"],
                "prompt_id":  item["id"],
                "user_input": item["prompt"],
                "response":   resp,
            })
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    logger.info("[%s] generated %d responses.", condition_name, len(records))
    return records


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", path)


def plot_ramp_vs_flat(
    df: pd.DataFrame,
    k_stats: dict[str, float],
    behavior: str,
    model_name: str,
    fig_dir: Path,
) -> None:
    """Bar chart comparing ramp-K vs flat-K conditions."""
    agg = df.groupby("condition")["judge_score"].mean().reindex(CONDITIONS).fillna(0)

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        range(len(CONDITIONS)),
        agg.values,
        color=[CONDITION_COLORS[c] for c in CONDITIONS],
        width=0.55,
        edgecolor="white",
    )
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels(
        [CONDITION_LABELS[c] for c in CONDITIONS], rotation=12, ha="right", fontsize=9
    )
    ax.set_ylabel("Mean refusal score (judge)")
    ax.set_ylim(0, 1.15)
    ax.set_title(
        f"{model_name} | {behavior}\n"
        f"Ramp-K (K range [{k_stats['k_min']:.3f}, {k_stats['k_max']:.3f}]) vs "
        f"Flat-K (mean={k_stats['k_mean']:.3f}, max={k_stats['k_max']:.3f})"
    )
    ax.grid(True, axis="y", alpha=0.3)
    _savefig(fig, fig_dir / "ramp_vs_flat.pdf")


def plot_k_schedule(
    directions: list,
    model_name: str,
    fig_dir: Path,
) -> None:
    """Visualise the ramp-K schedule vs flat-K alternatives."""
    layers = [d.layer_idx for d in directions]
    k_vals = np.array([d.k_value for d in directions])
    k_mean = float(np.mean(k_vals))
    k_max  = float(np.max(k_vals))

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(layers, k_vals, color=CONDITION_COLORS["ramp_pos"],
            linewidth=2.5, label="Ramp-K (K_ℓ = μ̄_ℓ / √d)")
    ax.axhline(k_mean, color=CONDITION_COLORS["flat_pos"],
               linewidth=1.8, linestyle="--", label=f"Flat-K mean ({k_mean:.3f})")
    ax.axhline(k_max, color=CONDITION_COLORS["flat_pos_max"],
               linewidth=1.8, linestyle=":", label=f"Flat-K max ({k_max:.3f})")

    ax.fill_between(layers, k_vals, k_mean,
                    where=(k_vals > k_mean), alpha=0.12, color=CONDITION_COLORS["ramp_pos"],
                    label="Ramp over-budget (deep layers)")
    ax.fill_between(layers, k_mean, k_vals,
                    where=(k_vals < k_mean), alpha=0.12, color=CONDITION_COLORS["flat_pos"],
                    label="Flat over-budget (shallow layers)")

    ax.set_xlabel("Layer index")
    ax.set_ylabel("K_ℓ magnitude")
    ax.set_title(f"{model_name} — ramp-K schedule vs flat-K alternatives")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _savefig(fig, fig_dir / "k_schedule.pdf")


def plot_per_layer_delta(
    df: pd.DataFrame,
    behavior: str,
    model_name: str,
    fig_dir: Path,
) -> None:
    """Heatmap: per-prompt refusal score across conditions."""
    pivot = (
        df.pivot_table(index="prompt_id", columns="condition",
                       values="judge_score", aggfunc="mean")
        .reindex(columns=CONDITIONS).fillna(0)
    )
    fig, ax = plt.subplots(figsize=(7, max(4, len(pivot) * 0.35)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITIONS], fontsize=8, rotation=10)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels([str(pid)[:30] for pid in pivot.index], fontsize=7)
    plt.colorbar(im, ax=ax, label="Refusal score")
    ax.set_title(f"{model_name} | {behavior} — per-prompt refusal")
    _savefig(fig, fig_dir / "per_prompt_heatmap.pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 04 — Flat-K vs Ramp-K comparison at matched budget"
    )
    parser.add_argument("--config", default="config/models.yml")
    parser.add_argument("--experiment-config", default="config/experiment.yml")
    parser.add_argument("--directions", default="results/directions")
    parser.add_argument("--eval-prompts", default="data/eval_prompts")
    parser.add_argument("--output-dir", default="results/flat_k_comparison")
    parser.add_argument(
        "--model", nargs="*",
        help="Model name(s) from config/models.yml. Default: all instruct models.",
    )
    parser.add_argument(
        "--behavior", required=True, choices=["safety", "refusal", "sycophancy"]
    )
    parser.add_argument("--load-in-4bit", action="store_true")
    return parser.parse_args()


def run_model(
    model_cfg: ModelConfig,
    behavior: str,
    eval_prompts: list[dict],
    directions_dir: Path,
    output_dir: Path,
    exp_cfg,
    registry: ModelRegistry,
) -> None:
    dir_path = directions_dir / model_cfg.name / f"{behavior}.npz"
    if not dir_path.exists():
        logger.error(
            "Directions not found at %s — run experiment 02 first.", dir_path
        )
        return

    directions = load_directions(str(dir_path))
    k_vals = np.array([d.k_value for d in directions])
    k_mean = float(np.mean(k_vals))
    k_max  = float(np.max(k_vals))
    k_stats = {"k_mean": k_mean, "k_max": k_max, "k_min": float(np.min(k_vals))}

    logger.info(
        "%s | %s — %d layers | K range [%.4f, %.4f] | flat mean=%.4f",
        model_cfg.name, behavior, len(directions), k_stats["k_min"], k_max, k_mean,
    )

    ramp_config     = build_ramp_config(directions)
    flat_mean_config = build_flat_config(directions, k_mean)
    flat_max_config  = build_flat_config(directions, k_max)
    configs = {
        "baseline":     {},
        "ramp_pos":     ramp_config,
        "flat_pos":     flat_mean_config,
        "flat_pos_max": flat_max_config,
    }

    out_dir = output_dir / model_cfg.name / behavior
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(exp_cfg.figures_dir) / "flat_k_comparison" / model_cfg.name / behavior

    # Phase 1: Generate
    all_records: list[dict] = []
    with registry.loaded(model_cfg) as (model, tokenizer):
        steerer = ActivationSteerer(model)
        for condition in CONDITIONS:
            logger.info("  Generating [%s]...", condition)
            records = generate_condition(
                model, tokenizer, steerer, configs[condition],
                eval_prompts, behavior, exp_cfg.max_new_tokens,
                model_cfg.extra, condition,
            )
            all_records.extend(records)

    raw_df = pd.DataFrame(all_records)
    raw_df.to_csv(out_dir / "raw_generations.csv", index=False)

    # Phase 2: Judge
    logger.info("  Judging %d responses...", len(all_records))
    judge = SmallModelJudge(model_id=exp_cfg.judge_model)
    scores = judge.score_records(all_records, batch_size=exp_cfg.judge_batch_size)
    judge.unload()

    for rec, score in zip(all_records, scores):
        rec["judge_score"] = score

    scored_df = pd.DataFrame(all_records)
    scored_df.to_csv(out_dir / "scored_results.csv", index=False)

    summary = scored_df.groupby("condition")["judge_score"].agg(["mean", "std", "count"])
    summary.columns = ["mean_score", "std_score", "n"]
    summary["k_used"] = summary.index.map({
        "baseline":     0.0,
        "ramp_pos":     float("nan"),
        "flat_pos":     k_mean,
        "flat_pos_max": k_max,
    })
    summary.to_csv(out_dir / "summary.csv")
    logger.info("\n%s", summary.to_string())

    # Phase 3: Plot
    plot_ramp_vs_flat(scored_df, k_stats, behavior, model_cfg.name, fig_dir)
    plot_k_schedule(directions, model_cfg.name, fig_dir)
    plot_per_layer_delta(scored_df, behavior, model_cfg.name, fig_dir)

    logger.info("%s | %s — done. Results → %s", model_cfg.name, behavior, out_dir)


def main() -> None:
    args = parse_args()

    all_configs: dict[str, ModelConfig] = load_model_configs(args.config)
    exp_cfg = load_experiment_config(args.experiment_config)

    if args.model:
        missing = set(args.model) - set(all_configs)
        if missing:
            logger.error("Unknown model name(s): %s", missing)
            sys.exit(1)
        target_names = args.model
    else:
        target_names = [n for n, c in all_configs.items() if c.is_instruct]

    eval_prompts = load_eval_prompts(
        Path(args.eval_prompts), exp_cfg.n_harmbench, exp_cfg.n_clearharm
    )
    registry = ModelRegistry(load_in_4bit=args.load_in_4bit)
    output_dir = Path(args.output_dir)

    for name in target_names:
        logger.info("=== %s | %s ===", name, args.behavior)
        run_model(
            model_cfg=all_configs[name],
            behavior=args.behavior,
            eval_prompts=eval_prompts,
            directions_dir=Path(args.directions),
            output_dir=output_dir,
            exp_cfg=exp_cfg,
            registry=registry,
        )

    logger.info("Experiment 04 complete.")


if __name__ == "__main__":
    main()
