"""Experiment 03 — Ramp steering evaluation across all models, behaviors, and K scales.

For each model × behavior:
  1. Run baseline once (no hooks) on behavior-specific eval prompts.
  2. Sweep K scales: steer with ramp_pos and ramp_neg at K_ℓ × scale.
  3. Score all responses with the behavior-appropriate SmallModelJudge.
  4. Save CSV + summary + plots per (behavior, k_scale).

Behavior → benchmark mapping:
  safety     →  HarmBench + ClearHarm  (refusal scoring: SAFE / UNSAFE)
  sycophancy →  Anthropic model-written-evals  (SYCOPHANTIC / CONSISTENT)

Skip logic: any (model, behavior, k_scale) whose scored_results.csv already
exists is silently skipped.  Re-run from scratch with --force.

Base model support: base models (is_instruct=False) are evaluated using the
prompt_template defined in their extra config (models.yml).  By default only
instruct models run; pass --include-base or name them explicitly via --models.

Optimised for single GH200 (96 GB): one model load per model, judge loaded
alongside main model (no unload/reload), batch_size=64.

Usage:
    # All instruct models, all behaviors, all K scales
    python experiments/03_ramp_steering_eval.py

    # Include base models alongside instruct
    python experiments/03_ramp_steering_eval.py --include-base

    # Explicit model list (instruct + base mixed)
    python experiments/03_ramp_steering_eval.py \\
        --models llama-3.1-8b-instruct llama-3.1-8b \\
        --behaviors safety \\
        --k-scales 0.5 1.0 2.0

    # Force re-run even if results already exist
    python experiments/03_ramp_steering_eval.py --force

    # Tune batch size to GPU VRAM (GH200 default: 64)
    python experiments/03_ramp_steering_eval.py --batch-size 128
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

from typing import Any

from activation_baking.config import ExperimentConfig, ModelConfig, load_experiment_config, load_model_configs
from activation_baking.direction_extractor import load_directions
from activation_baking.judges import SmallModelJudge
from activation_baking.model_utils import format_prompt
from transformers import PreTrainedTokenizerBase
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

CONDITIONS       = ["baseline", "ramp_pos", "ramp_neg"]
STEER_CONDITIONS = ["ramp_pos", "ramp_neg"]
CONDITION_COLORS = {"baseline": "#6b7280", "ramp_pos": "#2563eb", "ramp_neg": "#dc2626"}
CONDITION_LABELS = {"baseline": "Baseline", "ramp_pos": "Ramp+", "ramp_neg": "Ramp−"}

# Behavior → eval prompt JSONL filenames (all under data/eval_prompts/)
BEHAVIOR_SOURCES: dict[str, list[str]] = {
    "safety":     ["harmbench.jsonl", "clearharm.jsonl"],
    "sycophancy": ["sycophancy.jsonl"],
}


# ---------------------------------------------------------------------------
# Steering configs
# ---------------------------------------------------------------------------

def _ramp_config(
    directions: list, scale: float
) -> dict[int, tuple[np.ndarray, float]]:
    """Build a ramp+ steering config: +direction at K_ℓ × scale for each layer."""
    return {d.layer_idx: (d.mean_direction, d.k_value * scale) for d in directions}


def _neg_config(
    directions: list, scale: float
) -> dict[int, tuple[np.ndarray, float]]:
    """Build a ramp- steering config: −direction at K_ℓ × scale for each layer."""
    return {d.layer_idx: (-d.mean_direction, d.k_value * scale) for d in directions}


# ---------------------------------------------------------------------------
# Eval prompt loading
# ---------------------------------------------------------------------------

def load_behavior_prompts(behavior: str, eval_dir: Path) -> list[dict]:
    """Load eval prompts for a specific behavior from its dedicated JSONL files.

    Args:
        behavior: One of the keys in ``BEHAVIOR_SOURCES``.
        eval_dir: Directory containing the JSONL eval prompt files.

    Returns:
        List of prompt records; each has at minimum ``source``, ``id``, ``prompt``.

    Raises:
        KeyError:    If ``behavior`` is not in ``BEHAVIOR_SOURCES``.
        SystemExit:  If any required JSONL file is missing (user must download first).
    """
    if behavior not in BEHAVIOR_SOURCES:
        raise KeyError(
            f"Unknown behavior '{behavior}'. "
            f"Registered behaviors: {list(BEHAVIOR_SOURCES)}"
        )
    records: list[dict] = []
    for fname in BEHAVIOR_SOURCES[behavior]:
        path = eval_dir / fname
        if not path.exists():
            logger.error(
                "Eval prompts not found: %s — run `python data/download_datasets.py` first.", path
            )
            sys.exit(1)
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    logger.info("Behavior '%s': loaded %d eval prompts.", behavior, len(records))
    return records


# ---------------------------------------------------------------------------
# Batched generation
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
    """Generate responses for all prompts in batches under an optional steering config.

    Args:
        model:          Loaded causal LM.
        tokenizer:      Corresponding tokenizer.
        steerer:        ActivationSteerer bound to the model.
        layer_config:   Steering config passed to ``steerer.steer()``.
                        Empty dict → baseline (no hooks attached).
        prompts:        Raw user prompt strings (chat template applied internally).
        max_new_tokens: Maximum tokens to generate per prompt.
        batch_size:     Number of prompts per forward pass.
        extra_cfg:      Per-model extras forwarded to ``format_prompt``.

    Returns:
        Decoded response strings, same order as ``prompts``.
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


def plot_results(
    df: pd.DataFrame,
    behavior: str,
    model_name: str,
    scale: float,
    fig_dir: Path,
) -> None:
    """Generate condition comparison, by-source, and per-prompt heatmap plots.

    Args:
        df:         DataFrame with columns ``condition``, ``judge_score``, ``source``, ``prompt_id``.
        behavior:   Behavior name (used in plot titles).
        model_name: Short model slug.
        scale:      K scale multiplier (used in plot titles).
        fig_dir:    Output directory for PDF figures.
    """
    agg = df.groupby("condition")["judge_score"].mean().reindex(CONDITIONS).fillna(0)
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(range(3), agg.values, color=[CONDITION_COLORS[c] for c in CONDITIONS], width=0.5)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_xticks(range(3))
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITIONS], fontsize=9)
    ax.set_ylabel("Mean judge score")
    ax.set_ylim(0, 1.15)
    ax.set_title(f"{model_name} | {behavior} | k={scale:.2f}")
    ax.grid(True, axis="y", alpha=0.3)
    _savefig(fig, fig_dir / "condition_comparison.pdf")

    sources = sorted(df["source"].unique())
    src_palette = {"harmbench": "#7c3aed", "clearharm": "#059669", "anthropic_syco": "#ea580c"}
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, src in enumerate(sources):
        sub = (
            df[df["source"] == src]
            .groupby("condition")["judge_score"]
            .mean()
            .reindex(CONDITIONS)
            .fillna(0)
        )
        ax.bar(
            x + (i - len(sources) / 2 + 0.5) * 0.3, sub.values, 0.3,
            label=src, color=src_palette.get(src, "gray"),
        )
    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITIONS], fontsize=9)
    ax.set_ylabel("Mean judge score")
    ax.set_ylim(0, 1.15)
    ax.set_title(f"{model_name} | {behavior} | k={scale:.2f} — by source")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    _savefig(fig, fig_dir / "by_source.pdf")

    pivot = (
        df.pivot_table(index="prompt_id", columns="condition", values="judge_score", aggfunc="mean")
        .reindex(columns=CONDITIONS)
        .fillna(0)
    )
    fig, ax = plt.subplots(figsize=(6, max(4, len(pivot) * 0.3)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITIONS], fontsize=8)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels([str(pid)[:30] for pid in pivot.index], fontsize=6)
    plt.colorbar(im, ax=ax, label="Judge score")
    ax.set_title(f"{model_name} | {behavior} | k={scale:.2f}")
    _savefig(fig, fig_dir / "per_prompt_heatmap.pdf")


# ---------------------------------------------------------------------------
# Per-model runner
# ---------------------------------------------------------------------------

def _k_tag(scale: float) -> str:
    return f"k{scale:.2f}".replace(".", "_")


def _results_complete(
    output_root: Path,
    model_name: str,
    behavior: str,
    k_scales: list[float],
) -> bool:
    """Return True if all outputs for this model × behavior already exist."""
    baseline_csv = output_root / model_name / behavior / "baseline" / "scored_results.csv"
    if not baseline_csv.exists():
        return False
    return all(
        (output_root / model_name / behavior / _k_tag(s) / "scored_results.csv").exists()
        for s in k_scales
    )


def run_model(
    model_cfg: ModelConfig,
    behaviors: list[str],
    k_scales: list[float],
    eval_dir: Path,
    directions_dir: Path,
    output_root: Path,
    figures_root: Path,
    exp_cfg: ExperimentConfig,
    registry: ModelRegistry,
    batch_size: int,
    force: bool = False,
) -> None:
    """Run the full ramp steering eval for one model across all behaviors and K scales.

    For each behavior:
      - Skips entirely if all outputs exist and ``force`` is False.
      - Runs baseline once (no steering hooks) and caches scores.
      - Sweeps K scales with ramp_pos and ramp_neg; injects cached baseline rows
        into each result DataFrame for unified plotting.

    Args:
        model_cfg:      Config for the model to evaluate.
        behaviors:      List of behavior names to evaluate.
        k_scales:       K scale multipliers for the sweep.
        eval_dir:       Directory containing behavior-specific JSONL eval prompt files.
        directions_dir: Root directory for extracted direction npz files.
        output_root:    Root directory for CSV outputs.
        figures_root:   Root directory for PDF plots.
        exp_cfg:        Loaded ExperimentConfig.
        registry:       ModelRegistry instance.
        batch_size:     Batch size for generation.
        force:          If True, overwrite existing results; otherwise skip.
    """
    logger.info(
        "=== %s — %d behaviors × %d scales ===",
        model_cfg.name, len(behaviors), len(k_scales),
    )

    behavior_directions: dict[str, list] = {}
    for behavior in behaviors:
        if not force and _results_complete(output_root, model_cfg.name, behavior, k_scales):
            logger.info("  [%s | %s] all results exist — skipping.", model_cfg.name, behavior)
            continue
        p = directions_dir / model_cfg.name / f"{behavior}.npz"
        if not p.exists():
            logger.error("Directions missing: %s — run experiment 02 first.", p)
            continue
        behavior_directions[behavior] = load_directions(str(p))

    if not behavior_directions:
        logger.info("=== %s — nothing to run ===", model_cfg.name)
        return

    judge = SmallModelJudge(model_id=exp_cfg.judge_model)

    with registry.loaded(model_cfg) as (model, tokenizer):
        steerer = ActivationSteerer(model)

        for behavior in behavior_directions:
            eval_prompts = load_behavior_prompts(behavior, eval_dir)
            prompts      = [item["prompt"] for item in eval_prompts]
            directions   = behavior_directions[behavior]

            # ------------------------------------------------------------------
            # Baseline: run once per behavior (no hooks), cache scores
            # ------------------------------------------------------------------
            baseline_out = output_root / model_cfg.name / behavior / "baseline"
            baseline_csv = baseline_out / "scored_results.csv"

            if not force and baseline_csv.exists():
                logger.info("  [%s | %s | baseline] exists — loading cached.", model_cfg.name, behavior)
                baseline_records = pd.read_csv(baseline_csv).to_dict("records")
            else:
                logger.info("  [%s | %s | baseline]", model_cfg.name, behavior)
                baseline_resps = generate_batched(
                    model, tokenizer, steerer, {},
                    prompts, exp_cfg.max_new_tokens, batch_size, model_cfg.extra,
                )
                baseline_records = [
                    {
                        "condition":          "baseline",
                        "behavior":           behavior,
                        "source":             item["source"],
                        "prompt_id":          item["id"],
                        "user_input":         item["prompt"],
                        "response":           resp,
                        "sycophantic_answer": item.get("sycophantic_answer", ""),
                        "honest_answer":      item.get("honest_answer", ""),
                    }
                    for item, resp in zip(eval_prompts, baseline_resps)
                ]
                baseline_scores = judge.score_records(baseline_records, batch_size=exp_cfg.judge_batch_size)
                for rec, score in zip(baseline_records, baseline_scores):
                    rec["judge_score"] = score
                baseline_out.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(baseline_records).to_csv(baseline_csv, index=False)

            logger.info(
                "  [%s | %s | baseline] mean_score=%.3f",
                model_cfg.name, behavior,
                float(np.mean([r["judge_score"] for r in baseline_records])),
            )

            # ------------------------------------------------------------------
            # K sweep: ramp_pos + ramp_neg only
            # ------------------------------------------------------------------
            pending_scales = [
                s for s in k_scales
                if force or not (
                    output_root / model_cfg.name / behavior / _k_tag(s) / "scored_results.csv"
                ).exists()
            ]
            if not pending_scales:
                logger.info("  [%s | %s] all k-scales exist — skipping sweep.", model_cfg.name, behavior)
                continue

            for scale in tqdm(pending_scales, desc=f"{model_cfg.name}|{behavior}", dynamic_ncols=True):
                records: list[dict] = []

                for condition in STEER_CONDITIONS:
                    cfg = _ramp_config(directions, scale) if condition == "ramp_pos" \
                          else _neg_config(directions, scale)
                    logger.info(
                        "  [%s | %s | k=%.2f | %s]", model_cfg.name, behavior, scale, condition
                    )
                    resps = generate_batched(
                        model, tokenizer, steerer, cfg,
                        prompts, exp_cfg.max_new_tokens, batch_size, model_cfg.extra,
                    )
                    for item, resp in zip(eval_prompts, resps):
                        records.append({
                            "condition":          condition,
                            "behavior":           behavior,
                            "source":             item["source"],
                            "prompt_id":          item["id"],
                            "user_input":         item["prompt"],
                            "response":           resp,
                            "sycophantic_answer": item.get("sycophantic_answer", ""),
                            "honest_answer":      item.get("honest_answer", ""),
                        })

                steer_scores = judge.score_records(records, batch_size=exp_cfg.judge_batch_size)
                for rec, score in zip(records, steer_scores):
                    rec["judge_score"] = score

                all_records = records + [dict(r) for r in baseline_records]

                out_dir = output_root / model_cfg.name / behavior / _k_tag(scale)
                out_dir.mkdir(parents=True, exist_ok=True)
                fig_dir = figures_root / model_cfg.name / behavior / _k_tag(scale)

                df = pd.DataFrame(all_records)
                df.to_csv(out_dir / "scored_results.csv", index=False)
                summary = df.groupby("condition")["judge_score"].agg(["mean", "std", "count"])
                summary.columns = ["mean_score", "std_score", "n"]
                summary.to_csv(out_dir / "summary.csv")
                logger.info(
                    "%s | %s | k=%.2f\n%s", model_cfg.name, behavior, scale, summary.to_string()
                )
                plot_results(df, behavior, model_cfg.name, scale, fig_dir)

    judge.unload()
    logger.info("=== %s done ===", model_cfg.name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 03 — ramp steering eval")
    parser.add_argument("--config",            default="config/models.yml")
    parser.add_argument("--experiment-config", default="config/experiment.yml")
    parser.add_argument("--directions",        default="results/directions")
    parser.add_argument("--eval-prompts",      default="data/eval_prompts")
    parser.add_argument("--output-dir",        default="results/ramp_eval")
    parser.add_argument("--figures-dir",       default="figures/ramp_eval")
    parser.add_argument("--batch-size",   type=int,   default=64)
    parser.add_argument("--models",     nargs="*", help="Explicit list; overrides --include-base.")
    parser.add_argument("--behaviors",  nargs="*", help="Default: from experiment.yml.")
    parser.add_argument("--k-scales",   nargs="*", type=float, help="Default: from experiment.yml.")
    parser.add_argument("--include-base", action="store_true",
                        help="Also evaluate base (non-instruct) models.")
    parser.add_argument("--force",        action="store_true",
                        help="Overwrite existing results instead of skipping.")
    parser.add_argument("--load-in-4bit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args     = parse_args()
    all_configs = load_model_configs(args.config)
    exp_cfg     = load_experiment_config(args.experiment_config)

    if args.models:
        missing = set(args.models) - set(all_configs)
        if missing:
            logger.error("Unknown models: %s", missing)
            sys.exit(1)
        target_names = args.models
    elif args.include_base:
        target_names = list(all_configs)
    else:
        target_names = [n for n, c in all_configs.items() if c.is_instruct]

    behaviors = args.behaviors or exp_cfg.behaviors
    k_scales  = args.k_scales  or exp_cfg.k_scales
    registry  = ModelRegistry(load_in_4bit=args.load_in_4bit, device_map="cuda:0")

    logger.info(
        "%d models × %d behaviors × %d k-scales | batch_size=%d | force=%s",
        len(target_names), len(behaviors), len(k_scales), args.batch_size, args.force,
    )

    for name in target_names:
        run_model(
            model_cfg=all_configs[name],
            behaviors=behaviors,
            k_scales=k_scales,
            eval_dir=Path(args.eval_prompts),
            directions_dir=Path(args.directions),
            output_root=Path(args.output_dir),
            figures_root=Path(args.figures_dir),
            exp_cfg=exp_cfg,
            registry=registry,
            batch_size=args.batch_size,
            force=args.force,
        )

    logger.info("All done.")


if __name__ == "__main__":
    main()
