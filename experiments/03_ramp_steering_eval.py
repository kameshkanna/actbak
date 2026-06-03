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

# Conditions: CAA mean_direction, both injection modes, both signs
CONDITIONS = [
    "baseline",
    "caa_broadcast_pos",
    "caa_broadcast_neg",
    "caa_last_token_pos",
    "caa_last_token_neg",
]
STEER_CONDITIONS = [
    "caa_broadcast_pos",
    "caa_broadcast_neg",
    "caa_last_token_pos",
    "caa_last_token_neg",
]
CONDITION_COLORS = {
    "baseline":           "#6b7280",
    "caa_broadcast_pos":  "#2563eb",
    "caa_broadcast_neg":  "#93c5fd",
    "caa_last_token_pos": "#dc2626",
    "caa_last_token_neg": "#fca5a5",
}
CONDITION_LABELS = {
    "baseline":           "Baseline",
    "caa_broadcast_pos":  "CAA Broadcast +",
    "caa_broadcast_neg":  "CAA Broadcast −",
    "caa_last_token_pos": "CAA LastTok +",
    "caa_last_token_neg": "CAA LastTok −",
}

# Behavior → eval prompt JSONL filenames (all under data/eval_prompts/)
BEHAVIOR_SOURCES: dict[str, list[str]] = {
    "safety":     ["harmbench.jsonl", "clearharm.jsonl"],
    "sycophancy": ["sycophancy.jsonl"],
}


# ---------------------------------------------------------------------------
# Steering configs
# ---------------------------------------------------------------------------

def _caa_config(
    directions: list,
    scale: float,
    sign: float,
    inject_mode: str,
) -> dict[int, tuple]:
    """Build a CAA layer config using mean_direction (contrastive mean diff).

    Args:
        directions:   Extracted BehavioralDirection list (middle layers).
        scale:        K multiplier.
        sign:         +1.0 to amplify the behavior, -1.0 to suppress it.
        inject_mode:  ``"broadcast"`` or ``"last_token"``.
    """
    return {
        d.layer_idx: (sign * d.mean_direction, d.k_value * scale, inject_mode)
        for d in directions
    }

_CONDITION_CONFIGS = {
    "caa_broadcast_pos":  lambda dirs, s: _caa_config(dirs, s, +1.0, "broadcast"),
    "caa_broadcast_neg":  lambda dirs, s: _caa_config(dirs, s, -1.0, "broadcast"),
    "caa_last_token_pos": lambda dirs, s: _caa_config(dirs, s, +1.0, "last_token"),
    "caa_last_token_neg": lambda dirs, s: _caa_config(dirs, s, -1.0, "last_token"),
}


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
    n_cond = len(CONDITIONS)
    agg = df.groupby("condition")["judge_score"].mean().reindex(CONDITIONS).fillna(0)
    fig, ax = plt.subplots(figsize=(max(6, n_cond * 1.4), 4))
    bars = ax.bar(range(n_cond), agg.values, color=[CONDITION_COLORS[c] for c in CONDITIONS], width=0.5)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_xticks(range(n_cond))
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITIONS], fontsize=9)
    ax.set_ylabel("Mean judge score")
    ax.set_ylim(-0.6, 1.15)
    ax.set_title(f"{model_name} | {behavior} | k={scale:.2f}")
    ax.grid(True, axis="y", alpha=0.3)
    _savefig(fig, fig_dir / "condition_comparison.pdf")

    sources = sorted(df["source"].unique())
    src_palette = {"harmbench": "#7c3aed", "clearharm": "#059669", "anthropic_syco": "#ea580c"}
    x = np.arange(n_cond)
    fig, ax = plt.subplots(figsize=(max(7, n_cond * 1.6), 4))
    for i, src in enumerate(sources):
        sub = (
            df[df["source"] == src]
            .groupby("condition")["judge_score"]
            .mean()
            .reindex(CONDITIONS)
            .fillna(0)
        )
        ax.bar(
            x + (i - len(sources) / 2 + 0.5) * 0.25, sub.values, 0.25,
            label=src, color=src_palette.get(src, "gray"),
        )
    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITIONS], fontsize=9)
    ax.set_ylabel("Mean judge score")
    ax.set_ylim(-0.6, 1.15)
    ax.set_title(f"{model_name} | {behavior} | k={scale:.2f} — by source")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    _savefig(fig, fig_dir / "by_source.pdf")

    pivot = (
        df.pivot_table(index="prompt_id", columns="condition", values="judge_score", aggfunc="mean")
        .reindex(columns=CONDITIONS)
        .fillna(0)
    )
    fig, ax = plt.subplots(figsize=(max(6, n_cond * 1.2), max(4, len(pivot) * 0.3)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(n_cond))
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITIONS], fontsize=8)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels([str(pid)[:30] for pid in pivot.index], fontsize=6)
    plt.colorbar(im, ax=ax, label="Judge score")
    ax.set_title(f"{model_name} | {behavior} | k={scale:.2f}")
    _savefig(fig, fig_dir / "per_prompt_heatmap.pdf")


# ---------------------------------------------------------------------------
# Per-model runner
# ---------------------------------------------------------------------------

def _k_tag(scale) -> str:
    return f"k{float(scale):.2f}".replace(".", "_")


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
    """Generate all responses for one model and checkpoint raw responses to disk.

    Does NOT run the judge. Saves to ``output_root/{model}/raw_responses.csv``.
    Resumes cleanly from an existing checkpoint — already-checkpointed or
    already-scored rows are skipped without re-generation.
    """
    logger.info("=== [GEN] %s ===", model_cfg.name)

    # Load directions for behaviors that still need work
    behavior_directions: dict[str, list] = {}
    for behavior in behaviors:
        if not force and _results_complete(output_root, model_cfg.name, behavior, k_scales):
            logger.info("  %s | %s — fully scored, skipping", model_cfg.name, behavior)
            continue
        p = directions_dir / model_cfg.name / f"{behavior}.npz"
        if not p.exists():
            logger.error("Directions missing: %s — run experiment 02 first.", p)
            continue
        behavior_directions[behavior] = load_directions(str(p))

    if not behavior_directions:
        logger.info("  %s — nothing to generate", model_cfg.name)
        return

    ckpt_path = output_root / model_cfg.name / "raw_responses.csv"
    existing  = pd.read_csv(ckpt_path) if (not force and ckpt_path.exists()) else pd.DataFrame()
    if not existing.empty:
        logger.info("  Resuming from checkpoint: %d records already saved", len(existing))

    def _is_ckpt(behavior: str, condition: str, k_scale) -> bool:
        if existing.empty:
            return False
        mask = (existing["behavior"] == behavior) & (existing["condition"] == condition)
        if k_scale is None:
            mask &= existing["k_scale"].isna()
        else:
            mask &= existing["k_scale"].notna() & (existing["k_scale"].astype(float) == k_scale)
        return mask.any()

    new_records: list[dict] = []

    with registry.loaded(model_cfg) as (model, tokenizer):
        steerer = ActivationSteerer(model)

        for behavior in tqdm(behavior_directions, desc=f"{model_cfg.name} behaviors", leave=False, dynamic_ncols=True):
            eval_prompts = load_behavior_prompts(behavior, eval_dir)
            prompts      = [item["prompt"] for item in eval_prompts]
            directions   = behavior_directions[behavior]

            # Baseline
            bl_scored = output_root / model_cfg.name / behavior / "baseline" / "scored_results.csv"
            if not force and (bl_scored.exists() or _is_ckpt(behavior, "baseline", None)):
                logger.info("  %s | %s | baseline — skip", model_cfg.name, behavior)
            else:
                resps = generate_batched(
                    model, tokenizer, steerer, {},
                    prompts, exp_cfg.max_new_tokens, batch_size, model_cfg.extra,
                )
                for item, resp in zip(eval_prompts, resps):
                    new_records.append({
                        "model_name": model_cfg.name, "condition": "baseline",
                        "behavior": behavior, "source": item["source"],
                        "prompt_id": item["id"], "user_input": item["prompt"],
                        "response": resp,
                        "sycophantic_answer": item.get("sycophantic_answer", ""),
                        "honest_answer":      item.get("honest_answer", ""),
                        "k_scale": None,
                    })

            # K sweep
            pending_scales = [
                s for s in k_scales
                if force
                or not (output_root / model_cfg.name / behavior / _k_tag(s) / "scored_results.csv").exists()
                and not all(_is_ckpt(behavior, c, s) for c in STEER_CONDITIONS)
            ]

            for scale in tqdm(pending_scales, desc=f"{behavior} k-sweep", leave=False, dynamic_ncols=True):
                for condition in tqdm(STEER_CONDITIONS, desc=f"k={float(scale):.2f}", leave=False, dynamic_ncols=True):
                    if not force and _is_ckpt(behavior, condition, scale):
                        continue
                    cfg   = _CONDITION_CONFIGS[condition](directions, scale)
                    resps = generate_batched(
                        model, tokenizer, steerer, cfg,
                        prompts, exp_cfg.max_new_tokens, batch_size, model_cfg.extra,
                    )
                    for item, resp in zip(eval_prompts, resps):
                        new_records.append({
                            "model_name": model_cfg.name, "condition": condition,
                            "behavior": behavior, "source": item["source"],
                            "prompt_id": item["id"], "user_input": item["prompt"],
                            "response": resp,
                            "sycophantic_answer": item.get("sycophantic_answer", ""),
                            "honest_answer":      item.get("honest_answer", ""),
                            "k_scale": scale,
                        })

    if new_records:
        combined = pd.concat([existing, pd.DataFrame(new_records)], ignore_index=True) \
                   if not existing.empty else pd.DataFrame(new_records)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(ckpt_path, index=False)
        logger.info("  %s — checkpoint saved (%d total records)", model_cfg.name, len(combined))
    else:
        logger.info("  %s — no new records", model_cfg.name)


def score_and_save(
    model_name: str,
    behaviors: list[str],
    k_scales: list[float],
    output_root: Path,
    figures_root: Path,
    exp_cfg: ExperimentConfig,
    force: bool = False,
) -> None:
    """Score all raw responses for one model and write final CSVs + plots.

    Loads ``raw_responses.csv``, runs SmallModelJudge in one pass, saves
    scored CSVs per condition and per K-scale, then plots.
    """
    ckpt_path = output_root / model_name / "raw_responses.csv"
    if not ckpt_path.exists():
        logger.info("  %s — no checkpoint, skipping judge", model_name)
        return

    raw = pd.read_csv(ckpt_path)

    # Only score rows whose destination CSV doesn't exist yet
    to_score: list[dict] = []
    for _, row in raw.iterrows():
        behavior = row["behavior"]
        k_scale  = row["k_scale"]
        dest = (
            output_root / model_name / behavior / "baseline" / "scored_results.csv"
            if pd.isna(k_scale)
            else output_root / model_name / behavior / _k_tag(float(k_scale)) / "scored_results.csv"
        )
        if force or not dest.exists():
            to_score.append(row.to_dict())

    if not to_score:
        logger.info("  %s — all records already scored", model_name)
        return

    logger.info("=== [JUDGE] %s — %d records ===", model_name, len(to_score))
    judge  = SmallModelJudge(model_id=exp_cfg.judge_model)
    scores = judge.score_records(to_score, batch_size=exp_cfg.judge_batch_size)
    judge.unload()

    for rec, score in zip(to_score, scores):
        rec["judge_score"] = score
    scored = pd.DataFrame(to_score)

    def _baseline_rows(behavior: str) -> pd.DataFrame:
        p = output_root / model_name / behavior / "baseline" / "scored_results.csv"
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    for behavior in tqdm(behaviors, desc=f"{model_name} saving", leave=False, dynamic_ncols=True):
        # Save baseline
        bl = scored[(scored["behavior"] == behavior) & (scored["condition"] == "baseline")]
        if not bl.empty:
            out = output_root / model_name / behavior / "baseline"
            out.mkdir(parents=True, exist_ok=True)
            bl.to_csv(out / "scored_results.csv", index=False)
            logger.info("  %s | %s | baseline  mean=%.3f", model_name, behavior, bl["judge_score"].mean())

        # Save each K-scale
        baseline_df = _baseline_rows(behavior)
        for scale in tqdm(k_scales, desc=f"{behavior} k-scales", leave=False, dynamic_ncols=True):
            scale_rows = scored[
                (scored["behavior"] == behavior)
                & scored["k_scale"].notna()
                & (scored["k_scale"].astype(float) == scale)
            ]
            if scale_rows.empty:
                continue
            all_df  = pd.concat([scale_rows, baseline_df], ignore_index=True) \
                      if not baseline_df.empty else scale_rows
            out_dir = output_root / model_name / behavior / _k_tag(scale)
            fig_dir = figures_root / model_name / behavior / _k_tag(scale)
            out_dir.mkdir(parents=True, exist_ok=True)
            all_df.to_csv(out_dir / "scored_results.csv", index=False)
            summary = all_df.groupby("condition")["judge_score"].agg(["mean", "std", "count"])
            summary.columns = ["mean_score", "std_score", "n"]
            summary.to_csv(out_dir / "summary.csv")
            logger.info("  %s | %s | k=%.2f\n%s", model_name, behavior, scale, summary.to_string())
            plot_results(all_df, behavior, model_name, scale, fig_dir)

    logger.info("=== [DONE] %s ===", model_name)


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

    output_root  = Path(args.output_dir)
    figures_root = Path(args.figures_dir)
    eval_dir     = Path(args.eval_prompts)
    directions   = Path(args.directions)

    # -----------------------------------------------------------------------
    # Phase 1 — generate all models, checkpoint after each one
    # Judge is NOT loaded yet; GPU memory is fully dedicated to generation.
    # -----------------------------------------------------------------------
    for name in tqdm(target_names, desc="Generation", unit="model", dynamic_ncols=True):
        run_model(
            model_cfg=all_configs[name],
            behaviors=behaviors,
            k_scales=k_scales,
            eval_dir=eval_dir,
            directions_dir=directions,
            output_root=output_root,
            figures_root=figures_root,
            exp_cfg=exp_cfg,
            registry=registry,
            batch_size=args.batch_size,
            force=args.force,
        )

    # -----------------------------------------------------------------------
    # Phase 2 — all generation done; load judge and score each model in turn
    # -----------------------------------------------------------------------
    logger.info("All generation complete. Starting judge phase...")
    for name in tqdm(target_names, desc="Judging", unit="model", dynamic_ncols=True):
        score_and_save(
            model_name=name,
            behaviors=behaviors,
            k_scales=k_scales,
            output_root=output_root,
            figures_root=figures_root,
            exp_cfg=exp_cfg,
            force=args.force,
        )

    logger.info("All done.")


if __name__ == "__main__":
    main()
