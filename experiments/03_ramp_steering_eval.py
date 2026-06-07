"""Experiment 03 — Ramp steering evaluation across all models, behaviors, and K scales.

For each model × behavior:
  1. Run baseline once (no hooks) and all 4 steering conditions across all K scales.
  2. All conditions are distributed in parallel across N GPU workers via
     ``DataParallelGenerator`` (data-parallel; one model copy per GPU).
  3. After all generation completes, SmallModelJudge scores the full checkpoint
     using ``device_map="auto"`` across all GPUs.
  4. Saves scored CSVs + summary tables + plots per (behavior, k_scale).

Why data parallelism, not tensor parallelism (device_map="auto") or vLLM:
  - ``device_map="auto"`` pipeline-parallelises a single model across GPUs,
    serialising token flow and causing large pipeline bubbles — typically
    *slower* than 1 GPU for 7–9 B models that fit on one 40 GB card.
  - vLLM uses custom CUDA kernels that bypass ``register_forward_hook``
    entirely; our ``ActivationSteerer`` hooks would silently not fire.
  - Data parallelism replicates the full model on each GPU and fans out tasks;
    each worker owns its own model + steerer.  Full hook support, near-N× speed.

Behavior → benchmark mapping:
  safety     →  HarmBench + ClearHarm  (refusal scoring: SAFE / UNSAFE)
  sycophancy →  Anthropic model-written-evals  (SYCOPHANTIC / CONSISTENT)

Skip logic: any (model, behavior, k_scale) whose ``scored_results.csv`` already
exists is silently skipped.  Re-run from scratch with --force.

Base model support: base models (is_instruct=False) are evaluated using the
prompt_template defined in their extra config (models.yml).  By default only
instruct models run; pass --include-base or name them explicitly via --models.

Usage:
    # All instruct models, all behaviors — 8 GPU workers in parallel
    python experiments/03_ramp_steering_eval.py --n-gpus 8

    # Only safety behavior, explicit models
    python experiments/03_ramp_steering_eval.py \\
        --behaviors safety \\
        --models llama-3.1-8b-instruct qwen2.5-7b-instruct \\
        --n-gpus 8

    # Single-GPU debug run
    python experiments/03_ramp_steering_eval.py --n-gpus 1 --batch-size 8

    # Force re-run even if results already exist
    python experiments/03_ramp_steering_eval.py --force --n-gpus 8
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from activation_baking.config import ExperimentConfig, ModelConfig, load_experiment_config, load_model_configs
from activation_baking.direction_extractor import load_directions
from activation_baking.generation import DataParallelGenerator, generate_batched
from activation_baking.judges import SmallModelJudge
from activation_baking.model_utils import format_prompt
from activation_baking.registry import ModelRegistry
from activation_baking.steerer import ActivationSteerer
from transformers import PreTrainedTokenizerBase

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
        directions:  Extracted ``BehavioralDirection`` list (middle layers).
        scale:       K multiplier.
        sign:        +1.0 to amplify the behavior, -1.0 to suppress it.
        inject_mode: ``"broadcast"`` or ``"last_token"``.
    """
    return {
        d.layer_idx: (sign * d.mean_direction, d.k_value * scale, inject_mode)
        for d in directions
    }


_CONDITION_CONFIGS: dict[str, Any] = {
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
        KeyError:   If ``behavior`` is not in ``BEHAVIOR_SOURCES``.
        SystemExit: If any required JSONL file is missing.
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
        df:         DataFrame with columns ``condition``, ``judge_score``,
                    ``source``, ``prompt_id``.
        behavior:   Behavior name (used in plot titles).
        model_name: Short model slug.
        scale:      K scale multiplier (used in plot titles).
        fig_dir:    Output directory for PDF figures.
    """
    n_cond = len(CONDITIONS)
    agg = df.groupby("condition")["judge_score"].mean().reindex(CONDITIONS).fillna(0)

    fig, ax = plt.subplots(figsize=(max(6, n_cond * 1.4), 4))
    bars = ax.bar(
        range(n_cond), agg.values,
        color=[CONDITION_COLORS[c] for c in CONDITIONS], width=0.5,
    )
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_xticks(range(n_cond))
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITIONS], fontsize=9)
    ax.set_ylabel("Mean judge score")
    ax.set_ylim(-0.6, 1.15)
    ax.set_title(f"{model_name} | {behavior} | k={scale:.2f}")
    ax.grid(True, axis="y", alpha=0.3)
    _savefig(fig, fig_dir / "condition_comparison.pdf")

    sources = sorted(df["source"].unique())
    src_palette = {
        "harmbench": "#7c3aed",
        "clearharm": "#059669",
        "anthropic_syco": "#ea580c",
    }
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
        df.pivot_table(
            index="prompt_id", columns="condition", values="judge_score", aggfunc="mean"
        )
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
# Checkpoint helpers
# ---------------------------------------------------------------------------


def _k_tag(scale: Any) -> str:
    return f"k{float(scale):.2f}".replace(".", "_")


def _results_complete(
    output_root: Path,
    model_name: str,
    behavior: str,
    k_scales: list[float],
) -> bool:
    """Return True if all scored outputs for this model × behavior already exist."""
    baseline_csv = output_root / model_name / behavior / "baseline" / "scored_results.csv"
    if not baseline_csv.exists():
        return False
    return all(
        (output_root / model_name / behavior / _k_tag(s) / "scored_results.csv").exists()
        for s in k_scales
    )


def _load_checkpoint(ckpt_path: Path, force: bool) -> pd.DataFrame:
    if not force and ckpt_path.exists():
        df = pd.read_csv(ckpt_path)
        logger.info("  Resuming from checkpoint: %d records already saved", len(df))
        return df
    return pd.DataFrame()


def _is_ckpt(
    existing: pd.DataFrame,
    behavior: str,
    condition: str,
    k_scale: float | None,
) -> bool:
    if existing.empty:
        return False
    mask = (existing["behavior"] == behavior) & (existing["condition"] == condition)
    if k_scale is None:
        mask &= existing["k_scale"].isna()
    else:
        mask &= existing["k_scale"].notna() & (existing["k_scale"].astype(float) == k_scale)
    return bool(mask.any())


# ---------------------------------------------------------------------------
# Per-model generation — data-parallel fan-out
# ---------------------------------------------------------------------------


def run_model(
    model_cfg: ModelConfig,
    behaviors: list[str],
    k_scales: list[float],
    eval_dir: Path,
    directions_dir: Path,
    output_root: Path,
    figures_root: Path,
    exp_cfg: ExperimentConfig,
    n_gpus: int,
    batch_size: int,
    load_in_4bit: bool = False,
    force: bool = False,
    gpu_ids: list[int] | None = None,
) -> None:
    """Generate all responses for one model using N data-parallel GPU workers.

    Builds a flat task list of ``(task_id, layer_config, prompts)`` for every
    pending (behavior, condition, k_scale) triple, then fans all tasks out
    across ``n_gpus`` workers simultaneously.  Saves a raw checkpoint CSV after
    all tasks complete.  Does NOT run the judge.

    Args:
        model_cfg:      Config for the model to evaluate.
        behaviors:      Behavioral axes to evaluate.
        k_scales:       K scale multipliers.
        eval_dir:       Directory containing JSONL eval prompt files.
        directions_dir: Root directory for extracted direction npz files.
        output_root:    Root directory for CSV checkpoints and scored outputs.
        figures_root:   Root directory for PDF plots.
        exp_cfg:        Loaded ``ExperimentConfig``.
        n_gpus:         Number of GPU workers to spawn.
        batch_size:     Prompts per GPU forward pass.
        load_in_4bit:   Use 4-bit NF4 quantisation on every worker.
        force:          If True, ignore existing checkpoints and re-generate.
        gpu_ids:        Explicit GPU indices; defaults to ``range(n_gpus)``.
    """
    logger.info("=== [GEN] %s  (n_gpus=%d, batch=%d) ===", model_cfg.name, n_gpus, batch_size)

    # Load directions for behaviors that still have pending work
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
    existing = _load_checkpoint(ckpt_path, force)

    # Build flat task list — one task per (behavior, condition, k_scale)
    tasks: list[tuple[str, dict[int, tuple], list[str]]] = []
    task_meta: dict[str, dict] = {}  # task_id → {behavior, condition, k_scale, eval_prompts}

    for behavior, directions in behavior_directions.items():
        eval_prompts = load_behavior_prompts(behavior, eval_dir)
        prompts = [item["prompt"] for item in eval_prompts]

        # Baseline
        bl_dest = output_root / model_cfg.name / behavior / "baseline" / "scored_results.csv"
        if force or (not bl_dest.exists() and not _is_ckpt(existing, behavior, "baseline", None)):
            tid = f"{behavior}|baseline|None"
            tasks.append((tid, {}, prompts))
            task_meta[tid] = {
                "behavior": behavior,
                "condition": "baseline",
                "k_scale": None,
                "eval_prompts": eval_prompts,
            }

        # Steering conditions × k-scales
        for scale in k_scales:
            dest_done = (
                output_root / model_cfg.name / behavior / _k_tag(scale) / "scored_results.csv"
            ).exists()
            for condition in STEER_CONDITIONS:
                if not force and (
                    dest_done or _is_ckpt(existing, behavior, condition, scale)
                ):
                    continue
                layer_cfg = _CONDITION_CONFIGS[condition](directions, scale)
                tid = f"{behavior}|{condition}|{scale}"
                tasks.append((tid, layer_cfg, prompts))
                task_meta[tid] = {
                    "behavior": behavior,
                    "condition": condition,
                    "k_scale": scale,
                    "eval_prompts": eval_prompts,
                }

    if not tasks:
        logger.info("  %s — all tasks already checkpointed", model_cfg.name)
        return

    logger.info(
        "  %s — %d tasks queued across %d GPUs", model_cfg.name, len(tasks), n_gpus,
    )

    # Fan all tasks out across n_gpus workers simultaneously
    with DataParallelGenerator(
        n_gpus=n_gpus,
        model_cfg=model_cfg,
        load_in_4bit=load_in_4bit,
        gpu_ids=gpu_ids,
    ) as gen:
        all_responses = gen.generate_all(
            tasks=tasks,
            max_new_tokens=exp_cfg.max_new_tokens,
            batch_size=batch_size,
            extra_cfg=model_cfg.extra,
        )

    # Assemble new records
    new_records: list[dict] = []
    for tid, responses in all_responses.items():
        meta = task_meta[tid]
        for item, resp in zip(meta["eval_prompts"], responses):
            new_records.append({
                "model_name": model_cfg.name,
                "condition":  meta["condition"],
                "behavior":   meta["behavior"],
                "source":     item["source"],
                "prompt_id":  item["id"],
                "user_input": item["prompt"],
                "response":   resp,
                "sycophantic_answer": item.get("sycophantic_answer", ""),
                "honest_answer":      item.get("honest_answer", ""),
                "k_scale":    meta["k_scale"],
            })

    if new_records:
        combined = (
            pd.concat([existing, pd.DataFrame(new_records)], ignore_index=True)
            if not existing.empty
            else pd.DataFrame(new_records)
        )
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(ckpt_path, index=False)
        logger.info(
            "  %s — checkpoint saved (%d total records)", model_cfg.name, len(combined)
        )
    else:
        logger.info("  %s — no new records", model_cfg.name)


# ---------------------------------------------------------------------------
# Judge pass
# ---------------------------------------------------------------------------


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

    Loads the raw checkpoint, runs ``SmallModelJudge`` across all GPUs in one
    pass, then saves per-condition/per-K-scale CSVs and figures.

    Args:
        model_name:   Short model slug.
        behaviors:    Behavioral axes to score.
        k_scales:     K scale multipliers.
        output_root:  Root directory for scored CSV outputs.
        figures_root: Root directory for PDF plots.
        exp_cfg:      Loaded ``ExperimentConfig``.
        force:        If True, re-score even if outputs already exist.
    """
    ckpt_path = output_root / model_name / "raw_responses.csv"
    if not ckpt_path.exists():
        logger.info("  %s — no checkpoint, skipping judge", model_name)
        return

    raw = pd.read_csv(ckpt_path)

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
    judge  = SmallModelJudge(model_id=exp_cfg.judge_model, device_map="auto")
    scores = judge.score_records(to_score, batch_size=exp_cfg.judge_batch_size)
    judge.unload()

    for rec, score in zip(to_score, scores):
        rec["judge_score"] = score
    scored = pd.DataFrame(to_score)

    def _baseline_rows(behavior: str) -> pd.DataFrame:
        p = output_root / model_name / behavior / "baseline" / "scored_results.csv"
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    for behavior in tqdm(
        behaviors, desc=f"{model_name} saving", leave=False, dynamic_ncols=True
    ):
        bl = scored[(scored["behavior"] == behavior) & (scored["condition"] == "baseline")]
        if not bl.empty:
            out = output_root / model_name / behavior / "baseline"
            out.mkdir(parents=True, exist_ok=True)
            bl.to_csv(out / "scored_results.csv", index=False)
            logger.info(
                "  %s | %s | baseline  mean=%.3f",
                model_name, behavior, bl["judge_score"].mean(),
            )

        baseline_df = _baseline_rows(behavior)
        for scale in tqdm(
            k_scales, desc=f"{behavior} k-scales", leave=False, dynamic_ncols=True
        ):
            scale_rows = scored[
                (scored["behavior"] == behavior)
                & scored["k_scale"].notna()
                & (scored["k_scale"].astype(float) == scale)
            ]
            if scale_rows.empty:
                continue
            all_df = (
                pd.concat([scale_rows, baseline_df], ignore_index=True)
                if not baseline_df.empty
                else scale_rows
            )
            out_dir = output_root / model_name / behavior / _k_tag(scale)
            fig_dir = figures_root / model_name / behavior / _k_tag(scale)
            out_dir.mkdir(parents=True, exist_ok=True)
            all_df.to_csv(out_dir / "scored_results.csv", index=False)
            summary = all_df.groupby("condition")["judge_score"].agg(["mean", "std", "count"])
            summary.columns = ["mean_score", "std_score", "n"]
            summary.to_csv(out_dir / "summary.csv")
            logger.info(
                "  %s | %s | k=%.2f\n%s", model_name, behavior, scale, summary.to_string()
            )
            plot_results(all_df, behavior, model_name, scale, fig_dir)

    logger.info("=== [DONE] %s ===", model_name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 03 — ramp steering eval")
    parser.add_argument("--config",            default="config/models.yml")
    parser.add_argument("--experiment-config", default="config/experiment.yml")
    parser.add_argument("--directions",        default="results/directions")
    parser.add_argument("--eval-prompts",      default="data/eval_prompts")
    parser.add_argument("--output-dir",        default="results/ramp_eval")
    parser.add_argument("--figures-dir",       default="figures/ramp_eval")
    parser.add_argument("--batch-size",  type=int, default=64,
                        help="Prompts per GPU batch (default 64; use 128+ on 40 GB A100).")
    parser.add_argument("--n-gpus",      type=int, default=None,
                        help="GPU workers per model (default: all available GPUs).")
    parser.add_argument("--models",    nargs="*", help="Explicit list; overrides --include-base.")
    parser.add_argument("--behaviors", nargs="*", help="Default: from experiment.yml.")
    parser.add_argument("--k-scales",  nargs="*", type=float, help="Default: from experiment.yml.")
    parser.add_argument("--include-base", action="store_true",
                        help="Also evaluate base (non-instruct) models.")
    parser.add_argument("--force",        action="store_true",
                        help="Overwrite existing results instead of skipping.")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="4-bit NF4 quantisation on all GPU workers (halves VRAM).")
    return parser.parse_args()


def main() -> None:
    args        = parse_args()
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
    k_scales  = [float(s) for s in (args.k_scales or exp_cfg.k_scales)]

    n_gpus_available = torch.cuda.device_count()
    n_gpus = args.n_gpus if args.n_gpus is not None else n_gpus_available
    n_gpus = max(1, min(n_gpus, n_gpus_available))

    output_root  = Path(args.output_dir)
    figures_root = Path(args.figures_dir)

    # Total task count for upfront logging
    n_tasks_per_model = (1 + len(STEER_CONDITIONS) * len(k_scales)) * len(behaviors)
    logger.info(
        "%d models × %d behaviors × %d k-scales × %d cond = %d tasks/model | "
        "batch=%d | GPUs=%d/%d (workers per model)",
        len(target_names), len(behaviors), len(k_scales), len(CONDITIONS),
        n_tasks_per_model, args.batch_size, n_gpus, n_gpus_available,
    )

    # ---------------------------------------------------------------------------
    # Phase 1 — generation (data-parallel; one model at a time, N GPU workers)
    # ---------------------------------------------------------------------------
    logger.info("Phase 1: generation — %d GPU workers per model", n_gpus)
    failed: list[str] = []

    for name in tqdm(target_names, desc="Generation", unit="model", dynamic_ncols=True):
        try:
            run_model(
                model_cfg=all_configs[name],
                behaviors=behaviors,
                k_scales=k_scales,
                eval_dir=Path(args.eval_prompts),
                directions_dir=Path(args.directions),
                output_root=output_root,
                figures_root=figures_root,
                exp_cfg=exp_cfg,
                n_gpus=n_gpus,
                batch_size=args.batch_size,
                load_in_4bit=args.load_in_4bit,
                force=args.force,
            )
        except Exception as exc:
            logger.error("Generation failed for %s: %s", name, exc, exc_info=True)
            failed.append(name)

    if failed:
        logger.error("Generation failed for: %s", failed)

    # ---------------------------------------------------------------------------
    # Phase 2 — judge: Qwen32B loads across all GPUs once all generation is done
    # ---------------------------------------------------------------------------
    logger.info("Phase 2: all generation complete — starting judge pass (all %d GPUs)...", n_gpus_available)

    for name in tqdm(target_names, desc="Judging", unit="model", dynamic_ncols=True):
        if name in failed:
            logger.info("  Skipping %s (generation failed)", name)
            continue
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
