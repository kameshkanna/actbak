"""Optimised experiment 03 runner — one model load per model, batched, all GPUs.

Strategy
--------
Original exp03: 84 subprocess calls → model loaded 84 times (once per k-scale).
This runner:    4 model loads total (one per model family).

For each model:
  1. Load once across all GPUs (device_map='auto').
  2. Run ALL behavior × k-scale × condition combinations while loaded.
  3. Batch prompts together for faster generation.
  4. Unload model, load judge on GPU 0, score everything, save results.

Usage:
    python run_exp03.py                    # all models, all behaviours, all scales
    python run_exp03.py --batch-size 16    # larger batch (more VRAM per GPU)
    python run_exp03.py --models llama-3.1-8b-instruct --behaviors safety
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from itertools import product
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

CONDITIONS = ["baseline", "ramp_pos", "ramp_neg"]
CONDITION_COLORS = {"baseline": "#6b7280", "ramp_pos": "#2563eb", "ramp_neg": "#dc2626"}
CONDITION_LABELS = {"baseline": "Baseline", "ramp_pos": "Ramp+ (safety↑)", "ramp_neg": "Ramp− (safety↓)"}


# ---------------------------------------------------------------------------
# Steering config builders
# ---------------------------------------------------------------------------

def _build_ramp_config(directions: list, scale: float) -> dict:
    if scale == 0.0:
        return {}
    return {d.layer_idx: (d.mean_direction, d.k_value * scale) for d in directions}


def _build_neg_config(directions: list, scale: float) -> dict:
    if scale == 0.0:
        return {}
    return {d.layer_idx: (-d.mean_direction, d.k_value * scale) for d in directions}


# ---------------------------------------------------------------------------
# Batched generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_batched(
    model,
    tokenizer,
    steerer: ActivationSteerer,
    layer_config: dict,
    prompts: list[str],
    max_new_tokens: int,
    batch_size: int,
    extra_cfg: dict,
) -> list[str]:
    """Generate responses for all prompts in batches with steering active."""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    responses: list[str] = []
    device = next(model.parameters()).device

    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        formatted = [format_prompt(tokenizer, p, extra_cfg) for p in batch]
        inputs = tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
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
            new_tokens = output_ids[prompt_len:]
            responses.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()

    return responses


# ---------------------------------------------------------------------------
# Eval prompt loader
# ---------------------------------------------------------------------------

def load_eval_prompts(eval_dir: Path, n_harmbench: int, n_clearharm: int) -> list[dict]:
    records: list[dict] = []
    for fname, key, cap in (
        ("harmbench.jsonl", "harmbench", n_harmbench),
        ("clearharm.jsonl", "clearharm", n_clearharm),
    ):
        path = eval_dir / fname
        if not path.exists():
            logger.error("%s not found — run `python data/download_datasets.py` first.", path)
            sys.exit(1)
        batch: list[dict] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                if len(batch) >= cap:
                    break
                batch.append(json.loads(line))
        records.extend(batch)
    logger.info("Loaded %d eval prompts.", len(records))
    return records


# ---------------------------------------------------------------------------
# Plotting (mirrors exp03 output format)
# ---------------------------------------------------------------------------

def _savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", path)


def plot_condition_comparison(df: pd.DataFrame, behavior: str, model_name: str, scale: float, fig_dir: Path) -> None:
    agg = df.groupby("condition")["judge_score"].mean().reindex(CONDITIONS).fillna(0)
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(range(3), agg.values, color=[CONDITION_COLORS[c] for c in CONDITIONS], width=0.5)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_xticks(range(3))
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITIONS], fontsize=9)
    ax.set_ylabel("Mean refusal score")
    ax.set_ylim(0, 1.15)
    ax.set_title(f"{model_name} | {behavior} | k-scale={scale}")
    ax.grid(True, axis="y", alpha=0.3)
    _savefig(fig, fig_dir / "condition_comparison.pdf")


def plot_per_prompt_heatmap(df: pd.DataFrame, behavior: str, model_name: str, scale: float, fig_dir: Path) -> None:
    pivot = (
        df.pivot_table(index="prompt_id", columns="condition", values="judge_score", aggfunc="mean")
        .reindex(columns=CONDITIONS).fillna(0)
    )
    fig, ax = plt.subplots(figsize=(6, max(4, len(pivot) * 0.3)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITIONS], fontsize=8)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels([str(pid)[:30] for pid in pivot.index], fontsize=6)
    plt.colorbar(im, ax=ax, label="Refusal score")
    ax.set_title(f"{model_name} | {behavior} | k={scale}")
    _savefig(fig, fig_dir / "per_prompt_heatmap.pdf")


# ---------------------------------------------------------------------------
# Per-model runner
# ---------------------------------------------------------------------------

def run_model(
    model_cfg: ModelConfig,
    behaviors: list[str],
    k_scales: list[float],
    eval_prompts: list[dict],
    directions_dir: Path,
    output_root: Path,
    figures_root: Path,
    exp_cfg,
    registry: ModelRegistry,
    batch_size: int,
) -> None:
    logger.info("=== %s — loading once for %d behaviors × %d scales ===",
                model_cfg.name, len(behaviors), len(k_scales))

    # Load directions for all behaviors first (CPU, cheap)
    behavior_directions: dict[str, list] = {}
    for behavior in behaviors:
        dir_path = directions_dir / model_cfg.name / f"{behavior}.npz"
        if not dir_path.exists():
            logger.error("Directions missing: %s — run exp02 first.", dir_path)
            continue
        behavior_directions[behavior] = load_directions(str(dir_path))
    if not behavior_directions:
        logger.error("No directions found for %s — skipping.", model_cfg.name)
        return

    prompts = [item["prompt"] for item in eval_prompts]
    n = len(prompts)

    # Collect all raw records across all behavior × k-scale × condition combos
    # shape: {(behavior, scale): {condition: [responses]}}
    all_responses: dict[tuple, dict[str, list[str]]] = {}

    with registry.loaded(model_cfg) as (model, tokenizer):
        steerer = ActivationSteerer(model)

        for behavior, scale in tqdm(
            list(product(behavior_directions.keys(), k_scales)),
            desc=f"{model_cfg.name} — generating",
            dynamic_ncols=True,
        ):
            directions = behavior_directions[behavior]
            configs = {
                "baseline":  {},
                "ramp_pos":  _build_ramp_config(directions, scale),
                "ramp_neg":  _build_neg_config(directions, scale),
            }

            combo_responses: dict[str, list[str]] = {}
            for condition in CONDITIONS:
                logger.info("  %s | %s | k=%.2f | [%s]", model_cfg.name, behavior, scale, condition)
                resps = generate_batched(
                    model, tokenizer, steerer, configs[condition],
                    prompts, exp_cfg.max_new_tokens, batch_size, model_cfg.extra,
                )
                combo_responses[condition] = resps
            all_responses[(behavior, scale)] = combo_responses

    # Judge all responses (model unloaded, GPU free)
    logger.info("Judging all responses for %s...", model_cfg.name)
    judge = SmallModelJudge(model_id=exp_cfg.judge_model)

    for (behavior, scale), cond_responses in all_responses.items():
        records: list[dict] = []
        for condition, responses in cond_responses.items():
            for item, resp in zip(eval_prompts, responses):
                records.append({
                    "condition":  condition,
                    "behavior":   behavior,
                    "source":     item["source"],
                    "prompt_id":  item["id"],
                    "user_input": item["prompt"],
                    "response":   resp,
                })

        scores = judge.score_records(records, batch_size=exp_cfg.judge_batch_size)
        for rec, score in zip(records, scores):
            rec["judge_score"] = score

        k_tag = f"k{scale:.2f}".replace(".", "_")
        out_dir = output_root / model_cfg.name / behavior / k_tag
        out_dir.mkdir(parents=True, exist_ok=True)
        fig_dir = figures_root / model_cfg.name / behavior / k_tag

        df = pd.DataFrame(records)
        df.to_csv(out_dir / "raw_generations.csv", index=False)
        df.to_csv(out_dir / "scored_results.csv", index=False)

        summary = df.groupby("condition")["judge_score"].agg(["mean", "std", "count"])
        summary.columns = ["mean_score", "std_score", "n"]
        summary.to_csv(out_dir / "summary.csv")
        logger.info("\n%s | %s | k=%.2f\n%s", model_cfg.name, behavior, scale, summary.to_string())

        plot_condition_comparison(df, behavior, model_cfg.name, scale, fig_dir)
        plot_per_prompt_heatmap(df, behavior, model_cfg.name, scale, fig_dir)

    judge.unload()
    logger.info("=== %s done ===", model_cfg.name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimised exp03 — one model load, batched, all GPUs")
    parser.add_argument("--config",            default="config/models.yml")
    parser.add_argument("--experiment-config", default="config/experiment.yml")
    parser.add_argument("--directions",        default="results/directions")
    parser.add_argument("--eval-prompts",      default="data/eval_prompts")
    parser.add_argument("--output-dir",        default="results/ramp_eval")
    parser.add_argument("--figures-dir",       default="figures/ramp_eval")
    parser.add_argument("--batch-size",  type=int, default=8,
                        help="Prompts per generation batch (default: 8).")
    parser.add_argument("--models",   nargs="*",
                        help="Model names to run (default: all instruct models).")
    parser.add_argument("--behaviors", nargs="*",
                        help="Behaviors to run (default: from experiment.yml).")
    parser.add_argument("--k-scales",  nargs="*", type=float,
                        help="K-scales to sweep (default: from experiment.yml).")
    parser.add_argument("--load-in-4bit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    all_configs = load_model_configs(args.config)
    exp_cfg     = load_experiment_config(args.experiment_config)

    if args.models:
        missing = set(args.models) - set(all_configs)
        if missing:
            logger.error("Unknown models: %s", missing)
            sys.exit(1)
        target_names = args.models
    else:
        target_names = [n for n, c in all_configs.items() if c.is_instruct]

    behaviors = args.behaviors or exp_cfg.behaviors
    k_scales  = args.k_scales  or exp_cfg.k_scales

    eval_prompts = load_eval_prompts(
        Path(args.eval_prompts), exp_cfg.n_harmbench, exp_cfg.n_clearharm
    )

    # device_map="auto" spans all visible GPUs — no CUDA_VISIBLE_DEVICES restriction
    registry = ModelRegistry(load_in_4bit=args.load_in_4bit, device_map="auto")

    logger.info(
        "Running: %d models × %d behaviors × %d k-scales | batch_size=%d",
        len(target_names), len(behaviors), len(k_scales), args.batch_size,
    )

    for name in target_names:
        run_model(
            model_cfg=all_configs[name],
            behaviors=behaviors,
            k_scales=k_scales,
            eval_prompts=eval_prompts,
            directions_dir=Path(args.directions),
            output_root=Path(args.output_dir),
            figures_root=Path(args.figures_dir),
            exp_cfg=exp_cfg,
            registry=registry,
            batch_size=args.batch_size,
        )

    logger.info("All done.")


if __name__ == "__main__":
    main()
