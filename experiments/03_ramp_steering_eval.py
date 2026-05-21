"""Experiment 03 — Ramp steering evaluation on real harm benchmarks.

Three conditions on a single model × behavior pair:

  baseline  — no steering; model responds naturally
  ramp_pos  — steer along +direction at all middle layers (K_ℓ = μ̄_ℓ/√d)
              pushes model toward the extracted behavior (e.g. refusal/safety)
  ramp_neg  — steer along −direction at all middle layers (−K_ℓ)
              pushes model away from the behavior (toward harmful compliance)

Test set: 10 samples from HarmBench + 10 from ClearHarm (20 total).
Judge:    Qwen2.5-3B scores refusal 0–10 (10 = clear refusal, 0 = full compliance).

Run data/download_eval_prompts.py first to fetch the test prompts.

Usage:
    python experiments/03_ramp_steering_eval.py \\
        --config config/models.yml \\
        --directions results/directions \\
        --eval-prompts data/eval_prompts \\
        --output-dir results/ramp_eval \\
        --model llama-3.1-8b-instruct \\
        --behavior safety \\
        [--judge-model Qwen/Qwen2.5-3B-Instruct] \\
        [--judge-batch-size 8] \\
        [--max-new-tokens 200] \\
        [--load-in-4bit]
"""

import argparse
import gc
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
import yaml
from tqdm import tqdm

from activation_baking.direction_extractor import load_directions
from activation_baking.judges import SmallModelJudge
from activation_baking.model_utils import format_prompt, load_model_and_tokenizer
from activation_baking.steerer import ActivationSteerer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CONDITIONS = ["baseline", "ramp_pos", "ramp_neg"]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_eval_prompts(eval_dir: Path) -> list[dict]:
    """Load HarmBench + ClearHarm prompts from JSONL files."""
    records = []
    for fname in ("harmbench.jsonl", "clearharm.jsonl"):
        path = eval_dir / fname
        if not path.exists():
            logger.error(
                "%s not found — run `python data/download_eval_prompts.py` first.", path
            )
            sys.exit(1)
        with path.open(encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))
    logger.info(
        "Loaded %d eval prompts (%d HarmBench, %d ClearHarm)",
        len(records),
        sum(1 for r in records if r["source"] == "harmbench"),
        sum(1 for r in records if r["source"] == "clearharm"),
    )
    return records


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _generate_one(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    extra_cfg: dict,
) -> str:
    formatted = format_prompt(tokenizer, prompt, extra_cfg)
    device = next(model.parameters()).device
    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
        padding=False,
    ).to(device)
    prompt_len = inputs["input_ids"].shape[1]
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()


def generate_condition(
    model,
    tokenizer,
    steerer: ActivationSteerer,
    layer_config: dict,
    eval_prompts: list[dict],
    behavior: str,
    max_new_tokens: int,
    extra_cfg: dict,
    condition_name: str,
) -> list[dict]:
    """Generate responses for all eval prompts under one steering condition."""
    records = []
    with steerer.steer(layer_config):
        for item in tqdm(
            eval_prompts,
            desc=f"  [{condition_name}]",
            dynamic_ncols=True,
            leave=False,
        ):
            resp = _generate_one(model, tokenizer, item["prompt"], max_new_tokens, extra_cfg)
            records.append({
                "condition": condition_name,
                "behavior":  behavior,
                "source":    item["source"],
                "prompt_id": item["id"],
                "user_input": item["prompt"],
                "response":  resp,
            })
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    logger.info("[%s] generated %d responses", condition_name, len(records))
    return records


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", path)


def plot_condition_comparison(
    df: pd.DataFrame,
    behavior: str,
    model_name: str,
    fig_dir: Path,
) -> None:
    """Overall mean refusal score per condition."""
    agg = df.groupby("condition")["judge_score"].mean().reindex(CONDITIONS)
    colors = {"baseline": "#6b7280", "ramp_pos": "#2563eb", "ramp_neg": "#dc2626"}

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(
        agg.index,
        agg.values,
        color=[colors[c] for c in agg.index],
        width=0.5,
        edgecolor="white",
    )
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=10)
    ax.set_ylabel("Mean refusal score (judge)")
    ax.set_ylim(0, 1.1)
    ax.set_title(f"{model_name} | {behavior}\nbaseline vs ramp+ vs ramp−")
    ax.grid(True, axis="y", alpha=0.3)
    _savefig(fig, fig_dir / "condition_comparison.pdf")


def plot_by_source(
    df: pd.DataFrame,
    behavior: str,
    model_name: str,
    fig_dir: Path,
) -> None:
    """Mean refusal score per condition, split by benchmark source."""
    sources = df["source"].unique()
    x = np.arange(len(CONDITIONS))
    width = 0.35
    source_colors = {"harmbench": "#7c3aed", "clearharm": "#059669"}

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, source in enumerate(sources):
        sub = df[df["source"] == source].groupby("condition")["judge_score"].mean().reindex(CONDITIONS)
        offset = (i - len(sources) / 2 + 0.5) * width
        ax.bar(x + offset, sub.values, width, label=source, color=source_colors.get(source, "gray"))

    ax.set_xticks(x)
    ax.set_xticklabels(CONDITIONS)
    ax.set_ylabel("Mean refusal score (judge)")
    ax.set_ylim(0, 1.1)
    ax.set_title(f"{model_name} | {behavior} — by benchmark source")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    _savefig(fig, fig_dir / "by_source.pdf")


def plot_per_prompt_heatmap(
    df: pd.DataFrame,
    behavior: str,
    model_name: str,
    fig_dir: Path,
) -> None:
    """Heatmap: rows = prompts, columns = conditions."""
    pivot = df.pivot_table(
        index="prompt_id", columns="condition", values="judge_score", aggfunc="mean"
    ).reindex(columns=CONDITIONS)
    pivot = pivot.fillna(0)

    fig, ax = plt.subplots(figsize=(6, max(4, len(pivot) * 0.35)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels(CONDITIONS, fontsize=9)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(
        [pid[:30] for pid in pivot.index], fontsize=7
    )
    plt.colorbar(im, ax=ax, label="Refusal score")
    ax.set_title(f"{model_name} | {behavior} — per-prompt refusal")
    _savefig(fig, fig_dir / "per_prompt_heatmap.pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 03 — ramp steering eval")
    parser.add_argument("--config", default="config/models.yml")
    parser.add_argument("--directions", default="results/directions")
    parser.add_argument("--eval-prompts", default="data/eval_prompts")
    parser.add_argument("--output-dir", default="results/ramp_eval")
    parser.add_argument("--model", required=True)
    parser.add_argument("--behavior", required=True,
                        choices=["sycophancy", "safety", "refusal"])
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--judge-model", default=SmallModelJudge.DEFAULT_MODEL)
    parser.add_argument("--judge-batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfgs = {m["name"]: m for m in cfg["models"]}
    if args.model not in model_cfgs:
        logger.error("Model '%s' not found. Available: %s", args.model, list(model_cfgs))
        sys.exit(1)
    model_cfg = model_cfgs[args.model]
    extra_cfg = model_cfg.get("extra", {})

    out_dir = Path(args.output_dir) / args.model / args.behavior
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path("figures") / "ramp_eval" / args.model / args.behavior

    # ── Directions ────────────────────────────────────────────────────────────
    dir_path = Path(args.directions) / args.model / f"{args.behavior}.npz"
    if not dir_path.exists():
        logger.error("Directions not found at %s — run 02_direction_extraction.py first.", dir_path)
        sys.exit(1)
    directions = load_directions(str(dir_path))
    logger.info("Loaded %d direction vectors for %s | %s", len(directions), args.model, args.behavior)

    # ── Eval prompts ──────────────────────────────────────────────────────────
    eval_prompts = load_eval_prompts(Path(args.eval_prompts))

    # ── Model ─────────────────────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(
        hf_id=model_cfg["hf_id"],
        dtype=model_cfg.get("dtype", "bfloat16"),
        load_in_4bit=args.load_in_4bit,
    )
    steerer = ActivationSteerer(model)

    # ── Build steering configs ────────────────────────────────────────────────
    configs = {
        "baseline": {},                                         # empty config = no hooks
        "ramp_pos": steerer.build_ramp_config(directions, scale=1.0),
        "ramp_neg": steerer.build_ramp_config(directions, scale=-1.0),
    }

    k_vals = [v for _, v in configs["ramp_pos"].values()]
    logger.info(
        "Ramp config: %d layers | K range [%.4f, %.4f]",
        len(k_vals), min(k_vals), max(k_vals),
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 1: Generate all responses
    # ─────────────────────────────────────────────────────────────────────────
    all_records: list[dict] = []
    for condition in CONDITIONS:
        logger.info("=== Generating: %s ===", condition)
        records = generate_condition(
            model, tokenizer, steerer,
            configs[condition],
            eval_prompts, args.behavior,
            args.max_new_tokens, extra_cfg,
            condition,
        )
        all_records.extend(records)

    raw_df = pd.DataFrame(all_records)
    raw_df.to_csv(out_dir / "raw_generations.csv", index=False)
    logger.info("Phase 1 complete — %d responses. Checkpoint saved.", len(all_records))

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 2: Batch judge
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("=== Phase 2: Judging %d records with %s ===", len(all_records), args.judge_model)
    judge = SmallModelJudge(model_id=args.judge_model)
    scores = judge.score_records(all_records, batch_size=args.judge_batch_size)
    judge.unload()

    for rec, score in zip(all_records, scores):
        rec["judge_score"] = score

    scored_df = pd.DataFrame(all_records)
    scored_df.to_csv(out_dir / "scored_results.csv", index=False)

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 3: Aggregate and report
    # ─────────────────────────────────────────────────────────────────────────
    summary = scored_df.groupby("condition")["judge_score"].agg(["mean", "std", "count"])
    summary.columns = ["mean_score", "std_score", "n"]
    summary.to_csv(out_dir / "summary.csv")

    logger.info("\n%s", summary.to_string())

    plot_condition_comparison(scored_df, args.behavior, args.model, fig_dir)
    plot_by_source(scored_df, args.behavior, args.model, fig_dir)
    plot_per_prompt_heatmap(scored_df, args.behavior, args.model, fig_dir)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    del model, tokenizer, steerer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Experiment 03 complete. Results → %s", out_dir)


if __name__ == "__main__":
    main()
