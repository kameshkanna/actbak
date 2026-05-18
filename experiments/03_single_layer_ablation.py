"""Experiment 03 — Single-layer K sweep and injection-depth ablation.

Three ablation conditions are run for each (model, behavior) pair:

  A. K sweep (single layer l*):
     For K ∈ {0, 0.25, 0.5, 1, 2, 5, 10} × K_{l*}, steer at l* only.

  B. Injection depth:
     Fix K = K_{l*}. Vary injection layer across middle range.
     Layers before l* yield suppressed scores — growing norms subdue vectors.

  C. Schedule (ramp vs flat):
     All middle layers with ramped K (formula) vs flat K (mean).

Generation and scoring are fully decoupled:
  Phase 1 — run all generations, checkpoint raw_generations.csv
  Phase 2 — activation-space scoring via ActivationJudge (same main model, no aux judge)
             score = cosine_sim(last-token hidden state at target_layer, behavioral direction)
  Phase 3 — aggregate + plot per ablation

Replaces SmallModelJudge (Qwen2.5-3B) with activation-space scoring from the
AVAW / norms-k-calibration evaluator pattern — eliminates judge collapse risk.

Usage:
    python experiments/03_single_layer_ablation.py \\
        --config config/models.yml \\
        --directions results/directions \\
        --test-prompts data/test_prompts \\
        --output-dir results/ablation \\
        --model llama-3.1-8b-instruct \\
        --behavior sycophancy \\
        [--judge-batch-size 8] \\
        [--max-new-tokens 150] \\
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
from activation_baking.judges import ActivationJudge, SmallModelJudge
from activation_baking.model_utils import format_prompt, load_model_and_tokenizer
from activation_baking.steerer import ActivationSteerer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

K_SCALES = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_test_prompts(path: Path) -> list[dict]:
    prompts = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            prompts.append(json.loads(line))
    return prompts


def load_model_directions(directions_dir: Path, model_name: str, behavior: str) -> list:
    path = directions_dir / model_name / f"{behavior}.npz"
    if not path.exists():
        logger.error("Directions not found at %s — run 02 first.", path)
        sys.exit(1)
    return load_directions(str(path))


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _generate_one(model, tokenizer, prompt: str, max_new_tokens: int, extra_cfg: dict) -> str:
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


def _generate_condition(
    model,
    tokenizer,
    steerer: ActivationSteerer,
    layer_config: dict,
    test_prompts: list[dict],
    behavior: str,
    max_new_tokens: int,
    extra_cfg: dict,
    desc: str,
    meta: dict,
) -> list[dict]:
    """Generate responses for one steering condition; return flat record list."""
    records = []
    with steerer.steer(layer_config):
        for item in tqdm(test_prompts, desc=f"  gen {desc}", dynamic_ncols=True, leave=False):
            resp = _generate_one(model, tokenizer, item["prompt"], max_new_tokens, extra_cfg)
            records.append({
                "behavior":   behavior,
                "user_input": item["prompt"],
                "response":   resp,
                **meta,
            })
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return records


# ---------------------------------------------------------------------------
# Ablation A — K sweep
# ---------------------------------------------------------------------------

def collect_k_sweep(
    model, tokenizer, steerer, directions, target_layer,
    test_prompts, behavior, max_new_tokens, extra_cfg,
) -> list[dict]:
    dir_map = {d.layer_idx: d for d in directions}
    target_dir = dir_map[target_layer]
    all_records = []

    for scale in K_SCALES:
        k_val = target_dir.k_value * scale
        cfg = {target_layer: (target_dir.mean_direction, k_val)}
        records = _generate_condition(
            model, tokenizer, steerer, cfg, test_prompts,
            behavior, max_new_tokens, extra_cfg,
            desc=f"K×{scale}",
            meta={"ablation": "k_sweep", "k_scale": scale, "k_value": k_val,
                  "inject_layer": target_layer},
        )
        all_records.extend(records)
        logger.info("K×%.2f collected (%d responses)", scale, len(records))

    return all_records


# ---------------------------------------------------------------------------
# Ablation B — Injection depth
# ---------------------------------------------------------------------------

def collect_depth_ablation(
    model, tokenizer, steerer, directions, target_layer,
    test_prompts, behavior, max_new_tokens, extra_cfg,
) -> list[dict]:
    dir_map = {d.layer_idx: d for d in directions}
    k_val = dir_map[target_layer].k_value
    available = sorted(dir_map.keys())
    # every 2nd layer to keep runtime manageable
    probe_layers = available[::2]
    all_records = []

    for inject_layer in probe_layers:
        inject_dir = dir_map[inject_layer]
        cfg = {inject_layer: (inject_dir.mean_direction, k_val)}
        records = _generate_condition(
            model, tokenizer, steerer, cfg, test_prompts,
            behavior, max_new_tokens, extra_cfg,
            desc=f"depth={inject_layer}",
            meta={"ablation": "depth", "k_scale": 1.0, "k_value": k_val,
                  "inject_layer": inject_layer, "target_layer": target_layer,
                  "relative_depth": inject_layer - target_layer},
        )
        all_records.extend(records)
        logger.info("Depth layer=%d (Δ=%+d) collected", inject_layer, inject_layer - target_layer)

    return all_records


# ---------------------------------------------------------------------------
# Ablation C — Schedule
# ---------------------------------------------------------------------------

def collect_schedule_ablation(
    model, tokenizer, steerer, directions,
    test_prompts, behavior, max_new_tokens, extra_cfg,
) -> list[dict]:
    configs = {
        "ramp": steerer.build_ramp_config(directions, scale=1.0),
        "flat": steerer.build_flat_config(directions),
    }
    all_records = []

    for schedule_name, cfg in configs.items():
        k_vals = [v for _, v in cfg.values()]
        records = _generate_condition(
            model, tokenizer, steerer, cfg, test_prompts,
            behavior, max_new_tokens, extra_cfg,
            desc=f"schedule={schedule_name}",
            meta={"ablation": "schedule", "schedule": schedule_name,
                  "k_min": min(k_vals), "k_max": max(k_vals),
                  "k_mean": float(np.mean(k_vals))},
        )
        all_records.extend(records)
        logger.info("Schedule '%s' collected", schedule_name)

    return all_records


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", path)


def _dynamic_ylim(ax: plt.Axes, vals: pd.Series, lo: float = -1.0, hi: float = 1.0) -> None:
    margin = max(0.02, (vals.max() - vals.min()) * 0.3)
    ax.set_ylim(max(lo, vals.min() - margin), min(hi, vals.max() + margin))


def _add_llm_axis(ax: plt.Axes, x_vals: pd.Series, llm_vals: pd.Series,
                  x_is_categorical: bool = False) -> plt.Axes:
    """Overlay LLM score on a twin y-axis (right side, orange dashed)."""
    ax2 = ax.twinx()
    if x_is_categorical:
        ax2.plot(range(len(x_vals)), llm_vals.values, marker="s", lw=1.5,
                 ls="--", color="#f97316", alpha=0.8, label="LLM score")
    else:
        ax2.plot(x_vals, llm_vals, marker="s", lw=1.5,
                 ls="--", color="#f97316", alpha=0.8, label="LLM score")
    ax2.set_ylabel("LLM judge score (0–1)", color="#f97316")
    ax2.tick_params(axis="y", labelcolor="#f97316")
    _dynamic_ylim(ax2, llm_vals, 0.0, 1.0)
    return ax2


def plot_k_sweep(df: pd.DataFrame, behavior: str, model_name: str, fig_dir: Path) -> None:
    has_llm = "llm_score" in df.columns and df["llm_score"].notna().any()
    agg_cols = ["cosine_sim"] + (["llm_score"] if has_llm else [])
    agg = df.groupby("k_scale")[agg_cols].mean().reset_index()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(agg["k_scale"], agg["cosine_sim"], marker="o", lw=2,
            color="#2563eb", label="Cosine sim (activation)")
    ax.axvline(x=1.0, color="#dc2626", ls="--", lw=1.2, label="K = K_ℓ")
    ax.set_xlabel("K scale (× formula K_ℓ)")
    ax.set_ylabel("Cosine sim to behavioral direction", color="#2563eb")
    ax.tick_params(axis="y", labelcolor="#2563eb")
    ax.set_title(f"{model_name} | {behavior} — K sweep")
    _dynamic_ylim(ax, agg["cosine_sim"])

    lines, labels = ax.get_legend_handles_labels()
    if has_llm:
        ax2 = _add_llm_axis(ax, agg["k_scale"], agg["llm_score"])
        l2, lb2 = ax2.get_legend_handles_labels()
        lines += l2; labels += lb2

    ax.legend(lines, labels, loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    _savefig(fig, fig_dir / "k_sweep.pdf")


def plot_depth_ablation(df: pd.DataFrame, behavior: str, model_name: str, fig_dir: Path) -> None:
    has_llm = "llm_score" in df.columns and df["llm_score"].notna().any()
    agg_cols = ["cosine_sim"] + (["llm_score"] if has_llm else [])
    agg = df.groupby("inject_layer")[agg_cols].mean().reset_index()
    target_layer = int(df["target_layer"].iloc[0])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(agg["inject_layer"], agg["cosine_sim"], marker="o", lw=2,
            color="#16a34a", label="Cosine sim (activation)")
    ax.axvline(x=target_layer, color="#dc2626", ls="--", lw=1.2, label="Target layer l*")
    ax.set_xlabel("Injection layer")
    ax.set_ylabel("Cosine sim to behavioral direction", color="#16a34a")
    ax.tick_params(axis="y", labelcolor="#16a34a")
    ax.set_title(f"{model_name} | {behavior} — injection depth")
    _dynamic_ylim(ax, agg["cosine_sim"])

    lines, labels = ax.get_legend_handles_labels()
    if has_llm:
        ax2 = _add_llm_axis(ax, agg["inject_layer"], agg["llm_score"])
        l2, lb2 = ax2.get_legend_handles_labels()
        lines += l2; labels += lb2

    ax.legend(lines, labels, loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    _savefig(fig, fig_dir / "depth_ablation.pdf")


def plot_schedule(df: pd.DataFrame, behavior: str, model_name: str, fig_dir: Path) -> None:
    has_llm = "llm_score" in df.columns and df["llm_score"].notna().any()
    agg_cols = ["cosine_sim"] + (["llm_score"] if has_llm else [])
    agg = df.groupby("schedule")[agg_cols].mean().reset_index()
    schedules = agg["schedule"].tolist()
    x = np.arange(len(schedules))
    width = 0.35

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(x - (width / 2 if has_llm else 0), agg["cosine_sim"],
                  width if has_llm else 0.5, color=["#7c3aed", "#2563eb"][:len(schedules)],
                  label="Cosine sim (activation)")
    ax.set_xticks(x); ax.set_xticklabels(schedules)
    ax.set_ylabel("Cosine sim to behavioral direction", color="#2563eb")
    ax.tick_params(axis="y", labelcolor="#2563eb")
    ax.set_title(f"{model_name} | {behavior} — ramp vs flat")
    _dynamic_ylim(ax, agg["cosine_sim"])

    lines, labels = ax.get_legend_handles_labels()
    if has_llm:
        ax2 = ax.twinx()
        ax2.bar(x + width / 2, agg["llm_score"], width,
                color=["#f97316", "#fb923c"][:len(schedules)], alpha=0.7, label="LLM score")
        ax2.set_ylabel("LLM judge score (0–1)", color="#f97316")
        ax2.tick_params(axis="y", labelcolor="#f97316")
        _dynamic_ylim(ax2, agg["llm_score"], 0.0, 1.0)
        l2, lb2 = ax2.get_legend_handles_labels()
        lines += l2; labels += lb2

    ax.legend(lines, labels, fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    _savefig(fig, fig_dir / "schedule_ablation.pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 03 — ablation study")
    parser.add_argument("--config", default="config/models.yml")
    parser.add_argument("--directions", default="results/directions")
    parser.add_argument("--test-prompts", default="data/test_prompts")
    parser.add_argument("--output-dir", default="results/ablation")
    parser.add_argument("--model", required=True)
    parser.add_argument("--behavior", required=True,
                        choices=["sycophancy", "safety", "refusal"])
    parser.add_argument("--target-layer", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--ablations", nargs="*",
                        default=["k_sweep", "depth", "schedule"],
                        choices=["k_sweep", "depth", "schedule"])
    parser.add_argument("--judge-model", default=SmallModelJudge.DEFAULT_MODEL,
                        help="HuggingFace model ID for the LLM judge (run alongside ActivationJudge)")
    parser.add_argument("--judge-batch-size", type=int, default=8)
    parser.add_argument("--skip-llm-judge", action="store_true",
                        help="Run only ActivationJudge (skip SmallModelJudge)")
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
    fig_dir = Path("figures") / "ablation" / args.model / args.behavior

    # ── Directions ────────────────────────────────────────────────────────────
    directions = load_model_directions(
        Path(args.directions), args.model, args.behavior
    )
    available_layers = sorted(d.layer_idx for d in directions)
    target_layer = args.target_layer or available_layers[len(available_layers) // 2]
    logger.info("Loaded %d directions | target layer l* = %d", len(directions), target_layer)

    # ── Test prompts ──────────────────────────────────────────────────────────
    test_prompts = load_test_prompts(
        Path(args.test_prompts) / f"{args.behavior}.jsonl"
    )
    logger.info("Loaded %d test prompts", len(test_prompts))

    # ── Load main model ───────────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(
        hf_id=model_cfg["hf_id"],
        dtype=model_cfg.get("dtype", "bfloat16"),
        load_in_4bit=args.load_in_4bit,
    )
    steerer = ActivationSteerer(model)

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 1: Generate all responses
    # ─────────────────────────────────────────────────────────────────────────
    all_records: list[dict] = []

    if "k_sweep" in args.ablations:
        logger.info("=== Phase 1A: K sweep at layer %d ===", target_layer)
        all_records.extend(collect_k_sweep(
            model, tokenizer, steerer, directions, target_layer,
            test_prompts, args.behavior, args.max_new_tokens, extra_cfg,
        ))

    if "depth" in args.ablations:
        logger.info("=== Phase 1B: Injection depth ablation ===")
        all_records.extend(collect_depth_ablation(
            model, tokenizer, steerer, directions, target_layer,
            test_prompts, args.behavior, args.max_new_tokens, extra_cfg,
        ))

    if "schedule" in args.ablations:
        logger.info("=== Phase 1C: Schedule ablation ===")
        all_records.extend(collect_schedule_ablation(
            model, tokenizer, steerer, directions,
            test_prompts, args.behavior, args.max_new_tokens, extra_cfg,
        ))

    logger.info("Phase 1 complete — %d total responses collected", len(all_records))

    # Save raw generations before judging (safe checkpoint)
    raw_df = pd.DataFrame(all_records)
    raw_df.to_csv(out_dir / "raw_generations.csv", index=False)
    logger.info("Raw generations saved → %s/raw_generations.csv", out_dir)

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 2a: Activation-space scoring (ActivationJudge — same model, no aux)
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("=== Phase 2a: Activation scoring %d records at layer %d ===",
                len(all_records), target_layer)
    direction_map = {d.layer_idx: d.mean_direction for d in directions}
    act_judge = ActivationJudge(
        model=model,
        tokenizer=tokenizer,
        direction_map=direction_map,
    )
    cosine_scores = act_judge.score_records(
        all_records, target_layer=target_layer, batch_size=args.judge_batch_size
    )
    for rec, score in zip(all_records, cosine_scores):
        rec["cosine_sim"] = score
        rec["judge_score"] = score  # primary metric alias for downstream plotting

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 2b: LLM judge scoring (SmallModelJudge — loaded after generation)
    # ─────────────────────────────────────────────────────────────────────────
    if not args.skip_llm_judge:
        logger.info("=== Phase 2b: LLM judge scoring %d records (model: %s) ===",
                    len(all_records), args.judge_model)
        llm_judge = SmallModelJudge(model_id=args.judge_model)
        llm_scores = llm_judge.score_records(all_records, batch_size=args.judge_batch_size)
        llm_judge.unload()
        for rec, score in zip(all_records, llm_scores):
            rec["llm_score"] = score
        logger.info("LLM judge complete.")
    else:
        logger.info("Phase 2b skipped (--skip-llm-judge).")
        for rec in all_records:
            rec["llm_score"] = float("nan")

    scored_df = pd.DataFrame(all_records)
    scored_df.to_csv(out_dir / "scored_results.csv", index=False)
    logger.info("Scored results saved → %s/scored_results.csv", out_dir)

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 3: Aggregate and plot
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("=== Phase 3: Aggregating and plotting ===")

    if "k_sweep" in args.ablations:
        df_k = scored_df[scored_df["ablation"] == "k_sweep"].copy()
        df_k.to_csv(out_dir / "k_sweep.csv", index=False)
        plot_k_sweep(df_k, args.behavior, args.model, fig_dir)

    if "depth" in args.ablations:
        df_d = scored_df[scored_df["ablation"] == "depth"].copy()
        df_d.to_csv(out_dir / "depth_ablation.csv", index=False)
        plot_depth_ablation(df_d, args.behavior, args.model, fig_dir)

    if "schedule" in args.ablations:
        df_s = scored_df[scored_df["ablation"] == "schedule"].copy()
        df_s.to_csv(out_dir / "schedule_ablation.csv", index=False)
        plot_schedule(df_s, args.behavior, args.model, fig_dir)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    del model, tokenizer, steerer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Experiment 03 complete. Results → %s", out_dir)


if __name__ == "__main__":
    main()
