"""Experiment 03 — Single-layer K sweep and injection-depth ablation.

Three ablation conditions are run for each (model, behavior) pair:

  A. K sweep (single layer l*):
     For K ∈ {0, 0.5, 1, 2, 5, 10} × K_{l*}, steer at l* only and measure
     how behavior score and coherence vary with injection magnitude.
     At K >> K_{l*} output collapses (lobotomy), proving the formula ceiling.

  B. Injection depth:
     Fix K = K_{l*}. Vary the injection layer across early, middle, and late
     positions relative to the target behavioral manifold layer l*.
     Layers before l* yield suppressed behavior scores — "growing norms subdue
     vectors injected before the behavioral manifold forms."

  C. Schedule (ramp vs flat):
     Steer all middle layers with:
       - Ramped K: K_ℓ = μ̄_ℓ / √d  (formula value per layer)
       - Flat K:   K = mean(K_ℓ for ℓ in middle layers)
     Compare behavior score and coherence.

Usage:
    python experiments/03_single_layer_ablation.py \\
        --config config/models.yml \\
        --directions results/directions \\
        --test-prompts data/test_prompts \\
        --output-dir results/ablation \\
        [--model llama-3.1-8b-instruct] \\
        [--behavior sycophancy] \\
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
from activation_baking.judges import HeuristicScorer, PerplexityScorer
from activation_baking.model_utils import format_prompt, load_model_and_tokenizer
from activation_baking.steerer import ActivationSteerer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# K sweep multipliers
# ---------------------------------------------------------------------------
K_SCALES = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_test_prompts(path: Path) -> list[dict]:
    """Load test prompts JSONL and return list of dicts."""
    prompts = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            prompts.append(json.loads(line))
    return prompts


def load_model_directions(
    directions_dir: Path,
    model_name: str,
    behavior: str,
) -> list:
    """Load pre-extracted BehavioralDirections for a model/behavior pair."""
    path = directions_dir / model_name / f"{behavior}.npz"
    if not path.exists():
        logger.error(
            "Directions not found at %s — run 02_direction_extraction.py first.", path
        )
        sys.exit(1)
    return load_directions(str(path))


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    extra_cfg: dict,
) -> str:
    """Generate a single response string given a raw prompt."""
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
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()


def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    steerer: ActivationSteerer,
    layer_config: dict,
    max_new_tokens: int,
    extra_cfg: dict,
    desc: str,
) -> list[str]:
    """Generate responses for all test prompts under a given steering config."""
    responses = []
    with steerer.steer(layer_config):
        for item in tqdm(prompts, desc=f"  {desc}", dynamic_ncols=True, leave=False):
            prompt_text = item["prompt"]
            resp = generate_response(model, tokenizer, prompt_text, max_new_tokens, extra_cfg)
            responses.append(resp)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return responses


# ---------------------------------------------------------------------------
# Ablation A — K sweep
# ---------------------------------------------------------------------------

def run_k_sweep(
    model,
    tokenizer,
    steerer: ActivationSteerer,
    scorer: HeuristicScorer,
    directions: list,
    target_layer: int,
    test_prompts: list[dict],
    behavior: str,
    max_new_tokens: int,
    extra_cfg: dict,
) -> pd.DataFrame:
    """Sweep K scale at a single target layer.

    Returns a DataFrame with columns:
        k_scale, k_value, mean_behavior_score, mean_perplexity (placeholder 1.0)
    """
    dir_map = {d.layer_idx: d for d in directions}
    target_dir = dir_map[target_layer]

    records = []
    for scale in K_SCALES:
        k_val = target_dir.k_value * scale
        cfg = {target_layer: (target_dir.mean_direction, k_val)}

        responses = generate_batch(
            model, tokenizer, test_prompts, steerer, cfg,
            max_new_tokens, extra_cfg,
            desc=f"K={scale:.2f}×K_l",
        )
        scores = scorer.score_batch(responses, behavior)
        mean_score = float(np.mean([s.behavior_score for s in scores]))

        logger.info("  K×%.2f (K=%.4f) → behavior=%.3f", scale, k_val, mean_score)
        records.append({
            "k_scale":            scale,
            "k_value":            k_val,
            "mean_behavior_score": mean_score,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Ablation B — Injection depth
# ---------------------------------------------------------------------------

def run_depth_ablation(
    model,
    tokenizer,
    steerer: ActivationSteerer,
    scorer: HeuristicScorer,
    directions: list,
    target_layer: int,
    test_prompts: list[dict],
    behavior: str,
    max_new_tokens: int,
    extra_cfg: dict,
) -> pd.DataFrame:
    """Inject at varying layers relative to target_layer with K = K_{target}.

    Probes a window of ±6 layers around target_layer (clipped to middle range).
    """
    dir_map = {d.layer_idx: d for d in directions}
    target_dir = dir_map[target_layer]
    k_val = target_dir.k_value

    available_layers = sorted(dir_map.keys())
    min_l, max_l = available_layers[0], available_layers[-1]

    # Sample layers: every 2nd in the available range for manageable runtime
    probe_layers = [l for l in range(min_l, max_l + 1, 2) if l in dir_map]

    records = []
    for inject_layer in probe_layers:
        inject_dir = dir_map[inject_layer]
        cfg = {inject_layer: (inject_dir.mean_direction, k_val)}

        responses = generate_batch(
            model, tokenizer, test_prompts, steerer, cfg,
            max_new_tokens, extra_cfg,
            desc=f"layer={inject_layer}",
        )
        scores = scorer.score_batch(responses, behavior)
        mean_score = float(np.mean([s.behavior_score for s in scores]))

        relative_depth = inject_layer - target_layer
        logger.info(
            "  inject_layer=%d (Δ=%+d) → behavior=%.3f",
            inject_layer, relative_depth, mean_score,
        )
        records.append({
            "inject_layer":       inject_layer,
            "target_layer":       target_layer,
            "relative_depth":     relative_depth,
            "k_value":            k_val,
            "mean_behavior_score": mean_score,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Ablation C — Ramp vs flat schedule
# ---------------------------------------------------------------------------

def run_schedule_ablation(
    model,
    tokenizer,
    steerer: ActivationSteerer,
    scorer: HeuristicScorer,
    directions: list,
    test_prompts: list[dict],
    behavior: str,
    max_new_tokens: int,
    extra_cfg: dict,
) -> pd.DataFrame:
    """Compare ramped (formula) vs flat-K schedule over all middle layers."""
    schedules = {
        "ramp": steerer.build_ramp_config(directions, scale=1.0),
        "flat": steerer.build_flat_config(directions),
    }

    records = []
    for schedule_name, cfg in schedules.items():
        k_values = [v for _, v in cfg.values()]
        logger.info(
            "  %s schedule — K range [%.4f, %.4f]",
            schedule_name, min(k_values), max(k_values),
        )

        responses = generate_batch(
            model, tokenizer, test_prompts, steerer, cfg,
            max_new_tokens, extra_cfg,
            desc=f"schedule={schedule_name}",
        )
        scores = scorer.score_batch(responses, behavior)
        mean_score = float(np.mean([s.behavior_score for s in scores]))

        logger.info("  %s → behavior=%.3f", schedule_name, mean_score)
        records.append({
            "schedule":           schedule_name,
            "k_min":              min(k_values),
            "k_max":              max(k_values),
            "k_mean":             float(np.mean(k_values)),
            "mean_behavior_score": mean_score,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_k_sweep(df: pd.DataFrame, behavior: str, model_name: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["k_scale"], df["mean_behavior_score"], marker="o", lw=2, color="#2563eb")
    ax.axvline(x=1.0, color="#dc2626", ls="--", lw=1.2, label="K = K_ℓ (formula)")
    ax.set_xlabel("K scale (× formula K_ℓ)")
    ax.set_ylabel("Mean behavior score")
    ax.set_title(f"{model_name} | {behavior} — K sweep (single layer)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved K sweep plot → %s", out_path)


def plot_depth_ablation(
    df: pd.DataFrame, behavior: str, model_name: str, out_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        df["inject_layer"], df["mean_behavior_score"],
        marker="o", lw=2, color="#16a34a",
    )
    ax.axvline(
        x=df["target_layer"].iloc[0], color="#dc2626",
        ls="--", lw=1.2, label="Target layer l*",
    )
    ax.set_xlabel("Injection layer")
    ax.set_ylabel("Mean behavior score")
    ax.set_title(f"{model_name} | {behavior} — injection depth ablation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved depth ablation plot → %s", out_path)


def plot_schedule(
    df: pd.DataFrame, behavior: str, model_name: str, out_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    colors = {"ramp": "#7c3aed", "flat": "#d97706"}
    for _, row in df.iterrows():
        ax.bar(row["schedule"], row["mean_behavior_score"], color=colors[row["schedule"]])
    ax.set_ylabel("Mean behavior score")
    ax.set_title(f"{model_name} | {behavior} — ramp vs flat K")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved schedule ablation plot → %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 03 — ablation study")
    parser.add_argument("--config", default="config/models.yml")
    parser.add_argument("--directions", default="results/directions")
    parser.add_argument("--test-prompts", default="data/test_prompts")
    parser.add_argument("--output-dir", default="results/ablation")
    parser.add_argument("--model", required=True,
                        help="Model name as in config/models.yml")
    parser.add_argument("--behavior", required=True,
                        choices=["sycophancy", "safety", "refusal"])
    parser.add_argument("--target-layer", type=int, default=None,
                        help="Single target layer for K sweep and depth ablation. "
                             "Defaults to the median of extracted middle layers.")
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--ablations", nargs="*",
                        default=["k_sweep", "depth", "schedule"],
                        choices=["k_sweep", "depth", "schedule"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfgs = {m["name"]: m for m in cfg["models"]}
    if args.model not in model_cfgs:
        logger.error("Model '%s' not in config. Available: %s", args.model, list(model_cfgs))
        sys.exit(1)
    model_cfg = model_cfgs[args.model]
    extra_cfg = model_cfg.get("extra", {})

    # ── Directories ──────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir) / args.model / args.behavior
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path("figures") / "ablation" / args.model / args.behavior
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── Directions ────────────────────────────────────────────────────────────
    directions = load_model_directions(
        Path(args.directions), args.model, args.behavior
    )
    logger.info(
        "Loaded %d direction vectors for %s | %s",
        len(directions), args.model, args.behavior,
    )

    # ── Target layer ─────────────────────────────────────────────────────────
    available_layers = sorted(d.layer_idx for d in directions)
    if args.target_layer is not None:
        target_layer = args.target_layer
        if target_layer not in available_layers:
            logger.error(
                "--target-layer=%d not in extracted layers. Available: %s",
                target_layer, available_layers,
            )
            sys.exit(1)
    else:
        # median of middle layers
        target_layer = available_layers[len(available_layers) // 2]
    logger.info("Target layer l* = %d", target_layer)

    # ── Test prompts ──────────────────────────────────────────────────────────
    test_path = Path(args.test_prompts) / f"{args.behavior}.jsonl"
    if not test_path.exists():
        logger.error("Test prompts not found: %s", test_path)
        sys.exit(1)
    test_prompts = load_test_prompts(test_path)
    logger.info("Loaded %d test prompts", len(test_prompts))

    # ── Model ─────────────────────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(
        hf_id=model_cfg["hf_id"],
        dtype=model_cfg.get("dtype", "bfloat16"),
        load_in_4bit=args.load_in_4bit,
    )
    steerer = ActivationSteerer(model)
    scorer = HeuristicScorer()

    # ── Ablation A — K sweep ─────────────────────────────────────────────────
    if "k_sweep" in args.ablations:
        logger.info("=== Ablation A: K sweep at layer %d ===", target_layer)
        df_k = run_k_sweep(
            model, tokenizer, steerer, scorer,
            directions, target_layer, test_prompts,
            args.behavior, args.max_new_tokens, extra_cfg,
        )
        df_k.to_csv(out_dir / "k_sweep.csv", index=False)
        plot_k_sweep(df_k, args.behavior, args.model, fig_dir / "k_sweep.pdf")

    # ── Ablation B — Injection depth ──────────────────────────────────────────
    if "depth" in args.ablations:
        logger.info("=== Ablation B: injection depth ablation ===")
        df_depth = run_depth_ablation(
            model, tokenizer, steerer, scorer,
            directions, target_layer, test_prompts,
            args.behavior, args.max_new_tokens, extra_cfg,
        )
        df_depth.to_csv(out_dir / "depth_ablation.csv", index=False)
        plot_depth_ablation(df_depth, args.behavior, args.model, fig_dir / "depth_ablation.pdf")

    # ── Ablation C — Schedule ─────────────────────────────────────────────────
    if "schedule" in args.ablations:
        logger.info("=== Ablation C: ramp vs flat schedule ===")
        df_sched = run_schedule_ablation(
            model, tokenizer, steerer, scorer,
            directions, test_prompts,
            args.behavior, args.max_new_tokens, extra_cfg,
        )
        df_sched.to_csv(out_dir / "schedule_ablation.csv", index=False)
        plot_schedule(df_sched, args.behavior, args.model, fig_dir / "schedule_ablation.pdf")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    del model, tokenizer, steerer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Experiment 03 complete. Results saved to %s", out_dir)


if __name__ == "__main__":
    main()
