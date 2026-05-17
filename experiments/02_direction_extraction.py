"""Experiment 02 — Behavioral direction extraction via contrastive pairs.

For each model and behavioral axis:
  1. Load contrastive pairs (positive elicits behavior, negative suppresses it).
  2. Format with the model's chat template.
  3. Run DirectionExtractor over middle 50% of layers.
  4. Save per-layer PCA and mean-diff directions to results/directions/.
  5. Log PCA variance ratios to assess direction quality.

Usage:
    python experiments/02_direction_extraction.py \
        --config config/models.yml \
        --pairs-dir data/contrastive_pairs \
        --norm-profiles results/norm_profiles \
        --output-dir results/directions \
        [--behaviors sycophancy safety refusal] \
        [--models llama-3.1-8b-instruct] \
        [--load-in-4bit]
"""

import argparse
import gc
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from activation_baking.direction_extractor import (
    DirectionExtractor,
    save_directions,
)
from activation_baking.model_utils import load_model_and_tokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_pairs(path: Path) -> tuple[list[str], list[str], list[str]]:
    """Load completion-based contrastive pairs JSONL.

    Returns:
        (positive_full_prompts, negative_full_prompts, contexts) where each
        full prompt is context + completion. All strings carry embedded role
        labels and are tokenised directly (no chat template applied).
    """
    positives, negatives, contexts = [], [], []
    with path.open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            ctx = obj["context"]
            positives.append(ctx + obj["positive_completion"])
            negatives.append(ctx + obj["negative_completion"])
            contexts.append(ctx)
    return positives, negatives, contexts


def load_k_values(norm_profile_path: Path, num_layers: int) -> dict[int, float]:
    """Read per-layer K values from a norm profile CSV."""
    df = pd.read_csv(norm_profile_path)
    start = num_layers // 4
    end = (3 * num_layers) // 4
    mid = df[(df.layer_idx >= start) & (df.layer_idx < end)]
    return {int(row.layer_idx): float(row.k_value) for _, row in mid.iterrows()}


def run_extraction(
    model_cfg: dict,
    behavior: str,
    pairs_path: Path,
    norm_profile_path: Path,
    output_dir: Path,
    load_in_4bit: bool,
    extra_cfg: dict,
) -> None:
    model_name = model_cfg["name"]
    logger.info("--- %s | %s ---", model_name, behavior)

    positives, negatives, contexts = load_pairs(pairs_path)
    logger.info("Loaded %d contrastive pairs", len(positives))

    k_values = load_k_values(norm_profile_path, model_cfg["num_layers"])

    model, tokenizer = load_model_and_tokenizer(
        hf_id=model_cfg["hf_id"],
        dtype=model_cfg.get("dtype", "bfloat16"),
        load_in_4bit=load_in_4bit,
    )

    extractor = DirectionExtractor(
        model=model,
        tokenizer=tokenizer,
        hidden_size=model_cfg["hidden_size"],
        num_layers=model_cfg["num_layers"],
        k_values=k_values,
    )

    # Contrastive pairs carry embedded role labels; pass raw strings directly.
    # contexts enables completion-only pooling for a sharper directional signal.
    directions = extractor.extract(positives, negatives, contexts=contexts)

    out_dir = output_dir / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{behavior}.npz"
    save_directions(directions, str(out_path))

    var_ratios = [d.pca_variance_ratio for d in directions]
    logger.info(
        "%s | %s — saved %d layers | PCA var explained: min=%.3f mean=%.3f max=%.3f",
        model_name, behavior,
        len(directions),
        min(var_ratios), np.mean(var_ratios), max(var_ratios),
    )

    del model, tokenizer, extractor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 02 — direction extraction")
    parser.add_argument("--config", default="config/models.yml")
    parser.add_argument("--pairs-dir", default="data/contrastive_pairs")
    parser.add_argument("--norm-profiles", default="results/norm_profiles")
    parser.add_argument("--output-dir", default="results/directions")
    parser.add_argument("--behaviors", nargs="*",
                        default=["sycophancy", "safety", "refusal"])
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--load-in-4bit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfgs: list[dict] = cfg["models"]
    if args.models:
        model_cfgs = [m for m in model_cfgs if m["name"] in args.models]
        if not model_cfgs:
            logger.error("No models matched: %s", args.models)
            sys.exit(1)

    pairs_dir = Path(args.pairs_dir)
    norm_dir = Path(args.norm_profiles)
    output_dir = Path(args.output_dir)

    for model_cfg in model_cfgs:
        name = model_cfg["name"]
        extra_cfg = model_cfg.get("extra", {})
        norm_path = norm_dir / f"{name}.csv"

        if not norm_path.exists():
            logger.error("Norm profile not found for %s — run 01_norm_profiling first", name)
            sys.exit(1)

        for behavior in args.behaviors:
            pairs_path = pairs_dir / f"{behavior}.jsonl"
            if not pairs_path.exists():
                logger.warning("No pairs file for behavior '%s', skipping", behavior)
                continue

            out_path = output_dir / name / f"{behavior}.npz"
            if out_path.exists():
                logger.info("Skipping %s | %s (already exists)", name, behavior)
                continue

            run_extraction(
                model_cfg=model_cfg,
                behavior=behavior,
                pairs_path=pairs_path,
                norm_profile_path=norm_path,
                output_dir=output_dir,
                load_in_4bit=args.load_in_4bit,
                extra_cfg=extra_cfg,
            )

    logger.info("Experiment 02 complete.")


if __name__ == "__main__":
    main()
