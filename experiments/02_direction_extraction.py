"""Experiment 02 — Behavioral direction extraction via contrastive pairs.

For each model × behavior:
  1. Load contrastive pairs (positive elicits behavior, negative suppresses it).
  2. Load per-layer K values from the norm profile produced by experiment 01.
  3. Run DirectionExtractor over the middle 50% of layers.
  4. Save per-layer PCA and mean-diff directions to results/directions/.
  5. Log PCA variance ratios to assess direction quality.

Contrastive pairs are completion-based: identical context, two completions that
differ only on the behavioral axis. Activations are pooled over completion tokens
only for a sharper directional signal.

Usage:
    python experiments/02_direction_extraction.py \\
        --config config/models.yml \\
        --behaviors safety refusal sycophancy \\
        --models llama-3.1-8b-instruct
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from activation_baking.config import ModelConfig, load_model_configs
from activation_baking.direction_extractor import DirectionExtractor, save_directions
from activation_baking.norm_profiler import load_profile
from activation_baking.registry import ModelRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_pairs(path: Path) -> tuple[list[str], list[str], list[str]]:
    """Load completion-based contrastive pairs from a JSONL file.

    Returns:
        ``(positive_full_prompts, negative_full_prompts, contexts)`` where each
        full prompt is ``context + completion``.  Prompts carry embedded role
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
    logger.info("Loaded %d contrastive pairs from %s", len(positives), path)
    return positives, negatives, contexts


def load_k_values(norm_profile_path: Path, model_cfg: ModelConfig) -> dict[int, float]:
    """Read per-layer K values for the middle layers from a norm profile CSV.

    Args:
        norm_profile_path: CSV produced by ``save_profile`` in experiment 01.
        model_cfg:         Config for the model being processed; determines
                           which layer indices count as "middle".

    Returns:
        Mapping of ``layer_idx → K_ℓ`` for every middle layer.
    """
    stats = load_profile(norm_profile_path)
    middle = set(model_cfg.middle_layers)
    return {s.layer_idx: s.k_value for s in stats if s.layer_idx in middle}


def extract_behavior(
    model_cfg: ModelConfig,
    behavior: str,
    pairs_path: Path,
    norm_profile_path: Path,
    output_dir: Path,
    registry: ModelRegistry,
) -> None:
    """Extract and save behavioral directions for one model × behavior pair."""
    logger.info("--- %s | %s ---", model_cfg.name, behavior)

    positives, negatives, _ = load_pairs(pairs_path)
    k_values = load_k_values(norm_profile_path, model_cfg)

    with registry.loaded(model_cfg) as (model, tokenizer):
        extractor = DirectionExtractor(
            model=model,
            tokenizer=tokenizer,
            model_cfg=model_cfg,
            k_values=k_values,
        )
        directions = extractor.extract(positives, negatives)

    out_dir = output_dir / model_cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    save_directions(directions, str(out_dir / f"{behavior}.npz"))

    var_ratios = [d.pca_variance_ratio for d in directions]
    logger.info(
        "%s | %s — %d layers | PCA var: min=%.3f mean=%.3f max=%.3f",
        model_cfg.name, behavior, len(directions),
        min(var_ratios), float(np.mean(var_ratios)), max(var_ratios),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 02 — direction extraction")
    parser.add_argument("--config", default="config/models.yml")
    parser.add_argument("--pairs-dir", default="data/contrastive_pairs")
    parser.add_argument("--norm-profiles", default="results/norm_profiles")
    parser.add_argument("--output-dir", default="results/directions")
    parser.add_argument(
        "--behaviors", nargs="*", default=["safety", "sycophancy"]
    )
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument(
        "--device", default="cuda:0",
        help="Single GPU to use (default: cuda:0). Override via CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing direction files instead of skipping.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    all_configs: dict[str, ModelConfig] = load_model_configs(args.config)
    if args.models:
        missing = set(args.models) - set(all_configs)
        if missing:
            logger.error("Unknown model name(s): %s. Available: %s", missing, list(all_configs))
            sys.exit(1)
        selected = {n: all_configs[n] for n in args.models}
    else:
        selected = all_configs

    pairs_dir = Path(args.pairs_dir)
    norm_dir = Path(args.norm_profiles)
    output_dir = Path(args.output_dir)

    registry = ModelRegistry(load_in_4bit=args.load_in_4bit, device_map=args.device)

    for name, model_cfg in selected.items():
        norm_path = norm_dir / f"{name}.csv"
        if not norm_path.exists():
            logger.error(
                "Norm profile not found for %s at %s — run experiment 01 first.",
                name, norm_path,
            )
            sys.exit(1)

        for behavior in args.behaviors:
            pairs_path = pairs_dir / f"{behavior}.jsonl"
            if not pairs_path.exists():
                logger.warning("No pairs file for '%s'; skipping.", behavior)
                continue

            out_path = output_dir / name / f"{behavior}.npz"
            if out_path.exists() and not args.force:
                logger.info("Skipping %s | %s (already exists at %s).", name, behavior, out_path)
                continue

            extract_behavior(
                model_cfg=model_cfg,
                behavior=behavior,
                pairs_path=pairs_path,
                norm_profile_path=norm_path,
                output_dir=output_dir,
                registry=registry,
            )

    logger.info("Experiment 02 complete.")


if __name__ == "__main__":
    main()
