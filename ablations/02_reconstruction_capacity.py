"""Ablation 02 — Early-middle layer reconstruction capacity.

When a transformer layer in the 40–50% depth range is steered with K >> ‖h_ℓ‖,
can the model's subsequent layers reconstruct a coherent output?

Hypothesis:
  Early layers have smaller norms → a fixed large K is a proportionally massive
  perturbation (K/‖h_ℓ‖ >> 1).  Residual connections allow later layers to
  partially "undo" this perturbation, so responses may remain semi-coherent
  rather than pure gibberish.  The degree of recovery reveals the model's
  implicit error-correction capacity.

Method:
  For each model × layer at 40% and 50% depth:
    K = reconstruction_k_scale × K_ℓ  (default: 20×, far beyond operating point)
    Generate reconstruction_n_prompts HarmBench prompts
    Save side-by-side: baseline response vs heavily steered response

Outputs (qualitative — no judge scoring):
  results/ablations/reconstruction/{model}/layer_{pct}pct.jsonl
      Fields: prompt_id, user_input, baseline_response, steered_response,
              layer_idx, layer_pct, mean_norm, k_value, k_applied
  results/ablations/reconstruction/{model}/summary.txt
      Human-readable side-by-side comparison

Usage:
    python ablations/02_reconstruction_capacity.py
    python ablations/02_reconstruction_capacity.py --models llama-3.1-8b-instruct
    python ablations/02_reconstruction_capacity.py --k-scale 30.0 --n-prompts 5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from activation_baking.config import ExperimentConfig, ModelConfig, load_experiment_config, load_model_configs
from activation_baking.direction_extractor import load_directions
from activation_baking.model_utils import format_prompt
from activation_baking.norm_profiler import LayerNormStats, load_profile
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

_SEPARATOR = "─" * 80


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_harmbench_prompts(eval_dir: Path, n: int) -> list[dict]:
    """Load the first ``n`` HarmBench prompts from the eval JSONL.

    Args:
        eval_dir: Directory containing ``harmbench.jsonl``.
        n:        Maximum number of prompts to load.

    Returns:
        List of prompt dicts with at least ``id`` and ``prompt`` fields.

    Raises:
        SystemExit: If the JSONL file is missing.
    """
    path = eval_dir / "harmbench.jsonl"
    if not path.exists():
        logger.error(
            "harmbench.jsonl not found at %s — run `python data/download_datasets.py`.", path
        )
        sys.exit(1)
    records: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
            if len(records) >= n:
                break
    logger.info("Loaded %d prompts for reconstruction ablation.", len(records))
    return records


def load_norm_profile(norm_dir: Path, model_name: str) -> dict[int, LayerNormStats]:
    """Load norm profile CSV and index by layer_idx.

    Args:
        norm_dir:   Root directory containing per-model CSV files.
        model_name: Short model slug.

    Returns:
        Mapping of ``layer_idx → LayerNormStats``.

    Raises:
        SystemExit: If the CSV is missing.
    """
    path = norm_dir / f"{model_name}.csv"
    if not path.exists():
        logger.error("Norm profile missing: %s — run experiment 01 first.", path)
        sys.exit(1)
    stats = load_profile(path)
    return {s.layer_idx: s for s in stats}


def resolve_reconstruction_layers(num_layers: int, fracs: list[float]) -> list[int]:
    """Convert fractional depth targets to concrete layer indices.

    Args:
        num_layers: Total transformer layers.
        fracs:      Depth fractions, e.g. ``[0.4, 0.5]``.

    Returns:
        Unique, sorted list of layer indices.
    """
    return sorted(set(int(f * num_layers) for f in fracs))


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_single(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    steerer: ActivationSteerer,
    layer_config: dict[int, tuple[np.ndarray, float]],
    prompt: str,
    max_new_tokens: int,
    extra_cfg: dict[str, Any],
) -> str:
    """Generate a single response under an optional steering config.

    Args:
        model:          Loaded causal LM.
        tokenizer:      Corresponding tokenizer.
        steerer:        ActivationSteerer bound to the model.
        layer_config:   Steering config; empty dict = no steering.
        prompt:         Raw user message string.
        max_new_tokens: Token budget.
        extra_cfg:      Per-model extras forwarded to ``format_prompt``.

    Returns:
        Decoded response string (prompt prefix stripped).
    """
    device = next(model.parameters()).device
    formatted = format_prompt(tokenizer, prompt, extra_cfg)
    inputs = tokenizer(
        formatted, return_tensors="pt", truncation=True, max_length=1024, padding=False
    ).to(device)
    prompt_len = inputs["input_ids"].shape[1]

    with steerer.steer(layer_config):
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_comparison(
    item: dict,
    baseline_resp: str,
    steered_resp: str,
    layer_idx: int,
    layer_pct: int,
    mean_norm: float,
    k_value: float,
    k_applied: float,
    k_scale: float,
) -> str:
    """Format a single prompt comparison as a human-readable block.

    Args:
        item:          Original prompt record (``id``, ``prompt``).
        baseline_resp: Response with no steering.
        steered_resp:  Response under heavy steering.
        layer_idx:     Target layer index.
        layer_pct:     Layer depth as percentage of total layers.
        mean_norm:     Mean residual stream norm at this layer.
        k_value:       Formula K_ℓ = μ̄_ℓ / √d.
        k_applied:     Actual K used = k_value × k_scale.
        k_scale:       Scale multiplier.

    Returns:
        Multi-line string suitable for writing to a text file.
    """
    return (
        f"{_SEPARATOR}\n"
        f"PROMPT ID: {item['id']}\n"
        f"LAYER: {layer_idx} ({layer_pct}% depth) | "
        f"‖h_ℓ‖={mean_norm:.2f} | K_ℓ={k_value:.4f} | "
        f"K_applied={k_applied:.4f} ({k_scale}× K_ℓ)\n"
        f"PROMPT:\n{item['prompt']}\n\n"
        f"BASELINE:\n{baseline_resp}\n\n"
        f"STEERED (K={k_applied:.4f}):\n{steered_resp}\n"
    )


# ---------------------------------------------------------------------------
# Per-model runner
# ---------------------------------------------------------------------------

def run_model(
    model_cfg: ModelConfig,
    prompts: list[dict],
    profile: dict[int, LayerNormStats],
    directions: list,
    k_scale: float,
    output_root: Path,
    exp_cfg: ExperimentConfig,
    registry: ModelRegistry,
) -> None:
    """Run reconstruction capacity ablation for one model.

    Generates baseline and heavily-steered responses at early-middle layers.
    No judge scoring — outputs are saved for qualitative inspection.

    Args:
        model_cfg:   Config for the model.
        prompts:     HarmBench prompt records.
        profile:     Norm profile indexed by layer_idx.
        directions:  Extracted safety behavioral directions.
        k_scale:     K multiplier to apply (e.g. 20.0 — far above operating point).
        output_root: CSV/JSONL output root.
        exp_cfg:     Loaded ExperimentConfig.
        registry:    ModelRegistry instance.
    """
    recon_layers = resolve_reconstruction_layers(
        model_cfg.num_layers, exp_cfg.reconstruction_layer_fracs
    )
    dir_map = {d.layer_idx: d for d in directions}

    # Quietly skip any layer without extracted directions
    valid_layers = [l for l in recon_layers if l in dir_map and l in profile]
    if not valid_layers:
        logger.error("No valid reconstruction layers for %s — skipping.", model_cfg.name)
        return

    logger.info(
        "=== %s | reconstruction layers: %s | k_scale=%.1f ===",
        model_cfg.name, valid_layers, k_scale,
    )

    out_dir = output_root / model_cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_lines: list[str] = [
        f"Model: {model_cfg.name}",
        f"Layers: {valid_layers}",
        f"K scale: {k_scale}× K_ℓ",
        f"Prompts: {len(prompts)}",
        "",
    ]

    with registry.loaded(model_cfg) as (model, tokenizer):
        steerer = ActivationSteerer(model)

        # Generate baselines once for all prompts
        logger.info("  [%s | baseline]", model_cfg.name)
        baseline_responses: list[str] = []
        for item in tqdm(prompts, desc="  baseline", dynamic_ncols=True):
            resp = generate_single(
                model, tokenizer, steerer, {},
                item["prompt"], exp_cfg.max_new_tokens, model_cfg.extra,
            )
            baseline_responses.append(resp)

        for layer_idx in valid_layers:
            d          = dir_map[layer_idx]
            stats      = profile[layer_idx]
            k_applied  = d.k_value * k_scale
            layer_pct  = int(layer_idx / model_cfg.num_layers * 100)
            layer_config = {layer_idx: (d.mean_direction, k_applied)}

            logger.info(
                "  [%s | layer=%d (%d%%) | K_ℓ=%.4f | K_applied=%.4f (%.0f×)]",
                model_cfg.name, layer_idx, layer_pct, d.k_value, k_applied, k_scale,
            )

            jsonl_records: list[dict] = []
            layer_summary_lines: list[str] = [
                f"\n{'='*60}",
                f"LAYER {layer_idx} ({layer_pct}% depth) | "
                f"‖h_ℓ‖={stats.mean_norm:.2f} | K_ℓ={d.k_value:.4f} | "
                f"K_applied={k_applied:.4f} ({k_scale:.0f}× K_ℓ)",
                "=" * 60,
            ]

            for item, baseline_resp in tqdm(
                zip(prompts, baseline_responses),
                total=len(prompts),
                desc=f"  layer {layer_idx}",
                dynamic_ncols=True,
            ):
                steered_resp = generate_single(
                    model, tokenizer, steerer, layer_config,
                    item["prompt"], exp_cfg.max_new_tokens, model_cfg.extra,
                )

                jsonl_records.append({
                    "model_name":        model_cfg.name,
                    "layer_idx":         layer_idx,
                    "layer_pct":         layer_pct,
                    "mean_norm":         stats.mean_norm,
                    "k_value":           d.k_value,
                    "k_applied":         k_applied,
                    "k_scale":           k_scale,
                    "prompt_id":         item["id"],
                    "user_input":        item["prompt"],
                    "baseline_response": baseline_resp,
                    "steered_response":  steered_resp,
                })

                layer_summary_lines.append(
                    format_comparison(
                        item, baseline_resp, steered_resp,
                        layer_idx, layer_pct,
                        stats.mean_norm, d.k_value, k_applied, k_scale,
                    )
                )

            # Save per-layer JSONL
            jsonl_path = out_dir / f"layer_{layer_pct:03d}pct.jsonl"
            with jsonl_path.open("w", encoding="utf-8") as f:
                for rec in jsonl_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            logger.info("Saved %d records → %s", len(jsonl_records), jsonl_path)

            summary_lines.extend(layer_summary_lines)

    # Write full human-readable summary
    summary_path = out_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    logger.info("Summary written → %s", summary_path)
    logger.info("=== %s done ===", model_cfg.name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablation 02 — reconstruction capacity")
    parser.add_argument("--config",            default="config/models.yml")
    parser.add_argument("--experiment-config", default="config/experiment.yml")
    parser.add_argument("--norm-profiles",     default="results/norm_profiles")
    parser.add_argument("--directions",        default="results/directions")
    parser.add_argument("--eval-prompts",      default="data/eval_prompts")
    parser.add_argument("--output-dir",        default="results/ablations/reconstruction")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override exp_cfg.max_new_tokens.")
    parser.add_argument("--k-scale",     type=float, default=None, help="Override reconstruction_k_scale.")
    parser.add_argument("--n-prompts",   type=int,   default=None, help="Override reconstruction_n_prompts.")
    parser.add_argument("--models",    nargs="*",                  help="Default: all instruct models.")
    parser.add_argument("--load-in-4bit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_configs = load_model_configs(args.config)
    exp_cfg     = load_experiment_config(args.experiment_config)
    k_scale     = args.k_scale   if args.k_scale   is not None else exp_cfg.reconstruction_k_scale
    n_prompts   = args.n_prompts if args.n_prompts is not None else exp_cfg.reconstruction_n_prompts

    if args.models:
        missing = set(args.models) - set(all_configs)
        if missing:
            logger.error("Unknown models: %s", missing)
            sys.exit(1)
        target_names = args.models
    else:
        target_names = [n for n, c in all_configs.items() if c.is_instruct]

    prompts  = load_harmbench_prompts(Path(args.eval_prompts), n_prompts)
    registry = ModelRegistry(load_in_4bit=args.load_in_4bit, device_map="cuda:0")

    logger.info(
        "%d models | %d prompts | layers@%s | k_scale=%.1f×",
        len(target_names), n_prompts, exp_cfg.reconstruction_layer_fracs, k_scale,
    )

    for name in target_names:
        profile = load_norm_profile(Path(args.norm_profiles), name)
        directions_path = Path(args.directions) / name / "safety.npz"
        if not directions_path.exists():
            logger.error("Directions missing: %s — run experiment 02 first.", directions_path)
            continue
        directions = load_directions(str(directions_path))

        run_model(
            model_cfg=all_configs[name],
            prompts=prompts,
            profile=profile,
            directions=directions,
            k_scale=k_scale,
            output_root=Path(args.output_dir),
            exp_cfg=exp_cfg,
            registry=registry,
        )

    logger.info("Ablation 02 complete.")


if __name__ == "__main__":
    main()
