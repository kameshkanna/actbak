"""Centralised configuration dataclasses for activation baking experiments.

All experiment parameters flow through these dataclasses.  Instantiate from
YAML via ``ModelConfig.from_dict`` / ``ExperimentConfig.from_dict``, or
construct directly in code.  Nothing in this module imports torch or
transformers — it is safe to import anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Static description of a single model variant.

    Attributes:
        name:             Short slug used as a directory key (e.g. ``llama-3.1-8b``).
        hf_id:            HuggingFace model identifier.
        norm_type:        RMSNorm variant — ``"pre"`` (standard) or ``"pre_post"``
                          (Gemma-2 dual-norm).
        hidden_size:      Residual stream dimension ``d``.
        num_layers:       Total transformer layers.
        dtype:            Torch dtype string — ``"bfloat16"`` or ``"float16"``.
        is_instruct:      True if RLHF/instruction-tuned; False for base model.
        base_counterpart: ``name`` of the corresponding base model config, or
                          ``None`` if this *is* the base or no counterpart exists.
        extra:            Arbitrary per-model kwargs forwarded to tokenizer /
                          chat-template calls (e.g. ``enable_thinking: false``).
    """

    name: str
    hf_id: str
    norm_type: str
    hidden_size: int
    num_layers: int
    dtype: str = "bfloat16"
    is_instruct: bool = True
    base_counterpart: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def middle_layer_range(self) -> tuple[int, int]:
        """Inclusive (start, end) indices of the middle 50% of layers."""
        return self.num_layers // 4, (3 * self.num_layers) // 4

    @property
    def middle_layers(self) -> list[int]:
        """Layer indices spanning the middle 50% of the network."""
        start, end = self.middle_layer_range
        return list(range(start, end))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelConfig":
        """Construct from a raw YAML dict (unknown keys are silently dropped)."""
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """Runtime parameters shared across all experiment scripts.

    Attributes:
        behaviors:                  Behavioral axes to evaluate.
        k_scales:                   K multipliers for the ramp sweep.
        max_new_tokens:             Token budget for model generation.
        groq_judge_model:           Groq model id for GroqJudge (primary scorer).
        judge_model:                HuggingFace id of the SmallModelJudge (legacy fallback).
        judge_batch_size:           Prompts per judge forward pass.
        n_harmbench:                HarmBench samples for safety eval.
        n_clearharm:                ClearHarm samples for safety eval.
        n_sycophancy:               Anthropic sycophancy eval samples.
        n_gsm8k:                    GSM8K samples for capability check.
        n_mmlu:                     MMLU samples for capability check.
        n_truthfulqa:               TruthfulQA samples for capability check.
        ablation_n_prompts:         HarmBench prompts for norm calibration ablation.
        ablation_k_scales:          K scale multipliers for norm calibration ablation.
        ablation_layer_fracs:       Layer depth fractions for norm calibration (40–80%).
        reconstruction_k_scale:     K multiplier for reconstruction capacity ablation.
        reconstruction_n_prompts:   Prompt count for reconstruction capacity ablation.
        reconstruction_layer_fracs: Layer depth fractions for reconstruction ablation.
        seed:                       Global random seed for reproducibility.
        results_dir:                Root directory for CSV / npz outputs.
        figures_dir:                Root directory for PDF / PNG plots.
    """

    behaviors: list[str] = field(default_factory=lambda: ["safety", "sycophancy"])
    k_scales: list[float] = field(default_factory=lambda: [0.25, 0.5, 1.0, 2.0, 3.0, 5.0])
    max_new_tokens: int = 300
    groq_judge_model: str = "llama-3.3-70b-versatile"
    judge_model: str = "Qwen/Qwen2.5-3B-Instruct"
    judge_batch_size: int = 8
    n_harmbench: int = 200
    n_clearharm: int = 200
    n_sycophancy: int = 200
    n_gsm8k: int = 500
    n_mmlu: int = 500
    n_truthfulqa: int = 500
    ablation_n_prompts: int = 50
    ablation_k_scales: list[float] = field(default_factory=lambda: [0.1, 1.0, 10.0])
    ablation_layer_fracs: list[float] = field(default_factory=lambda: [0.4, 0.5, 0.6, 0.7, 0.8])
    reconstruction_k_scale: float = 20.0
    reconstruction_n_prompts: int = 10
    reconstruction_layer_fracs: list[float] = field(default_factory=lambda: [0.4, 0.5])
    seed: int = 42
    results_dir: str = "results"
    figures_dir: str = "figures"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExperimentConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# Registry loader
# ---------------------------------------------------------------------------


def load_model_configs(path: str | Path) -> dict[str, ModelConfig]:
    """Load all model configurations from a YAML file.

    Args:
        path: Path to ``config/models.yml``.

    Returns:
        Mapping of ``name → ModelConfig`` for every entry in the file.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)
    configs = {}
    for entry in raw["models"]:
        cfg = ModelConfig.from_dict(entry)
        configs[cfg.name] = cfg
    return configs


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load experiment configuration from a YAML file.

    Args:
        path: Path to ``config/experiment.yml``.

    Returns:
        Populated ``ExperimentConfig``.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)
    return ExperimentConfig.from_dict(raw.get("experiment", {}))
