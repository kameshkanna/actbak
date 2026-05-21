"""Activation Baking — persistent behavioral modification via weight biases.

Public API surface.  Import directly from this package in experiment scripts.
"""

from activation_baking.config import (
    ModelConfig,
    ExperimentConfig,
    load_model_configs,
    load_experiment_config,
)
from activation_baking.registry import ModelRegistry
from activation_baking.model_utils import (
    load_model_and_tokenizer,
    format_prompt,
    generate_response,
)
from activation_baking.norm_profiler import (
    NormProfiler,
    LayerNormStats,
    save_profile,
    load_profile,
)
from activation_baking.direction_extractor import (
    DirectionExtractor,
    BehavioralDirection,
    save_directions,
    load_directions,
)
from activation_baking.steerer import ActivationSteerer
from activation_baking.baker import ModelBaker, bake_directions
from activation_baking.evaluators import (
    EvalResult,
    BaseEvaluator,
    GSM8KEvaluator,
    MMLUEvaluator,
    TruthfulQAEvaluator,
    HarmBenchEvaluator,
    run_capability_suite,
)
from activation_baking.judges import (
    HeuristicScorer,
    PerplexityScorer,
    ActivationJudge,
    SmallModelJudge,
    JudgePromptBuilder,
)

__all__ = [
    # config
    "ModelConfig",
    "ExperimentConfig",
    "load_model_configs",
    "load_experiment_config",
    # registry
    "ModelRegistry",
    # model utils
    "load_model_and_tokenizer",
    "format_prompt",
    "generate_response",
    # norm profiling
    "NormProfiler",
    "LayerNormStats",
    "save_profile",
    "load_profile",
    # direction extraction
    "DirectionExtractor",
    "BehavioralDirection",
    "save_directions",
    "load_directions",
    # steering / baking
    "ActivationSteerer",
    "ModelBaker",
    "bake_directions",
    # evaluators
    "EvalResult",
    "BaseEvaluator",
    "GSM8KEvaluator",
    "MMLUEvaluator",
    "TruthfulQAEvaluator",
    "HarmBenchEvaluator",
    "run_capability_suite",
    # judges
    "HeuristicScorer",
    "PerplexityScorer",
    "ActivationJudge",
    "SmallModelJudge",
    "JudgePromptBuilder",
]
