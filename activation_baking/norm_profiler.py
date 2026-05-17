"""Per-layer residual stream norm profiler for transformer causal LMs."""

import gc
import logging
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


@dataclass
class LayerNormStats:
    """Norm statistics and calibrated K value for a single transformer layer."""

    layer_idx: int
    mean_norm: float
    std_norm: float
    k_value: float
    sample_count: int

    def to_dict(self) -> dict:
        return asdict(self)


class NormProfiler:
    """Measures per-layer residual stream L2 norms across a set of prompts.

    For each decoder layer the profiler hooks the post-residual hidden state,
    computes the mean L2 norm across the sequence dimension, and aggregates
    statistics across all prompts.  The calibrated steering magnitude is then:

        K_ℓ = μ̄_ℓ / √d

    where μ̄_ℓ is the mean norm at layer ℓ and d is the model hidden size.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        hidden_size: int,
        num_layers: int,
        max_length: int = 512,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._max_length = max_length
        self._device = next(model.parameters()).device
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._layer_norms: dict[int, list[float]] = defaultdict(list)

    def _make_hook(self, layer_idx: int) -> Callable:
        """Return a forward hook that records mean sequence-level L2 norms."""

        def hook(
            module: nn.Module,
            input: tuple,
            output: tuple | torch.Tensor,
        ) -> None:
            hidden: torch.Tensor = output[0] if isinstance(output, tuple) else output
            # hidden: [batch, seq_len, hidden_size]
            norms = hidden.detach().float().norm(dim=-1)  # [batch, seq_len]
            seq_means = norms.mean(dim=-1).cpu().tolist()  # [batch]
            self._layer_norms[layer_idx].extend(seq_means)

        return hook

    def _register_hooks(self) -> None:
        layers = self._model.model.layers
        for idx, layer in enumerate(layers):
            handle = layer.register_forward_hook(self._make_hook(idx))
            self._hooks.append(handle)

    def _remove_hooks(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def profile(self, prompts: list[str]) -> list[LayerNormStats]:
        """Run norm profiling over the provided prompts.

        Args:
            prompts: List of pre-formatted prompt strings.

        Returns:
            One LayerNormStats per layer, sorted by layer index.
        """
        self._layer_norms.clear()
        self._register_hooks()

        try:
            for prompt in tqdm(prompts, desc="Norm profiling", dynamic_ncols=True):
                inputs = self._tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self._max_length,
                    padding=False,
                ).to(self._device)

                with torch.no_grad():
                    self._model(**inputs)

                if self._device.type == "cuda":
                    torch.cuda.empty_cache()
        finally:
            self._remove_hooks()

        gc.collect()
        return self._compute_stats()

    def _compute_stats(self) -> list[LayerNormStats]:
        stats: list[LayerNormStats] = []
        for layer_idx in range(self._num_layers):
            norms = np.array(self._layer_norms[layer_idx], dtype=np.float64)
            if len(norms) == 0:
                logger.warning("No norms recorded for layer %d", layer_idx)
                continue
            mean_norm = float(np.mean(norms))
            std_norm = float(np.std(norms))
            k_value = mean_norm / math.sqrt(self._hidden_size)
            stats.append(LayerNormStats(
                layer_idx=layer_idx,
                mean_norm=mean_norm,
                std_norm=std_norm,
                k_value=k_value,
                sample_count=len(norms),
            ))
        return stats
