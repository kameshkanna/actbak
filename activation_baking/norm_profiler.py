"""Per-layer residual stream norm profiler for transformer causal LMs."""

from __future__ import annotations

import gc
import logging
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from activation_baking.config import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class LayerNormStats:
    """Norm statistics and calibrated steering magnitude for one transformer layer.

    Attributes:
        model_name:   Short model slug from ``ModelConfig.name``.
        layer_idx:    Zero-based index of the decoder layer.
        mean_norm:    Mean L2 norm of the residual stream across all profiled
                      prompts and sequence positions.
        std_norm:     Standard deviation of those per-prompt mean norms.
        k_value:      Calibrated steering magnitude ``μ̄_ℓ / √d`` where ``d``
                      is the model hidden size.
        sample_count: Number of (prompt, position) samples contributing to the
                      statistics (equals the number of profiled prompts since
                      norms are averaged over the sequence dimension first).
    """

    model_name: str
    layer_idx: int
    mean_norm: float
    std_norm: float
    k_value: float
    sample_count: int

    def to_dict(self) -> dict:
        """Serialize the dataclass to a plain dictionary."""
        return asdict(self)


class NormProfiler:
    """Measures per-layer residual stream L2 norms across a set of prompts.

    For each decoder layer the profiler attaches a forward hook to the post-
    residual hidden state, computes the mean L2 norm across the sequence
    dimension, and accumulates per-prompt scalar values.  After profiling,
    the calibrated steering magnitude is derived as:

        K_ℓ = μ̄_ℓ / √d

    where μ̄_ℓ is the mean norm at layer ℓ and d is the model hidden size.
    All layers are profiled; the caller is responsible for selecting a subset
    (e.g. ``ModelConfig.middle_layers``) for downstream use.

    Hooks are guaranteed to be removed even if an exception occurs during the
    forward pass.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        model_cfg: ModelConfig,
        max_length: int = 512,
    ) -> None:
        """Initialise the profiler.

        Args:
            model: Loaded causal LM in eval mode.
            tokenizer: Corresponding tokenizer with pad token set.
            model_cfg: ``ModelConfig`` for the loaded model; ``hidden_size``
                and ``num_layers`` are pulled from this object.
            max_length: Maximum tokenized sequence length; longer inputs are
                truncated.
        """
        self._model = model
        self._tokenizer = tokenizer
        self._model_cfg = model_cfg
        self._hidden_size: int = model_cfg.hidden_size
        self._num_layers: int = model_cfg.num_layers
        self._max_length = max_length
        self._device: torch.device = self._resolve_input_device(model)
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._layer_norms: dict[int, list[float]] = defaultdict(list)

    @staticmethod
    def _resolve_input_device(model: PreTrainedModel) -> torch.device:
        """Return the device where input tokens should be placed.

        With device_map='auto' the embedding layer lands on the first device in
        hf_device_map.  Falling back to next(model.parameters()).device works
        for single-GPU and CPU models.
        """
        hf_map: dict | None = getattr(model, "hf_device_map", None)
        if hf_map:
            embed_device: str = hf_map.get("model.embed_tokens") or next(iter(hf_map.values()))
            return torch.device(embed_device) if embed_device != "cpu" else torch.device("cpu")
        return next(model.parameters()).device

    @staticmethod
    def _clear_cuda_cache() -> None:
        """Empty the CUDA allocator cache on every visible device."""
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()

    def _make_hook(self, layer_idx: int) -> Callable:
        """Return a forward hook that records mean sequence-level L2 norms.

        The hook operates on the first element of tuple outputs (standard for
        decoder layers) or directly on tensor outputs.  Norms are computed in
        float32 to avoid precision loss and immediately moved to CPU.

        Args:
            layer_idx: Index of the layer whose output is being hooked.

        Returns:
            A ``register_forward_hook``-compatible callable.
        """

        def hook(
            module: nn.Module,
            input: tuple,
            output: tuple | torch.Tensor,
        ) -> None:
            hidden: torch.Tensor = output[0] if isinstance(output, tuple) else output
            norms = hidden.detach().float().norm(dim=-1)  # [batch, seq_len]
            seq_means = norms.mean(dim=-1).cpu().tolist()  # [batch]
            self._layer_norms[layer_idx].extend(seq_means)

        return hook

    def _register_hooks(self) -> None:
        """Attach forward hooks to every decoder layer in the model."""
        layers = self._model.model.layers
        for idx, layer in enumerate(layers):
            handle = layer.register_forward_hook(self._make_hook(idx))
            self._hooks.append(handle)

    def _remove_hooks(self) -> None:
        """Remove all registered hooks and clear the internal handle list."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def profile(self, prompts: list[str]) -> list["LayerNormStats"]:
        """Run norm profiling over the provided prompts.

        Performs a single no-grad forward pass per prompt with hooks active,
        then computes summary statistics.  CUDA caches are cleared after each
        forward pass to prevent OOM accumulation on large prompt sets, and
        ``gc.collect`` is called after the loop to release any residual
        Python-side references.

        Args:
            prompts: Pre-formatted prompt strings (e.g. output of
                ``model_utils.format_prompt``).

        Returns:
            One :class:`LayerNormStats` per layer, sorted by layer index.

        Raises:
            RuntimeError: If ``model.model.layers`` is not accessible (i.e. the
                model architecture does not expose a standard layers attribute).
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

                self._clear_cuda_cache()
        finally:
            self._remove_hooks()

        gc.collect()
        return self._compute_stats()

    def _compute_stats(self) -> list["LayerNormStats"]:
        """Aggregate raw per-prompt norms into :class:`LayerNormStats` objects.

        Returns:
            A list of :class:`LayerNormStats`, one per layer with recorded data,
            sorted by ascending layer index.  Layers with no data are skipped
            with a warning.
        """
        stats: list[LayerNormStats] = []
        for layer_idx in range(self._num_layers):
            norms = np.array(self._layer_norms[layer_idx], dtype=np.float64)
            if len(norms) == 0:
                logger.warning("No norms recorded for layer %d — skipping.", layer_idx)
                continue
            mean_norm = float(np.mean(norms))
            std_norm = float(np.std(norms))
            k_value = mean_norm / math.sqrt(self._hidden_size)
            stats.append(
                LayerNormStats(
                    model_name=self._model_cfg.name,
                    layer_idx=layer_idx,
                    mean_norm=mean_norm,
                    std_norm=std_norm,
                    k_value=k_value,
                    sample_count=len(norms),
                )
            )
        return stats


def save_profile(stats: list[LayerNormStats], path: str | Path) -> None:
    """Persist a list of :class:`LayerNormStats` to a CSV file via pandas.

    The output columns match the dataclass field order exactly so that
    :func:`load_profile` can round-trip the data without schema ambiguity.

    Args:
        stats: Non-empty list of :class:`LayerNormStats` to serialise.
        path: Destination file path; parent directories must exist.

    Raises:
        ValueError: If ``stats`` is empty.
        OSError: If the path is not writable.
    """
    if not stats:
        raise ValueError("Cannot save an empty stats list.")

    path = Path(path)
    rows = [s.to_dict() for s in stats]
    df = pd.DataFrame(rows, columns=[f.name for f in fields(LayerNormStats)])
    df.to_csv(path, index=False)
    logger.info("Saved %d layer stats to %s", len(stats), path)


def load_profile(path: str | Path) -> list[LayerNormStats]:
    """Load :class:`LayerNormStats` records from a CSV file produced by :func:`save_profile`.

    Args:
        path: Path to the CSV file written by :func:`save_profile`.

    Returns:
        List of :class:`LayerNormStats` sorted by ascending layer index.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        KeyError: If the CSV is missing required columns.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Profile CSV not found: {path}")

    df = pd.read_csv(path)
    required_cols = {f.name for f in fields(LayerNormStats)}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Profile CSV is missing columns: {missing}")

    stats: list[LayerNormStats] = [
        LayerNormStats(
            model_name=str(row["model_name"]),
            layer_idx=int(row["layer_idx"]),
            mean_norm=float(row["mean_norm"]),
            std_norm=float(row["std_norm"]),
            k_value=float(row["k_value"]),
            sample_count=int(row["sample_count"]),
        )
        for row in df.sort_values("layer_idx").to_dict(orient="records")
    ]
    logger.info("Loaded %d layer stats from %s", len(stats), path)
    return stats
