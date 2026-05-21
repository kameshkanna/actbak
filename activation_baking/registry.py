"""Model registry — load once, reuse everywhere.

The ``ModelRegistry`` maintains a per-process cache of loaded models and
tokenizers keyed by HuggingFace model id.  Every experiment that needs a model
goes through the registry; the model is never loaded more than once per
process, and memory is released deterministically via ``release`` or the
``loaded`` context manager.

Typical usage::

    registry = ModelRegistry()

    with registry.loaded(model_cfg) as (model, tokenizer):
        profiler   = NormProfiler(model, tokenizer, model_cfg)
        directions = DirectionExtractor(model, tokenizer, model_cfg, k_values).extract(...)
        # ... all operations on the same loaded model

    # model released here — GPU memory returned
"""

from __future__ import annotations

import gc
import logging
from contextlib import contextmanager
from typing import Generator

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from activation_baking.config import ModelConfig
from activation_baking.model_utils import load_model_and_tokenizer

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Thread-local cache of (model, tokenizer) pairs.

    Models are loaded lazily on first access and kept alive until
    ``release`` is called.  Use ``loaded`` as a context manager to
    guarantee cleanup even if an exception is raised mid-experiment.

    Attributes:
        dtype:       Default torch dtype for all loaded models.
        device_map:  Device placement strategy passed to ``from_pretrained``.
        load_in_4bit: Enable 4-bit quantisation via bitsandbytes.
    """

    def __init__(
        self,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        load_in_4bit: bool = False,
    ) -> None:
        self._dtype = dtype
        self._device_map = device_map
        self._load_in_4bit = load_in_4bit
        self._cache: dict[str, tuple[PreTrainedModel, PreTrainedTokenizerBase]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self, model_cfg: ModelConfig
    ) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """Return a cached (model, tokenizer) pair, loading if necessary.

        Args:
            model_cfg: Configuration of the model to load.

        Returns:
            ``(model, tokenizer)`` ready for inference.
        """
        key = model_cfg.hf_id
        if key not in self._cache:
            logger.info("Loading %s into registry...", model_cfg.name)
            model, tokenizer = load_model_and_tokenizer(
                hf_id=model_cfg.hf_id,
                dtype=self._dtype or model_cfg.dtype,
                load_in_4bit=self._load_in_4bit,
                device_map=self._device_map,
            )
            self._cache[key] = (model, tokenizer)
            logger.info("Registry now holds: %s", list(self._cache))
        return self._cache[key]

    def release(self, model_cfg: ModelConfig) -> None:
        """Unload a model and free GPU memory.

        Args:
            model_cfg: Config of the model to evict from the cache.
        """
        key = model_cfg.hf_id
        if key in self._cache:
            del self._cache[key]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Released %s from registry.", model_cfg.name)

    def release_all(self) -> None:
        """Unload every cached model."""
        keys = list(self._cache.keys())
        for key in keys:
            del self._cache[key]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Registry cleared (%d models released).", len(keys))

    @contextmanager
    def loaded(
        self, model_cfg: ModelConfig
    ) -> Generator[tuple[PreTrainedModel, PreTrainedTokenizerBase], None, None]:
        """Context manager that guarantees model cleanup on exit.

        Yields:
            ``(model, tokenizer)`` for the duration of the block.

        Example::

            with registry.loaded(cfg) as (model, tok):
                run_norm_profiling(model, tok, cfg)
                run_direction_extraction(model, tok, cfg)
        """
        model, tokenizer = self.get(model_cfg)
        try:
            yield model, tokenizer
        finally:
            self.release(model_cfg)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def loaded_models(self) -> list[str]:
        """HuggingFace ids of currently cached models."""
        return list(self._cache.keys())

    def vram_summary(self) -> str:
        """Return a human-readable VRAM usage string (CUDA only)."""
        if not torch.cuda.is_available():
            return "CUDA unavailable"
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved  = torch.cuda.memory_reserved()  / 1e9
        total     = torch.cuda.get_device_properties(0).total_memory / 1e9
        return (
            f"VRAM  allocated={allocated:.1f}GB  "
            f"reserved={reserved:.1f}GB  "
            f"total={total:.1f}GB"
        )
