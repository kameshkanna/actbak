"""Inference-time activation steering via forward hooks.

Adds K·direction to the residual stream at configured transformer layers without
modifying model weights. For persistent weight-level baking see baker.py.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator

import numpy as np
import torch
import torch.nn as nn
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


class ActivationSteerer:
    """Injects K·direction into transformer residual streams at runtime.

    For each registered layer l, adds K_l · d̂_l to every token position's
    hidden state, where d̂_l is a unit-norm behavioral direction and K_l is
    the calibrated injection magnitude.

    Hooks are installed on entry and removed on exit of the ``steer`` context
    manager. The model's weights are never modified.

    Example::

        steerer = ActivationSteerer(model)
        with steerer.steer({16: (direction_vec, k_value)}):
            ids = model.generate(**inputs, max_new_tokens=200)
    """

    def __init__(self, model: PreTrainedModel) -> None:
        """
        Args:
            model: HuggingFace transformer with a ``model.model.layers`` attribute.
        """
        self._model = model
        self._hooks: list = []

    # ------------------------------------------------------------------
    # Hook construction
    # ------------------------------------------------------------------

    def _make_hook(
        self,
        direction: np.ndarray,
        k_value: float,
    ):
        """Return a forward hook that adds k_value * direction to hidden states.

        Args:
            direction: Unit-norm 1-D float32 array of shape (hidden_size,).
            k_value: Injection magnitude.
        """
        dir_tensor = torch.from_numpy(direction.copy()).float()  # (d,)
        delta_cpu = dir_tensor * k_value  # pre-scale once

        def hook(
            module: nn.Module,
            input: tuple,
            output: tuple | torch.Tensor,
        ) -> tuple | torch.Tensor:
            hidden: torch.Tensor = output[0] if isinstance(output, tuple) else output
            delta = delta_cpu.to(device=hidden.device, dtype=hidden.dtype)
            steered = hidden + delta.unsqueeze(0).unsqueeze(0)  # broadcast (1, 1, d)
            return (steered,) + output[1:] if isinstance(output, tuple) else steered

        return hook

    # ------------------------------------------------------------------
    # Hook lifecycle
    # ------------------------------------------------------------------

    def _install(self, layer_configs: dict[int, tuple[np.ndarray, float]]) -> None:
        for layer_idx, (direction, k_value) in layer_configs.items():
            handle = self._model.model.layers[layer_idx].register_forward_hook(
                self._make_hook(direction, k_value)
            )
            self._hooks.append(handle)
            logger.debug("Steering hook at layer %d (K=%.4f)", layer_idx, k_value)

    def _remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    @contextmanager
    def steer(
        self,
        layer_configs: dict[int, tuple[np.ndarray, float]],
    ) -> Generator[None, None, None]:
        """Temporarily apply steering hooks for the duration of the block.

        Args:
            layer_configs: Mapping of ``layer_idx → (unit_direction, k_value)``.
                           Directions must be unit-norm 1-D float32 arrays.

        Yields:
            None; the model is temporarily modified in-place via hooks.
        """
        self._remove()
        self._install(layer_configs)
        try:
            yield
        finally:
            self._remove()

    # ------------------------------------------------------------------
    # Config builders
    # ------------------------------------------------------------------

    def build_ramp_config(
        self,
        directions: list,
        scale: float = 1.0,
    ) -> dict[int, tuple[np.ndarray, float]]:
        """Build a ramped layer config using per-layer formula K values.

        K_ℓ = μ̄_ℓ / √d is encoded in each BehavioralDirection; ``scale``
        multiplies every K (default 1.0 = formula value).

        Args:
            directions: Output of ``DirectionExtractor.extract()`` or
                        ``load_directions()``.
            scale: Global multiplier applied to all K values.

        Returns:
            layer_configs dict ready for ``steer()``.
        """
        return {d.layer_idx: (d.mean_direction, d.k_value * scale) for d in directions}

    def build_flat_config(
        self,
        directions: list,
        k_override: float | None = None,
        scale: float = 1.0,
    ) -> dict[int, tuple[np.ndarray, float]]:
        """Build a flat-K layer config where all layers receive the same K.

        Args:
            directions: Output of ``DirectionExtractor.extract()`` or
                        ``load_directions()``.
            k_override: Explicit K value; uses mean(K_ℓ) if None.
            scale: Multiplier applied to K.

        Returns:
            layer_configs dict ready for ``steer()``.
        """
        k = k_override if k_override is not None else float(
            np.mean([d.k_value for d in directions])
        )
        return {d.layer_idx: (d.mean_direction, k * scale) for d in directions}

    def build_single_layer_config(
        self,
        directions: list,
        layer_idx: int,
        k_scale: float = 1.0,
    ) -> dict[int, tuple[np.ndarray, float]]:
        """Build config for a single-layer injection.

        Args:
            directions: Full direction set; the entry for ``layer_idx`` is used.
            layer_idx: Target layer index.
            k_scale: Multiplier applied to the layer's formula K value.

        Returns:
            layer_configs dict with exactly one entry.

        Raises:
            ValueError: If ``layer_idx`` is not present in ``directions``.
        """
        dir_map = {d.layer_idx: d for d in directions}
        if layer_idx not in dir_map:
            raise ValueError(
                f"layer_idx={layer_idx} not in extracted directions. "
                f"Available: {sorted(dir_map)}"
            )
        d = dir_map[layer_idx]
        return {layer_idx: (d.mean_direction, d.k_value * k_scale)}
