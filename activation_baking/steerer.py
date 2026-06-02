"""Inference-time activation steering via forward hooks.

Adds K·direction to the residual stream at configured transformer layers without
modifying model weights. For persistent weight-level baking see baker.py.

Two injection modes are supported per layer:

- ``"broadcast"`` (default): adds delta to every token position.
  Identical to the ``resid_bias.view(1,1,-1)`` in the baked HuggingFace models
  (Kameshr/Llama-3B-AntiSafety-Low, Qwen-anti-safety-*).  "Iron Wall" semantics.

- ``"last_token"``: adds delta only at the final sequence position.
  Classic CAA / RepE inference-time steering — only influences the next-token
  prediction without rewriting every prompt-token representation.

The layer config tuple is ``(direction, k_value)`` or
``(direction, k_value, inject_mode)``; the third element defaults to
``"broadcast"`` when omitted for backwards compatibility.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator, Literal

import numpy as np
import torch
import torch.nn as nn
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)

InjectMode = Literal["broadcast", "last_token"]


class ActivationSteerer:
    """Injects K·direction into transformer residual streams at runtime.

    For each registered layer l, adds K_l · d̂_l to hidden states, where
    d̂_l is a unit-norm behavioral direction and K_l is the calibrated magnitude.
    Injection can be broadcast across all positions or restricted to the last token.

    Hooks are installed on entry and removed on exit of the ``steer`` context
    manager. The model's weights are never modified.

    Example::

        steerer = ActivationSteerer(model)
        # broadcast (baking-equivalent)
        with steerer.steer({16: (direction_vec, k_value)}):
            ids = model.generate(**inputs, max_new_tokens=200)

        # last-token only (CAA-style)
        with steerer.steer({16: (direction_vec, k_value, "last_token")}):
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
        inject_mode: InjectMode = "broadcast",
    ):
        """Return a forward hook for the given direction, magnitude, and injection mode.

        Args:
            direction:    Unit-norm 1-D float32 array of shape ``(hidden_size,)``.
            k_value:      Injection magnitude.
            inject_mode:  ``"broadcast"`` adds delta to all token positions (matches
                          baked model ``resid_bias.view(1,1,-1)`` semantics).
                          ``"last_token"`` adds delta only at position ``[:, -1, :]``
                          (CAA / RepE inference-time convention).
        """
        dir_tensor = torch.from_numpy(direction.copy()).float()  # (d,)
        delta_cpu = dir_tensor * k_value                         # pre-scale once

        if inject_mode == "last_token":
            def hook(
                module: nn.Module,
                input: tuple,
                output: tuple | torch.Tensor,
            ) -> tuple | torch.Tensor:
                hidden: torch.Tensor = output[0] if isinstance(output, tuple) else output
                delta = delta_cpu.to(device=hidden.device, dtype=hidden.dtype)
                steered = hidden.clone()
                steered[:, -1, :] = steered[:, -1, :] + delta
                return (steered,) + output[1:] if isinstance(output, tuple) else steered
        else:  # broadcast — Iron Wall, matches baked resid_bias
            def hook(
                module: nn.Module,
                input: tuple,
                output: tuple | torch.Tensor,
            ) -> tuple | torch.Tensor:
                hidden: torch.Tensor = output[0] if isinstance(output, tuple) else output
                delta = delta_cpu.to(device=hidden.device, dtype=hidden.dtype)
                steered = hidden + delta.view(1, 1, -1)
                return (steered,) + output[1:] if isinstance(output, tuple) else steered

        return hook

    # ------------------------------------------------------------------
    # Hook lifecycle
    # ------------------------------------------------------------------

    def _install(
        self,
        layer_configs: dict[int, tuple[np.ndarray, float] | tuple[np.ndarray, float, InjectMode]],
    ) -> None:
        for layer_idx, cfg in layer_configs.items():
            direction, k_value = cfg[0], cfg[1]
            inject_mode: InjectMode = cfg[2] if len(cfg) == 3 else "broadcast"  # type: ignore[misc]
            handle = self._model.model.layers[layer_idx].register_forward_hook(
                self._make_hook(direction, k_value, inject_mode)
            )
            self._hooks.append(handle)
            logger.debug("Steering hook at layer %d (K=%.4f, mode=%s)", layer_idx, k_value, inject_mode)

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
        layer_configs: dict[int, tuple[np.ndarray, float] | tuple[np.ndarray, float, InjectMode]],
    ) -> Generator[None, None, None]:
        """Temporarily apply steering hooks for the duration of the block.

        Args:
            layer_configs: Mapping of ``layer_idx → (direction, k_value)`` or
                           ``layer_idx → (direction, k_value, inject_mode)``.
                           Directions must be unit-norm 1-D float32 arrays.
                           ``inject_mode`` defaults to ``"broadcast"`` if omitted.

        Yields:
            None; the model is temporarily modified in-place via hooks.
        """
        self._remove()
        self._install(layer_configs)
        try:
            yield
        finally:
            self._remove()
