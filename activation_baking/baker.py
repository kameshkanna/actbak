"""Persistent activation baking via rank-1 MLP bias injection.

Instead of adding K·direction to the residual stream at inference time through
forward hooks (see ``steerer.ActivationSteerer``), this module permanently bakes
the effect into the model's weights.  The mechanism is:

For each target layer ℓ, the MLP output projection (``down_proj``) computes::

    output = W_down @ a  (+ bias if present)

Adding K·direction to the residual stream on every forward pass is algebraically
equivalent to adding K·direction to the bias of ``down_proj``.  If that layer has
no explicit bias term, one is created.  This makes the intervention zero-cost at
inference time — no hooks, no extra memory bandwidth.
"""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from transformers import PreTrainedModel

from activation_baking.direction_extractor import BehavioralDirection

logger = logging.getLogger(__name__)


class ModelBaker:
    """Bakes behavioral directions into MLP ``down_proj`` biases.

    For each target layer ℓ with direction ĉ and magnitude K, the operation is::

        down_proj.bias += K * direction

    If ``down_proj`` has no bias, a zero bias ``nn.Parameter`` is created first.
    The effect on every subsequent forward pass is identical to an activation
    steering hook at that layer, but with no runtime overhead.

    Reversibility is supported via ``unbake()``, which subtracts the same delta
    and removes the bias parameter entirely if it returns to all-zeros (i.e. the
    layer had no original bias).

    Example::

        baker = ModelBaker(model)
        baked = baker.bake({16: (direction_vec, k_val), 20: (direction_vec, k_val)})
        baker.save_baked_model(baked, "baked_model/")
    """

    def __init__(self, model: PreTrainedModel) -> None:
        """
        Args:
            model: HuggingFace causal LM with a ``model.model.layers`` attribute
                   where each layer exposes ``mlp.down_proj`` as an ``nn.Linear``.
        """
        self._model = model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_down_proj(self, model: PreTrainedModel, layer_idx: int) -> nn.Linear:
        """Retrieve the ``down_proj`` linear layer for a given transformer layer.

        Args:
            model:     The model to inspect.
            layer_idx: Zero-based transformer layer index.

        Returns:
            The ``nn.Linear`` module corresponding to ``down_proj``.

        Raises:
            AttributeError: If the expected path
                            ``model.model.layers[layer_idx].mlp.down_proj``
                            does not exist.
        """
        return model.model.layers[layer_idx].mlp.down_proj

    def _validate_direction(
        self,
        direction: np.ndarray,
        layer_idx: int,
        down_proj: nn.Linear,
    ) -> None:
        """Validate that ``direction`` is compatible with ``down_proj``'s output dimension.

        Args:
            direction:  1-D float array of shape ``(hidden_size,)``.
            layer_idx:  Layer index (used only for error messages).
            down_proj:  The target linear module; ``out_features`` must equal
                        ``direction.shape[0]``.

        Raises:
            ValueError: If ``direction`` is not 1-D or its length does not match
                        ``down_proj.out_features``.
        """
        if direction.ndim != 1:
            raise ValueError(
                f"Layer {layer_idx}: direction must be 1-D, got shape {direction.shape}."
            )
        hidden_size = down_proj.out_features
        if direction.shape[0] != hidden_size:
            raise ValueError(
                f"Layer {layer_idx}: direction length {direction.shape[0]} does not match "
                f"down_proj.out_features={hidden_size}."
            )

    def _apply_delta(
        self,
        down_proj: nn.Linear,
        direction: np.ndarray,
        k_value: float,
        layer_idx: int,
    ) -> None:
        """Add ``k_value * direction`` to ``down_proj``'s bias in-place.

        If ``down_proj`` has no bias, a zero ``nn.Parameter`` is registered
        before adding the delta so that the operation is reversible.

        Args:
            down_proj:  Target ``nn.Linear`` module.
            direction:  Unit-norm 1-D float array of shape ``(hidden_size,)``.
            k_value:    Injection magnitude.
            layer_idx:  Layer index (used for logging).
        """
        device: torch.device = down_proj.weight.device
        dtype: torch.dtype = down_proj.weight.dtype

        delta: torch.Tensor = (
            torch.from_numpy(direction.copy())
            .to(device=device, dtype=dtype)
            .mul_(k_value)
        )

        if down_proj.bias is None:
            down_proj.bias = nn.Parameter(
                torch.zeros(down_proj.out_features, device=device, dtype=dtype)
            )
            logger.debug(
                "Layer %d: created zero bias for down_proj (out_features=%d).",
                layer_idx,
                down_proj.out_features,
            )

        down_proj.bias.data.add_(delta)
        logger.info(
            "Layer %d: baked K=%.4f * direction into down_proj.bias.",
            layer_idx,
            k_value,
        )

    def _remove_delta(
        self,
        down_proj: nn.Linear,
        direction: np.ndarray,
        k_value: float,
        layer_idx: int,
    ) -> None:
        """Subtract ``k_value * direction`` from ``down_proj``'s bias in-place.

        If the bias is all-zeros after subtraction (within floating-point
        tolerance), it is removed entirely by setting the parameter to ``None``.

        Args:
            down_proj:  Target ``nn.Linear`` module.
            direction:  The same unit-norm array used during baking.
            k_value:    The same injection magnitude used during baking.
            layer_idx:  Layer index (used for logging).

        Raises:
            ValueError: If ``down_proj`` has no bias (nothing to unbake).
        """
        if down_proj.bias is None:
            raise ValueError(
                f"Layer {layer_idx}: down_proj has no bias; cannot unbake."
            )

        device: torch.device = down_proj.weight.device
        dtype: torch.dtype = down_proj.weight.dtype

        delta: torch.Tensor = (
            torch.from_numpy(direction.copy())
            .to(device=device, dtype=dtype)
            .mul_(k_value)
        )

        down_proj.bias.data.sub_(delta)
        logger.info(
            "Layer %d: removed K=%.4f * direction from down_proj.bias.",
            layer_idx,
            k_value,
        )

        if torch.allclose(
            down_proj.bias.data,
            torch.zeros_like(down_proj.bias.data),
            atol=1e-6,
        ):
            down_proj.bias = None
            logger.debug(
                "Layer %d: bias returned to zero; parameter removed.", layer_idx
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def bake(
        self,
        layer_configs: dict[int, tuple[np.ndarray, float]],
        inplace: bool = False,
    ) -> PreTrainedModel:
        """Bake behavioral directions into ``down_proj`` biases.

        For each ``layer_idx`` in ``layer_configs``, adds
        ``k_value * direction`` to the corresponding MLP ``down_proj``'s bias
        parameter.  This is mathematically equivalent to an activation steering
        hook that adds ``K · ĉ`` to the residual stream at every forward pass.

        Args:
            layer_configs: Mapping of ``layer_idx → (direction, k_value)``.
                           Each ``direction`` is a unit-norm 1-D float32 array
                           of shape ``(hidden_size,)``.
            inplace:       If ``False`` (default), deep-copies the model before
                           modifying it, leaving the original untouched.  If
                           ``True``, modifies the original model in-place.

        Returns:
            The modified model (a deep copy when ``inplace=False``).

        Raises:
            ValueError: If any direction shape is incompatible with the
                        corresponding layer's ``down_proj.out_features``.
        """
        target: PreTrainedModel = self._model if inplace else copy.deepcopy(self._model)

        logger.info(
            "Baking %d layer(s): %s",
            len(layer_configs),
            sorted(layer_configs.keys()),
        )
        for layer_idx, (direction, k_value) in sorted(layer_configs.items()):
            down_proj = self._get_down_proj(target, layer_idx)
            self._validate_direction(direction, layer_idx, down_proj)
            self._apply_delta(down_proj, direction, k_value, layer_idx)

        return target

    def unbake(
        self,
        layer_configs: dict[int, tuple[np.ndarray, float]],
    ) -> None:
        """Reverse a prior bake by subtracting the same deltas from the original model.

        Operates **in-place** on ``self._model``.  The caller is responsible for
        passing the same ``layer_configs`` that were used during ``bake()``.  If
        any layer's bias returns to all-zeros it is removed.

        Args:
            layer_configs: Mapping of ``layer_idx → (direction, k_value)``
                           matching those originally passed to ``bake()``.

        Raises:
            ValueError: If any targeted ``down_proj`` has no bias (nothing to
                        subtract from).
        """
        logger.info(
            "Unbaking %d layer(s): %s",
            len(layer_configs),
            sorted(layer_configs.keys()),
        )
        for layer_idx, (direction, k_value) in sorted(layer_configs.items()):
            down_proj = self._get_down_proj(self._model, layer_idx)
            self._validate_direction(direction, layer_idx, down_proj)
            self._remove_delta(down_proj, direction, k_value, layer_idx)

    def save_baked_model(self, model: PreTrainedModel, path: str) -> None:
        """Persist a baked model to disk via ``save_pretrained``.

        The saved artefact can be reloaded with ``AutoModelForCausalLM.from_pretrained``
        and will have the baked biases present without any hooks.

        Args:
            model: A baked model (typically the return value of ``bake()``).
            path:  Local directory path.  Created if it does not exist.
        """
        model.save_pretrained(path)
        logger.info("Saved baked model to %s.", path)


# ---------------------------------------------------------------------------
# Convenience module-level function
# ---------------------------------------------------------------------------


def bake_directions(
    model: PreTrainedModel,
    directions: list[BehavioralDirection],
    scale: float = 1.0,
    inplace: bool = False,
) -> PreTrainedModel:
    """Bake a list of ``BehavioralDirection`` objects into a model's MLP biases.

    Convenience wrapper around ``ModelBaker.bake()`` that builds the
    ``layer_configs`` dict automatically from each direction's ``mean_direction``
    and ``k_value`` fields.

    Args:
        model:      HuggingFace causal LM to modify.
        directions: Output of ``DirectionExtractor.extract()`` or
                    ``load_directions()``.
        scale:      Global multiplier applied to every layer's ``k_value``
                    (default ``1.0`` uses the formula-derived values as-is).
        inplace:    If ``False`` (default), deep-copies the model first.

    Returns:
        The modified model (a deep copy when ``inplace=False``).

    Raises:
        ValueError: If any direction is incompatible with the target layer.
    """
    layer_configs: dict[int, tuple[np.ndarray, float]] = {
        d.layer_idx: (d.mean_direction, d.k_value * scale)
        for d in directions
    }
    baker = ModelBaker(model)
    return baker.bake(layer_configs, inplace=inplace)
