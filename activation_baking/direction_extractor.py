"""Behavioral direction extraction via contrastive activation differences."""

from __future__ import annotations

import gc
import logging
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from activation_baking.config import ModelConfig

logger = logging.getLogger(__name__)

_UNSET = object()  # sentinel for optional completion_start


@dataclass
class BehavioralDirection:
    """Extracted behavioral direction for a single transformer layer.

    Attributes:
        layer_idx:           Index of the transformer layer this direction was
                             extracted from.
        k_value:             Calibrated injection magnitude for this layer, equal
                             to the layer's formula-derived K (μ̄_ℓ / √d) at
                             extraction time.
        pca_direction:       Leading principal component of the contrastive
                             activation differences; unit-norm, shape
                             ``(hidden_size,)``.
        mean_direction:      Normalised mean of the contrastive activation
                             differences; unit-norm, shape ``(hidden_size,)``.
        pca_variance_ratio:  Fraction of total variance in the diff matrix
                             explained by the leading PC.
    """

    layer_idx: int
    k_value: float
    pca_direction: np.ndarray
    mean_direction: np.ndarray
    pca_variance_ratio: float


class DirectionExtractor:
    """Extracts per-layer behavioral directions from contrastive prompt pairs.

    For each middle layer ℓ (indices spanning the middle 50% of the network),
    stacks activation differences ΔH_ℓ = H_ℓ(positive) − H_ℓ(negative) into a
    matrix and extracts two complementary direction estimates:

    - **PCA direction**: the leading principal component of ΔH_ℓ, capturing the
      axis of maximum variance across the contrastive pairs.
    - **Mean direction**: mean(ΔH_ℓ) / ‖mean(ΔH_ℓ)‖, capturing the average
      shift in representation.

    Both directions are unit-normalised and ready for steering (see
    ``steerer.ActivationSteerer``) or persistent weight baking (see
    ``baker.ModelBaker``).

    Attributes are intentionally private; interact via ``extract()``.

    Example::

        extractor = DirectionExtractor(model, tokenizer, model_cfg, k_values)
        directions = extractor.extract(pos_prompts, neg_prompts, contexts=ctxs)
        save_directions(directions, "safety_directions.npz")
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        model_cfg: ModelConfig,
        k_values: dict[int, float],
        max_length: int = 512,
    ) -> None:
        """
        Args:
            model:       HuggingFace causal LM with a ``model.model.layers``
                         attribute.
            tokenizer:   Corresponding tokenizer; pad token must be set.
            model_cfg:   ``ModelConfig`` for the loaded model.  ``hidden_size``
                         and ``num_layers`` are read from this object.
            k_values:    Mapping of ``layer_idx → K`` injection magnitude.
                         Layers not present default to 0.0.
            max_length:  Maximum tokenized sequence length; longer inputs are
                         truncated.
        """
        self._model = model
        self._tokenizer = tokenizer
        self._hidden_size: int = model_cfg.hidden_size
        self._num_layers: int = model_cfg.num_layers
        self._k_values = k_values
        self._max_length = max_length
        self._device: torch.device = next(model.parameters()).device
        self._hooks: list = []
        self._activations: dict[int, list[np.ndarray]] = {}
        self._completion_start: int = 0

    # ------------------------------------------------------------------
    # Layer selection
    # ------------------------------------------------------------------

    @property
    def middle_layers(self) -> list[int]:
        """Layer indices spanning the middle 50% of the network.

        Returns:
            Sorted list of integer layer indices from ``num_layers // 4`` to
            ``(3 * num_layers) // 4 − 1`` inclusive.
        """
        start = self._num_layers // 4
        end = (3 * self._num_layers) // 4
        return list(range(start, end))

    # ------------------------------------------------------------------
    # Hook construction
    # ------------------------------------------------------------------

    def _make_hook(self, layer_idx: int) -> Callable:
        """Return a forward hook that captures mean-pooled hidden states.

        When ``_completion_start > 0``, pooling is restricted to the token
        positions following that index, isolating the completion representation
        from the shared context prefix.

        Args:
            layer_idx: Index of the transformer layer being hooked.

        Returns:
            A callable compatible with ``register_forward_hook``.
        """
        def hook(module: nn.Module, input: tuple, output: tuple | torch.Tensor) -> None:
            hidden: torch.Tensor = output[0] if isinstance(output, tuple) else output
            start = self._completion_start
            segment = hidden[:, start:, :] if start > 0 and start < hidden.shape[1] else hidden
            vec = segment.detach().float().mean(dim=1).squeeze(0).cpu().numpy()
            self._activations[layer_idx].append(vec)
        return hook

    # ------------------------------------------------------------------
    # Hook lifecycle
    # ------------------------------------------------------------------

    def _register_hooks(self) -> None:
        """Register forward hooks on all middle layers."""
        for idx in self.middle_layers:
            self._activations[idx] = []
            handle = self._model.model.layers[idx].register_forward_hook(
                self._make_hook(idx)
            )
            self._hooks.append(handle)

    def _remove_hooks(self) -> None:
        """Remove all currently registered forward hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    # Activation collection
    # ------------------------------------------------------------------

    def _collect_activations(
        self,
        prompts: list[str],
        context_lens: list[int] | None = None,
    ) -> dict[int, np.ndarray]:
        """Run prompts through the model and return per-layer activation matrices.

        Each matrix has shape ``(n_prompts, hidden_size)``, containing the
        mean-pooled hidden state at that layer for each prompt.

        Args:
            prompts:      Full tokenizable strings (context + completion).
            context_lens: If provided, pool activations over completion tokens
                          only.  Each entry is the number of context tokens for
                          the corresponding prompt.

        Returns:
            Mapping of ``layer_idx → np.ndarray`` of shape ``(n, hidden_size)``.
        """
        for idx in self.middle_layers:
            self._activations[idx] = []

        self._register_hooks()
        try:
            for i, prompt in enumerate(
                tqdm(prompts, desc="  collecting", dynamic_ncols=True, leave=False)
            ):
                self._completion_start = context_lens[i] if context_lens is not None else 0
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
            self._completion_start = 0

        return {idx: np.stack(self._activations[idx]) for idx in self.middle_layers}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        pos_prompts: list[str],
        neg_prompts: list[str],
        n_pca_components: int = 5,
        contexts: list[str] | None = None,
    ) -> list[BehavioralDirection]:
        """Extract behavioral directions from contrastive prompt pairs.

        Passes both prompt sets through the model, computes per-layer activation
        differences ΔH_ℓ = H_ℓ(pos) − H_ℓ(neg), and fits PCA to obtain both a
        PCA direction (leading PC) and a mean direction (normalised mean diff).

        Args:
            pos_prompts:      Full strings (context + positive completion), one
                              per contrastive pair.
            neg_prompts:      Full strings (context + negative completion);
                              must be the same length as ``pos_prompts``.
            n_pca_components: Number of PCA components to fit.  Only PC1 is
                              retained; higher values improve numerical stability.
            contexts:         Raw context strings shared by both poles.  When
                              provided, activations are pooled over completion
                              tokens only, yielding a sharper directional estimate
                              uncontaminated by the identical context prefix.

        Returns:
            List of ``BehavioralDirection`` objects, one per middle layer,
            sorted by ascending layer index.

        Raises:
            AssertionError: If ``pos_prompts`` and ``neg_prompts`` lengths differ,
                            or if ``contexts`` length does not match the pair count.
        """
        assert len(pos_prompts) == len(neg_prompts), \
            "Positive and negative prompt counts must match."

        context_lens: list[int] | None = None
        if contexts is not None:
            assert len(contexts) == len(pos_prompts), \
                "contexts length must match number of pairs."
            context_lens = [
                len(self._tokenizer(ctx, add_special_tokens=True)["input_ids"])
                for ctx in contexts
            ]
            logger.info(
                "Completion-only pooling enabled. Mean context len: %.1f tokens",
                sum(context_lens) / len(context_lens),
            )

        logger.info("Collecting positive activations (%d prompts)...", len(pos_prompts))
        pos_acts = self._collect_activations(pos_prompts, context_lens)

        logger.info("Collecting negative activations (%d prompts)...", len(neg_prompts))
        neg_acts = self._collect_activations(neg_prompts, context_lens)

        directions: list[BehavioralDirection] = []
        for layer_idx in self.middle_layers:
            diff = pos_acts[layer_idx] - neg_acts[layer_idx]  # (n, d)

            n_comp = min(n_pca_components, diff.shape[0] - 1)
            pca = PCA(n_components=n_comp)
            pca.fit(diff)
            pca_dir = pca.components_[0]
            pca_dir = pca_dir / (np.linalg.norm(pca_dir) + 1e-8)
            var_ratio = float(pca.explained_variance_ratio_[0])

            mean_diff = diff.mean(axis=0)
            mean_dir = mean_diff / (np.linalg.norm(mean_diff) + 1e-8)

            directions.append(BehavioralDirection(
                layer_idx=layer_idx,
                k_value=self._k_values.get(layer_idx, 0.0),
                pca_direction=pca_dir,
                mean_direction=mean_dir,
                pca_variance_ratio=var_ratio,
            ))

        gc.collect()
        logger.info(
            "Extracted directions for %d layers (layers %d–%d).",
            len(directions),
            directions[0].layer_idx,
            directions[-1].layer_idx,
        )
        return directions


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def save_directions(
    directions: list[BehavioralDirection],
    path: str,
) -> None:
    """Persist a list of behavioral directions to a compressed ``.npz`` archive.

    The archive stores five parallel arrays keyed by field name.  Reload with
    ``load_directions``.

    Args:
        directions: Output of ``DirectionExtractor.extract()``.
        path:       Destination file path; ``.npz`` extension appended by NumPy
                    if absent.
    """
    np.savez(
        path,
        layer_indices=np.array([d.layer_idx for d in directions]),
        k_values=np.array([d.k_value for d in directions]),
        pca_directions=np.stack([d.pca_direction for d in directions]),
        mean_directions=np.stack([d.mean_direction for d in directions]),
        pca_variance_ratios=np.array([d.pca_variance_ratio for d in directions]),
    )
    logger.info("Saved %d directions to %s.", len(directions), path)


def load_directions(path: str) -> list[BehavioralDirection]:
    """Load behavioral directions from a ``.npz`` archive.

    Args:
        path: Path to the archive produced by ``save_directions``.

    Returns:
        List of ``BehavioralDirection`` objects in the order they were saved.

    Raises:
        OSError:   If the file does not exist or cannot be opened.
        KeyError:  If the archive is missing expected array keys.
    """
    data = np.load(path)
    directions = [
        BehavioralDirection(
            layer_idx=int(data["layer_indices"][i]),
            k_value=float(data["k_values"][i]),
            pca_direction=data["pca_directions"][i],
            mean_direction=data["mean_directions"][i],
            pca_variance_ratio=float(data["pca_variance_ratios"][i]),
        )
        for i in range(len(data["layer_indices"]))
    ]
    logger.info("Loaded %d directions from %s.", len(directions), path)
    return directions
