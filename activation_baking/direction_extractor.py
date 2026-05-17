"""Behavioral direction extraction via contrastive activation differences."""

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

logger = logging.getLogger(__name__)


@dataclass
class BehavioralDirection:
    """Extracted behavioral direction for a single layer."""

    layer_idx: int
    k_value: float
    pca_direction: np.ndarray    # leading PC of contrastive diffs, shape (hidden_size,)
    mean_direction: np.ndarray   # normalised mean diff, shape (hidden_size,)
    pca_variance_ratio: float    # variance explained by PC1


class DirectionExtractor:
    """Extracts per-layer behavioral directions from contrastive prompt pairs.

    For each middle layer l, stacks activation differences
    ΔH_l = H_l(positive) - H_l(negative) into a matrix and extracts:
      - PCA direction: leading principal component of ΔH_l
      - Mean direction: mean(ΔH_l) / ||mean(ΔH_l)||

    Both directions are unit-normalised and ready for steering or baking.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        hidden_size: int,
        num_layers: int,
        k_values: dict[int, float],
        max_length: int = 512,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._k_values = k_values
        self._max_length = max_length
        self._device = next(model.parameters()).device
        self._hooks: list = []
        self._activations: dict[int, list[np.ndarray]] = {}

    @property
    def middle_layers(self) -> list[int]:
        start = self._num_layers // 4
        end = (3 * self._num_layers) // 4
        return list(range(start, end))

    def _make_hook(self, layer_idx: int) -> Callable:
        def hook(module: nn.Module, input: tuple, output: tuple | torch.Tensor) -> None:
            hidden: torch.Tensor = output[0] if isinstance(output, tuple) else output
            # mean-pool over sequence → (hidden_size,)
            vec = hidden.detach().float().mean(dim=1).squeeze(0).cpu().numpy()
            self._activations[layer_idx].append(vec)
        return hook

    def _register_hooks(self) -> None:
        for idx in self.middle_layers:
            self._activations[idx] = []
            handle = self._model.model.layers[idx].register_forward_hook(
                self._make_hook(idx)
            )
            self._hooks.append(handle)

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _collect_activations(self, prompts: list[str]) -> dict[int, np.ndarray]:
        """Run prompts through model, return per-layer activation matrix (n, d)."""
        for idx in self.middle_layers:
            self._activations[idx] = []

        self._register_hooks()
        try:
            for prompt in tqdm(prompts, desc="  collecting", dynamic_ncols=True, leave=False):
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

        return {idx: np.stack(self._activations[idx]) for idx in self.middle_layers}

    def extract(
        self,
        pos_prompts: list[str],
        neg_prompts: list[str],
        n_pca_components: int = 5,
    ) -> list[BehavioralDirection]:
        """Extract behavioral directions from contrastive prompt pairs.

        Args:
            pos_prompts: Prompts strongly eliciting the target behavior.
            neg_prompts: Prompts strongly suppressing the target behavior.
            n_pca_components: Number of PCA components to fit (PC1 is used).

        Returns:
            One BehavioralDirection per middle layer, sorted by layer index.
        """
        assert len(pos_prompts) == len(neg_prompts), \
            "Positive and negative prompt counts must match."

        logger.info("Collecting positive activations (%d prompts)...", len(pos_prompts))
        pos_acts = self._collect_activations(pos_prompts)

        logger.info("Collecting negative activations (%d prompts)...", len(neg_prompts))
        neg_acts = self._collect_activations(neg_prompts)

        directions: list[BehavioralDirection] = []
        for layer_idx in self.middle_layers:
            diff = pos_acts[layer_idx] - neg_acts[layer_idx]  # (n, d)

            # PCA direction
            n_comp = min(n_pca_components, diff.shape[0] - 1)
            pca = PCA(n_components=n_comp)
            pca.fit(diff)
            pca_dir = pca.components_[0]
            pca_dir = pca_dir / (np.linalg.norm(pca_dir) + 1e-8)
            var_ratio = float(pca.explained_variance_ratio_[0])

            # Mean diff direction
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
        return directions


def save_directions(
    directions: list[BehavioralDirection],
    path: str,
) -> None:
    """Persist directions to a .npz archive."""
    np.savez(
        path,
        layer_indices=np.array([d.layer_idx for d in directions]),
        k_values=np.array([d.k_value for d in directions]),
        pca_directions=np.stack([d.pca_direction for d in directions]),
        mean_directions=np.stack([d.mean_direction for d in directions]),
        pca_variance_ratios=np.array([d.pca_variance_ratio for d in directions]),
    )


def load_directions(path: str) -> list[BehavioralDirection]:
    """Load directions from a .npz archive."""
    data = np.load(path)
    return [
        BehavioralDirection(
            layer_idx=int(data["layer_indices"][i]),
            k_value=float(data["k_values"][i]),
            pca_direction=data["pca_directions"][i],
            mean_direction=data["mean_directions"][i],
            pca_variance_ratio=float(data["pca_variance_ratios"][i]),
        )
        for i in range(len(data["layer_indices"]))
    ]
