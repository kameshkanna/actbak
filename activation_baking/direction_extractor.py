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

_UNSET = object()  # sentinel for optional completion_start


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
        self._completion_start: int = 0  # tokens before completion; 0 = full sequence

    @property
    def middle_layers(self) -> list[int]:
        start = self._num_layers // 4
        end = (3 * self._num_layers) // 4
        return list(range(start, end))

    def _make_hook(self, layer_idx: int) -> Callable:
        def hook(module: nn.Module, input: tuple, output: tuple | torch.Tensor) -> None:
            hidden: torch.Tensor = output[0] if isinstance(output, tuple) else output
            start = self._completion_start
            # pool over completion tokens only when a context boundary is set
            segment = hidden[:, start:, :] if start > 0 and start < hidden.shape[1] else hidden
            vec = segment.detach().float().mean(dim=1).squeeze(0).cpu().numpy()
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

    def _collect_activations(
        self,
        prompts: list[str],
        context_lens: list[int] | None = None,
    ) -> dict[int, np.ndarray]:
        """Run prompts through model, return per-layer activation matrix (n, d).

        Args:
            prompts: Full tokenizable strings (context + completion).
            context_lens: If provided, pool activations over completion tokens only.
                          Each entry is the number of context tokens for that prompt.
        """
        for idx in self.middle_layers:
            self._activations[idx] = []

        self._register_hooks()
        try:
            for i, prompt in enumerate(tqdm(prompts, desc="  collecting", dynamic_ncols=True, leave=False)):
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

    def extract(
        self,
        pos_prompts: list[str],
        neg_prompts: list[str],
        n_pca_components: int = 5,
        contexts: list[str] | None = None,
    ) -> list[BehavioralDirection]:
        """Extract behavioral directions from contrastive prompt pairs.

        Args:
            pos_prompts: Full strings (context + positive completion).
            neg_prompts: Full strings (context + negative completion).
            n_pca_components: Number of PCA components to fit (PC1 is used).
            contexts: Raw context strings shared by both poles. When provided,
                      activations are pooled over completion tokens only, giving
                      a sharper directional estimate uncontaminated by the
                      identical context prefix.

        Returns:
            One BehavioralDirection per middle layer, sorted by layer index.
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
