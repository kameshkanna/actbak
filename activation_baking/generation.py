"""Data-parallel batched generation for activation steering experiments.

Replicates a model across N GPUs via independent subprocesses, distributing
``(condition_config, prompts)`` tasks through a shared task queue.  This
achieves near-N× generation throughput while keeping PyTorch forward hooks
(``ActivationSteerer``) fully functional.

vLLM and ``device_map="auto"`` tensor parallelism cannot be used here:
  - vLLM bypasses ``register_forward_hook`` entirely via custom CUDA kernels.
  - Pipeline parallelism (``device_map="auto"``) serialises token flow across
    GPUs, causing large pipeline bubbles — often *slower* than 1 GPU for 7-9B
    models that fit on a single 40 GB card.

Data parallelism avoids both issues: each worker has its own complete model
copy and its own ``ActivationSteerer``.  VRAM cost: model_size × n_gpus
(e.g. 16 GB × 8 = 128 GB on 8 × A100-40GB, well within 320 GB).

Typical usage::

    with DataParallelGenerator(n_gpus=8, model_cfg=cfg) as gen:
        results = gen.generate_all(tasks, max_new_tokens=300, batch_size=64)
    # GPU memory freed on context exit; results: {task_id: [response, ...]}
"""

from __future__ import annotations

import gc
import logging
import multiprocessing as mp
import os
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from activation_baking.config import ModelConfig
from activation_baking.model_utils import format_prompt, load_model_and_tokenizer
from activation_baking.steerer import ActivationSteerer

logger = logging.getLogger(__name__)

_SENTINEL: None = None  # poison-pill to signal worker shutdown


# ---------------------------------------------------------------------------
# Core batched generation (used by both single-GPU and worker paths)
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_batched(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    steerer: ActivationSteerer,
    layer_config: dict[int, tuple],
    prompts: list[str],
    max_new_tokens: int,
    batch_size: int,
    extra_cfg: dict[str, Any],
) -> list[str]:
    """Generate responses for all prompts in batches under an optional steering config.

    Applies steering hooks only for the duration of each ``model.generate()``
    call.  An empty ``layer_config`` produces unsteered (baseline) output.

    Args:
        model:          Loaded causal LM in eval mode.
        tokenizer:      Corresponding tokenizer; pad token must be set.
        steerer:        ``ActivationSteerer`` bound to ``model``.
        layer_config:   Layer → ``(direction, k_value[, inject_mode])`` mapping.
                        Pass an empty dict for baseline generation.
        prompts:        Raw user message strings (chat template applied internally).
        max_new_tokens: Token budget per response.
        batch_size:     Prompts per forward pass.
        extra_cfg:      Per-model kwargs forwarded to ``format_prompt``
                        (e.g. ``{"enable_thinking": False}`` for Qwen3).

    Returns:
        Decoded response strings in the same order as ``prompts``.
    """
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    device = next(model.parameters()).device
    responses: list[str] = []

    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        batch = [format_prompt(tokenizer, p, extra_cfg) for p in chunk]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=1024
        ).to(device)
        prompt_len: int = inputs["input_ids"].shape[1]

        with steerer.steer(layer_config):
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        for output_ids in out:
            responses.append(
                tokenizer.decode(output_ids[prompt_len:], skip_special_tokens=True).strip()
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return responses


# ---------------------------------------------------------------------------
# Subprocess worker
# ---------------------------------------------------------------------------


def _worker_main(
    gpu_id: int,
    model_cfg_dict: dict[str, Any],
    task_queue: "mp.Queue[Any]",
    result_queue: "mp.Queue[Any]",
    load_in_4bit: bool,
) -> None:
    """Per-GPU worker: load model once, process tasks until sentinel.

    Called via ``mp.Process`` with start method ``"spawn"``.  Sets
    ``CUDA_VISIBLE_DEVICES`` before any CUDA import so the process owns
    exactly one physical GPU visible as ``cuda:0``.

    Task format (enqueued by ``DataParallelGenerator``)::

        (task_id, layer_config, prompts, max_new_tokens, batch_size, extra_cfg)

    Result format (put to ``result_queue``)::

        (task_id, responses)

    Args:
        gpu_id:          Physical GPU index to pin this worker to.
        model_cfg_dict:  Serialised ``ModelConfig`` fields (dict for spawn safety).
        task_queue:      Shared in-queue; ``None`` sentinel triggers exit.
        result_queue:    Shared out-queue for ``(task_id, responses)`` results.
        load_in_4bit:    Whether to load in 4-bit NF4 mode.
    """
    import logging as _log

    _log.basicConfig(
        level=_log.INFO,
        format=f"%(asctime)s | GPU{gpu_id} | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    _logger = _log.getLogger(__name__)

    # Pin this process to one GPU before any CUDA initialisation
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from activation_baking.config import ModelConfig as _MC  # noqa: PLC0415
    from activation_baking.model_utils import load_model_and_tokenizer as _load  # noqa: PLC0415
    from activation_baking.steerer import ActivationSteerer as _Steerer  # noqa: PLC0415

    model_cfg = _MC(**model_cfg_dict)
    _logger.info("GPU %d | loading %s", gpu_id, model_cfg.name)

    model, tokenizer = _load(
        hf_id=model_cfg.hf_id,
        dtype=model_cfg.dtype,
        load_in_4bit=load_in_4bit,
        device_map="cuda:0",
    )
    steerer = _Steerer(model)
    _logger.info("GPU %d | ready", gpu_id)

    while True:
        task = task_queue.get()
        if task is _SENTINEL:
            break

        task_id: str
        layer_config: dict[int, tuple]
        prompts: list[str]
        max_new_tokens: int
        batch_size: int
        extra_cfg: dict[str, Any]
        task_id, layer_config, prompts, max_new_tokens, batch_size, extra_cfg = task

        _logger.info("GPU %d | task=%s  n_prompts=%d", gpu_id, task_id, len(prompts))
        responses = generate_batched(
            model, tokenizer, steerer, layer_config,
            prompts, max_new_tokens, batch_size, extra_cfg,
        )
        result_queue.put((task_id, responses))
        _logger.info("GPU %d | task=%s done", gpu_id, task_id)

    del model, tokenizer, steerer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _logger.info("GPU %d | exiting", gpu_id)


# ---------------------------------------------------------------------------
# Data-parallel generator
# ---------------------------------------------------------------------------


class DataParallelGenerator:
    """Replicate a model across N GPUs for parallel activation-steered generation.

    Each GPU runs an independent model copy in its own subprocess.  Work items
    are distributed across GPUs via a shared task queue — each task is a
    ``(task_id, layer_config, prompts)`` triple.  Workers pull tasks
    dynamically so load is balanced regardless of prompt-count variation.

    Forward hooks are fully supported: every worker has its own ``model`` and
    ``ActivationSteerer`` instance; no weight sharing or IPC-level hook
    coordination is required.

    VRAM requirement: ``model_size × n_gpus``.  A 7–9 B model in bfloat16
    (~16 GB) across 8 × 40 GB A100s uses 128 GB of 320 GB available — safe.

    Args:
        n_gpus:       Number of GPU workers to spawn.
        model_cfg:    Model configuration replicated on every worker.
        load_in_4bit: Use 4-bit NF4 quantisation (halves VRAM, ~10–15% slower).
        gpu_ids:      Explicit physical GPU indices to use.  Defaults to
                      ``list(range(n_gpus))``.  Pass the indices from the
                      parent's ``CUDA_VISIBLE_DEVICES`` env when applicable.

    Raises:
        ValueError: If ``n_gpus < 1``.
    """

    def __init__(
        self,
        n_gpus: int,
        model_cfg: ModelConfig,
        load_in_4bit: bool = False,
        gpu_ids: list[int] | None = None,
    ) -> None:
        if n_gpus < 1:
            raise ValueError(f"n_gpus must be >= 1, got {n_gpus}")

        self.n_gpus = n_gpus
        self.model_cfg = model_cfg
        self._gpu_ids: list[int] = (gpu_ids or list(range(n_gpus)))[:n_gpus]

        ctx = mp.get_context("spawn")
        self._task_q: mp.Queue = ctx.Queue()
        self._result_q: mp.Queue = ctx.Queue()

        # Serialise ModelConfig to a plain dict (spawn-safe — no torch state)
        model_cfg_dict: dict[str, Any] = {
            "name":             model_cfg.name,
            "hf_id":            model_cfg.hf_id,
            "norm_type":        model_cfg.norm_type,
            "hidden_size":      model_cfg.hidden_size,
            "num_layers":       model_cfg.num_layers,
            "dtype":            model_cfg.dtype,
            "is_instruct":      model_cfg.is_instruct,
            "base_counterpart": model_cfg.base_counterpart,
            "extra":            model_cfg.extra,
        }

        self._workers: list[mp.Process] = []
        for gid in self._gpu_ids:
            p = ctx.Process(
                target=_worker_main,
                args=(gid, model_cfg_dict, self._task_q, self._result_q, load_in_4bit),
                name=f"gen-gpu{gid}",
                daemon=True,
            )
            p.start()
            self._workers.append(p)

        logger.info(
            "DataParallelGenerator: %d workers started for %s  GPUs=%s",
            n_gpus, model_cfg.name, self._gpu_ids,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_all(
        self,
        tasks: list[tuple[str, dict[int, tuple], list[str]]],
        max_new_tokens: int,
        batch_size: int,
        extra_cfg: dict[str, Any],
    ) -> dict[str, list[str]]:
        """Distribute tasks across all GPU workers and collect responses.

        All tasks are enqueued at once; workers pull them dynamically for
        natural load balancing.  The call blocks until every task is complete.

        Args:
            tasks:          ``[(task_id, layer_config, prompts), ...]``.
                            ``task_id`` must be unique across the list — it
                            keys the returned dict.  An empty ``layer_config``
                            produces baseline (unsteered) output.
            max_new_tokens: Token budget for generation.
            batch_size:     Prompts per forward pass inside each task.
            extra_cfg:      Per-model kwargs forwarded to ``format_prompt``.

        Returns:
            ``{task_id: [response, ...]}`` — responses in the same order as
            each task's ``prompts`` list.

        Raises:
            RuntimeError: If a worker process exits with a non-zero code while
                tasks are still outstanding.
        """
        if not tasks:
            return {}

        n_tasks = len(tasks)
        for task_id, layer_config, prompts in tasks:
            self._task_q.put(
                (task_id, layer_config, prompts, max_new_tokens, batch_size, extra_cfg)
            )

        results: dict[str, list[str]] = {}
        with tqdm(
            total=n_tasks,
            desc=f"  [{self.model_cfg.name}] gen",
            unit="task",
            dynamic_ncols=True,
            leave=False,
        ) as pbar:
            while len(results) < n_tasks:
                dead = [p for p in self._workers if not p.is_alive() and p.exitcode != 0]
                if dead:
                    raise RuntimeError(
                        f"Worker(s) crashed: {[p.name for p in dead]}  "
                        f"exit codes: {[p.exitcode for p in dead]}"
                    )
                try:
                    task_id, responses = self._result_q.get(timeout=5.0)
                    results[task_id] = responses
                    pbar.update(1)
                    pbar.set_postfix(done=len(results), total=n_tasks)
                except Exception:  # queue.Empty on timeout — retry  # noqa: BLE001
                    pass

        return results

    def shutdown(self) -> None:
        """Send sentinels and wait for all workers to exit cleanly."""
        for _ in self._workers:
            self._task_q.put(_SENTINEL)
        for p in tqdm(
            self._workers,
            desc="  Shutting down GPU workers",
            leave=False,
            dynamic_ncols=True,
        ):
            p.join(timeout=60)
            if p.is_alive():
                logger.warning("Worker %s did not exit within 60 s; terminating.", p.name)
                p.terminate()
                p.join()
        logger.info("DataParallelGenerator: all workers stopped.")

    def __enter__(self) -> "DataParallelGenerator":
        return self

    def __exit__(self, *_: object) -> None:
        self.shutdown()
