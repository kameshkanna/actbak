"""Utilities for loading models, formatting prompts, and generating responses."""

from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from activation_baking.config import ModelConfig

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(
    hf_id: str,
    dtype: str = "bfloat16",
    load_in_4bit: bool = False,
    device_map: str = "auto",
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a causal LM and its tokenizer from HuggingFace.

    Optionally quantizes to 4-bit NF4 via bitsandbytes.  The tokenizer's pad
    token is set to eos when absent so batched generation does not require
    special handling by the caller.

    Args:
        hf_id: HuggingFace model identifier (e.g. ``"meta-llama/Llama-3.1-8B"``).
        dtype: Torch dtype string — ``"bfloat16"`` or ``"float16"``.
        load_in_4bit: Enable 4-bit NF4 quantization via bitsandbytes.
        device_map: Device placement strategy passed to ``from_pretrained``; use
            ``"auto"`` for automatic multi-GPU sharding.

    Returns:
        A ``(model, tokenizer)`` tuple, both ready for inference.

    Raises:
        AttributeError: If ``dtype`` is not a valid ``torch`` attribute.
        OSError: If the model or tokenizer cannot be found on HuggingFace.
    """
    torch_dtype: torch.dtype = getattr(torch, dtype)

    quantization_config: BitsAndBytesConfig | None = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    logger.info("Loading model: %s  (dtype=%s, 4bit=%s)", hf_id, dtype, load_in_4bit)
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        quantization_config=quantization_config,
        trust_remote_code=True,
    )
    model.eval()

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        hf_id,
        trust_remote_code=True,
        clean_up_tokenization_spaces=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(
        "Loaded %s — %d layers, hidden_size=%d",
        hf_id,
        model.config.num_hidden_layers,
        model.config.hidden_size,
    )
    return model, tokenizer


def format_prompt(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    extra_cfg: dict[str, Any] | None = None,
) -> str:
    """Apply the model's chat template to a bare user message.

    Handles per-model extras such as Qwen3's ``enable_thinking`` flag.  Falls
    back to a plain ``User: … / Assistant:`` format when the tokenizer has no
    chat template.

    Args:
        tokenizer: Loaded tokenizer for the target model.
        text: Raw user message string.
        extra_cfg: Optional per-model overrides forwarded to
            ``apply_chat_template`` (e.g. ``{"enable_thinking": False}``).

    Returns:
        Formatted prompt string ready for tokenization.
    """
    extra_cfg = extra_cfg or {}
    messages: list[dict[str, str]] = [{"role": "user", "content": text}]

    template_kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if "enable_thinking" in extra_cfg:
        template_kwargs["enable_thinking"] = extra_cfg["enable_thinking"]

    try:
        return tokenizer.apply_chat_template(messages, **template_kwargs)
    except (ValueError, AttributeError) as exc:
        logger.warning(
            "Chat template unavailable (%s); using plain fallback format.", exc
        )
        return f"User: {text}\nAssistant:"


def generate_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    max_new_tokens: int = 150,
    extra_cfg: dict[str, Any] | None = None,
    seed: int | None = None,
) -> str:
    """Generate a single text response for a formatted or unformatted prompt.

    The method formats the prompt via the chat template, tokenizes it, runs a
    greedy or sampled forward pass, and decodes only the newly generated tokens
    (the prompt prefix is stripped from the output).

    Args:
        model: Loaded causal LM in eval mode.
        tokenizer: Corresponding tokenizer (pad token must be set).
        prompt: Raw user message; chat-template formatting is applied internally.
        max_new_tokens: Maximum number of tokens to generate.
        extra_cfg: Forwarded to ``format_prompt`` (e.g. ``{"enable_thinking": False}``).
        seed: Optional integer seed for reproducible sampling.  Has no effect
            when the model uses greedy decoding.

    Returns:
        Decoded string containing only the model's generated reply (no prompt).

    Raises:
        RuntimeError: If tokenization produces an empty input sequence.
    """
    if seed is not None:
        torch.manual_seed(seed)

    formatted: str = format_prompt(tokenizer, prompt, extra_cfg)

    device: torch.device = next(model.parameters()).device
    inputs: dict[str, torch.Tensor] = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=True,
        padding=False,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    prompt_length: int = inputs["input_ids"].shape[-1]
    if prompt_length == 0:
        raise RuntimeError("Tokenization produced an empty input sequence.")

    with torch.no_grad():
        output_ids: torch.Tensor = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens: torch.Tensor = output_ids[0, prompt_length:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)
