"""Utilities for loading models and formatting prompts across architectures."""

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

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(
    hf_id: str,
    dtype: str = "bfloat16",
    load_in_4bit: bool = False,
    device_map: str = "auto",
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a causal LM and its tokenizer.

    Args:
        hf_id: HuggingFace model identifier.
        dtype: Torch dtype string — 'bfloat16' or 'float16'.
        load_in_4bit: Enable 4-bit quantization via bitsandbytes.
        device_map: Passed to from_pretrained; 'auto' handles multi-GPU.

    Returns:
        (model, tokenizer) both ready for inference.
    """
    torch_dtype = getattr(torch, dtype)

    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    logger.info("Loading model: %s", hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        quantization_config=quantization_config,
        trust_remote_code=True,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        hf_id,
        trust_remote_code=True,
        clean_up_tokenization_spaces=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loaded %s — %d layers, hidden_size=%d",
                hf_id, model.config.num_hidden_layers, model.config.hidden_size)
    return model, tokenizer


def format_prompt(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    extra_cfg: dict[str, Any] | None = None,
) -> str:
    """Apply the model's chat template to a bare user message.

    Handles Qwen3's enable_thinking flag and falls back gracefully for
    tokenizers without a chat template.

    Args:
        tokenizer: Loaded tokenizer for the target model.
        text: Raw user message string.
        extra_cfg: Optional per-model extra config (e.g. enable_thinking).

    Returns:
        Formatted prompt string ready for tokenization.
    """
    extra_cfg = extra_cfg or {}
    messages = [{"role": "user", "content": text}]

    template_kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }

    if "enable_thinking" in extra_cfg:
        template_kwargs["enable_thinking"] = extra_cfg["enable_thinking"]

    try:
        return tokenizer.apply_chat_template(messages, **template_kwargs)
    except Exception:
        return f"User: {text}\nAssistant:"
