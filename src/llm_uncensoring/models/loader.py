"""
Model and tokenizer loading utilities.
Abstracts architecture-specific attribute access (LM head, layer list, final norm)
so the rest of the codebase works across Qwen2.5 / Qwen3 / LLaMA / Mistral.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    BitsAndBytesConfig,
)


def load_model_and_tokenizer(
    model_name: str,
    dtype: str = "bfloat16",
    device: str = "cuda",
    attn_implementation: Optional[str] = None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    torch_dtype = _resolve_dtype(dtype)

    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model_kwargs: dict = dict(
        torch_dtype=torch_dtype,
        device_map=device,
        quantization_config=quantization_config,
        trust_remote_code=True,
    )
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# ---------------------------------------------------------------------------
# Architecture-agnostic accessors
# ---------------------------------------------------------------------------

def get_transformer_layers(model: PreTrainedModel) -> nn.ModuleList:
    """Return the list of transformer decoder blocks."""
    inner = _get_inner_model(model)
    for attr in ("layers", "h", "blocks", "decoder_layers"):
        if hasattr(inner, attr):
            return getattr(inner, attr)
    raise AttributeError(
        f"Cannot locate transformer layer list in {type(model).__name__}. "
        "Extend get_transformer_layers() for this architecture."
    )


def get_final_norm(model: PreTrainedModel) -> Optional[nn.Module]:
    """Return the final LayerNorm applied before the LM head (may be None)."""
    inner = _get_inner_model(model)
    for attr in ("norm", "ln_f", "final_layernorm", "norm_f"):
        if hasattr(inner, attr):
            return getattr(inner, attr)
    return None


def get_lm_head(model: PreTrainedModel) -> nn.Module:
    for attr in ("lm_head", "embed_out", "output"):
        if hasattr(model, attr):
            return getattr(model, attr)
    raise AttributeError(f"Cannot locate LM head in {type(model).__name__}.")


def _get_inner_model(model: PreTrainedModel) -> nn.Module:
    """Unwrap the outer CausalLM wrapper to access the transformer body."""
    for attr in ("model", "transformer", "gpt_neox"):
        if hasattr(model, attr):
            return getattr(model, attr)
    return model


def _resolve_dtype(dtype: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype]
