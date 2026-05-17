"""Model loading helpers for supported TransformerLens families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformer_lens import HookedTransformer
from transformer_lens.config import HookedTransformerConfig
from transformers import PreTrainedTokenizerBase

# The registry gives stable family aliases while still allowing full HF ids.

@dataclass(frozen=True)
class ModelBundle:
    """Loaded model, tokenizer, and TransformerLens config.

    Mechanistic note: keeping the config beside the model makes hook shapes and
    architectural assumptions explicit before running interventions.

    Args:
        model: A loaded TransformerLens HookedTransformer.
        tokenizer: The tokenizer attached to the model, if one is available.
        cfg: The HookedTransformerConfig used by the model.

    Returns:
        A typed container for shared model state.
    """

    model: HookedTransformer
    tokenizer: PreTrainedTokenizerBase | None
    cfg: HookedTransformerConfig


@dataclass(frozen=True)
class FamilyLoadConfig:
    """Static loading preferences for a model family.

    Mechanistic note: these flags choose the transformed weight basis used for
    all downstream cache and hook analysis.

    Args:
        hf_id: Full HuggingFace model id.
        family: Human-readable family label.
        fold_ln: Whether to fold LayerNorm/RMSNorm weights where supported.
        center_unembed: Whether to center the unembedding matrix.
        refactor_factored_attn_matrices: Whether to refactor QK/OV factored matrices.
        default_dtype: Default load dtype for large instruction models.

    Returns:
        A frozen family loading configuration.
    """

    hf_id: str
    family: str
    fold_ln: bool = True
    center_unembed: bool = True
    refactor_factored_attn_matrices: bool = False
    default_dtype: str = "bfloat16"


MODEL_REGISTRY: dict[str, str] = {
    "llama3": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen3": "Qwen/Qwen3-4B-Instruct-2507",
    "gemma3": "google/gemma-3-4b-it",
}

MODEL_FAMILY_CONFIGS: dict[str, FamilyLoadConfig] = {
    "llama3": FamilyLoadConfig(
        hf_id=MODEL_REGISTRY["llama3"],
        family="Llama 3",
    ),
    "qwen3": FamilyLoadConfig(
        hf_id=MODEL_REGISTRY["qwen3"],
        family="Qwen 3",
    ),
    "gemma3": FamilyLoadConfig(
        hf_id=MODEL_REGISTRY["gemma3"],
        family="Gemma 3",
    ),
}


def resolve_model_name(model_name: str) -> str:
    """Resolve a short model alias to a HuggingFace id.

    Mechanistic note: a stable alias prevents experiments from silently moving
    between model families with different hook shapes.

    Args:
        model_name: Short alias from MODEL_REGISTRY or a full HuggingFace id.

    Returns:
        The HuggingFace id to pass to HookedTransformer.from_pretrained().
    """

    return MODEL_REGISTRY.get(model_name, model_name)


def family_config_for(model_name: str) -> FamilyLoadConfig | None:
    """Return the family loading config for a short or full model name.

    Mechanistic note: family metadata documents architecture-specific hook
    expectations such as RMSNorm, RoPE, GQA, and Q/K normalization.

    Args:
        model_name: Short alias or full HuggingFace id.

    Returns:
        The matching FamilyLoadConfig, or None for an arbitrary external model.
    """

    if model_name in MODEL_FAMILY_CONFIGS:
        return MODEL_FAMILY_CONFIGS[model_name]

    resolved = resolve_model_name(model_name)
    for config in MODEL_FAMILY_CONFIGS.values():
        if config.hf_id == resolved:
            return config
    return None


def load_model(
    model_name: str,
    *,
    device: str | torch.device | None = None,
    dtype: str | torch.dtype | None = None,
    fold_ln: bool | None = None,
    center_unembed: bool | None = None,
    refactor_factored_attn_matrices: bool | None = None,
    **from_pretrained_kwargs: Any,
) -> ModelBundle:
    """Load a supported model through HookedTransformer.from_pretrained().

    Mechanistic note: all modules use this single entry point so activation
    names and config fields are consistent across Llama 3, Qwen 3, and Gemma 3.

    Args:
        model_name: Short alias ("llama3", "qwen3", "gemma3") or HF model id.
        device: Optional torch device override.
        dtype: Optional dtype override. Defaults to the family preference.
        fold_ln: Optional override for TransformerLens fold_ln.
        center_unembed: Optional override for TransformerLens center_unembed.
        refactor_factored_attn_matrices: Optional override for attention refactoring.
        **from_pretrained_kwargs: Extra arguments forwarded to from_pretrained().

    Returns:
        A ModelBundle containing the loaded model, tokenizer, and cfg.
    """

    resolved_name = resolve_model_name(model_name)
    family_config = family_config_for(model_name)
    effective_dtype = dtype or (family_config.default_dtype if family_config else "bfloat16")

    model = HookedTransformer.from_pretrained(
        resolved_name,
        device=device,
        dtype=effective_dtype,
        fold_ln=fold_ln if fold_ln is not None else (family_config.fold_ln if family_config else True),
        center_unembed=(
            center_unembed
            if center_unembed is not None
            else (family_config.center_unembed if family_config else True)
        ),
        refactor_factored_attn_matrices=(
            refactor_factored_attn_matrices
            if refactor_factored_attn_matrices is not None
            else (
                family_config.refactor_factored_attn_matrices
                if family_config
                else False
            )
        ),
        **from_pretrained_kwargs,
    )
    return ModelBundle(model=model, tokenizer=model.tokenizer, cfg=model.cfg)


def inspect_cfg(bundle: ModelBundle) -> dict[str, Any]:
    """Summarize the TransformerLens config fields most relevant to hooks.

    Mechanistic note: these fields predict the tensor shapes seen at residual,
    attention-head, and MLP hook points.

    Args:
        bundle: Loaded model bundle.

    Returns:
        A dictionary of architecture and hook-shape metadata.
    """

    cfg = bundle.cfg
    return {
        "model_name": cfg.model_name,
        "original_architecture": cfg.original_architecture,
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "n_key_value_heads": cfg.n_key_value_heads,
        "d_model": cfg.d_model,
        "d_head": cfg.d_head,
        "d_mlp": cfg.d_mlp,
        "d_vocab": cfg.d_vocab,
        "n_ctx": cfg.n_ctx,
        "normalization_type": cfg.normalization_type,
        "positional_embedding_type": cfg.positional_embedding_type,
        "gated_mlp": cfg.gated_mlp,
        "use_qk_norm": cfg.use_qk_norm,
        "use_local_attn": cfg.use_local_attn,
        "attn_types": cfg.attn_types,
        "device": str(cfg.device),
    }
