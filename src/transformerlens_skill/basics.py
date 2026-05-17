"""TransformerLens workflows."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
from jaxtyping import Float, Int
from torch import Tensor
from transformer_lens import ActivationCache, HookedTransformer
from transformer_lens.hook_points import HookPoint

from transformerlens_skill.models import ModelBundle, inspect_cfg, load_model
from transformerlens_skill.utils import PromptLike, cfg_summary, ensure_tokens

# Basic examples are written as reusable functions rather than notebook snippets.

def load_and_inspect_model(
    model_name: str,
    **load_kwargs: Any,
) -> tuple[ModelBundle, dict[str, Any]]:
    """Load a model and return its hook-relevant config summary.

    Mechanistic note: config inspection tells you the valid layer/head/position
    axes before you cache or intervene on activations.

    Args:
        model_name: Short alias or HuggingFace id accepted by load_model().
        **load_kwargs: Keyword arguments forwarded to load_model().

    Returns:
        A loaded ModelBundle and a dictionary of config metadata.
    """

    bundle = load_model(model_name, **load_kwargs)
    return bundle, inspect_cfg(bundle)


def cache_all_activations(
    model: HookedTransformer,
    prompts_or_tokens: PromptLike,
    *,
    prepend_bos: bool = True,
) -> tuple[Float[Tensor, "batch seq vocab"], ActivationCache]:
    """Run the model and cache all TransformerLens hook activations.

    Mechanistic note: a full ActivationCache is the baseline object for later
    residual, attention, and MLP circuit analysis.

    Args:
        model: TransformerLens model.
        prompts_or_tokens: Prompt text, prompt batch, or token tensor.
        prepend_bos: Whether to prepend BOS when tokenizing strings.

    Returns:
        Model logits and an ActivationCache containing all hook activations.
    """

    tokens = ensure_tokens(model, prompts_or_tokens, prepend_bos=prepend_bos)
    logits, cache = model.run_with_cache(tokens)
    return logits, cache


def read_core_activations(
    cache: ActivationCache,
    layer: int,
) -> dict[str, Float[Tensor, "batch seq d_model"]]:
    """Read common residual, attention-output, and MLP-output activations.

    Mechanistic note: these four tensors expose the main residual-stream
    checkpoints before and after attention and MLP computation.

    Args:
        cache: ActivationCache returned by run_with_cache().
        layer: Transformer block index to read.

    Returns:
        Dictionary with resid_pre, resid_post, attn_out, and mlp_out tensors.
    """

    return {
        "resid_pre": cache[f"blocks.{layer}.hook_resid_pre"],
        "resid_post": cache[f"blocks.{layer}.hook_resid_post"],
        "attn_out": cache[f"blocks.{layer}.hook_attn_out"],
        "mlp_out": cache[f"blocks.{layer}.hook_mlp_out"],
    }


def cache_named_activations(
    model: HookedTransformer,
    prompts_or_tokens: PromptLike,
    hook_names: Sequence[str],
    *,
    prepend_bos: bool = True,
) -> tuple[Float[Tensor, "batch seq vocab"], dict[str, Tensor]]:
    """Cache a selected set of named activations.

    Mechanistic note: TransformerLens uses cache filters where nnsight would
    use explicit .save() calls on activation proxies.

    Args:
        model: TransformerLens model.
        prompts_or_tokens: Prompt text, prompt batch, or token tensor.
        hook_names: Canonical hook names to cache.
        prepend_bos: Whether to prepend BOS when tokenizing strings.

    Returns:
        Model logits and a dictionary keyed by requested hook name.
    """

    tokens = ensure_tokens(model, prompts_or_tokens, prepend_bos=prepend_bos)
    logits, cache_dict = model.run_with_cache(
        tokens,
        names_filter=list(hook_names),
        return_cache_object=False,
    )
    return logits, cache_dict


def run_with_inplace_activation_modification(
    model: HookedTransformer,
    prompts_or_tokens: PromptLike,
    hook_name: str,
    edit_fn: Callable[[Tensor], Tensor],
    *,
    prepend_bos: bool = True,
) -> Float[Tensor, "batch seq vocab"]:
    """Run a model after mutating one activation in an add_hook callback.

    Mechanistic note: activation editing tests whether a component is causally
    sufficient to change the model's downstream computation.

    Args:
        model: TransformerLens model.
        prompts_or_tokens: Prompt text, prompt batch, or token tensor.
        hook_name: Canonical hook name to edit.
        edit_fn: Function that returns a same-shaped edited activation tensor.
        prepend_bos: Whether to prepend BOS when tokenizing strings.

    Returns:
        Logits from the edited forward pass.
    """

    tokens = ensure_tokens(model, prompts_or_tokens, prepend_bos=prepend_bos)

    def hook_fn(activation: Tensor, hook: HookPoint) -> Tensor:
        edited = edit_fn(activation)
        activation[...] = edited
        return activation

    try:
        model.add_hook(hook_name, hook_fn)
        logits = model(tokens)
    finally:
        model.reset_hooks()
    return logits


def run_batch_with_bos(
    model: HookedTransformer,
    prompts: Sequence[str],
    *,
    prepend_bos: bool = True,
) -> tuple[
    Int[Tensor, "batch seq"],
    Float[Tensor, "batch seq vocab"],
    ActivationCache,
]:
    """Tokenize a prompt batch with explicit BOS policy and cache activations.

    Mechanistic note: BOS and padding policy determine positional alignment,
    which is critical for clean/corrupted patching comparisons.

    Args:
        model: TransformerLens model.
        prompts: Batch of prompt strings.
        prepend_bos: Whether to prepend BOS during tokenization.

    Returns:
        Tokens, logits, and ActivationCache for the batch.
    """

    tokens = model.to_tokens(list(prompts), prepend_bos=prepend_bos)
    logits, cache = model.run_with_cache(tokens)
    return tokens, logits, cache


def extract_gradients_with_cache(
    model: HookedTransformer,
    prompts_or_tokens: PromptLike,
    hook_names: Sequence[str],
    *,
    prepend_bos: bool = True,
) -> tuple[Float[Tensor, ""], dict[str, Tensor]]:
    """Extract activation gradients for a language-model loss.

    Mechanistic note: backward hooks identify which saved activations locally
    influence the next-token loss under the current prompt distribution.

    Args:
        model: TransformerLens model.
        prompts_or_tokens: Prompt text, prompt batch, or token tensor.
        hook_names: Forward hook names whose backward gradients should be cached.
        prepend_bos: Whether to prepend BOS when tokenizing strings.

    Returns:
        Scalar loss and a dictionary mapping hook names to gradient tensors.
    """

    tokens = ensure_tokens(model, prompts_or_tokens, prepend_bos=prepend_bos)
    model.zero_grad(set_to_none=True)
    loss, cache_dict = model.run_with_cache(
        tokens,
        names_filter=list(hook_names),
        return_type="loss",
        incl_bwd=True,
        return_cache_object=False,
    )
    grads = {
        name: cache_dict[f"{name}_grad"]
        for name in hook_names
        if f"{name}_grad" in cache_dict
    }
    model.zero_grad(set_to_none=True)
    return loss.detach(), grads


def summarize_loaded_model(model: HookedTransformer) -> dict[str, Any]:
    """Summarize an already-loaded HookedTransformer.

    Mechanistic note: the summary is a lightweight checklist for valid hook
    names, tensor ranks, and device placement.

    Args:
        model: TransformerLens model.

    Returns:
        Dictionary of common HookedTransformer config fields.
    """

    return cfg_summary(model)
