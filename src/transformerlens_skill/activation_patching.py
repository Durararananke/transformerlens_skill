"""Activation patching with TransformerLens hooks."""

from __future__ import annotations

from collections.abc import Sequence

import seaborn as sns
import torch
from jaxtyping import Float, Int
from matplotlib.axes import Axes
from torch import Tensor
from transformer_lens import ActivationCache, HookedTransformer
from transformer_lens.hook_points import HookPoint

from transformerlens_skill.utils import logit_diff


def get_logit_diff(
    logits: Float[Tensor, "batch seq vocab"],
    correct_token: int,
    incorrect_token: int,
) -> Float[Tensor, ""]:
    """Return the clean-answer minus corrupted-answer logit difference.

    Mechanistic note: this scalar measures whether a patch restores the clean
    behavior at the final prediction position.

    Args:
        logits: Model logits with shape [batch, seq, vocab].
        correct_token: Token id for the clean answer.
        incorrect_token: Token id for the corrupted answer.

    Returns:
        Scalar logit difference.
    """

    return logit_diff(logits, correct_token, incorrect_token)


def patch_residual_stream(
    model: HookedTransformer,
    corrupted_tokens: Int[Tensor, "batch seq"],
    clean_cache: ActivationCache,
    layer: int,
    pos: int,
    correct_token: int,
    incorrect_token: int,
) -> float:
    """Patch resid_pre at one layer and position from clean into corrupted.

    Mechanistic note: residual-stream patching localizes where task-relevant
    information is represented in the model's running state.

    Args:
        model: TransformerLens model.
        corrupted_tokens: Corrupted prompt tokens.
        clean_cache: ActivationCache from the clean prompt.
        layer: Layer index to patch.
        pos: Token position to patch.
        correct_token: Clean-answer token id.
        incorrect_token: Corrupted-answer token id.

    Returns:
        Logit-difference score after the patch.
    """

    hook_name = f"blocks.{layer}.hook_resid_pre"

    def patch_hook(activation: Tensor, hook: HookPoint) -> Tensor:
        patched = activation.clone()
        clean_activation = clean_cache[hook_name].to(device=activation.device, dtype=activation.dtype)
        patched[:, pos, :] = clean_activation[:, pos, :]
        return patched

    logits = model.run_with_hooks(corrupted_tokens, fwd_hooks=[(hook_name, patch_hook)])
    return float(get_logit_diff(logits, correct_token, incorrect_token).detach().item())


def patch_attention_head_output(
    model: HookedTransformer,
    corrupted_tokens: Int[Tensor, "batch seq"],
    clean_cache: ActivationCache,
    layer: int,
    head: int,
    correct_token: int,
    incorrect_token: int,
) -> float:
    """Patch one attention head's hook_z output across all positions.

    Mechanistic note: head-output patching tests whether a specific attention
    head carries causal information for the behavior.

    Args:
        model: TransformerLens model.
        corrupted_tokens: Corrupted prompt tokens.
        clean_cache: ActivationCache from the clean prompt.
        layer: Layer index to patch.
        head: Attention head index to patch.
        correct_token: Clean-answer token id.
        incorrect_token: Corrupted-answer token id.

    Returns:
        Logit-difference score after the patch.
    """

    hook_name = f"blocks.{layer}.attn.hook_z"

    def patch_hook(activation: Tensor, hook: HookPoint) -> Tensor:
        patched = activation.clone()
        clean_activation = clean_cache[hook_name].to(device=activation.device, dtype=activation.dtype)
        patched[:, :, head, :] = clean_activation[:, :, head, :]
        return patched

    logits = model.run_with_hooks(corrupted_tokens, fwd_hooks=[(hook_name, patch_hook)])
    return float(get_logit_diff(logits, correct_token, incorrect_token).detach().item())


def patch_mlp_output(
    model: HookedTransformer,
    corrupted_tokens: Int[Tensor, "batch seq"],
    clean_cache: ActivationCache,
    layer: int,
    correct_token: int,
    incorrect_token: int,
) -> float:
    """Patch one layer's MLP output from clean into corrupted.

    Mechanistic note: MLP-output patching tests whether feed-forward features
    are causally responsible for the target behavior.

    Args:
        model: TransformerLens model.
        corrupted_tokens: Corrupted prompt tokens.
        clean_cache: ActivationCache from the clean prompt.
        layer: Layer index to patch.
        correct_token: Clean-answer token id.
        incorrect_token: Corrupted-answer token id.

    Returns:
        Logit-difference score after the patch.
    """

    hook_name = f"blocks.{layer}.hook_mlp_out"

    def patch_hook(activation: Tensor, hook: HookPoint) -> Tensor:
        clean_activation = clean_cache[hook_name].to(device=activation.device, dtype=activation.dtype)
        return clean_activation

    logits = model.run_with_hooks(corrupted_tokens, fwd_hooks=[(hook_name, patch_hook)])
    return float(get_logit_diff(logits, correct_token, incorrect_token).detach().item())


def run_activation_patching_grid(
    model: HookedTransformer,
    clean_tokens: Int[Tensor, "batch seq"],
    corrupted_tokens: Int[Tensor, "batch seq"],
    correct_token: int,
    incorrect_token: int,
) -> Float[Tensor, "layer pos"]:
    """Sweep residual-stream patches over all layers and positions.

    Mechanistic note: a layer-by-position grid identifies localized residual
    states whose restoration recovers the clean answer.

    Args:
        model: TransformerLens model.
        clean_tokens: Clean prompt tokens.
        corrupted_tokens: Corrupted prompt tokens with matching sequence length.
        correct_token: Clean-answer token id.
        incorrect_token: Corrupted-answer token id.

    Returns:
        Tensor of logit-difference scores with shape [n_layers, seq].
    """

    if clean_tokens.shape != corrupted_tokens.shape:
        raise ValueError("clean_tokens and corrupted_tokens must have identical shape")

    hook_names = [f"blocks.{layer}.hook_resid_pre" for layer in range(model.cfg.n_layers)]
    _, clean_cache = model.run_with_cache(clean_tokens, names_filter=hook_names)
    seq_len = clean_tokens.shape[-1]
    results = torch.empty((model.cfg.n_layers, seq_len), device=clean_tokens.device)
    for layer in range(model.cfg.n_layers):
        for pos in range(seq_len):
            results[layer, pos] = patch_residual_stream(
                model,
                corrupted_tokens,
                clean_cache,
                layer,
                pos,
                correct_token,
                incorrect_token,
            )
    return results


def plot_patching_heatmap(
    results: Float[Tensor, "layer pos"],
    *,
    token_labels: Sequence[str] | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot activation patching scores as a layer-by-position heatmap.

    Mechanistic note: visual clusters reveal where causal information is
    concentrated across depth and token position.

    Args:
        results: Patching scores with shape [n_layers, seq].
        token_labels: Optional x-axis token labels.
        ax: Optional matplotlib axes to draw into.

    Returns:
        The matplotlib axes containing the heatmap.
    """

    if ax is None:
        import matplotlib.pyplot as plt

        _, axis = plt.subplots()
    else:
        axis = ax
    x_labels: Sequence[str] | bool = token_labels if token_labels is not None else True
    sns.heatmap(
        results.detach().cpu(),
        cmap="RdBu_r",
        center=0.0,
        xticklabels=x_labels,
        yticklabels=[f"L{layer}" for layer in range(results.shape[0])],
        ax=axis,
    )
    axis.set_xlabel("Token position")
    axis.set_ylabel("Layer")
    axis.set_title("Activation patching logit-diff recovery")
    return axis
