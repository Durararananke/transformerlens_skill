"""Gradient-based attribution patching for TransformerLens models."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import einops
import seaborn as sns
import torch
from jaxtyping import Float, Int
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from torch import Tensor
from transformer_lens import HookedTransformer

from transformerlens_skill.utils import infer_clean_target_token, logit_diff


def attribution_hook_names(model: HookedTransformer) -> list[str]:
    """Return canonical hooks used by attribution patching.

    Mechanistic note: these hooks cover residual stream states, attention head
    outputs, and MLP outputs at every layer.

    Args:
        model: TransformerLens model.

    Returns:
        List of full hook names.
    """

    names: list[str] = []
    for layer in range(model.cfg.n_layers):
        names.extend(
            [
                f"blocks.{layer}.hook_resid_pre",
                f"blocks.{layer}.attn.hook_z",
                f"blocks.{layer}.hook_mlp_out",
            ]
        )
    return names


def run_with_gradient_cache(
    model: HookedTransformer,
    tokens: Int[Tensor, "batch seq"],
    metric_fn: Callable[[Float[Tensor, "batch seq vocab"]], Float[Tensor, ""]],
    hook_names: Sequence[str],
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Run a forward/backward pass and cache activations plus gradients.

    Mechanistic note: attribution patching replaces many explicit interventions
    with one local linear approximation around the measured activations.

    Args:
        model: TransformerLens model.
        tokens: Prompt tokens.
        metric_fn: Scalar metric to differentiate.
        hook_names: Hook names to cache in forward and backward passes.

    Returns:
        Forward activations and backward gradients keyed by hook name.
    """

    cache_dict, fwd_hooks, bwd_hooks = model.get_caching_hooks(
        names_filter=list(hook_names),
        incl_bwd=True,
        device=model.cfg.device,
    )
    model.zero_grad(set_to_none=True)
    with model.hooks(fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks, reset_hooks_end=False):
        logits = model(tokens)
        metric = metric_fn(logits)
        metric.backward()
    model.reset_hooks()
    model.zero_grad(set_to_none=True)
    acts = {name: cache_dict[name] for name in hook_names if name in cache_dict}
    grads = {
        name: cache_dict[f"{name}_grad"]
        for name in hook_names
        if f"{name}_grad" in cache_dict
    }
    return acts, grads


def compute_gradients(
    model: HookedTransformer,
    clean_tokens: Int[Tensor, "batch seq"],
    metric_fn: Callable[[Float[Tensor, "batch seq vocab"]], Float[Tensor, ""]],
) -> dict[str, Tensor]:
    """Compute gradients of a metric with respect to clean activations.

    Mechanistic note: these gradients estimate the local causal sensitivity of
    each cached component to the behavioral metric.

    Args:
        model: TransformerLens model.
        clean_tokens: Clean prompt tokens.
        metric_fn: Scalar metric computed from logits.

    Returns:
        Dictionary mapping hook names to gradient tensors.
    """

    hook_names = attribution_hook_names(model)
    _, grads = run_with_gradient_cache(model, clean_tokens, metric_fn, hook_names)
    return grads


def compute_attribution_scores(
    model: HookedTransformer,
    clean_tokens: Int[Tensor, "batch seq"],
    corrupted_tokens: Int[Tensor, "batch seq"],
) -> dict[str, Tensor]:
    """Compute one-shot attribution scores for residual, attention, and MLP hooks.

    Mechanistic note: the core approximation is grad(clean) * (act_clean -
    act_corrupted), aggregated over non-component axes.

    Args:
        model: TransformerLens model.
        clean_tokens: Clean prompt tokens.
        corrupted_tokens: Corrupted prompt tokens.

    Returns:
        Dictionary with resid_pre [layer, pos], attn_z [layer, head], and
        mlp_out [layer] attribution scores.
    """

    if clean_tokens.shape != corrupted_tokens.shape:
        raise ValueError("clean_tokens and corrupted_tokens must have identical shape")

    with torch.no_grad():
        clean_logits = model(clean_tokens)
        corrupted_logits = model(corrupted_tokens)
    correct_token = infer_clean_target_token(clean_logits)
    incorrect_token = infer_clean_target_token(corrupted_logits)

    def metric_fn(logits: Float[Tensor, "batch seq vocab"]) -> Float[Tensor, ""]:
        return logit_diff(logits, correct_token, incorrect_token)

    hook_names = attribution_hook_names(model)
    clean_acts, clean_grads = run_with_gradient_cache(model, clean_tokens, metric_fn, hook_names)
    _, corrupted_cache = model.run_with_cache(
        corrupted_tokens,
        names_filter=list(hook_names),
        return_cache_object=False,
    )

    seq_len = clean_tokens.shape[-1]
    resid_scores = torch.empty((model.cfg.n_layers, seq_len), device=clean_tokens.device)
    attn_scores = torch.empty((model.cfg.n_layers, model.cfg.n_heads), device=clean_tokens.device)
    mlp_scores = torch.empty((model.cfg.n_layers,), device=clean_tokens.device)

    for layer in range(model.cfg.n_layers):
        resid_name = f"blocks.{layer}.hook_resid_pre"
        resid_attr = clean_grads[resid_name] * (clean_acts[resid_name] - corrupted_cache[resid_name])
        resid_scores[layer] = einops.reduce(
            resid_attr,
            "batch pos d_model -> pos",
            "sum",
        )

        attn_name = f"blocks.{layer}.attn.hook_z"
        attn_attr = clean_grads[attn_name] * (clean_acts[attn_name] - corrupted_cache[attn_name])
        attn_scores[layer] = einops.reduce(
            attn_attr,
            "batch pos head d_head -> head",
            "sum",
        )

        mlp_name = f"blocks.{layer}.hook_mlp_out"
        mlp_attr = clean_grads[mlp_name] * (clean_acts[mlp_name] - corrupted_cache[mlp_name])
        mlp_scores[layer] = einops.reduce(
            mlp_attr,
            "batch pos d_model ->",
            "sum",
        )

    return {"resid_pre": resid_scores, "attn_z": attn_scores, "mlp_out": mlp_scores}


def compare_patching_vs_attribution(
    patching_scores: Float[Tensor, "layer pos"],
    attribution_scores: Mapping[str, Tensor],
    *,
    component: str = "resid_pre",
    axes: Sequence[Axes] | None = None,
) -> tuple[Figure, Sequence[Axes]]:
    """Plot exact patching scores next to attribution patching scores.

    Mechanistic note: agreement between the two plots validates the local
    linear approximation before scaling to many components.

    Args:
        patching_scores: Exact activation patching scores.
        attribution_scores: Scores returned by compute_attribution_scores().
        component: Attribution component to compare.
        axes: Optional pair of matplotlib axes.

    Returns:
        Matplotlib figure and axes containing side-by-side heatmaps.
    """

    if axes is None:
        import matplotlib.pyplot as plt

        fig, new_axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        axes = list(new_axes)
    else:
        fig = axes[0].figure

    sns.heatmap(patching_scores.detach().cpu(), cmap="RdBu_r", center=0.0, ax=axes[0])
    axes[0].set_title("Activation patching")
    axes[0].set_xlabel("Position")
    axes[0].set_ylabel("Layer")

    sns.heatmap(attribution_scores[component].detach().cpu(), cmap="RdBu_r", center=0.0, ax=axes[1])
    axes[1].set_title(f"Attribution: {component}")
    axes[1].set_xlabel("Position/head")
    axes[1].set_ylabel("Layer")
    return fig, axes


def plot_attribution_grid(
    scores: Mapping[str, Tensor],
    *,
    component: str = "resid_pre",
    ax: Axes | None = None,
) -> Axes:
    """Plot an attribution score grid for one component type.

    Mechanistic note: attribution heatmaps reveal candidate circuit components
    before spending compute on exact patching.

    Args:
        scores: Scores returned by compute_attribution_scores().
        component: Component key to plot.
        ax: Optional matplotlib axes to draw into.

    Returns:
        The matplotlib axes containing the heatmap.
    """

    if ax is None:
        import matplotlib.pyplot as plt

        _, axis = plt.subplots()
    else:
        axis = ax
    data = scores[component].detach().cpu()
    if data.ndim == 1:
        data = einops.rearrange(data, "layer -> layer 1")
    sns.heatmap(data, cmap="RdBu_r", center=0.0, ax=axis)
    axis.set_title(f"Attribution patching: {component}")
    axis.set_xlabel("Position/head")
    axis.set_ylabel("Layer")
    return axis
