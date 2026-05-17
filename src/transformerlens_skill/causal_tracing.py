"""Causal tracing and interchange interventions for TransformerLens."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd
import seaborn as sns
import torch
from jaxtyping import Float, Int
from matplotlib.axes import Axes
from torch import Tensor
from transformer_lens import ActivationCache, HookedTransformer
from transformer_lens.hook_points import HookPoint

from transformerlens_skill.utils import (
    PromptLike,
    ensure_tokens,
    infer_clean_target_token,
    token_probability,
)


def replacement_token_id(model: HookedTransformer) -> int:
    """Choose a neutral token id for subject corruption.

    Mechanistic note: replacing subject tokens creates the corrupted baseline
    used to estimate causal mediation through internal states.

    Args:
        model: TransformerLens model.

    Returns:
        Token id used to corrupt subject positions.
    """

    tokenizer = model.tokenizer
    if tokenizer is None:
        return 0
    for candidate in (tokenizer.unk_token_id, tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id):
        if candidate is not None:
            return int(candidate)
    return 0


def compute_total_effect(
    model: HookedTransformer,
    clean: PromptLike,
    corrupted: PromptLike,
    *,
    prepend_bos: bool = True,
) -> float:
    """Compute the clean-target probability gap between clean and corrupted runs.

    Mechanistic note: the total effect is the behavioral change induced by the
    input corruption before any internal mediation is isolated.

    Args:
        model: TransformerLens model.
        clean: Clean prompt or tokens.
        corrupted: Corrupted prompt or tokens.
        prepend_bos: Whether to prepend BOS when tokenizing strings.

    Returns:
        Clean probability minus corrupted probability for the clean top token.
    """

    clean_tokens = ensure_tokens(model, clean, prepend_bos=prepend_bos)
    corrupted_tokens = ensure_tokens(model, corrupted, prepend_bos=prepend_bos)
    with torch.no_grad():
        clean_logits = model(clean_tokens)
        corrupted_logits = model(corrupted_tokens)
    target_token = infer_clean_target_token(clean_logits)
    return float(
        (
            token_probability(clean_logits, target_token)
            - token_probability(corrupted_logits, target_token)
        )
        .detach()
        .item()
    )


def interchange_intervention(
    model: HookedTransformer,
    source_cache: ActivationCache,
    target_tokens: Int[Tensor, "batch seq"],
    hook_name: str,
) -> Float[Tensor, "batch seq vocab"]:
    """Run target tokens while replacing one activation from a source cache.

    Mechanistic note: interchange interventions test whether a source internal
    state is sufficient to transfer source behavior into a target run.

    Args:
        model: TransformerLens model.
        source_cache: Cache containing the source activation.
        target_tokens: Tokens for the target/base run.
        hook_name: Hook name to replace.

    Returns:
        Logits from the intervened target run.
    """

    def patch_hook(activation: Tensor, hook: HookPoint) -> Tensor:
        return source_cache[hook_name].to(device=activation.device, dtype=activation.dtype)

    return model.run_with_hooks(target_tokens, fwd_hooks=[(hook_name, patch_hook)])


def compute_direct_effect(
    model: HookedTransformer,
    clean: PromptLike,
    corrupted: PromptLike,
    layer: int,
    pos: int,
    *,
    prepend_bos: bool = True,
) -> float:
    """Patch clean resid_pre into corrupted and measure target recovery.

    Mechanistic note: direct effect estimates how much a clean internal state
    can restore behavior in an otherwise corrupted computation.

    Args:
        model: TransformerLens model.
        clean: Clean prompt or tokens.
        corrupted: Corrupted prompt or tokens.
        layer: Layer index to patch.
        pos: Token position to patch.
        prepend_bos: Whether to prepend BOS when tokenizing strings.

    Returns:
        Patched probability minus corrupted probability for the clean top token.
    """

    clean_tokens = ensure_tokens(model, clean, prepend_bos=prepend_bos)
    corrupted_tokens = ensure_tokens(model, corrupted, prepend_bos=prepend_bos)
    hook_name = f"blocks.{layer}.hook_resid_pre"
    with torch.no_grad():
        clean_logits = model(clean_tokens)
        corrupted_logits = model(corrupted_tokens)
    target_token = infer_clean_target_token(clean_logits)
    _, clean_cache = model.run_with_cache(clean_tokens, names_filter=[hook_name])

    def patch_hook(activation: Tensor, hook: HookPoint) -> Tensor:
        patched = activation.clone()
        source = clean_cache[hook_name].to(device=activation.device, dtype=activation.dtype)
        patched[:, pos, :] = source[:, pos, :]
        return patched

    patched_logits = model.run_with_hooks(corrupted_tokens, fwd_hooks=[(hook_name, patch_hook)])
    return float(
        (
            token_probability(patched_logits, target_token)
            - token_probability(corrupted_logits, target_token)
        )
        .detach()
        .item()
    )


def compute_indirect_effect(
    model: HookedTransformer,
    clean: PromptLike,
    corrupted: PromptLike,
    layer: int,
    pos: int,
    *,
    prepend_bos: bool = True,
) -> float:
    """Patch corrupted resid_pre into clean and measure target disruption.

    Mechanistic note: indirect effect estimates how necessary a clean internal
    state is by replacing it with the corrupted counterpart.

    Args:
        model: TransformerLens model.
        clean: Clean prompt or tokens.
        corrupted: Corrupted prompt or tokens.
        layer: Layer index to patch.
        pos: Token position to patch.
        prepend_bos: Whether to prepend BOS when tokenizing strings.

    Returns:
        Clean probability minus patched probability for the clean top token.
    """

    clean_tokens = ensure_tokens(model, clean, prepend_bos=prepend_bos)
    corrupted_tokens = ensure_tokens(model, corrupted, prepend_bos=prepend_bos)
    hook_name = f"blocks.{layer}.hook_resid_pre"
    with torch.no_grad():
        clean_logits = model(clean_tokens)
    target_token = infer_clean_target_token(clean_logits)
    _, corrupted_cache = model.run_with_cache(corrupted_tokens, names_filter=[hook_name])

    def patch_hook(activation: Tensor, hook: HookPoint) -> Tensor:
        patched = activation.clone()
        source = corrupted_cache[hook_name].to(device=activation.device, dtype=activation.dtype)
        patched[:, pos, :] = source[:, pos, :]
        return patched

    patched_logits = model.run_with_hooks(clean_tokens, fwd_hooks=[(hook_name, patch_hook)])
    return float(
        (
            token_probability(clean_logits, target_token)
            - token_probability(patched_logits, target_token)
        )
        .detach()
        .item()
    )


def trace_important_states(
    model: HookedTransformer,
    prompt: str,
    subject_tokens: Sequence[int],
    *,
    prepend_bos: bool = True,
) -> dict[str, Tensor | list[int] | int]:
    """Sweep layer-position interventions after corrupting subject positions.

    Mechanistic note: this ROME-style causal trace highlights internal states
    that mediate subject-specific factual recall.

    Args:
        model: TransformerLens model.
        prompt: Clean prompt string.
        subject_tokens: Token positions corresponding to the subject span.
        prepend_bos: Whether to prepend BOS during tokenization.

    Returns:
        Dictionary containing resid_pre, attention, and mlp effect grids plus
        marked subject and final-token positions.
    """

    clean_tokens = model.to_tokens(prompt, prepend_bos=prepend_bos)
    corrupted_tokens = clean_tokens.clone()
    corrupt_id = replacement_token_id(model)
    for pos in subject_tokens:
        corrupted_tokens[:, pos] = corrupt_id

    with torch.no_grad():
        clean_logits = model(clean_tokens)
        corrupted_logits = model(corrupted_tokens)
    target_token = infer_clean_target_token(clean_logits)
    corrupted_prob = token_probability(corrupted_logits, target_token)

    hook_names = []
    for layer in range(model.cfg.n_layers):
        hook_names.extend(
            [
                f"blocks.{layer}.hook_resid_pre",
                f"blocks.{layer}.hook_attn_out",
                f"blocks.{layer}.hook_mlp_out",
            ]
        )
    _, clean_cache = model.run_with_cache(clean_tokens, names_filter=hook_names)
    seq_len = clean_tokens.shape[-1]
    resid_scores = torch.empty((model.cfg.n_layers, seq_len), device=clean_tokens.device)
    attn_scores = torch.empty((model.cfg.n_layers, seq_len), device=clean_tokens.device)
    mlp_scores = torch.empty((model.cfg.n_layers, seq_len), device=clean_tokens.device)

    for layer in range(model.cfg.n_layers):
        for pos in range(seq_len):
            for component, scores in (
                ("hook_resid_pre", resid_scores),
                ("hook_attn_out", attn_scores),
                ("hook_mlp_out", mlp_scores),
            ):
                hook_name = f"blocks.{layer}.{component}"

                def patch_hook(
                    activation: Tensor,
                    hook: HookPoint,
                    hook_name: str = hook_name,
                    pos: int = pos,
                ) -> Tensor:
                    patched = activation.clone()
                    source = clean_cache[hook_name].to(device=activation.device, dtype=activation.dtype)
                    patched[:, pos, :] = source[:, pos, :]
                    return patched

                patched_logits = model.run_with_hooks(
                    corrupted_tokens,
                    fwd_hooks=[(hook_name, patch_hook)],
                )
                scores[layer, pos] = token_probability(patched_logits, target_token) - corrupted_prob

    return {
        "resid_pre": resid_scores,
        "attention": attn_scores,
        "mlp": mlp_scores,
        "subject_positions": list(subject_tokens),
        "last_position": seq_len - 1,
    }


def decompose_mlp_vs_attention(
    results: Mapping[str, Tensor | list[int] | int],
) -> pd.DataFrame:
    """Summarize causal trace attribution by attention and MLP components.

    Mechanistic note: splitting attention from MLP effects separates routing
    mechanisms from feature-transformation mechanisms.

    Args:
        results: Results returned by trace_important_states().

    Returns:
        DataFrame with total, mean, and max effect per component type.
    """

    rows = []
    for component in ("attention", "mlp"):
        values = results[component]
        if not isinstance(values, Tensor):
            raise TypeError(f"{component} results must be a tensor")
        rows.append(
            {
                "component": component,
                "total_effect": float(values.sum().detach().cpu()),
                "mean_effect": float(values.mean().detach().cpu()),
                "max_effect": float(values.max().detach().cpu()),
            }
        )
    return pd.DataFrame(rows)


def plot_causal_trace(
    results: Mapping[str, Tensor | list[int] | int] | Float[Tensor, "layer pos"],
    *,
    component: str = "resid_pre",
    ax: Axes | None = None,
) -> Axes:
    """Plot a causal trace heatmap with subject and final positions marked.

    Mechanistic note: the marked positions make it easier to see whether causal
    states cluster around subject encoding or final-token prediction.

    Args:
        results: Result dictionary from trace_important_states() or a score tensor.
        component: Component key to plot when a result dictionary is provided.
        ax: Optional matplotlib axes to draw into.

    Returns:
        The matplotlib axes containing the heatmap.
    """

    if ax is None:
        import matplotlib.pyplot as plt

        _, axis = plt.subplots()
    else:
        axis = ax
    if isinstance(results, Tensor):
        data = results
        subject_positions: Sequence[int] = []
        last_position: int | None = data.shape[-1] - 1
    else:
        value = results[component]
        if not isinstance(value, Tensor):
            raise TypeError(f"{component} results must be a tensor")
        data = value
        raw_subjects = results.get("subject_positions", [])
        subject_positions = raw_subjects if isinstance(raw_subjects, list) else []
        raw_last = results.get("last_position", data.shape[-1] - 1)
        last_position = int(raw_last) if isinstance(raw_last, int) else data.shape[-1] - 1

    sns.heatmap(data.detach().cpu(), cmap="viridis", ax=axis)
    for pos in subject_positions:
        axis.axvline(pos + 0.5, color="white", linewidth=1.2)
    if last_position is not None:
        axis.axvline(last_position + 0.5, color="black", linewidth=1.2, linestyle="--")
    axis.set_xlabel("Token position")
    axis.set_ylabel("Layer")
    axis.set_title(f"Causal trace: {component}")
    return axis
