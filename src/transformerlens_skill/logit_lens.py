"""Logit Lens utilities for TransformerLens residual streams."""

from __future__ import annotations

from collections.abc import Sequence

import seaborn as sns
import torch
from jaxtyping import Float, Int
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from torch import Tensor
from transformer_lens import HookedTransformer
from transformers import PreTrainedTokenizerBase

from transformerlens_skill.utils import ensure_tokens

# Logit-lens utilities expose intermediate predictions without mutating model state.

def get_logit_lens_predictions(
    model: HookedTransformer,
    tokens: Int[Tensor, "batch seq"],
) -> Float[Tensor, "layer seq vocab"]:
    """Decode each layer's residual stream through final norm and unembedding.

    Mechanistic note: logit lens shows how vocabulary predictions evolve as
    information moves through the residual stream.

    Args:
        model: TransformerLens model.
        tokens: Prompt tokens with batch dimension.

    Returns:
        Layer-by-position vocabulary logits with shape [n_layers, seq, vocab].
    """

    hook_names = [f"blocks.{layer}.hook_resid_post" for layer in range(model.cfg.n_layers)]
    _, cache = model.run_with_cache(tokens, names_filter=hook_names)
    layer_logits = []
    for layer in range(model.cfg.n_layers):
        hidden = cache[f"blocks.{layer}.hook_resid_post"]
        if hasattr(model, "ln_final"):
            hidden = model.ln_final(hidden)
        logits = model.unembed(hidden)
        layer_logits.append(logits[0])
    return torch.stack(layer_logits, dim=0)


def get_top_predictions_per_layer(
    predictions: Float[Tensor, "layer seq vocab"],
    tokenizer: PreTrainedTokenizerBase,
    *,
    k: int = 5,
) -> list[list[list[dict[str, int | float | str]]]]:
    """Decode top-k token predictions for every layer and position.

    Mechanistic note: decoded top tokens make intermediate representations
    inspectable without only relying on scalar metrics.

    Args:
        predictions: Logit lens logits with shape [layer, seq, vocab].
        tokenizer: Tokenizer used to decode vocabulary ids.
        k: Number of top tokens per layer-position cell.

    Returns:
        Nested list indexed by layer, position, and rank.
    """

    values, indices = torch.topk(predictions.detach(), k=k, dim=-1)
    decoded: list[list[list[dict[str, int | float | str]]]] = []
    for layer_idx in range(predictions.shape[0]):
        layer_rows: list[list[dict[str, int | float | str]]] = []
        for pos_idx in range(predictions.shape[1]):
            cell: list[dict[str, int | float | str]] = []
            for value, token_id in zip(values[layer_idx, pos_idx].cpu(), indices[layer_idx, pos_idx].cpu(), strict=True):
                int_id = int(token_id)
                cell.append(
                    {
                        "token": tokenizer.decode([int_id]),
                        "token_id": int_id,
                        "logit": float(value),
                    }
                )
            layer_rows.append(cell)
        decoded.append(layer_rows)
    return decoded


def plot_logit_lens_sequence(
    predictions: Float[Tensor, "layer seq vocab"],
    tokenizer: PreTrainedTokenizerBase,
    token_strs: Sequence[str],
    *,
    ax: Axes | None = None,
) -> Axes:
    """Plot the top decoded token at each layer and position.

    Mechanistic note: a sequence heatmap reveals when the model's intermediate
    predictions become task-relevant or collapse to the final answer.

    Args:
        predictions: Logit lens logits with shape [layer, seq, vocab].
        tokenizer: Tokenizer used to decode token ids.
        token_strs: Labels for the input token positions.
        ax: Optional matplotlib axes to draw into.

    Returns:
        The matplotlib axes containing the heatmap.
    """

    if ax is None:
        import matplotlib.pyplot as plt

        _, axis = plt.subplots()
    else:
        axis = ax
    probs = torch.softmax(predictions.detach(), dim=-1)
    top_probs, top_tokens = torch.max(probs, dim=-1)
    labels = [
        [tokenizer.decode([int(token_id)]) for token_id in layer_tokens.cpu()]
        for layer_tokens in top_tokens
    ]
    sns.heatmap(
        top_probs.cpu(),
        cmap="Blues",
        annot=labels,
        fmt="",
        xticklabels=list(token_strs),
        yticklabels=[f"L{layer}" for layer in range(predictions.shape[0])],
        ax=axis,
    )
    axis.set_xlabel("Input token")
    axis.set_ylabel("Layer")
    axis.set_title("Logit lens top predictions")
    return axis


def compare_prompts_logit_lens(
    model: HookedTransformer,
    prompts: Sequence[str],
    *,
    prepend_bos: bool = True,
    axes: Sequence[Axes] | None = None,
) -> tuple[Figure, Sequence[Axes]]:
    """Create one logit lens heatmap per prompt.

    Mechanistic note: comparing prompts shows which intermediate prediction
    changes are prompt-specific versus architecture-wide.

    Args:
        model: TransformerLens model.
        prompts: Prompt strings to compare.
        prepend_bos: Whether to prepend BOS during tokenization.
        axes: Optional axes with one entry per prompt.

    Returns:
        Matplotlib figure and axes containing faceted heatmaps.
    """

    if axes is None:
        import matplotlib.pyplot as plt

        fig, new_axes = plt.subplots(len(prompts), 1, figsize=(10, 3 * len(prompts)), constrained_layout=True)
        if len(prompts) == 1:
            axes = [new_axes]
        else:
            axes = list(new_axes)
    else:
        fig = axes[0].figure

    if model.tokenizer is None:
        raise ValueError("compare_prompts_logit_lens requires a tokenizer")

    for prompt, axis in zip(prompts, axes, strict=True):
        tokens = ensure_tokens(model, prompt, prepend_bos=prepend_bos)
        predictions = get_logit_lens_predictions(model, tokens)
        token_strs = model.to_str_tokens(tokens[0])
        plot_logit_lens_sequence(predictions, model.tokenizer, token_strs, ax=axis)
        axis.set_title(prompt)
    return fig, axes


def get_rank_of_correct_token(
    predictions: Float[Tensor, "layer seq vocab"],
    correct_token_id: int,
) -> Int[Tensor, "layer seq"]:
    """Compute the rank trajectory of a correct token across layers.

    Mechanistic note: rank trajectories show when a target answer becomes
    linearly decodable from the residual stream.

    Args:
        predictions: Logit lens logits with shape [layer, seq, vocab].
        correct_token_id: Token id whose rank should be tracked.

    Returns:
        Integer tensor of 1-indexed ranks with shape [layer, seq].
    """

    correct_logits = predictions[..., correct_token_id]
    return (predictions > correct_logits[..., None]).sum(dim=-1) + 1
