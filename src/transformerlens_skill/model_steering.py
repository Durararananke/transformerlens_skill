"""Activation steering and probing with TransformerLens hooks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import einops
import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from transformerlens_skill.utils import PromptLike, ensure_tokens


HookSpec = tuple[str, Any]


def cached_final_resid_pre(
    model: HookedTransformer,
    prompts: Sequence[str],
    layer: int,
    *,
    prepend_bos: bool = True,
) -> Float[Tensor, "sample d_model"]:
    """Cache final-position resid_pre activations for a prompt set.

    Mechanistic note: contrastive steering vectors are usually built from the
    difference between class-conditional residual-stream means.

    Args:
        model: TransformerLens model.
        prompts: Prompt strings to encode.
        layer: Layer whose resid_pre activation should be cached.
        prepend_bos: Whether to prepend BOS during tokenization.

    Returns:
        Tensor of final-position activations with shape [sample, d_model].
    """

    hook_name = f"blocks.{layer}.hook_resid_pre"
    activations = []
    for prompt in prompts:
        tokens = model.to_tokens(prompt, prepend_bos=prepend_bos)
        _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
        activations.append(cache[hook_name][:, -1, :])
    return torch.cat(activations, dim=0)


def extract_steering_vector(
    model: HookedTransformer,
    positive_prompts: Sequence[str],
    negative_prompts: Sequence[str],
    layer: int,
    *,
    prepend_bos: bool = True,
) -> Float[Tensor, "d_model"]:
    """Compute a contrastive mean-difference steering vector.

    Mechanistic note: the vector estimates a residual-stream direction for a
    concept by subtracting negative activations from positive activations.

    Args:
        model: TransformerLens model.
        positive_prompts: Prompts expressing the target concept.
        negative_prompts: Prompts expressing the contrast concept.
        layer: Layer whose resid_pre direction should be extracted.
        prepend_bos: Whether to prepend BOS during tokenization.

    Returns:
        Steering vector with shape [d_model].
    """

    positive = cached_final_resid_pre(model, positive_prompts, layer, prepend_bos=prepend_bos)
    negative = cached_final_resid_pre(model, negative_prompts, layer, prepend_bos=prepend_bos)
    positive_mean = einops.reduce(positive, "sample d_model -> d_model", "mean")
    negative_mean = einops.reduce(negative, "sample d_model -> d_model", "mean")
    return positive_mean - negative_mean


def apply_actadd(
    model: HookedTransformer,
    steering_vector: Float[Tensor, "d_model"],
    layer: int,
    alpha: float,
    *,
    pos: int = -1,
) -> HookSpec:
    """Create an ActAdd hook that adds a vector at hook_resid_pre.

    Mechanistic note: activation addition tests whether a residual direction is
    sufficient to steer downstream logits without changing weights.

    Args:
        model: TransformerLens model.
        steering_vector: Direction to add to the residual stream.
        layer: Layer index where the vector is added.
        alpha: Steering strength multiplier.
        pos: Position to steer, defaulting to the final token.

    Returns:
        A TransformerLens forward hook spec suitable for model.hooks().
    """

    if layer < 0 or layer >= model.cfg.n_layers:
        raise ValueError(f"layer must be in [0, {model.cfg.n_layers})")

    hook_name = f"blocks.{layer}.hook_resid_pre"

    def hook_fn(activation: Tensor, hook: HookPoint) -> Tensor:
        patched = activation.clone()
        vector = steering_vector.to(device=activation.device, dtype=activation.dtype)
        patched[:, pos, :] = patched[:, pos, :] + alpha * vector
        return patched

    return hook_name, hook_fn


def generate_with_steering(
    model: HookedTransformer,
    prompt: str,
    steering_vector: Float[Tensor, "d_model"],
    layer: int,
    alpha: float,
    max_new_tokens: int,
    *,
    prepend_bos: bool = True,
) -> str | list[str]:
    """Generate text while applying an ActAdd steering hook.

    Mechanistic note: generation-time steering applies the same residual
    direction at each autoregressive forward pass.

    Args:
        model: TransformerLens model.
        prompt: Text prompt to continue.
        steering_vector: Direction to add to resid_pre.
        layer: Layer index where the vector is added.
        alpha: Steering strength multiplier.
        max_new_tokens: Number of new tokens to generate.
        prepend_bos: Whether to prepend BOS during tokenization.

    Returns:
        Generated string or batch of strings from TransformerLens generate().
    """

    hook = apply_actadd(model, steering_vector, layer, alpha)
    with model.hooks(fwd_hooks=[hook]):
        generated = model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            prepend_bos=prepend_bos,
            use_past_kv_cache=False,
            return_type="str",
            verbose=False,
        )
    return generated


def multi_step_intervention(
    model: HookedTransformer,
    prompt: PromptLike,
    interventions: list[dict[str, Any]],
    *,
    prepend_bos: bool = True,
) -> Float[Tensor, "batch seq vocab"]:
    """Apply multiple activation interventions in one forward pass.

    Mechanistic note: multi-hook interventions test whether a distributed set
    of components jointly implements a behavior.

    Args:
        model: TransformerLens model.
        prompt: Prompt text, prompt batch, or token tensor.
        interventions: List of dictionaries with hook_name or layer/component,
            plus optional vector, value, alpha, scale, and pos fields.
        prepend_bos: Whether to prepend BOS when tokenizing strings.

    Returns:
        Logits from the intervened forward pass.
    """

    tokens = ensure_tokens(model, prompt, prepend_bos=prepend_bos)
    hooks: list[HookSpec] = []
    for intervention in interventions:
        hook_name = intervention.get("hook_name")
        if not isinstance(hook_name, str):
            layer = int(intervention["layer"])
            component = str(intervention.get("component", "hook_resid_pre"))
            hook_name = f"blocks.{layer}.{component}"
        alpha = float(intervention.get("alpha", 1.0))
        pos = int(intervention.get("pos", -1))
        vector = intervention.get("vector")
        value = intervention.get("value")
        scale = intervention.get("scale")

        def hook_fn(
            activation: Tensor,
            hook: HookPoint,
            vector: object = vector,
            value: object = value,
            scale: object = scale,
            alpha: float = alpha,
            pos: int = pos,
        ) -> Tensor:
            patched = activation.clone()
            if value is not None:
                replacement = value.to(device=activation.device, dtype=activation.dtype) if isinstance(value, Tensor) else torch.as_tensor(value, device=activation.device, dtype=activation.dtype)
                patched[:, pos, ...] = replacement
            if vector is not None:
                direction = vector.to(device=activation.device, dtype=activation.dtype) if isinstance(vector, Tensor) else torch.as_tensor(vector, device=activation.device, dtype=activation.dtype)
                patched[:, pos, ...] = patched[:, pos, ...] + alpha * direction
            if scale is not None:
                patched[:, pos, ...] = patched[:, pos, ...] * float(scale)
            return patched

        hooks.append((hook_name, hook_fn))
    return model.run_with_hooks(tokens, fwd_hooks=hooks)


def ablate_head(
    model: HookedTransformer,
    layer: int,
    head: int,
) -> HookSpec:
    """Create a zero-ablation hook for one attention head's hook_z output.

    Mechanistic note: head ablation tests whether a head is necessary by
    removing its contribution before the output projection.

    Args:
        model: TransformerLens model.
        layer: Layer index containing the head.
        head: Attention head index to ablate.

    Returns:
        A TransformerLens forward hook spec suitable for model.hooks().
    """

    if layer < 0 or layer >= model.cfg.n_layers:
        raise ValueError(f"layer must be in [0, {model.cfg.n_layers})")
    if head < 0 or head >= model.cfg.n_heads:
        raise ValueError(f"head must be in [0, {model.cfg.n_heads})")

    hook_name = f"blocks.{layer}.attn.hook_z"

    def hook_fn(activation: Tensor, hook: HookPoint) -> Tensor:
        patched = activation.clone()
        patched[:, :, head, :] = 0.0
        return patched

    return hook_name, hook_fn


def prepare_probe_data(
    activations: Float[Tensor, "... d_model"],
    labels: Int[Tensor, "..."] | Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert activations and labels into sklearn-ready arrays.

    Mechanistic note: linear probes test whether a concept is linearly decodable
    from a chosen activation space.

    Args:
        activations: Activation tensor whose final axis is d_model.
        labels: Integer labels aligned to all non-feature activation axes.

    Returns:
        Feature matrix X and label vector y as numpy arrays.
    """

    if activations.ndim == 2:
        features = activations
    elif activations.ndim == 3:
        features = einops.rearrange(activations, "batch pos d_model -> (batch pos) d_model")
    else:
        raise ValueError("activations must have shape [sample, d_model] or [batch, pos, d_model]")

    if isinstance(labels, Tensor):
        label_tensor = labels
        if label_tensor.ndim == 2:
            label_tensor = einops.rearrange(label_tensor, "batch pos -> (batch pos)")
        elif label_tensor.ndim != 1:
            raise ValueError("labels tensor must be rank 1 or rank 2")
        label_array = label_tensor.detach().cpu().numpy()
    else:
        label_array = np.asarray(labels)

    return features.detach().cpu().numpy(), label_array


def train_linear_probe(
    activations: Float[Tensor, "... d_model"],
    labels: Int[Tensor, "..."] | Sequence[int],
    *,
    max_iter: int = 1000,
) -> Any:
    """Train a sklearn logistic-regression probe on cached activations.

    Mechanistic note: a linear probe estimates whether concept information is
    explicitly represented in a linearly readable direction.

    Args:
        activations: Activation tensor with final axis d_model.
        labels: Labels aligned with samples or batch-position cells.
        max_iter: Maximum logistic-regression iterations.

    Returns:
        A fitted sklearn.linear_model.LogisticRegression instance.
    """

    from sklearn.linear_model import LogisticRegression

    features, label_array = prepare_probe_data(activations, labels)
    probe = LogisticRegression(max_iter=max_iter)
    probe.fit(features, label_array)
    return probe


def measure_probe_accuracy(
    probe: Any,
    activations: Float[Tensor, "... d_model"],
    labels: Int[Tensor, "..."] | Sequence[int],
) -> float:
    """Measure accuracy of a fitted linear probe.

    Mechanistic note: probe accuracy quantifies how reliably the activation
    space separates the labeled concept.

    Args:
        probe: Fitted sklearn-compatible classifier.
        activations: Activation tensor with final axis d_model.
        labels: Labels aligned with samples or batch-position cells.

    Returns:
        Classification accuracy as a float.
    """

    features, label_array = prepare_probe_data(activations, labels)
    return float(probe.score(features, label_array))
