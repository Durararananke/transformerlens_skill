"""Shared helpers for TransformerLens skill modules."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from jaxtyping import Float, Int
from torch import Tensor
from transformer_lens import HookedTransformer
from transformers import PreTrainedTokenizerBase


PromptLike = str | Sequence[str] | Int[Tensor, "batch seq"]


def get_model_device(model: HookedTransformer) -> torch.device:
    """Return the torch device used by a HookedTransformer.

    Mechanistic note: keeping interventions on the model device avoids hidden
    host-device transfers during hook execution.

    Args:
        model: TransformerLens model.

    Returns:
        The model device as a torch.device.
    """

    return torch.device(model.cfg.device)


def ensure_tokens(
    model: HookedTransformer,
    prompts_or_tokens: PromptLike,
    *,
    prepend_bos: bool = True,
) -> Int[Tensor, "batch seq"]:
    """Convert prompts to tokens or move existing tokens to the model device.

    Mechanistic note: patching compares positions, so tokenization policy must
    be explicit and stable across clean and corrupted prompts.

    Args:
        model: TransformerLens model.
        prompts_or_tokens: Prompt string, prompt batch, or token tensor.
        prepend_bos: Whether to prepend the BOS token when tokenizing strings.

    Returns:
        Token ids with shape [batch, seq] on the model device.
    """

    if isinstance(prompts_or_tokens, Tensor):
        return prompts_or_tokens.to(get_model_device(model))
    return model.to_tokens(prompts_or_tokens, prepend_bos=prepend_bos)


def logit_diff(
    logits: Float[Tensor, "batch seq vocab"],
    correct_token: int | Int[Tensor, ""],
    incorrect_token: int | Int[Tensor, ""],
    *,
    batch_index: int = 0,
    pos: int = -1,
) -> Float[Tensor, ""]:
    """Compute the correct-minus-incorrect next-token logit difference.

    Mechanistic note: logit difference is a signed scalar behavioral metric for
    whether an intervention restores the clean answer direction.

    Args:
        logits: Model logits with shape [batch, seq, vocab].
        correct_token: Token id for the clean/correct answer.
        incorrect_token: Token id for the corrupted/incorrect answer.
        batch_index: Batch row to score.
        pos: Sequence position to score.

    Returns:
        Scalar tensor equal to correct logit minus incorrect logit.
    """

    correct = int(correct_token.item()) if isinstance(correct_token, Tensor) else correct_token
    incorrect = int(incorrect_token.item()) if isinstance(incorrect_token, Tensor) else incorrect_token
    return logits[batch_index, pos, correct] - logits[batch_index, pos, incorrect]


def final_token_logits(
    logits: Float[Tensor, "batch seq vocab"],
    *,
    batch_index: int = 0,
    pos: int = -1,
) -> Float[Tensor, "vocab"]:
    """Extract logits at a single batch row and sequence position.

    Mechanistic note: most causal interventions are evaluated at the final
    prediction position where the behavior is expressed.

    Args:
        logits: Model logits with shape [batch, seq, vocab].
        batch_index: Batch row to select.
        pos: Sequence position to select.

    Returns:
        Vocabulary logits for the chosen position.
    """

    return logits[batch_index, pos, :]


def topk_tokens(
    logits: Float[Tensor, "batch seq vocab"] | Float[Tensor, "vocab"],
    tokenizer: PreTrainedTokenizerBase,
    *,
    k: int = 10,
    batch_index: int = 0,
    pos: int = -1,
) -> list[tuple[str, int, float]]:
    """Decode the top-k tokens from logits.

    Mechanistic note: top-k decoding turns raw logit movement into readable
    token-level hypotheses about what the model is representing.

    Args:
        logits: Full logits [batch, seq, vocab] or a vocabulary vector.
        tokenizer: HuggingFace tokenizer used by the model.
        k: Number of tokens to return.
        batch_index: Batch row used when logits are rank 3.
        pos: Sequence position used when logits are rank 3.

    Returns:
        Tuples of decoded token string, token id, and logit value.
    """

    vocab_logits = (
        final_token_logits(logits, batch_index=batch_index, pos=pos)
        if logits.ndim == 3
        else logits
    )
    values, indices = torch.topk(vocab_logits.detach(), k=k)
    return [
        (tokenizer.decode([int(token_id)]), int(token_id), float(value))
        for value, token_id in zip(values.cpu(), indices.cpu(), strict=True)
    ]


def component_hook_names(model: HookedTransformer, components: Sequence[str]) -> list[str]:
    """Build canonical per-layer hook names for component suffixes.

    Mechanistic note: canonical names remove ambiguity between residual,
    attention, and MLP intervention targets.

    Args:
        model: TransformerLens model.
        components: Component suffixes such as "hook_resid_pre" or "attn.hook_z".

    Returns:
        Full hook names for every layer and requested component.
    """

    return [f"blocks.{layer}.{component}" for layer in range(model.cfg.n_layers) for component in components]


def detach_to_cpu(tensor: Tensor) -> Tensor:
    """Detach a tensor and move it to CPU.

    Mechanistic note: moving completed scores to CPU frees accelerator memory
    before plotting or tabular analysis.

    Args:
        tensor: Tensor to detach.

    Returns:
        Detached CPU tensor.
    """

    return tensor.detach().cpu()


def infer_clean_target_token(
    logits: Float[Tensor, "batch seq vocab"],
    *,
    batch_index: int = 0,
    pos: int = -1,
) -> int:
    """Infer the clean target as the top final-position token.

    Mechanistic note: this gives causal tracing a default behavioral target
    when the caller has not specified an answer token.

    Args:
        logits: Clean-run logits.
        batch_index: Batch row to inspect.
        pos: Position to inspect.

    Returns:
        Integer token id with the largest logit at the selected position.
    """

    return int(torch.argmax(final_token_logits(logits, batch_index=batch_index, pos=pos)).item())


def token_probability(
    logits: Float[Tensor, "batch seq vocab"],
    token_id: int,
    *,
    batch_index: int = 0,
    pos: int = -1,
) -> Float[Tensor, ""]:
    """Return the softmax probability of a token at one position.

    Mechanistic note: probability is a bounded effect metric for causal
    mediation when logit differences are not pre-defined.

    Args:
        logits: Model logits with shape [batch, seq, vocab].
        token_id: Vocabulary token id to score.
        batch_index: Batch row to select.
        pos: Sequence position to select.

    Returns:
        Scalar probability tensor.
    """

    probs = torch.softmax(final_token_logits(logits, batch_index=batch_index, pos=pos), dim=-1)
    return probs[token_id]


def cfg_summary(model: HookedTransformer) -> dict[str, Any]:
    """Return a compact model configuration summary.

    Mechanistic note: these fields determine the valid axes for layer, head,
    position, and residual-stream sweeps.

    Args:
        model: TransformerLens model.

    Returns:
        Dictionary of common config fields.
    """

    cfg = model.cfg
    return {
        "model_name": cfg.model_name,
        "architecture": cfg.original_architecture,
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
        "device": str(cfg.device),
    }
