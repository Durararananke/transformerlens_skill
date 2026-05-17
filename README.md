# 🔬 transformerlens-skill

Reusable TransformerLens modules for mechanistic interpretability workflows: model loading, activation caching, causal intervention, gradient attribution, causal tracing, logit-lens decoding, activation steering, and probing utilities for transformer internals.

This repository was completed with assistance from Codex.

## 🛠️ Installation

```bash
uv sync
```

### 🧩 Supported Models

| Model Family | Status |
|--------------|--------|
| 🦙 Llama 3 | ✅ |
| 🧠 Qwen 3 | ✅ |
| ✨ Gemma 3 | ✅ |


## 🗂️ Project Structure

```text
src/transformerlens_skill/
├── models.py
├── utils.py
├── basics.py
├── activation_patching.py
├── attribution_patching.py
├── causal_tracing.py
├── logit_lens.py
└── model_steering.py
```

## 📦 Modules

`nnsight_basics.py` covers TransformerLens fundamentals: `run_with_cache`, named cache filters, core residual/attention/MLP activations, `add_hook` interventions, batched tokenization with `prepend_bos=True`, and backward-gradient caching.

`activation_patching.py` implements the clean-corrupted-patched paradigm used in causal circuit localization. It supports residual stream patches, attention head `hook_z` patches, MLP output patches, full layer-by-position grids, and seaborn heatmaps.

`attribution_patching.py` implements the gradient approximation to activation patching: `grad(clean) * (act_clean - act_corrupted)`. It is useful when exact patching over many components would require too many forward passes.

`causal_tracing.py` implements causal mediation analysis in the style of Meng et al.'s ROME work. It computes total, direct, and indirect effects, performs interchange interventions, and traces subject-mediated states across layers and positions.

`logit_lens.py` decodes intermediate residual streams through the final normalization and unembedding. It tracks how vocabulary predictions and correct-token ranks evolve across layers.

`model_steering.py` implements activation addition and probing: contrastive steering vectors, generation-time ActAdd hooks, multi-hook interventions, attention-head ablation, and sklearn logistic-regression probes.

## 🚀 Quick Start

### Model Loading

```python
from transformerlens_skill.models import load_model

bundle = load_model("llama3", device="cuda")
model, tokenizer, cfg = bundle.model, bundle.tokenizer, bundle.cfg
```

### Basics

```python
from transformerlens_skill.nnsight_basics import cache_all_activations, read_core_activations

logits, cache = cache_all_activations(model, "The Eiffel Tower is in")
layer0 = read_core_activations(cache, layer=0)
```

### Activation Patching

```python
from transformerlens_skill.activation_patching import run_activation_patching_grid

clean = model.to_tokens("The Eiffel Tower is in", prepend_bos=True)
corrupt = model.to_tokens("The Colosseum is in", prepend_bos=True)
paris = tokenizer(" Paris")["input_ids"][0]
rome = tokenizer(" Rome")["input_ids"][0]
grid = run_activation_patching_grid(model, clean, corrupt, paris, rome)
```

### Attribution Patching

```python
from transformerlens_skill.attribution_patching import compute_attribution_scores

scores = compute_attribution_scores(model, clean, corrupt)
resid_scores = scores["resid_pre"]
```

### Causal Tracing

```python
from transformerlens_skill.causal_tracing import compute_direct_effect, trace_important_states

effect = compute_direct_effect(model, clean, corrupt, layer=10, pos=3)
trace = trace_important_states(model, "The Eiffel Tower is located in", subject_tokens=[1, 2, 3])
```

### Logit Lens

```python
from transformerlens_skill.logit_lens import get_logit_lens_predictions, get_rank_of_correct_token

tokens = model.to_tokens("The capital of France is", prepend_bos=True)
predictions = get_logit_lens_predictions(model, tokens)
rank = get_rank_of_correct_token(predictions, tokenizer(" Paris")["input_ids"][0])
```

### Model Steering

```python
from transformerlens_skill.model_steering import extract_steering_vector, generate_with_steering

vector = extract_steering_vector(
    model,
    positive_prompts=["I love this", "This is excellent"],
    negative_prompts=["I hate this", "This is terrible"],
    layer=10,
)
text = generate_with_steering(model, "The movie was", vector, layer=10, alpha=1.5, max_new_tokens=30)
```

## 📄 Project Statement

The author and affiliation of this project:

```text
Project Name: transformerlens-skill
Author: Durararananke
Affiliation: College of Cyber Security, Jinan University
```

## 📚 References

- Nanda, N. and Bloom, J. "TransformerLens." [https://github.com/TransformerLensOrg/TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
