# Contributing

This repository is intentionally small: each module should stay focused on one mechanistic interpretability workflow and should remain importable without circular dependencies.

## Local Checks

```bash
python3 -m compileall src tests
python3 -m unittest discover -s tests
```

When dependencies are installed, also run:

```bash
uv run ruff check .
uv run pytest
```

## Code Standards

- Keep public functions typed and documented with Google-style docstrings.
- Use TransformerLens APIs only; do not introduce `nnsight` imports.
- Keep tensor operations device-agnostic and avoid hard-coded CPU/GPU movement except at plotting boundaries.
- Prefer `einops.rearrange` and `einops.reduce` when changing tensor shapes.
- Plotting helpers should accept optional `ax` parameters and should not call `plt.show()`.
