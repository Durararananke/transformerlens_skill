"""TransformerLens skill modules for mechanistic interpretability workflows.

The package root is intentionally lightweight. Heavy dependencies such as torch
and TransformerLens are imported only when model-loading helpers are requested.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

# Keep package-root imports lightweight for metadata and discovery tools.
try:
    __version__ = version("transformerlens-skill")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["MODEL_REGISTRY", "ModelBundle", "__version__", "load_model"]


def __getattr__(name: str) -> object:
    """Lazily expose model helpers.

    Args:
        name: Attribute requested from the package root.

    Returns:
        The requested model helper object.
    """

    if name in {"MODEL_REGISTRY", "ModelBundle", "load_model"}:
        from transformerlens_skill.models import MODEL_REGISTRY, ModelBundle, load_model

        exports = {
            "MODEL_REGISTRY": MODEL_REGISTRY,
            "ModelBundle": ModelBundle,
            "load_model": load_model,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
