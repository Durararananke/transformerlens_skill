"""Structure tests that do not require heavyweight ML dependencies."""

from __future__ import annotations

import ast
import os
import pathlib
import subprocess
import sys
import tomllib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "transformerlens_skill"

# These tests avoid importing torch or TransformerLens so they can run in a bare checkout.

class ProjectStructureTest(unittest.TestCase):
    """Validate packaging, module layout, and dependency-light conventions."""

    def test_expected_modules_exist(self) -> None:
        """The public module layout should match the skill contract."""

        expected = {
            "__init__.py",
            "activation_patching.py",
            "attribution_patching.py",
            "causal_tracing.py",
            "logit_lens.py",
            "model_steering.py",
            "models.py",
            "nnsight_basics.py",
            "utils.py",
            "py.typed",
        }
        actual = {path.name for path in PACKAGE.iterdir() if path.is_file()}
        self.assertTrue(expected.issubset(actual))

    def test_pyproject_declares_runtime_dependencies(self) -> None:
        """Runtime dependencies should be explicit in project metadata."""

        pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
        dependencies = set(pyproject["project"]["dependencies"])
        required = {
            "transformer-lens",
            "torch",
            "einops",
            "fancy-einsum",
            "jaxtyping",
            "typeguard",
            "numpy",
            "pandas",
            "matplotlib",
            "seaborn",
            "plotly",
            "tqdm",
            "huggingface-hub",
            "transformers",
            "scikit-learn",
        }
        self.assertTrue(required.issubset(dependencies))

    def test_package_root_import_is_lightweight(self) -> None:
        """The package root should import without importing torch or TransformerLens."""

        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src")
        code = (
            "import sys; "
            "import transformerlens_skill; "
            "assert transformerlens_skill.__version__ == '0.1.0'; "
            "assert 'torch' not in sys.modules; "
            "assert 'transformer_lens' not in sys.modules"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            env=env,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_no_nnsight_imports_or_plot_show_calls(self) -> None:
        """Modules should stay TransformerLens-native and caller-controlled for plots."""

        for path in PACKAGE.glob("*.py"):
            source = path.read_text()
            self.assertNotIn("import nnsight", source)
            self.assertNotIn("from nnsight", source)
            self.assertNotIn("plt.show(", source)

    def test_public_functions_are_typed_and_documented(self) -> None:
        """Public module-level functions should keep a typed, documented surface."""

        for path in PACKAGE.glob("*.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in tree.body:
                if not isinstance(node, ast.FunctionDef) or node.name.startswith("_"):
                    continue
                args = [
                    *node.args.posonlyargs,
                    *node.args.args,
                    *node.args.kwonlyargs,
                ]
                missing_annotations = [
                    arg.arg
                    for arg in args
                    if arg.arg not in {"self", "cls"} and arg.annotation is None
                ]
                self.assertEqual(missing_annotations, [], f"{path}:{node.name}")
                self.assertIsNotNone(node.returns, f"{path}:{node.name}")
                self.assertIsNotNone(ast.get_docstring(node), f"{path}:{node.name}")


if __name__ == "__main__":
    unittest.main()
