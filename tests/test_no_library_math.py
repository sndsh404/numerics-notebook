"""Prove the README promise: calccode/ contains no library math.

Every .py file in calccode/ is parsed and walked. The test fails on
numpy.linalg, numpy.fft, and numpy.random (any use, call or reference),
on calls of np.polyfit, np.polyval, and np.gradient, and on any import
of scipy, sympy, or sklearn. Import aliases are resolved per file, so
"import numpy as np" and a bare "import numpy" are both caught. Nothing
is whitelisted; if a real exception ever appears it belongs here as an
explicit, commented entry.
"""

import ast
from pathlib import Path

CALCCODE_DIR = Path(__file__).resolve().parents[1] / "calccode"

# numpy submodules that reimplement the math this repo writes by hand.
BANNED_NUMPY_SUBMODULES = ("linalg", "fft", "random")
# numpy functions ditto: polynomial fitting and finite differences.
BANNED_NUMPY_FUNCTIONS = ("polyfit", "polyval", "gradient")
# whole packages the repo never touches.
BANNED_IMPORT_ROOTS = ("scipy", "sympy", "sklearn")


def _dotted_name(node: ast.expr) -> str | None:
    """Resolve an Attribute/Name chain like np.linalg.norm to a dotted string."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _numpy_aliases(tree: ast.Module) -> set[str]:
    """Names bound to numpy in this file: np, numpy, or any custom alias."""
    aliases = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                if name.name == "numpy" or name.name.startswith("numpy."):
                    aliases.add(name.asname or name.name.split(".")[0])
    return aliases


def _check_file(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    aliases = _numpy_aliases(tree)
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                parts = name.name.split(".")
                if parts[0] in BANNED_IMPORT_ROOTS:
                    problems.append(f"{path.name}:{node.lineno} imports {name.name}")
                elif parts[0] == "numpy" and len(parts) > 1 and parts[1] in BANNED_NUMPY_SUBMODULES:
                    problems.append(f"{path.name}:{node.lineno} imports {name.name}")
        elif isinstance(node, ast.ImportFrom):
            parts = (node.module or "").split(".")
            if parts[0] in BANNED_IMPORT_ROOTS:
                problems.append(f"{path.name}:{node.lineno} imports from {node.module}")
            elif parts[0] == "numpy":
                if len(parts) > 1 and parts[1] in BANNED_NUMPY_SUBMODULES:
                    problems.append(f"{path.name}:{node.lineno} imports from {node.module}")
                for name in node.names:
                    banned = (*BANNED_NUMPY_SUBMODULES, *BANNED_NUMPY_FUNCTIONS)
                    if name.name in banned:
                        problems.append(f"{path.name}:{node.lineno} imports numpy.{name.name}")
        elif isinstance(node, ast.Attribute):
            dotted = _dotted_name(node)
            if dotted is None:
                continue
            parts = dotted.split(".")
            if parts[0] in aliases and len(parts) > 1:
                if parts[1] in (*BANNED_NUMPY_SUBMODULES, *BANNED_NUMPY_FUNCTIONS):
                    problems.append(f"{path.name}:{node.lineno} uses {dotted}")
    return problems


def test_no_library_math_in_calccode():
    files = sorted(CALCCODE_DIR.glob("*.py"))
    assert files, "calccode/ should contain Python files"
    problems = []
    for path in files:
        problems.extend(_check_file(path))
    assert not problems, "library math found in calccode/:\n" + "\n".join(problems)
