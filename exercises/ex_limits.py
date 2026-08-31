"""Exercise: numerical limits.

Implement each function. Your tests are in tests/test_exercises.py,
run them with: python -m pytest tests/test_exercises.py --run-exercises
The reference implementation is calccode/limits.py.
"""

from typing import Callable


def sample_near(f: Callable[[float], float], a: float, side: str, n: int = 24) -> list[float]:
    """Return f evaluated at n points approaching a from the given side.

    side is "left" or "right". Offsets should shrink geometrically from
    0.1 by a factor of 0.25 per step.
    """
    raise NotImplementedError


def one_sided_limit(f: Callable[[float], float], a: float, side: str) -> float:
    """Estimate the one-sided limit from the last few samples."""
    raise NotImplementedError


def is_oscillating(f: Callable[[float], float], a: float) -> bool:
    """Return True if the samples near a stay bounded but never settle.

    sin(1/x) at 0 should give True; sin(x)/x at 0 should give False.
    """
    raise NotImplementedError


def two_sided_limit(f: Callable[[float], float], a: float) -> float | None:
    """Return the limit if both sides agree, else None."""
    raise NotImplementedError
