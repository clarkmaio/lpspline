"""Bound constraint (created by ``Spline.bounded``)."""
from typing import Any, Iterable, Optional

from .base import Constraint, _fmt, describe_window


class Bound(Constraint):
    """
    Keep the spline values within ``[lower, upper]`` on a window.

    The spline is evaluated on a grid of ``n`` points plus its breakpoints
    (knots), so the bound is exact for piecewise-linear splines and holds on the
    grid for smoother ones. Prefer the spline method ``bounded(...)``.

    Parameters
    ----------
    lower : float, default=None
        Lower bound (none if None).
    upper : float, default=None
        Upper bound (none if None).
    n : int, default=100
        Number of grid points.
    start : float, default=None
        Lower edge of the region (spline domain if None).
    end : float, default=None
        Upper edge of the region (spline domain if None).
    groups : Iterable, default=None
        `by` groups the constraint applies to (all if None).
    """

    method = "bounded"

    def __init__(self, lower: Optional[float] = None, upper: Optional[float] = None, n: int = 100,
                 start: Optional[float] = None, end: Optional[float] = None,
                 groups: Optional[Iterable[Any]] = None) -> None:
        super().__init__(start=start, end=end, groups=groups)
        if lower is None and upper is None:
            raise ValueError("bounded() needs at least one of `lower` or `upper`.")
        if lower is not None and upper is not None and lower > upper:
            raise ValueError(f"lower ({lower}) must be <= upper ({upper}).")
        if n < 2:
            raise ValueError("n must be >= 2.")
        self.lower = lower
        self.upper = upper
        self.n = n

    def build_constraint(self, s) -> list:
        """Bound the spline values at the evaluation points of the window."""
        basis = s._bound_basis(self.start, self.end, self.n)
        constraints = []
        for v in s._coefficient_groups(self.groups):
            expr = basis @ v
            if self.lower is not None:
                constraints.append(expr >= self.lower)
            if self.upper is not None:
                constraints.append(expr <= self.upper)
        return constraints

    def describe(self, term: str = "x") -> str:
        lo = "-∞" if self.lower is None else _fmt(self.lower)
        hi = "∞" if self.upper is None else _fmt(self.upper)
        return f"bounded in [{lo}, {hi}]{describe_window(self.start, self.end, term)}{self._describe_groups()}"
