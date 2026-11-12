"""Anchor constraint (created by ``Spline.passes_through``)."""
import warnings
from typing import Any, Iterable, Optional, Sequence, Tuple

import numpy as np

from .base import Constraint, _fmt


class Anchor(Constraint):
    """
    Force the spline through the given ``(x, y)`` points.

    Prefer the spline method ``passes_through(points=[(x, y), ...])``.

    Parameters
    ----------
    points : Sequence[Tuple[Any, float]]
        List of ``(x, y)`` tuples the spline must pass through.
    groups : Iterable, default=None
        `by` groups the constraint applies to (all if None).
    """

    method = "passes_through"

    def __init__(self, *args: Tuple[Any, float], points: Optional[Sequence[Tuple[Any, float]]] = None,
                 groups: Optional[Iterable[Any]] = None) -> None:
        super().__init__(groups=groups)
        if args:
            if points is not None:
                raise ValueError("Pass the points either positionally or with `points=`, not both.")
            warnings.warn(
                "Anchor(*points) is deprecated: use passes_through(points=[(x, y), ...]).",
                DeprecationWarning,
                stacklevel=2,
            )
            points = list(args)
        self.points = self._validate_points(points)

    @staticmethod
    def _validate_points(points) -> list:
        if points is None or isinstance(points, (str, bytes)) or not hasattr(points, "__len__"):
            raise TypeError("points must be a list of (x, y) tuples, e.g. points=[(5, 2), (1, -1)].")
        points = list(points)
        if not points:
            raise ValueError("points must not be empty.")
        for p in points:
            if not hasattr(p, "__len__") or isinstance(p, (str, bytes)) or len(p) != 2:
                raise ValueError(f"Each point must be an (x, y) tuple, got {p!r}.")
        return [(x, float(y)) for x, y in points]

    @property
    def xy(self) -> list:
        """The anchor points (kept for backward compatibility)."""
        return self.points

    def build_constraint(self, s) -> list:
        """Equality constraints ``spline(x_i) == y_i`` for every point and selected group."""
        basis = s._build_basis(np.array([x for x, _ in self.points]))
        y = np.array([y for _, y in self.points])
        return [basis @ v == y for v in s._coefficient_groups(self.groups)]

    def describe(self, term: str = "x") -> str:
        pts = ", ".join(f"({_fmt(x)}, {_fmt(y)})" for x, y in self.points)
        return f"passes through {pts}{self._describe_groups()}"
