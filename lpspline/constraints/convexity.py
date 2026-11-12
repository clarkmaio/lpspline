"""Convexity constraint (created by ``Spline.convex``)."""
from typing import Any, Iterable, Optional

from .base import ShapeConstraint


class Convex(ShapeConstraint):
    """
    Convex spline (non-decreasing slope) on a window.

    Prefer the spline method ``convex(...)``.

    Parameters
    ----------
    start : float, default=None
        Lower edge of the region (unbounded if None).
    end : float, default=None
        Upper edge of the region (unbounded if None).
    groups : Iterable, default=None
        `by` groups the constraint applies to (all if None).
    """

    def __init__(self, start: Optional[float] = None, end: Optional[float] = None,
                 groups: Optional[Iterable[Any]] = None) -> None:
        super().__init__(order=2, sign=1, start=start, end=end, groups=groups)
