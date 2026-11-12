"""Monotonicity constraint (created by ``Spline.increasing`` / ``Spline.decreasing``)."""
from typing import Any, Iterable, Optional

from .base import ShapeConstraint


class Monotonic(ShapeConstraint):
    """
    Non-decreasing (or non-increasing) spline on a window.

    Prefer the spline methods ``increasing(...)`` / ``decreasing(...)``.

    Parameters
    ----------
    start : float, default=None
        Lower edge of the region (unbounded if None).
    end : float, default=None
        Upper edge of the region (unbounded if None).
    decreasing : bool, default=False
        If True, enforce a non-increasing spline.
    groups : Iterable, default=None
        `by` groups the constraint applies to (all if None).
    """

    def __init__(self, start: Optional[float] = None, end: Optional[float] = None,
                 decreasing: bool = False, groups: Optional[Iterable[Any]] = None) -> None:
        super().__init__(order=1, sign=-1 if decreasing else 1, start=start, end=end, groups=groups)

    @property
    def decreasing(self) -> bool:
        """Whether the constraint enforces a non-increasing spline."""
        return self.sign < 0
