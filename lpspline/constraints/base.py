"""Base class and shared helpers for spline shape constraints.

Constraints are internal objects built by the spline methods
(``increasing``, ``decreasing``, ``convex``, ``concave``, ``bounded``,
``passes_through``). Each constraint only describes *what* is required; the
spline translates it into CVXPY constraints on its own coefficients.
"""
import abc
from typing import Any, Iterable, List, Optional, Tuple

import numpy as np


def window_bounds(start: Optional[float], end: Optional[float]) -> Tuple[float, float]:
    """Convert optional ``start``/``end`` into a closed interval.

    A missing bound means the window is unbounded on that side, so
    ``start=5`` alone constrains everything right of 5.

    Args:
        start: Lower edge of the window, or None.
        end: Upper edge of the window, or None.

    Returns:
        The ``(lo, hi)`` window, using infinities for missing edges.

    Raises:
        ValueError: If ``start > end``.
    """
    lo = -np.inf if start is None else float(start)
    hi = np.inf if end is None else float(end)
    if lo > hi:
        raise ValueError(f"Constraint window start ({start}) must be <= end ({end}).")
    return lo, hi


def _fmt(value: float) -> str:
    """Format a number compactly for labels."""
    return f"{value:g}" if isinstance(value, (int, float, np.number)) else str(value)


def describe_window(start: Optional[float], end: Optional[float], term: str = "x") -> str:
    """Human readable window, e.g. ``"for 2 ≤ x ≤ 6"`` (empty if unbounded)."""
    if start is None and end is None:
        return ""
    if end is None:
        return f" for {term} ≥ {_fmt(start)}"
    if start is None:
        return f" for {term} ≤ {_fmt(end)}"
    return f" for {_fmt(start)} ≤ {term} ≤ {_fmt(end)}"


class Constraint(abc.ABC):
    """
    Abstract base class of all constraints.

    Parameters
    ----------
    start : float, default=None
        Lower edge of the region where the constraint applies (unbounded if None).
    end : float, default=None
        Upper edge of the region where the constraint applies (unbounded if None).
    groups : Iterable, default=None
        For splines with a `by` column: the groups the constraint applies to.
        None means every group.
    """

    #: Name of the spline method that creates this constraint.
    method: str = ""

    def __init__(self, start: Optional[float] = None, end: Optional[float] = None,
                 groups: Optional[Iterable[Any]] = None) -> None:
        window_bounds(start, end)  # validate early
        self.start = start
        self.end = end
        self.groups = None if groups is None else list(groups)

    @property
    def window(self) -> Tuple[float, float]:
        """The ``(lo, hi)`` region where the constraint applies."""
        return window_bounds(self.start, self.end)

    @abc.abstractmethod
    def build_constraint(self, s) -> list:
        """
        Build the CVXPY constraints for the given (initialised) spline.

        Parameters
        ----------
        s : Spline
            The spline the constraint applies to.

        Returns
        -------
        list
            A list of CVXPY constraint objects.
        """

    @abc.abstractmethod
    def describe(self, term: str = "x") -> str:
        """
        Human readable description used in the model summary.

        Parameters
        ----------
        term : str, default="x"
            Name of the spline feature, used in the window description.

        Returns
        -------
        str
            E.g. ``"increasing for hour ≥ 6"``.
        """

    def _describe_groups(self) -> str:
        if self.groups is None:
            return ""
        return f" (groups: {', '.join(map(str, self.groups))})"

    def __repr__(self) -> str:
        return f"<{self.describe()}>"


class ShapeConstraint(Constraint):
    """Constraint on the slope (``order=1``) or curvature (``order=2``) of a spline.

    Parameters
    ----------
    order : int
        1 for monotonicity, 2 for convexity/concavity.
    sign : int
        +1 for increasing/convex, -1 for decreasing/concave.
    start, end, groups :
        See :class:`Constraint`.
    """

    _names = {(1, 1): "increasing", (1, -1): "decreasing", (2, 1): "convex", (2, -1): "concave"}

    def __init__(self, order: int, sign: int, start: Optional[float] = None,
                 end: Optional[float] = None, groups: Optional[Iterable[Any]] = None) -> None:
        super().__init__(start=start, end=end, groups=groups)
        if (order, sign) not in self._names:
            raise ValueError(f"Invalid shape constraint order={order}, sign={sign}.")
        self.order = order
        self.sign = sign

    @property
    def method(self) -> str:  # type: ignore[override]
        return self._names[(self.order, self.sign)]

    def build_constraint(self, s) -> list:
        """Ask the spline for its slope/curvature constraints on each selected group."""
        lo, hi = self.window
        constraints: List = []
        for v in s._coefficient_groups(self.groups):
            if self.order == 1:
                constraints.extend(s._slope_constraints(v, lo, hi, self.sign))
            else:
                constraints.extend(s._curvature_constraints(v, lo, hi, self.sign))
        return constraints

    def describe(self, term: str = "x") -> str:
        return f"{self.method}{describe_window(self.start, self.end, term)}{self._describe_groups()}"
