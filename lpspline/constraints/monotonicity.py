"""Monotonicity constraint."""
from typing import Optional

import cvxpy as cp
import numpy as np

from .base import Constraint, bspline_coef_range, group_chunks, window_bounds


class Monotonic(Constraint):
    """
    Monotonicity constraint enforcing non-decreasing or non-increasing behaviour.
    """
    def __init__(self, start: Optional[float] = None, end: Optional[float] = None, decreasing: bool = False):
        """
        Initialize the Monotonic constraint.

        Parameters
        ----------
        start : float, default=None
            The domain starting coordinate to bound the constraint enforcement region.
            If None, the region is unbounded on the left.
        end : float, default=None
            The domain ending coordinate for the constraint region.
            If None, the region is unbounded on the right.
        decreasing : bool, default=False
            If True, enforces a monotonically decreasing behavior. Otherwise, non-decreasing.
        """
        window_bounds(start, end)  # validate early
        self.start = start
        self.end = end
        self.decreasing = decreasing

    def build_constraint(self, s) -> list:
        """
        Constructs the appropriate CVXPY monotonic formulations according to the basis type.

        Parameters
        ----------
        s : Spline
            The parent Spline applying this restriction.

        Returns
        -------
        list
            A sequence containing formulation rules as CVXPY boolean expressions.

        Raises
        ------
        ValueError
            If the supplied Spline instance is functionally unsupported.
        """
        from ..spline import BSpline, Linear, PiecewiseLinear

        sign = -1 if self.decreasing else 1

        if isinstance(s, Linear):
            return self._constraint_Linear(s=s, sign=sign)
        if isinstance(s, PiecewiseLinear):
            return self._constraint_PiecewiseLinear(s=s, sign=sign)
        if isinstance(s, BSpline):
            return self._constraint_BSpline(s=s, sign=sign)
        raise ValueError(f"Monotonic constraint not supported for spline of type '{type(s).__name__}'")

    def _constraint_Linear(self, s, sign: int) -> list:
        """
        Constrain the slope of a Linear spline (the window is irrelevant for a line).

        Parameters
        ----------
        s : Linear
            Linear spline component to limit.
        sign : int
            Slope direction multiplier.

        Returns
        -------
        list
            One constraint on the slope coefficient(s).
        """
        variables = s._build_variables()
        slope_idx = 1 if s.bias else 0
        if len(variables.shape) == 2:
            return [sign * variables[slope_idx, :] >= 0]
        return [sign * variables[slope_idx] >= 0]

    def _constraint_PiecewiseLinear(self, s, sign: int) -> list:
        """
        Constrain the slope of every linear segment overlapping the window.

        With basis ``[1, x, max(0, x - k_0), ..., max(0, x - k_{K-1})]`` the
        segment ``j`` (``j = 0`` left of the first knot, ``j = K`` right of the
        last one) has slope ``sum(v[1:j + 2])``.

        Parameters
        ----------
        s : PiecewiseLinear
            Approximation basis component.
        sign : int
            Direction multiplier.

        Returns
        -------
        list
            Constraints on the slopes of the selected segments.
        """
        lo, hi = window_bounds(self.start, self.end)
        knots = np.asarray(s.knots, dtype=float)
        edges = np.concatenate([[-np.inf], knots, [np.inf]])
        segments = [
            j for j in range(len(knots) + 1)
            if edges[j] < hi and edges[j + 1] > lo
        ]

        constraints = []
        for v in group_chunks(s, s._build_variables()):
            for j in segments:
                constraints.append(sign * cp.sum(v[1:j + 2]) >= 0)
        return constraints

    def _constraint_BSpline(self, s, sign: int) -> list:
        """
        Constrain consecutive differences of all coefficients active in the window.

        Non-decreasing coefficients are a sufficient condition for a
        non-decreasing B-spline, for any (sorted) knot sequence.

        Parameters
        ----------
        s : BSpline
            B-spline formulation component.
        sign : int
            Direction multiplier.

        Returns
        -------
        list
            Constraints on coefficient differences.
        """
        rng = bspline_coef_range(s, self.start, self.end)
        if rng is None or rng[1] == rng[0]:
            return []
        first, last = rng
        return [
            sign * (v[first + 1:last + 1] - v[first:last]) >= 0
            for v in group_chunks(s, s._build_variables())
        ]
