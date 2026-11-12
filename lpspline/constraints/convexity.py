"""Convexity constraint."""
from typing import Optional

from .base import Constraint, bspline_curvature_constraints, pwl_kink_constraints, window_bounds


class _CurvatureConstraint(Constraint):
    """Shared implementation of :class:`Convex` and :class:`Concave`."""

    _sign = 1

    def __init__(self, start: Optional[float] = None, end: Optional[float] = None) -> None:
        """
        Initialize the curvature constraint.

        Parameters
        ----------
        start : float, default=None
            The domain starting coordinate to bound the constraint enforcement region.
            If None, the region is unbounded on the left.
        end : float, default=None
            The domain ending coordinate for the constraint region.
            If None, the region is unbounded on the right.
        """
        window_bounds(start, end)  # validate early
        self.start = start
        self.end = end

    def build_constraint(self, s) -> list:
        """
        Constructs the appropriate CVXPY curvature formulations according to the basis type.

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
        NotImplementedError
            If the supplied Spline instance does not support curvature restrictions.
        """
        from ..spline import BSpline, PiecewiseLinear

        if isinstance(s, BSpline):
            return bspline_curvature_constraints(s, self.start, self.end, self._sign)
        if isinstance(s, PiecewiseLinear):
            return pwl_kink_constraints(s, self.start, self.end, self._sign)
        raise NotImplementedError(
            f"{type(self).__name__} constraint not implemented for spline type '{type(s).__name__}'"
        )


class Convex(_CurvatureConstraint):
    """
    Convexity constraint enforcing a positive second derivative globally or locally.
    """

    _sign = 1
