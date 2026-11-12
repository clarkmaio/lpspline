"""Base class and shared helpers for spline shape constraints."""
import abc
from typing import List, Optional, Tuple

import cvxpy as cp
import numpy as np


class Constraint(abc.ABC):
    """
    Abstract base class defining the shape constraint interface.
    """
    @abc.abstractmethod
    def build_constraint(self, s) -> list:
        """
        Builds CVXPY constraints functionally mapping to the given spline structure.

        Parameters
        ----------
        s : Spline
            The initialized Spline instance to construct constraints upon.

        Returns
        -------
        list
            A list containing CVXPY constraint objects.
        """
        pass


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


def group_chunks(s, variables: cp.Variable) -> List[cp.Expression]:
    """Split a spline's coefficient variable into one 1D vector per ``by`` group.

    Args:
        s: The spline owning the variables.
        variables: The CVXPY variable of the spline.

    Returns:
        A list of 1D coefficient expressions, one per group (or one in total).
    """
    if getattr(s, "by", None) is not None and len(variables.shape) == 2:
        return [variables[:, c] for c in range(variables.shape[1])]
    return [variables]


def bspline_coef_range(s, start: Optional[float], end: Optional[float]) -> Optional[Tuple[int, int]]:
    """Return the inclusive range of B-spline coefficients active inside a window.

    On the knot interval ``[knots[i], knots[i+1]]`` a spline of degree ``k`` is a
    combination of coefficients ``i, ..., i + k``. The window is mapped to all
    knot intervals it overlaps.

    Args:
        s: A fitted-ready BSpline (knots already initialised).
        start: Lower edge of the window, or None.
        end: Upper edge of the window, or None.

    Returns:
        ``(first, last)`` coefficient indices, or None if the window does not
        overlap the spline domain.
    """
    knots = np.asarray(s.knots, dtype=float)
    lo, hi = window_bounds(start, end)
    intervals = [
        i for i in range(len(knots) - 1)
        if knots[i] < hi and knots[i + 1] > lo
    ]
    if not intervals:
        return None
    return intervals[0], intervals[-1] + s.degree


def bspline_derivative_coefs(s, v: cp.Expression, first: int, last: int) -> cp.Expression:
    """Coefficients of the derivative B-spline restricted to ``first..last``.

    The derivative of a degree-``k`` B-spline with coefficients ``c`` is a
    degree ``k-1`` B-spline with coefficients
    ``k * (c[j+1] - c[j]) / (t[j+k+1] - t[j+1])``. Using these (rather than raw
    differences) keeps convexity constraints exact for unevenly spaced knots.

    Args:
        s: The BSpline.
        v: 1D coefficient expression for one group.
        first: First coefficient index (inclusive).
        last: Last coefficient index (inclusive).

    Returns:
        Expression of length ``last - first``.
    """
    k = s.degree
    diffs = v[first + 1:last + 1] - v[first:last]
    if k == 0:
        return diffs
    t = s._pad_knots(s.knots, k)
    j = np.arange(first, last)
    scale = k / (t[j + k + 1] - t[j + 1])
    return cp.multiply(scale, diffs)


def bspline_curvature_constraints(s, start: Optional[float], end: Optional[float], sign: int) -> list:
    """Build convexity (``sign=1``) or concavity (``sign=-1``) constraints for a BSpline.

    The spline is convex where its derivative is non-decreasing, which is
    guaranteed when the derivative's B-spline coefficients are non-decreasing.

    Args:
        s: The BSpline.
        start: Lower edge of the window, or None.
        end: Upper edge of the window, or None.
        sign: ``1`` for convex, ``-1`` for concave.

    Returns:
        A list of CVXPY constraints.
    """
    rng = bspline_coef_range(s, start, end)
    if rng is None or rng[1] - rng[0] < 2:
        return []
    first, last = rng
    constraints = []
    for v in group_chunks(s, s._build_variables()):
        d = bspline_derivative_coefs(s, v, first, last)
        constraints.append(sign * (d[1:] - d[:-1]) >= 0)
    return constraints


def pwl_kink_constraints(s, start: Optional[float], end: Optional[float], sign: int) -> list:
    """Build convexity (``sign=1``) or concavity (``sign=-1``) constraints for a PiecewiseLinear.

    The slope change at knot ``i`` is the coefficient ``v[i + 2]`` of the hinge
    ``max(0, x - knot_i)``; only knots inside the window are constrained.

    Args:
        s: The PiecewiseLinear spline.
        start: Lower edge of the window, or None.
        end: Upper edge of the window, or None.
        sign: ``1`` for convex, ``-1`` for concave.

    Returns:
        A list of CVXPY constraints.
    """
    lo, hi = window_bounds(start, end)
    knots = np.asarray(s.knots, dtype=float)
    # A kink exactly on the window edge does not affect curvature inside it,
    # but including it is harmless and matches the previous inclusive behaviour.
    indices = np.where((knots >= lo) & (knots <= hi))[0]
    if len(indices) == 0:
        return []
    return [sign * v[indices + 2] >= 0 for v in group_chunks(s, s._build_variables())]
