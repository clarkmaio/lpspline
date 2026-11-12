"""Concavity constraint."""
from .convexity import _CurvatureConstraint


class Concave(_CurvatureConstraint):
    """
    Concavity constraint enforcing a negative second derivative globally or locally.

    Shares the implementation of :class:`~lpspline.constraints.Convex` with the sign flipped.

    Parameters
    ----------
    start : float, default=None
        The domain starting coordinate to bound the constraint enforcement region.
    end : float, default=None
        The domain ending coordinate for the constraint region.
    """

    _sign = -1
