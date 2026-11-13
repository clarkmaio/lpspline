"""Constraint objects, built by the spline methods (increasing, bounded, ...).

They are not meant to be created directly: use ``spline.increasing()``,
``spline.bounded(lower=0)``, ``spline.passes_through(points=[...])``, ...
The objects are listed in ``spline.constraints``.
"""
from .anchor import Anchor
from .base import Constraint, ShapeConstraint
from .bound import Bound

__all__ = ["Constraint", "ShapeConstraint", "Bound", "Anchor"]
