"""Coefficient penalties, added with ``Spline.add_penalty``."""
from .base import Penalty
from .smooth import Lasso, Ridge

__all__ = ["Penalty", "Ridge", "Lasso"]
