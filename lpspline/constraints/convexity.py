import cvxpy as cp
import numpy as np
from .base import Constraint

class Convex(Constraint):
    def build_constraint(self, s) -> list:
        raise ValueError(f"Convexity constraint not supported for spline of type '{type(s).__name__}'")
