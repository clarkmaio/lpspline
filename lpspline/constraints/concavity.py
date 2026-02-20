import cvxpy as cp
import numpy as np
from .base import Constraint

class Concave(Constraint):
    def build_constraint(self, s) -> list:
        raise ValueError(f"Concavity constraint not supported for spline of type '{type(s).__name__}'")
            
