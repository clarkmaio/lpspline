import cvxpy as cp
import numpy as np
from .base import Constraint

class Monotonic(Constraint):
    def __init__(self, decreasing: bool = False):
        self.decreasing = decreasing
        
    def build_constraint(self, s) -> list:
        from ..spline import Linear, PiecewiseLinear, BSpline
        variables = s._build_variables()[0]
        constraints = []
        
        sign = -1 if self.decreasing else 1
        
        if isinstance(s, Linear):
            # linear: variables[0] is slope
            constraints.append(sign * variables[0] >= 0)
            
        elif isinstance(s, PiecewiseLinear):
            # piecewiselinear: variables[0] is intercept, variables[1] is initial slope, variables[2:] are slope changes
            constraints.append(sign * variables[1] >= 0)
            for i in range(2, variables.shape[0]):
                constraints.append(sign * cp.sum(variables[1:i+1]) >= 0)
                
        elif isinstance(s, BSpline):
            """
            To make BSpline monotonic we impose coefficients to be monotonic: 
            c_i >= c_{i-1} for increasing, c_i <= c_{i-1} for decreasing.
            """
            constraints.append(sign * (variables[1:] - variables[:-1]) >= 0)

        else:
            raise ValueError(f"Monotonic constraint not supported for spline of type '{type(s).__name__}'")
            
        return constraints
