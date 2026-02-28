import cvxpy as cp
import numpy as np
from .base import Constraint

class Monotonic(Constraint):
    def __init__(self, start: float = None, end: float = None, decreasing: bool = False):
        self.start = start
        self.end = end
        self.decreasing = decreasing
        
    def build_constraint(self, s) -> list:
        from ..spline import Linear, PiecewiseLinear, BSpline
        variables = s._build_variables()[0]
        constraints = []
        
        sign = -1 if self.decreasing else 1
        
        if isinstance(s, Linear):
            return self._constraint_Linear(s, sign)
            
        elif isinstance(s, PiecewiseLinear):
            return self._constraint_PiecewiseLinear(s, sign)
                
        elif isinstance(s, BSpline):
            return self._constraint_BSpline(s, sign)

        else:
            raise ValueError(f"Monotonic constraint not supported for spline of type '{type(s).__name__}'")

    def _constraint_Linear(self, s, sign: int):
        variables = s._build_variables()[0]
        return [sign * variables[0] >= 0]

    def _constraint_PiecewiseLinear(self, s, sign):
        variables = s._build_variables()[0]
        constraints = []
        
        if self.start is not None and self.end is not None:
            knots = np.array(s.knots)
            indices = np.where((knots >= self.start) & (knots <= self.end))[0]
            
            if len(indices) == 0:
                return []
            
            for i in indices:
                constraints.append(sign * cp.sum(variables[1:i+2]) >= 0)
        else:
            constraints.append(sign * variables[1] >= 0)
            for i in range(2, variables.shape[0]):
                constraints.append(sign * cp.sum(variables[1:i+1]) >= 0)
                
        return constraints

    def _constraint_BSpline(self, s, sign):
        variables = s._build_variables()[0]
        
        if self.start is not None and self.end is not None:
            knots = np.array(s.knots)
            indices = np.where((knots >= self.start) & (knots <= self.end))[0]
            max_idx = len(knots) - 2
            indices = indices[indices <= max_idx]
            
            if len(indices) == 0:
                return []
                
            return [sign * (variables[indices+1] - variables[indices]) >= 0]
        else:
            return [sign * (variables[1:] - variables[:-1]) >= 0]
