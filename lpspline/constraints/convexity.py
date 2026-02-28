import cvxpy as cp
import numpy as np
from .base import Constraint
from ..spline import Linear, PiecewiseLinear, BSpline


class Convex(Constraint):

    def __init__(self, start: float = None, end: float = None) -> None:
        self.start = start
        self.end = end

    def build_constraint(self, s) -> list:        
        variables = s._build_variables()[0]

        if isinstance(s, BSpline):
            return self._constraint_BSpline(s)
        elif isinstance(s, PiecewiseLinear):
            return self._constraint_PiecewiseLinear(s)
        else:
            raise NotImplementedError(f"Convexity constraint not implemented for spline type '{type(s).__name__}'")


    def _constraint_BSpline(self, s):
        variables = s._build_variables()[0]

        if self.start is not None and self.end is not None:
            knots = np.array(s.knots)
            indices = np.where((knots >= self.start) & (knots <= self.end))[0]
            max_idx = len(knots) - 3
            indices = indices[indices <= max_idx]
            
            if len(indices) == 0:
                return []
            
            return [variables[indices+2] - 2 * variables[indices+1] + variables[indices] >= 0]
        else:
            return [variables[2:] - 2 * variables[1:-1] + variables[:-2] >= 0]

    def _constraint_PiecewiseLinear(self, s):
        variables = s._build_variables()[0]
        
        if self.start is not None and self.end is not None:
            knots = np.array(s.knots)
            indices = np.where((knots >= self.start) & (knots <= self.end))[0]
            
            if len(indices) == 0:
                return []
                
            return [variables[indices+2] >= 0]
        else:
            return [variables[2:] >= 0]