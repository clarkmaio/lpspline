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
        variables = s._build_variables()
        constraints = []
        
        sign = -1 if self.decreasing else 1
        
        if isinstance(s, Linear):
            return self._constraint_Linear(s=s, sign=sign)
            
        elif isinstance(s, PiecewiseLinear):
            return self._constraint_PiecewiseLinear(s=s, sign=sign)
                
        elif isinstance(s, BSpline):
            return self._constraint_BSpline(s=s, sign=sign)

        else:
            raise ValueError(f"Monotonic constraint not supported for spline of type '{type(s).__name__}'")

    def _constraint_Linear(self, s, sign: int):
        variables = s._build_variables()
        slope_idx = 1 if s.bias else 0
        return [sign * variables[slope_idx] >= 0]

    def _constraint_PiecewiseLinear(self, s, sign):
        variables = s._build_variables()
        constraints = []
        M = len(s._by_classes) if s.by is not None else 1
        dim_base = 2 + len(s.knots)
        
        for c in range(M):
            v_chunk = variables[:, c] if s.by is not None else variables
            if self.start is not None and self.end is not None:
                knots = np.array(s.knots)
                indices = np.where((knots >= self.start) & (knots <= self.end))[0]
                
                if len(indices) > 0:
                    for i in indices:
                        constraints.append(sign * cp.sum(v_chunk[1:i+2]) >= 0)
            else:
                constraints.append(sign * v_chunk[1] >= 0)
                for i in range(2, dim_base):
                    constraints.append(sign * cp.sum(v_chunk[1:i+1]) >= 0)
                
        return constraints

    def _constraint_BSpline(self, s, sign):
        variables = s._build_variables()
        constraints = []
        M = len(s._by_classes) if s.by is not None else 1
        dim_base = len(s.knots) + s.degree - 1
        
        for c in range(M):
            # Select variables for current group spline
            v_chunk = variables[:, c] if s.by is not None else variables

            if self.start is not None and self.end is not None:
                knots = np.array(s.knots)
                indices = np.where((knots >= self.start) & (knots <= self.end))[0]
                max_idx = len(knots) - 2
                indices = indices[indices <= max_idx]
                
                if len(indices) > 0:
                    constraints.extend([sign * (v_chunk[indices+1] - v_chunk[indices]) >= 0])
            else:
                constraints.extend([sign * (v_chunk[1:] - v_chunk[:-1]) >= 0])
        return constraints
