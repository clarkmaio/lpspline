import cvxpy as cp
import numpy as np
from .base import Constraint
from ..spline import Linear, PiecewiseLinear, BSpline


class Convex(Constraint):

    def __init__(self, start: float = None, end: float = None) -> None:
        self.start = start
        self.end = end

    def build_constraint(self, s) -> list:        
        variables = s._build_variables()

        if isinstance(s, BSpline):
            return self._constraint_BSpline(s)
        elif isinstance(s, PiecewiseLinear):
            return self._constraint_PiecewiseLinear(s)
        else:
            raise NotImplementedError(f"Convexity constraint not implemented for spline type '{type(s).__name__}'")


    def _constraint_BSpline(self, s):
        variables = s._build_variables()
        constraints = []
        M = len(s._by_classes) if s.by is not None else 1
        dim_base = len(s.knots) + s.degree - 1

        for c in range(M):
            v_chunk = variables[c * dim_base : (c + 1) * dim_base]
            if self.start is not None and self.end is not None:
                knots = np.array(s.knots)
                indices = np.where((knots >= self.start) & (knots <= self.end))[0]
                max_idx = len(knots) - 3
                indices = indices[indices <= max_idx]
                
                if len(indices) > 0:
                    constraints.extend([v_chunk[indices+2] - 2 * v_chunk[indices+1] + v_chunk[indices] >= 0])
            else:
                constraints.extend([v_chunk[2:] - 2 * v_chunk[1:-1] + v_chunk[:-2] >= 0])
        return constraints

    def _constraint_PiecewiseLinear(self, s):
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
                    constraints.extend([v_chunk[indices+2] >= 0])
            else:
                constraints.extend([v_chunk[2:] >= 0])
        return constraints