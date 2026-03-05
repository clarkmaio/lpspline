import cvxpy as cp
import numpy as np
from .base import Constraint

class Concave(Constraint):
    """
    Concavity constraint enforcing a negative second derivative globally or locally.
    """

    def __init__(self, start: float = None, end: float = None) -> None:
        """
        Initialize the Concavity constraint.

        Parameters
        ----------
        start : float, default=None
            The domain starting coordinate to bound the constraint enforcement region.
        end : float, default=None
            The domain ending coordinate for the constraint region.
        """
        self.start = start
        self.end = end

    def build_constraint(self, s) -> list:
        """
        Constructs the appropriate CVXPY concave formulations according to the basis type.

        Parameters
        ----------
        s : Spline
            The parent Spline applying this restriction.

        Returns
        -------
        list
            A sequence containing formulation rules as CVXPY boolean expressions.

        Raises
        ------
        NotImplementedError
            If the supplied Spline instance functionally lacks concavity restrictions.
        """
        from ..spline import Linear, PiecewiseLinear, BSpline
        
        variables = s._build_variables()

        if isinstance(s, BSpline):
            return self._constraint_BSpline(s)
        elif isinstance(s, PiecewiseLinear):
            return self._constraint_PiecewiseLinear(s)
        else:
            raise NotImplementedError(f"Concavity constraint not implemented for spline type '{type(s).__name__}'")    

    def _constraint_BSpline(self, s):
        """
        Bounds sequential B-Splines evaluating directly on the recursive knot control points natively.

        Parameters
        ----------
        s : BSpline
            B-spline formulation component.

        Returns
        -------
        list
            Resultant list enforcing sequential second-order numerical negative bounds.
        """
        variables = s._build_variables()
        constraints = []
        M = len(s._by_classes) if s.by is not None else 1
        dim_base = len(s.knots) + s.degree - 1

        for c in range(M):
            v_chunk = variables[:, c] if s.by is not None else variables
            if self.start is not None and self.end is not None:
                knots = np.array(s.knots)
                indices = np.where((knots >= self.start) & (knots <= self.end))[0]
                max_idx = len(knots) - 3
                indices = indices[indices <= max_idx]
                
                if len(indices) > 0:
                    constraints.extend([v_chunk[indices+2] - 2 * v_chunk[indices+1] + v_chunk[indices] <= 0])
            else:
                constraints.extend([v_chunk[2:] - 2 * v_chunk[1:-1] + v_chunk[:-2] <= 0])
        return constraints

    def _constraint_PiecewiseLinear(self, s):
        """
        Enforces piecewise linear negative second differences tracking changes in slopes natively.

        Parameters
        ----------
        s : PiecewiseLinear
            Approximation basis component.

        Returns
        -------
        list
            Formulations targeting piecewise basis differences targeting negative knot adjustments.
        """
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
                    constraints.extend([v_chunk[indices+2] <= 0])
            else:
                constraints.extend([v_chunk[2:] <= 0])
        return constraints
