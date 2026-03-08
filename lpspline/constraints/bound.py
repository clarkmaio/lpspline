import cvxpy as cp
import numpy as np
from .base import Constraint

class Bound(Constraint):
    """
    Constraint enforcing lower and upper bounds on the spline output values
    using a grid-based discretization approach.
    """
    def __init__(self, lower: float = None, upper: float = None, n: int = 100, start: float = None, end: float = None):
        """
        Initialize the Bound constraint.

        Parameters
        ----------
        lower : float, default=None
            The lower bound for the spline output. If None, no lower bound is enforced.
        upper : float, default=None
            The upper bound for the spline output. If None, no upper bound is enforced.
        n : int, default=100
            The number of grid points used to discretize the constraint.
        start : float, default=None
            The domain starting coordinate to bound the constraint enforcement region.
        end : float, default=None
            The domain ending coordinate for the constraint region.
        """
        self.lower = lower
        self.upper = upper
        self.n = n
        self.start = start
        self.end = end

    def build_constraint(self, s) -> list:
        """
        Constructs the CVXPY bound constraints by evaluating the spline basis on a grid.

        Parameters
        ----------
        s : Spline
            The parent Spline applying this restriction.

        Returns
        -------
        list
            A list containing CVXPY constraint objects.
        """
        from ..spline import BSpline, PiecewiseLinear, CyclicSpline, Linear
        
        # Determine the default domain range for the grid
        if isinstance(s, BSpline):
            x_min_default, x_max_default = np.min(s.knots), np.max(s.knots) # TODO This is a proxy of spline domain, to improve
        elif isinstance(s, PiecewiseLinear):
            x_min_default, x_max_default = np.min(s.knots), np.max(s.knots) # TODO This is a proxy of spline domain, to improve
        elif isinstance(s, CyclicSpline):
            x_min_default, x_max_default = 0, s.period
        else:
            raise ValueError(f"Bound constraint not supported for spline of type '{type(s).__name__}'")

        # Use explicitly provided start/end if available
        x_min = self.start if self.start is not None else x_min_default
        x_max = self.end if self.end is not None else x_max_default

        grid = np.linspace(x_min, x_max, self.n)
        basis = s._build_basis(grid)
        variables = s._build_variables()
        
        constraints = []
        M = len(s._by_classes) if (getattr(s, 'by', None) is not None and s._by_classes is not None) else 1
        
        for c in range(M):
            v_chunk = variables[:, c] if (M > 1) else variables
            expr = basis @ v_chunk
            
            if self.lower is not None:
                constraints.append(expr >= self.lower)
            if self.upper is not None:
                constraints.append(expr <= self.upper)
                
        return constraints
