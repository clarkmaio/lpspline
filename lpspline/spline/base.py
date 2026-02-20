
import abc
import numpy as np
import cvxpy as cp
from typing import List, Optional

class Spline(abc.ABC):
    """
    Abstract base class for all spline types.
    """
    def __init__(self, term: str, tag: Optional[str] = None):
        """
        Initialize the Spline.

        Args:
            term: The name of the feature column this spline models.
        """
        self.term = term
        self.tag = tag
        self.constraints = []

    def add_constraint(self, *constraints):
        """
        Takes in input Constraints.
        Not all splines can accept all constraints.
        Raises ValueError depending on the spline type and constraint.
        """
        from ..constraints import Monotonic, Convex, Concave
        from .cyclic_spline import CyclicSpline
        from .factor import Factor
        from .linear import Linear
        from .piecewise_linear import PiecewiseLinear
        
        for c in constraints:
            if isinstance(self, (CyclicSpline, Factor)):
                if isinstance(c, (Monotonic, Convex, Concave)):
                    raise ValueError(f"{type(self).__name__} cannot accept {type(c).__name__} constraint.")
            if isinstance(self, (Linear, PiecewiseLinear)):
                if isinstance(c, (Convex, Concave)):
                    raise ValueError(f"{type(self).__name__} cannot accept {type(c).__name__} constraint.")
                
            self.constraints.append(c)
        return self

    @abc.abstractmethod
    def _build_basis(self, x: np.ndarray) -> np.ndarray:
        """
        Builds the basis functions for the spline.

        Args:
            x: Input feature array.

        Returns:
            A numpy array of shape (n_samples, n_basis_funcs).
        """
        pass

    @abc.abstractmethod
    def _build_variables(self) -> List[cp.Variable]:
        """
        Returns a list of cvxpy variables associated with this spline.
        """
        pass

    def __call__(self, x: np.ndarray) -> cp.Expression:
        """
        Evaluates the spline expression for the given input x.

        Args:
            x: Input feature array.

        Returns:
            A cvxpy Expression representing the spline values.
        
        Raises:
            ValueError: If no variables are defined for the spline.
        """
        basis = self._build_basis(x)
        variables = self._build_variables()
        
        if not variables:
            raise ValueError("No variables defined for this spline.")
            
        return basis @ variables[0]

    def __add__(self, other):
        """
        Implements addition to allow combining Splines into an LpRegressor model.
        
        Spline + Spline -> LpRegressor
        Spline + LpRegressor -> LpRegressor
        """
        from ..optimizer import LpRegressor
        
        if isinstance(other, Spline):
            return LpRegressor([self, other])
        elif isinstance(other, LpRegressor):
             other.splines.append(self)
             return other
        else:
            raise TypeError(f"Cannot add Spline and {type(other)}")

    def __pos__(self):
        """
        Unary + operator to create an LpRegressor with a single spline.
        Usage: model = +spline
        """
        from ..optimizer import LpRegressor
        return LpRegressor(self)

    def __repr__(self):
        return f"{self.__class__.__name__}(term='{self.term}')"
