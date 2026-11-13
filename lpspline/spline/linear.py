
import numpy as np
import cvxpy as cp
from typing import Optional

from .base import Spline

class Linear(Spline):
    """
    Standard Linear feature expansion modeling.
    """

    supported_constraints = frozenset({"increasing", "decreasing", "bounded", "passes_through"})
    def __init__(self, term: str, bias: bool = True, tag: Optional[str] = 'linear', by: Optional[str] = None):
        """
        Initialize the Linear Spline mapping.

        Parameters
        ----------
        term : str
            The column name representing the continuous covariate.
        bias : bool, default=True
            Whether an intercept column of ones should be natively prefixed.
        tag : Optional[str], default='linear'
            The descriptive tag denoting spline implementation type.
        by : Optional[str], default=None
            The categorical array if modeling independent grouped slopes.
        """
        super().__init__(term=term, tag=tag)
        self.bias = bias
        self._by = by

    def _build_basis(self, x: np.ndarray) -> np.ndarray:
        """
        Builds the raw direct mapping matrix for linear expansions.
        
        The basis consists of `x` combined with a constant intercept of `1` if `bias=True`.

        Parameters
        ----------
        x : np.ndarray
            The 1D input numeric feature sequence array.

        Returns
        -------
        np.ndarray
            A 2D mathematical basis matrix.
        """
        assert x.ndim == 1, "x must be a 1D array"
        return np.hstack([np.ones((len(x), 1)), x.reshape(-1, 1)]) if self.bias else x.reshape(-1, 1)

    def _build_variables(self) -> cp.Variable:
        """
        Create the respective individual mapping variables mapping directly to slopes and bounds.

        Returns
        -------
        cp.Variable
            A CVXPY Variable dimensioned according to the bias configuration length.
        """
        if not self._variables:
            basedim = 2 if self.bias else 1
            if self.by is None:
                self._variables = cp.Variable(shape=(basedim,), name=f"{self.term}_linear")
            else:
                self._variables = cp.Variable(shape=(basedim, len(self._by_classes)), name=f"{self.term}_linear")
        return self._variables


    def _slope_constraints(self, v: cp.Expression, lo: float, hi: float, sign: int) -> list:
        """A line has a single slope, whatever the window."""
        return [sign * v[1 if self.bias else 0] >= 0]

    def __repr__(self):
        return f"Linear(term='{self.term}', bias={self.bias}, by={self._by})"
