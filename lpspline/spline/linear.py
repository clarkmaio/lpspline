
import numpy as np
import cvxpy as cp
from typing import List, Optional
from .base import Spline

class Linear(Spline):
    """
    Standard Linear feature expansion modeling.
    """
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
        self._variables = []
        self._by = by
        self._by_classes = None

    def init_spline(self, x: np.ndarray, by: np.ndarray = None):
        """
        Introspectively determines unique categories for nested grouping mappings.

        Parameters
        ----------
        x : np.ndarray
            The evaluation group.
        by : np.ndarray, default=None
            The grouped indexing column if modeling interactions.
        """
        super().init_spline(x, by)

    def _build_basis(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Builds the raw direct mapping matrix for linear expansions.
        
        The basis consists of `x` combined with a constant intercept of `1` if `bias=True`.

        Parameters
        ----------
        x : np.ndarray
            The 1D input numeric feature sequence array.
        **kwargs : dict
            Additional arguments.

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


    def __repr__(self):
        return f"Linear(term='{self.term}', bias={self.bias}, by={self._by})"
