
import numpy as np
import cvxpy as cp
from typing import List, Optional
from .base import Spline

class Linear(Spline):
    def __init__(self, term: str, bias: bool = True, tag: Optional[str] = 'linear', by: Optional[str] = None):
        super().__init__(term=term, tag=tag)
        self.bias = bias
        self._variables = []
        self._by = by
        self._by_classes = None

    def init_spline(self, x: np.ndarray, by: np.ndarray = None):
        if self._by is not None:
            self._by_classes = np.unique(by)

    def _build_basis(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Basis consist in x and 1 if bias is True.
        """
        assert x.ndim == 1, "x must be a 1D array"
        return np.hstack([np.ones((len(x), 1)), x.reshape(-1, 1)]) if self.bias else x.reshape(-1, 1)

    def _build_variables(self) -> cp.Variable:
        """
        Build variables matrix
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
