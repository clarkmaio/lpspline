
import numpy as np
import cvxpy as cp
from typing import List
from .base import Spline

class Linear(Spline):
    def __init__(self, term: str, bias: bool = True, tag: Optional[str] = 'linear'):
        super().__init__(term=term, tag=tag)
        self.bias = bias
        self._variables = []

    def _build_basis(self, x: np.ndarray) -> np.ndarray:
        # x is expected to be 1D array of shape (N,) or (N, 1)
        x = np.array(x).flatten()
        if self.bias:
            return np.vstack([x, np.ones_like(x)]).T
        else:
            return x.reshape(-1, 1)

    def _build_variables(self) -> List[cp.Variable]:
        if not self._variables:
            dim = 2 if self.bias else 1
            self._variables = [cp.Variable(shape=(dim,), name=f"{self.term}_linear")]
        return self._variables

    def __repr__(self):
        return f"Linear(term='{self.term}', bias={self.bias})"
