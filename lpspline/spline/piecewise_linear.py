
import numpy as np
import cvxpy as cp
from typing import List, Optional, Union
from .base import Spline

class PiecewiseLinear(Spline):
    def __init__(self, term: str, knots: Union[int, np.ndarray], tag: Optional[str] = 'pwl', by: Optional[str] = None):
        super().__init__(term=term, tag=tag)
        self._knots = knots
        self._by = by
        self._variables = []

    @property
    def knots(self):
        return self._knots

    @property
    def by(self):
        return self._by

    def init_spline(self, x: np.ndarray, by: np.ndarray = None):
        if isinstance(self._knots, int):
            self._knots = np.linspace(np.min(x), np.max(x), self._knots)
        else:
            self._knots = np.sort(self._knots)

        if self._by is not None:
            self._by_classes = np.unique(by)

    def _build_basis(self, x: np.ndarray, **kwargs) -> np.ndarray:
        x = np.array(x).flatten()
        n = len(x)
        
        basis_list = [np.ones(n), x]
        for k in self.knots:
            basis_list.append(np.maximum(0, x - k))
            
        base_basis = np.vstack(basis_list).T
        return base_basis

    def _build_variables(self) -> cp.Variable:
        if isinstance(self._variables, list) and not self._variables:
            dim_base = 2 + len(self.knots)
            if self._by is not None:
                self._variables = cp.Variable(shape=(dim_base, len(self._by_classes)), name=f"{self.term}_pwl")
            else:
                self._variables = cp.Variable(shape=(dim_base,), name=f"{self.term}_pwl")
        return self._variables

    def __repr__(self):
        return f"PiecewiseLinear(term='{self.term}', knots={self.knots}, by={self._by})"
