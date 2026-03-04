
import numpy as np
import cvxpy as cp
from typing import List, Optional
from .base import Spline

class CyclicSpline(Spline):
    def __init__(self, term: str, order: int, period: float = None, tag: Optional[str] = 'cyclicspline', by: Optional[str] = None):
        super().__init__(term=term, tag=tag)
        self._period = period
        self._order = order
        self._by = by
        self._variables = []

    @property
    def period(self):
        return self._period

    @property
    def order(self):
        return self._order

    def init_spline(self, x: np.ndarray, by: np.ndarray = None):
        if self._period is None:
            self._period = np.max(x) - np.min(x)
            
        if self._by is not None:
            self._by_classes = np.unique(by)

    def _build_basis(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Build basis matrix for cyclic spline.
        1, sin(2*pi*x/P), cos(2*pi*x/P), sin(4*pi*x/P), cos(4*pi*x/P), ...
        Order K means K pairs of sin/cos.
        
        Args:
            x: Input array.
            **kwargs: Additional keyword arguments.
        
        Returns:
            np.ndarray: Basis matrix.
        """
        x = np.array(x).flatten()
        n = len(x)
        basis_list = [np.ones_like(x)]
        
        for k in range(1, self.order + 1):
            omega = 2 * np.pi * k / self.period
            basis_list.append(np.sin(omega * x))
            basis_list.append(np.cos(omega * x))
            
        base_basis = np.vstack(basis_list).T    
        return base_basis

    def _build_variables(self) -> cp.Variable:
        if isinstance(self._variables, list) and not self._variables:
            dim_base = 1 + 2 * self.order
            if self._by is not None:
                self._variables = cp.Variable(shape=(dim_base, len(self._by_classes)), name=f"{self.term}_cyclic")
            else:
                self._variables = cp.Variable(shape=(dim_base,), name=f"{self.term}_cyclic")
        return self._variables

    def __repr__(self):
        return f"CyclicSpline(term='{self.term}', period={self.period}, order={self.order}, by={self._by})"
