
import numpy as np
import cvxpy as cp
from typing import List, Optional
from .base import Spline

class CyclicSpline(Spline):
    def __init__(self, term: str, order: int, period: float = None, tag: Optional[str] = 'cyclicspline'):
        super().__init__(term=term, tag=tag)
        self._period = period
        self._order = order
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

    def _build_basis(self, x: np.ndarray) -> np.ndarray:
        # Cyclic representation using Fourier series
        # 1, sin(2*pi*x/P), cos(2*pi*x/P), sin(4*pi*x/P), cos(4*pi*x/P), ...
        # Order K means K pairs of sin/cos? Or Order means degree?
        # Usually Fourier order K means terms up to k*omega.
        
        x = np.array(x).flatten()
        basis_list = [np.ones_like(x)]
        
        for k in range(1, self.order + 1):
            omega = 2 * np.pi * k / self.period
            basis_list.append(np.sin(omega * x))
            basis_list.append(np.cos(omega * x))
            
        return np.vstack(basis_list).T

    def _build_variables(self) -> List[cp.Variable]:
        if not self._variables:
            # 1 (intercept) + 2 * order (sin/cos pairs)
            dim = 1 + 2 * self.order
            self._variables = [cp.Variable(shape=(dim,), name=f"{self.term}_cyclic")]
        return self._variables

    def __repr__(self):
        return f"CyclicSpline(term='{self.term}', period={self.period}, order={self.order})"
