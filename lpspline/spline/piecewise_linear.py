
import numpy as np
import cvxpy as cp
from typing import List, Optional
from .base import Spline

class PiecewiseLinear(Spline):
    def __init__(self, term: str, knots: List[float], tag: Optional[str] = 'pwl'):
        super().__init__(term=term, tag=tag)
        self.knots = sorted(knots)
        self._variables = []

    def _build_basis(self, x: np.ndarray) -> np.ndarray:
        # Basis for PWL: 1, x, (x-k1)+, (x-k2)+ ...
        # Or just (x-k)+ for each knot?
        # Usually it's intercept + slope * x + sum(beta_k * (x-k)+)
        # Let's check common definitions.
        # Often: 1, x, max(0, x-k1), max(0, x-k2), ... 
        
        x = np.array(x).flatten()
        n = len(x)
        # Columns: 1, x, (x-k1)+, (x-k2)+, ...
        # This parameterization makes it easy to enforce continuity.
        
        # However, standard basis is often truncated power basis.
        # Let's use:
        # basis[:, 0] = 1
        # basis[:, 1] = x
        # basis[:, i+2] = max(0, x - knots[i])
        
        basis_list = [np.ones(n), x]
        for k in self.knots:
            basis_list.append(np.maximum(0, x - k))
            
        return np.vstack(basis_list).T

    def _build_variables(self) -> List[cp.Variable]:
        if not self._variables:
            # 1 (intercept) + 1 (slope) + len(knots) (changes in slope)
            dim = 2 + len(self.knots)
            self._variables = [cp.Variable(shape=(dim,), name=f"{self.term}_pwl")]
        return self._variables

    def __repr__(self):
        return f"PiecewiseLinear(term='{self.term}', knots={self.knots})"
