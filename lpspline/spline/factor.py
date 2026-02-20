
import numpy as np
import cvxpy as cp
from typing import List
from .base import Spline

class Factor(Spline):
    def __init__(self, term: str, n_classes: int, tag: Optional[str] = 'factor'):
        super().__init__(term=term, tag=tag)
        self.n_classes = n_classes
        self._variables = []

    def _build_basis(self, x: np.ndarray) -> np.ndarray:
        # One-hot encoding
        x = np.array(x).flatten().astype(int)
        n = len(x)
        basis = np.zeros((n, self.n_classes))
        
        # Clip or handle out of bounds? 
        # Assuming x is in [0, n_classes-1]
        mask = (x >= 0) & (x < self.n_classes)
        basis[np.arange(n)[mask], x[mask]] = 1.0
        
        return basis

    def _build_variables(self) -> List[cp.Variable]:
        if not self._variables:
            dim = self.n_classes
            self._variables = [cp.Variable(shape=(dim,), name=f"{self.term}_factor")]
        return self._variables

    def __repr__(self):
        return f"Factor(term='{self.term}', n_classes={self.n_classes})"
