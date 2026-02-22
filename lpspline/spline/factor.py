
import numpy as np
import cvxpy as cp
from typing import List, Optional
from .base import Spline

class Factor(Spline):
    def __init__(self, term: str, tag: Optional[str] = 'factor', n_classes: Optional[int] = None):
        super().__init__(term=term, tag=tag)
        self._n_classes = n_classes
        self._variables = []

    @property
    def n_classes(self):
        return self._n_classes

    def init_spline(self, x: np.ndarray, by: np.ndarray = None):
        if self._n_classes is None:
            self._n_classes = len(np.unique(x))
        

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
