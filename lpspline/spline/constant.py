


from .base import Spline
import cvxpy as cp
from typing import List, Optional
import numpy as np


class Constant(Spline):
    def __init__(self, term: str, tag: Optional[str] = 'constant'):
        super().__init__(term=term, tag=tag)
        self._variables = []

    def _build_basis(self, x: np.ndarray) -> np.ndarray:
        """
        Builds the Constant basis functions.
        """
        return np.ones((len(x), 1))

    def _build_variables(self) -> List[cp.Variable]:
        """
        Create cvxpy variables for the spline coefficients.
        """
        if not self._variables:
            self._variables = [cp.Variable(shape=(1,), name=f"{self.term}_constant")]
        return self._variables

    def __repr__(self):
        return f"Constant(term='{self.term}')"