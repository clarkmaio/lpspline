


from .base import Spline
import cvxpy as cp
from typing import List, Optional
import numpy as np


class Constant(Spline):
    """
    Constant spline intercept representation.
    """
    def __init__(self, term: str, tag: Optional[str] = 'constant'):
        """
        Initialize the Constant intercept model component.

        Parameters
        ----------
        term : str
            A descriptive string, e.g. "intercept".
        tag : Optional[str], default='constant'
            A tag for categorical identification.
        """
        super().__init__(term=term, tag=tag)
        self._variables = []

    def _build_basis(self, x: np.ndarray) -> np.ndarray:
        """
        Builds the Constant basis function uniformly evaluating to `1.0`.

        Parameters
        ----------
        x : np.ndarray
            Input distribution array used functionally to match the matrix dimensional length.

        Returns
        -------
        np.ndarray
            A 2D matrix of shape `(n_samples, 1)` entirely populated of ones.
        """
        return np.ones((len(x), 1))

    def _build_variables(self) -> cp.Variable:
        """
        Create the CVXPY variable for the sole intercept coefficient.

        Returns
        -------
        cp.Variable
            A scalar CVXPY Variable initialized effectively as an array of shape `(1,)`.
        """
        if not self._variables:
            self._variables = cp.Variable(shape=(1,), name=f"{self.term}_constant")
        return self._variables

    def __repr__(self):
        return f"Constant(term='{self.term}')"