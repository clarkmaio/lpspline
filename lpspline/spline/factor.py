
import numpy as np
import cvxpy as cp
from typing import List, Optional
from .base import Spline

class Factor(Spline):
    """
    Categorical Factor mapping utilizing a one-hot-encoded basis.
    """
    def __init__(self, term: str, tag: Optional[str] = 'factor', n_classes: Optional[int] = None):
        """
        Initialize the Factor.

        Parameters
        ----------
        term : str
            The column name representing the categorical features.
        tag : Optional[str], default='factor'
            The descriptive tag denoting spline implementation type.
        n_classes : Optional[int], default=None
            The number of explicitly known classes or categories natively included in the data array `x`.
            If not populated, it uses the length of exactly uniquely present feature subsets.
        """
        super().__init__(term=term, tag=tag)
        self._n_classes = n_classes
        self._variables = []

    @property
    def n_classes(self) -> Optional[int]:
        """
        Returns the number of unique identified categorical classes.

        Returns
        -------
        Optional[int]
            The number of classes evaluated in the factor basis system.
        """
        return self._n_classes

    def init_spline(self, x: np.ndarray, by: np.ndarray = None):
        """
        Introspectively determines the number of categorical instances locally contained in `x`
        assuming none were globally established at initialization.

        Parameters
        ----------
        x : np.ndarray
            The evaluation group.
        by : np.ndarray, default=None
            The grouped indexing column if modeling interactions.
        """
        if self._n_classes is None:
            self._n_classes = len(np.unique(x))
        

    def _build_basis(self, x: np.ndarray) -> np.ndarray:
        """
        Generates the one-hot-encoded transformation matrix for evaluated inputs.

        Parameters
        ----------
        x : np.ndarray
            Categorical representation list implicitly structured as indexed zero-based classes.

        Returns
        -------
        np.ndarray
            A 2D binary matrix of shape `(n_samples, n_classes)`.
        """
        # One-hot encoding
        x = np.array(x).flatten().astype(int)
        n = len(x)
        basis = np.zeros((n, self.n_classes))
        
        # Clip or handle out of bounds? 
        # Assuming x is in [0, n_classes-1]
        mask = (x >= 0) & (x < self.n_classes)
        basis[np.arange(n)[mask], x[mask]] = 1.0
        
        return basis

    def _build_variables(self) -> cp.Variable:
        """
        Create the respective individual mapping variables corresponding cleanly to individual elements encoded.

        Returns
        -------
        cp.Variable
            A 1D dimensional vector tracking factor biases sized `(n_classes,)`.
        """
        if not self._variables:
            dim = self.n_classes
            self._variables = cp.Variable(shape=(dim,), name=f"{self.term}_factor")
        return self._variables

    def __repr__(self):
        return f"Factor(term='{self.term}', n_classes={self.n_classes})"
