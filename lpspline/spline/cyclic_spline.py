
import numpy as np
import cvxpy as cp
from typing import List, Optional
from .base import Spline

class CyclicSpline(Spline):
    """
    Periodic Cyclic Spline defined by Fourier series expansions.
    """
    def __init__(self, term: str, order: int, period: float = None, tag: Optional[str] = 'cyclicspline', by: Optional[str] = None):
        """
        Initialize the Cyclic Spline.

        Parameters
        ----------
        term : str
            The column name denoting the periodic feature.
        order : int
            The number of sine/cosine pairs to generate.
        period : float, default=None
            The periodicity interval. If None, it is inferred from data.
        tag : Optional[str], default='cyclicspline'
            A tag for identification.
        by : Optional[str], default=None
            The column name denoting group classes if interaction modeling is required.
        """
        super().__init__(term=term, tag=tag)
        self._period = period
        self._order = order
        self._by = by
        self._variables = []

    @property
    def period(self) -> float:
        """
        Returns the determined period of the cyclic spline.

        Returns
        -------
        float
            The numeric period length.
        """
        return self._period

    @property
    def order(self) -> int:
        """
        Returns the number of generated Fourier term pairs.

        Returns
        -------
        int
            The cyclic order describing polynomial flexibility.
        """
        return self._order

    def init_spline(self, x: np.ndarray, by: np.ndarray = None):
        """
        Initializes the periodic boundary conditions based on data.

        If `period` is not specified directly during Initialization, it calculates 
        the period automatically across the maximum observed range in `x`.

        Parameters
        ----------
        x : np.ndarray
            The 1D input array of training periodic measurements.
        by : np.ndarray, default=None
            The array of grouping values.
        """
        super().init_spline(x, by)
        if self._period is None:
            self._period = np.max(x) - np.min(x)

    def _build_basis(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Builds the basis matrix for the cyclic spline using sine/cosine decompositions.

        Generates expansions evaluating sequentially as:
        $1, \\sin(2\\pi x / P), \\cos(2\\pi x / P), \\sin(4\\pi x / P), \\dots$
        
        Parameters
        ----------
        x : np.ndarray
            Input 1D array of cyclic measurements.
        **kwargs : dict
            Additional format arguments.
        
        Returns
        -------
        np.ndarray
            A 2D mathematical basis matrix of shape `(n_samples, 1 + 2 * order)`.
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
        """
        Create CVXPY variables representing the Fourier coefficients.

        Returns
        -------
        cp.Variable
            A CVXPY Variable of shape `(1 + 2 * order, len(by_classes) if by else 1)`.
        """
        if isinstance(self._variables, list) and not self._variables:
            dim_base = 1 + 2 * self.order
            if self._by is not None:
                self._variables = cp.Variable(shape=(dim_base, len(self._by_classes)), name=f"{self.term}_cyclic")
            else:
                self._variables = cp.Variable(shape=(dim_base,), name=f"{self.term}_cyclic")
        return self._variables

    def __repr__(self):
        return f"CyclicSpline(term='{self.term}', period={self.period}, order={self.order}, by={self._by})"
