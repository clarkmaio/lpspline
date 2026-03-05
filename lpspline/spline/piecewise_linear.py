
import numpy as np
import cvxpy as cp
from typing import List, Optional, Union
from .base import Spline

class PiecewiseLinear(Spline):
    """
    Piecewise Linear Spine framework built primarily around discrete ReLU knot bases.
    """
    def __init__(self, term: str, knots: Union[int, np.ndarray], tag: Optional[str] = 'pwl', by: Optional[str] = None):
        """
        Initialize the Piecewise Linear Spline.

        Parameters
        ----------
        term : str
            The column name representing the continuous covariate.
        knots : Union[int, np.ndarray]
            An explicit array sequence of sorted breakpoint knot coordinates, or an integer specifying count.
        tag : Optional[str], default='pwl'
            The descriptive tag denoting spline implementation type.
        by : Optional[str], default=None
            The column representing interaction categorical groupings.
        """
        super().__init__(term=term, tag=tag)
        self._knots = knots
        self._by = by
        self._variables = []

    @property
    def knots(self) -> Union[int, np.ndarray]:
        """
        Returns the knot breaking parameters natively tracked within.

        Returns
        -------
        Union[int, np.ndarray]
            The 1D locations sequentially mapping piecewise shift points.
        """
        return self._knots

    @property
    def by(self) -> Optional[str]:
        """
        Returns the grouping variable column name.

        Returns
        -------
        Optional[str]
            The mapping group attribute column name.
        """
        return self._by

    def init_spline(self, x: np.ndarray, by: np.ndarray = None):
        """
        Initializes parameter bounds dynamically assuming knots were expressed as integer lengths.

        Parameters
        ----------
        x : np.ndarray
            The 1D input target observation set.
        by : np.ndarray, default=None
            Assigned category classes if grouping interactively.
        """
        if isinstance(self._knots, int):
            self._knots = np.linspace(np.min(x), np.max(x), self._knots)
        else:
            self._knots = np.sort(self._knots)

        if self._by is not None:
            self._by_classes = np.unique(by)

    def _build_basis(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Builds the localized Rectified Linear basis terms around internal knots.

        Evaluates natively sequentially across features utilizing $max(0, x-k_i)$.

        Parameters
        ----------
        x : np.ndarray
            Input dataset array points.
        **kwargs : dict
            Supplemental arguments.

        Returns
        -------
        np.ndarray
            Matrix containing structural features with shape `(n_samples, 2 + len(knots))`.
        """
        x = np.array(x).flatten()
        n = len(x)
        
        basis_list = [np.ones(n), x]
        for k in self.knots:
            basis_list.append(np.maximum(0, x - k))
            
        base_basis = np.vstack(basis_list).T
        return base_basis

    def _build_variables(self) -> cp.Variable:
        """
        Creates optimized CVXPY configurations sequentially describing parameter weights.

        Returns
        -------
        cp.Variable
            Structured matrix matching length required for evaluating full sequence.
        """
        if isinstance(self._variables, list) and not self._variables:
            dim_base = 2 + len(self.knots)
            if self._by is not None:
                self._variables = cp.Variable(shape=(dim_base, len(self._by_classes)), name=f"{self.term}_pwl")
            else:
                self._variables = cp.Variable(shape=(dim_base,), name=f"{self.term}_pwl")
        return self._variables

    def __repr__(self):
        return f"PiecewiseLinear(term='{self.term}', knots={self.knots}, by={self._by})"
