
import numpy as np
import cvxpy as cp
from typing import Optional, Tuple, Union

from .base import CONSTRAINT_METHODS, Spline

class PiecewiseLinear(Spline):
    """
    Piecewise Linear Spline framework built primarily around discrete ReLU knot bases.
    """

    supported_constraints = frozenset(CONSTRAINT_METHODS)
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
        self._knots_spec = knots
        self._by = by

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
        super().init_spline(x, by)
        # Recompute from the user's original spec so refitting on new data works.
        spec = getattr(self, '_knots_spec', self._knots)
        if isinstance(spec, (int, np.integer)):
            self._knots = np.linspace(np.min(x), np.max(x), int(spec))
        else:
            self._knots = np.sort(np.asarray(spec, dtype=float))

    def _build_basis(self, x: np.ndarray) -> np.ndarray:
        """
        Builds the localized Rectified Linear basis terms around internal knots.

        Evaluates natively sequentially across features utilizing $max(0, x-k_i)$.

        Parameters
        ----------
        x : np.ndarray
            Input dataset array points.

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

    # ------------------------------------------------------------- constraints

    def _breakpoints(self) -> np.ndarray:
        return np.asarray(self.knots, dtype=float)

    def _bound_domain(self) -> Tuple[float, float]:
        knots = np.asarray(self.knots, dtype=float)
        x_range = getattr(self, '_x_range', None) or (knots.min(), knots.max())
        return min(knots.min(), x_range[0]), max(knots.max(), x_range[1])

    def _slope_constraints(self, v: cp.Expression, lo: float, hi: float, sign: int) -> list:
        """
        Constrain the slope of every linear segment overlapping ``[lo, hi]``.

        With basis ``[1, x, max(0, x - k_0), ...]`` segment ``j`` (0 = left of the
        first knot) has slope ``sum(v[1:j + 2])``.
        """
        knots = np.asarray(self.knots, dtype=float)
        edges = np.concatenate([[-np.inf], knots, [np.inf]])
        segments = [j for j in range(len(knots) + 1) if edges[j] < hi and edges[j + 1] > lo]
        return [sign * cp.sum(v[1:j + 2]) >= 0 for j in segments]

    def _curvature_constraints(self, v: cp.Expression, lo: float, hi: float, sign: int) -> list:
        """The slope change at knot ``i`` is the hinge coefficient ``v[i + 2]``."""
        knots = np.asarray(self.knots, dtype=float)
        indices = np.where((knots >= lo) & (knots <= hi))[0]
        if len(indices) == 0:
            return []
        return [sign * v[indices + 2] >= 0]

    def __repr__(self):
        return f"PiecewiseLinear(term='{self.term}', knots={self.knots}, by={self._by})"
