
import abc
import numpy as np
import cvxpy as cp
from typing import List, Optional

class Spline(abc.ABC):
    """
    Abstract base class for all spline types.
    """
    def __init__(self, term: str, tag: Optional[str] = None):
        """
        Initialize the Spline component.

        Parameters
        ----------
        term : str
            The name of the feature column this spline models.
        tag : Optional[str], default=None
            An optional tag to identify this specific spline component.
        """
        self._term = term
        self._tag = tag
        self._constraints = []
        self._penalties = []
        self._variables = []

        self._by = None # column name of by reference values
        self._by_classes = None # set of unique by values
        self._by_int_map = None # map of by values to integer indices


    @property
    def by(self) -> Optional[str]:
        """
        Returns the grouping column name used to group spline coefficients.

        Returns
        -------
        Optional[str]
            The name of the column used for the `by` argument, or None.
        """
        return self._by

    @property
    def variables(self) -> List[cp.Variable]:
        """
        Returns the CVXPY variables representing the spline coefficients.

        Returns
        -------
        List[cp.Variable]
            The list of CVXPY variables constructed during fitting.
        """
        return self._variables

    @property
    def constraints(self) -> List[cp.Constraint]:
        """
        Returns the CVXPY constraints associated with the spline.

        Returns
        -------
        List[cp.Constraint]
            A list of all constraints that apply to this spline.
        """
        return self._constraints

    @property
    def penalties(self) -> List[cp.Expression]:
        """
        Returns the penalty expressions associated with the spline.

        Returns
        -------
        List[cp.Expression]
            A list of CVXPY expressions representing the penalties.
        """
        return self._penalties

    @property
    def term(self) -> str:
        """
        Returns the column name or term this spline models.

        Returns
        -------
        str
            The column name or term string.
        """
        return self._term

    @property
    def tag(self) -> Optional[str]:
        """
        Returns the custom tag for this spline component.

        Returns
        -------
        Optional[str]
            The tag assigned to this spline component, or None.
        """
        return self._tag

    @property
    def coefficients(self) -> np.ndarray:
        """
        Returns the fitted coefficients of the spline.

        Returns
        -------
        np.ndarray
            The computed coefficient values from the CVXPY variables.
        """
        return np.array(self._variables.value)

    def init_spline(self, x: np.ndarray, by: np.ndarray = None):
        """
        Initialize core parameters to build the spline basis according to the training data.

        This method is meant to be overridden by subclasses to initialize knot locations, 
        base configurations, or other data-dependent structures.

        Parameters
        ----------
        x : np.ndarray
            The 1D input array of training features for this spline.
        by : np.ndarray, default=None
            The array of grouping values, if a `by` variable is specified.
        """
        pass


    def init_by(self, by_classes: np.ndarray):
        """
        Initialize internal tracking for the unique classes in the `by` variable.

        Parameters
        ----------
        by_classes : np.ndarray
            A 1D array of the unique categorical/grouping values found in the data.
        """
        self._by_classes = by_classes
        self._by_int_map = {c: i for i, c in enumerate(by_classes)}

    def add_constraint(self, *constraints):
        """
        Add one or more shape constraints to the spline.

        Not all splines can accept all constraints. This function validates the
        compatibility of the constraints with the current spline type.

        Parameters
        ----------
        *constraints : Constraint
            One or more Constraint objects to apply (e.g., Monotonic, Convex).

        Returns
        -------
        Spline
            Returns the spline instance to allow methodical chaining.

        Raises
        ------
        ValueError
            If a given constraint is incompatible with this spline type.
        """
        from ..constraints import Monotonic, Convex, Concave, Bound
        from .cyclic_spline import CyclicSpline
        from .factor import Factor
        from .linear import Linear
        from .piecewise_linear import PiecewiseLinear
        
        for c in constraints:
            if isinstance(self, (CyclicSpline, Factor)):
                if isinstance(c, (Monotonic, Convex, Concave)):
                    raise ValueError(f"{type(self).__name__} cannot accept {type(c).__name__} constraint.")
            if isinstance(self, (Linear)):
                if isinstance(c, (Convex, Concave)):
                    raise ValueError(f"{type(self).__name__} cannot accept {type(c).__name__} constraint.")
                
            self._constraints.append(c)
        return self

    def add_penalty(self, *penalties):
        """
        Add one or more regularization penalties to the spline coefficients.

        Parameters
        ----------
        *penalties : Penalty
            One or more Penalty objects to apply (e.g., Ridge, Lasso).

        Returns
        -------
        Spline
            Returns the spline instance to allow methodical chaining.

        Raises
        ------
        TypeError
            If the supplied argument is not a Penalty instance.
        """
        from ..penalties import Penalty
        for p in penalties:
            if not isinstance(p, Penalty):
                raise TypeError(f"Expected a Penalty instance, got {type(p).__name__}")
            self._penalties.append(p)
        return self

    @abc.abstractmethod
    def _build_basis(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Builds the basis matrix evaluated at the input features.

        Parameters
        ----------
        x : np.ndarray
            The 1D input feature array.
        **kwargs : dict
            Additional arguments for basis construction.

        Returns
        -------
        np.ndarray
            A 2D numpy array of shape `(n_samples, n_basis_funcs)`.
        """
        pass

    @abc.abstractmethod
    def _build_variables(self) -> cp.Variable:
        """
        Returns the CVXPY variables associated with this spline.

        Returns
        -------
        cp.Variable
            The CVXPY variables matrix or vector used for convex optimization.
        """
        pass


    def _build_one_hot_matrix(self, by: np.ndarray) -> np.ndarray:
        """
        Returns a one-hot encoded matrix for the given array of categorical values based on `_by_classes`.

        Parameters
        ----------
        by : np.ndarray
            An integer encoded 1D array of group indices.

        Returns
        -------
        np.ndarray
            A 2D binary numpy array of shape `(n_samples, n_classes)`.
        """
        return np.eye(len(self._by_classes))[by]

    def __call__(self, x: np.ndarray, by: np.ndarray = None) -> cp.Expression:
        """
        Evaluates the symbolic CVXPY spline expression for the given input `x`.

        Parameters
        ----------
        x : np.ndarray
            The 1D input feature array for the spline evaluation.
        by : np.ndarray, default=None
            The 1D integer-encoded grouping array, if the `by` argument is specified.

        Returns
        -------
        cp.Expression
            A CVXPY Expression representing the spline values for optimization.
        
        Raises
        ------
        ValueError
            If no CVXPY variables are defined for the spline prior to evaluation.
        """

        variables = self._build_variables()
        if not variables:
            raise ValueError("No variables defined for this spline.")
        
        basis = self._build_basis(x)

        if by is None:
            return basis @ variables
        else:
            onehotby = self._build_one_hot_matrix(by=by)
            out = basis @ variables
            out = cp.multiply(out, onehotby)
            out = cp.sum(out, axis=1)
            return out


    def eval(self, x: np.ndarray, return_basis: bool = False, by: np.ndarray = None) -> np.ndarray:
        """
        Evaluates the fitted numeric spline values for the given input `x`.

        Parameters
        ----------
        x : np.ndarray
            The 1D input feature array to evaluate the spline on.
        return_basis : bool, default=False
            Whether to return the raw basis matrix instead of the evaluated spline.
        by : np.ndarray, default=None
            The 1D integer-encoded grouping array, if the `by` argument is specified.

        Returns
        -------
        np.ndarray
            A 1D numpy array of shape `(n_samples,)` representing the predicted values.
        
        Raises
        ------
        AssertionError
            If the spline has not been fitted and coefficients are not available.
        """
        assert self.coefficients is not None, "Spline has not been fitted."
        
        basis = self._build_basis(x)

        if by is None:
            return basis @ self.coefficients
        else:
            onehotby = self._build_one_hot_matrix(by=by)
            out = basis @ self.coefficients
            out = np.multiply(out, onehotby)
            out = np.sum(out, axis=1)
            return out

    def __add__(self, other):
        """
        Implements addition to allow combining Splines into an `LpRegressor` model.
        
        Parameters
        ----------
        other : Spline or LpRegressor
            The other component to combine with this spline.

        Returns
        -------
        LpRegressor
            An LpRegressor model combining the terms:
            - `Spline + Spline -> LpRegressor`
            - `Spline + LpRegressor -> LpRegressor`

        Raises
        ------
        TypeError
            If the object being added is not a Spline or LpRegressor instance.
        """
        from ..optimizer import LpRegressor
        
        if isinstance(other, Spline):
            return LpRegressor([self, other])
        elif isinstance(other, LpRegressor):
             other.splines.append(self)
             return other
        else:
            raise TypeError(f"Cannot add Spline and {type(other)}")

    def __pos__(self):
        """
        Unary `+` operator to create an `LpRegressor` with a single spline constraint.
        
        Usage: `model = +spline(...)`
        
        Returns
        -------
        LpRegressor
            An initialized regression model containing solely this spline.
        """
        from ..optimizer import LpRegressor
        return LpRegressor(self)

    def __repr__(self):
        return f"{self.__class__.__name__}(term='{self.term}')"
