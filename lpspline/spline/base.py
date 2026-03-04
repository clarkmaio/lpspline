
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
        Initialize the Spline.

        Args:
            term: The name of the feature column this spline models.
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
        Returns the by variable of the spline.
        """
        return self._by

    @property
    def variables(self) -> List[cp.Variable]:
        """
        Returns the variables of the spline.
        """
        return self._variables

    @property
    def constraints(self) -> List[cp.Constraint]:
        """
        Returns the constraints of the spline.
        """
        return self._constraints

    @property
    def penalties(self) -> List[cp.Expression]:
        """
        Returns the penalties of the spline.
        """
        return self._penalties

    @property
    def term(self) -> str:
        """
        Returns the term of the spline.
        """
        return self._term

    @property
    def tag(self) -> Optional[str]:
        """
        Returns the tag of the spline.
        """
        return self._tag

    @property
    def coefficients(self) -> np.ndarray:
        """
        Returns the coefficients of the spline.
        """
        return np.array(self._variables.value)

    def init_spline(self, x: np.ndarray, by: np.ndarray = None):
        """
        Initialize core parameters to build spline basis according to train data.
        """
        pass


    def init_by(self, by_classes: np.ndarray):
        """
        Initialize the by integer map.
        """
        self._by_classes = by_classes
        self._by_int_map = {c: i for i, c in enumerate(by_classes)}

    def add_constraint(self, *constraints):
        """
        Takes in input Constraints.
        Not all splines can accept all constraints.
        Raises ValueError depending on the spline type and constraint.
        """
        from ..constraints import Monotonic, Convex, Concave
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
        Takes in input Penalties.
        Not all splines can accept all penalties.
        Raises ValueError depending on the spline type and penalty.
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
        Builds the basis functions for the spline.

        Args:
            x: Input feature array.

        Returns:
            A numpy array of shape (n_samples, n_basis_funcs).
        """
        pass

    @abc.abstractmethod
    def _build_variables(self) -> cp.Variable:
        """
        Returns a list of cvxpy variables associated with this spline.
        """
        pass


    def _build_one_hot_matrix(self, by: np.ndarray) -> np.ndarray:
        """
        Returns a one-hot encoded array for the given array of values. One hot based on self._by_classes.
        """
        return np.eye(len(self._by_classes))[by]

    def __call__(self, x: np.ndarray, by: np.ndarray = None) -> cp.Expression:
        """
        Evaluates the spline expression for the given input x.

        Args:
            x: Input feature array.

        Returns:
            A cvxpy Expression representing the spline values.
        
        Raises:
            ValueError: If no variables are defined for the spline.
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
        Evaluates the spline expression for the given input x.

        Args:
            x: Input feature array.

        Returns:
            A numpy array of shape (n_samples,) representing the spline values.
        
        Raises:
            ValueError: If no variables are defined for the spline.
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
        Implements addition to allow combining Splines into an LpRegressor model.
        
        Spline + Spline -> LpRegressor
        Spline + LpRegressor -> LpRegressor
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
        Unary + operator to create an LpRegressor with a single spline.
        Usage: model = +spline
        """
        from ..optimizer import LpRegressor
        return LpRegressor(self)

    def __repr__(self):
        return f"{self.__class__.__name__}(term='{self.term}')"
