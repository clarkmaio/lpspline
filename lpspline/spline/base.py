
import abc
import warnings
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import cvxpy as cp
import numpy as np
import polars as pl

from ..constraints import Anchor, Bound, Concave, Constraint, Convex, Monotonic
from ..constraints.base import ShapeConstraint

#: All constraint methods, in the order they are documented.
CONSTRAINT_METHODS = ("increasing", "decreasing", "convex", "concave", "bounded", "passes_through")


def evaluate_condition(condition: pl.Expr, X: pl.DataFrame) -> np.ndarray:
    """
    Evaluate a boolean polars expression row-wise on a DataFrame.

    Parameters
    ----------
    condition : pl.Expr
        Boolean expression, e.g. ``pl.col("day_index") < 1000``.
    X : pl.DataFrame
        The data to evaluate the condition on.

    Returns
    -------
    np.ndarray
        Boolean array of shape ``(len(X),)``. Nulls are treated as False.

    Raises
    ------
    ValueError
        If the expression references missing columns or does not return one value per row.
    TypeError
        If the expression does not evaluate to booleans.
    """
    try:
        values = X.select(condition.alias("__lpspline_mask__")).to_series()
    except pl.exceptions.ColumnNotFoundError as e:
        raise ValueError(f"Condition {condition} references a column not in the data: {e}") from e
    if values.dtype != pl.Boolean:
        raise TypeError(f"Condition {condition} must evaluate to booleans, got {values.dtype}.")
    if len(values) != len(X):
        raise ValueError(
            f"Condition {condition} must return one value per row "
            f"(got {len(values)} values for {len(X)} rows)."
        )
    return values.fill_null(False).to_numpy()


def _check_condition(condition: pl.Expr, method: str) -> pl.Expr:
    """Validate the argument of `train_on` / `active_where`."""
    if not isinstance(condition, pl.Expr):
        raise TypeError(
            f"{method} expects a polars expression such as pl.col('x') > 0, "
            f"got {type(condition).__name__}."
        )
    return condition

class Spline(abc.ABC):
    """
    Abstract base class for all spline types.
    """

    #: Constraint methods this spline type supports (see `CONSTRAINT_METHODS`).
    supported_constraints: frozenset = frozenset()
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

        self._train_condition: Optional[pl.Expr] = None  # rows used to fit
        self._active_condition: Optional[pl.Expr] = None  # rows where the spline is non-zero


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
        if isinstance(self._variables, list) or self._variables.value is None:
            raise ValueError(f"{self!r} has not been fitted yet.")
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
        # Drop variables from any previous fit: their shape depends on the data
        # (knots, number of `by` groups, number of classes...).
        self._variables = []
        x_arr = np.asarray(x)
        self._x_range = (
            (float(np.nanmin(x_arr)), float(np.nanmax(x_arr)))
            if x_arr.size and np.issubdtype(x_arr.dtype, np.number) else None
        )
        if self._by is not None and by is not None:
            self.init_by(by)


    def init_by(self, by: np.ndarray):
        """
        Initialize internal tracking for the unique classes in the `by` variable.

        Parameters
        ----------
        by : np.ndarray
            A 1D array of the categorical/grouping values found in the data.
        """
        self._by_classes = np.unique(by)
        self._by_int_map = {c: i for i, c in enumerate(self._by_classes)}

    @property
    def train_condition(self) -> Optional[pl.Expr]:
        """
        Returns the condition selecting the training rows, set with `train_on`.

        Returns
        -------
        Optional[pl.Expr]
            The polars expression, or None if the spline trains on all rows.
        """
        return getattr(self, '_train_condition', None)

    @property
    def active_condition(self) -> Optional[pl.Expr]:
        """
        Returns the condition selecting the rows where the spline is active, set with `active_where`.

        Returns
        -------
        Optional[pl.Expr]
            The polars expression, or None if the spline is active on all rows.
        """
        return getattr(self, '_active_condition', None)

    def train_on(self, condition: pl.Expr) -> "Spline":
        """
        Fit this spline only on the rows matching `condition`.

        During `fit` the spline contributes 0 on the other rows, so the rest of the
        model explains them. Data-driven settings (automatic knots, cyclic period,
        `by` groups, factor classes) are computed from the training rows only.
        At prediction time the condition is ignored: use `active_where` to also
        switch the spline off when predicting.

        Parameters
        ----------
        condition : pl.Expr
            Boolean polars expression, e.g. ``pl.col("day_index") < 1000``.
            Rows where it is null count as False.

        Returns
        -------
        Spline
            The spline itself, to allow chaining.
        """
        self._train_condition = _check_condition(condition, "train_on")
        return self

    def active_where(self, condition: pl.Expr) -> "Spline":
        """
        Make the spline return 0 on the rows not matching `condition`.

        Applies both when fitting (the inactive rows do not inform this spline and
        are explained by the rest of the model) and when predicting.

        Parameters
        ----------
        condition : pl.Expr
            Boolean polars expression, e.g. ``pl.col("feature3") <= 0``.
            Rows where it is null count as False (inactive).

        Returns
        -------
        Spline
            The spline itself, to allow chaining.
        """
        self._active_condition = _check_condition(condition, "active_where")
        return self

    def train_mask(self, X: pl.DataFrame) -> np.ndarray:
        """
        Boolean mask of the rows selected by `train_on` (all True if unset).

        Parameters
        ----------
        X : pl.DataFrame
            The data.

        Returns
        -------
        np.ndarray
            Boolean array of shape ``(len(X),)``.
        """
        if self.train_condition is None:
            return np.ones(len(X), dtype=bool)
        return evaluate_condition(self.train_condition, X)

    def active_mask(self, X: pl.DataFrame) -> np.ndarray:
        """
        Boolean mask of the rows where the spline is active (all True if unset).

        Parameters
        ----------
        X : pl.DataFrame
            The data.

        Returns
        -------
        np.ndarray
            Boolean array of shape ``(len(X),)``.
        """
        if self.active_condition is None:
            return np.ones(len(X), dtype=bool)
        return evaluate_condition(self.active_condition, X)

    def fit_mask(self, X: pl.DataFrame) -> np.ndarray:
        """
        Rows on which the spline contributes during fitting: `train_on` AND `active_where`.

        Parameters
        ----------
        X : pl.DataFrame
            The training data.

        Returns
        -------
        np.ndarray
            Boolean array of shape ``(len(X),)``.
        """
        return self.train_mask(X) & self.active_mask(X)

    # ------------------------------------------------------------------ constraints

    def increasing(self, start: Optional[float] = None, end: Optional[float] = None,
                   groups: Optional[Iterable[Any]] = None) -> "Spline":
        """
        Constrain the spline to be non-decreasing.

        Parameters
        ----------
        start : float, default=None
            Only from this value of the feature on (unbounded if None).
        end : float, default=None
            Only up to this value of the feature (unbounded if None).
        groups : Iterable, default=None
            For splines with `by`: the groups to constrain (all if None).

        Returns
        -------
        Spline
            The spline itself, to allow chaining.
        """
        return self._add_constraint(Monotonic(start=start, end=end, groups=groups))

    def decreasing(self, start: Optional[float] = None, end: Optional[float] = None,
                   groups: Optional[Iterable[Any]] = None) -> "Spline":
        """
        Constrain the spline to be non-increasing.

        Parameters
        ----------
        start : float, default=None
            Only from this value of the feature on (unbounded if None).
        end : float, default=None
            Only up to this value of the feature (unbounded if None).
        groups : Iterable, default=None
            For splines with `by`: the groups to constrain (all if None).

        Returns
        -------
        Spline
            The spline itself, to allow chaining.
        """
        return self._add_constraint(Monotonic(start=start, end=end, decreasing=True, groups=groups))

    def convex(self, start: Optional[float] = None, end: Optional[float] = None,
               groups: Optional[Iterable[Any]] = None) -> "Spline":
        """
        Constrain the spline to be convex (non-decreasing slope).

        Parameters
        ----------
        start : float, default=None
            Only from this value of the feature on (unbounded if None).
        end : float, default=None
            Only up to this value of the feature (unbounded if None).
        groups : Iterable, default=None
            For splines with `by`: the groups to constrain (all if None).

        Returns
        -------
        Spline
            The spline itself, to allow chaining.
        """
        return self._add_constraint(Convex(start=start, end=end, groups=groups))

    def concave(self, start: Optional[float] = None, end: Optional[float] = None,
                groups: Optional[Iterable[Any]] = None) -> "Spline":
        """
        Constrain the spline to be concave (non-increasing slope).

        Parameters
        ----------
        start : float, default=None
            Only from this value of the feature on (unbounded if None).
        end : float, default=None
            Only up to this value of the feature (unbounded if None).
        groups : Iterable, default=None
            For splines with `by`: the groups to constrain (all if None).

        Returns
        -------
        Spline
            The spline itself, to allow chaining.
        """
        return self._add_constraint(Concave(start=start, end=end, groups=groups))

    def bounded(self, lower: Optional[float] = None, upper: Optional[float] = None,
                start: Optional[float] = None, end: Optional[float] = None, n: int = 100,
                groups: Optional[Iterable[Any]] = None) -> "Spline":
        """
        Keep the spline values within ``[lower, upper]``.

        The spline is checked on a grid of `n` points plus its knots: exact for
        piecewise-linear splines, on the grid for smoother ones.

        Parameters
        ----------
        lower : float, default=None
            Lower bound (none if None).
        upper : float, default=None
            Upper bound (none if None).
        start : float, default=None
            Only from this value of the feature on (domain start if None).
        end : float, default=None
            Only up to this value of the feature (domain end if None).
        n : int, default=100
            Number of grid points.
        groups : Iterable, default=None
            For splines with `by`: the groups to constrain (all if None).

        Returns
        -------
        Spline
            The spline itself, to allow chaining.
        """
        return self._add_constraint(Bound(lower=lower, upper=upper, n=n, start=start, end=end, groups=groups))

    def passes_through(self, points: Sequence[Tuple[Any, float]],
                       groups: Optional[Iterable[Any]] = None) -> "Spline":
        """
        Force the spline through the given points.

        Parameters
        ----------
        points : Sequence[Tuple[Any, float]]
            List of ``(x, y)`` tuples, e.g. ``[(5, 2), (1, -1)]``.
        groups : Iterable, default=None
            For splines with `by`: the groups to constrain (all if None).

        Returns
        -------
        Spline
            The spline itself, to allow chaining.
        """
        return self._add_constraint(Anchor(points=points, groups=groups))

    def _add_constraint(self, constraint: Constraint) -> "Spline":
        """Validate a constraint against this spline type and store it."""
        if constraint.method not in self.supported_constraints:
            supported = [m for m in CONSTRAINT_METHODS if m in self.supported_constraints]
            raise TypeError(
                f"{type(self).__name__} does not support {constraint.method}(); "
                f"supported: {', '.join(supported) if supported else 'none'}."
            )
        if constraint.groups is not None and self.by is None:
            raise ValueError(f"{constraint.method}(groups=...) needs a spline with a `by` column.")
        self._validate_constraint(constraint)
        self._constraints.append(constraint)
        return self

    def _validate_constraint(self, constraint: Constraint) -> None:
        """Hook for spline-specific checks when a constraint is added."""

    def add_constraint(self, *constraints: Constraint) -> "Spline":
        """
        Add constraint objects (deprecated).

        .. deprecated::
            Use the spline methods instead: ``increasing``, ``decreasing``,
            ``convex``, ``concave``, ``bounded``, ``passes_through``.

        Parameters
        ----------
        *constraints : Constraint
            Constraint objects such as ``Monotonic()`` or ``Bound(lower=0)``.

        Returns
        -------
        Spline
            The spline itself, to allow chaining.
        """
        for c in constraints:
            if not isinstance(c, Constraint):
                raise TypeError(f"Expected a Constraint instance, got {type(c).__name__}")
            warnings.warn(
                f"add_constraint({type(c).__name__}(...)) is deprecated: use "
                f".{c.method}(...) on the spline instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self._add_constraint(c)
        return self

    def _coefficient_groups(self, groups: Optional[Iterable[Any]] = None) -> List[cp.Expression]:
        """
        Coefficient vectors to constrain: one per selected `by` group (or the only one).

        Parameters
        ----------
        groups : Iterable, default=None
            Group values; None selects every group.

        Returns
        -------
        List[cp.Expression]
            1D coefficient expressions.

        Raises
        ------
        ValueError
            If a requested group is not present in the training data.
        """
        variables = self._build_variables()
        if self.by is None or len(variables.shape) == 1:
            return [variables]
        if groups is None:
            return [variables[:, i] for i in range(variables.shape[1])]
        missing = [g for g in groups if g not in self._by_int_map]
        if missing:
            raise ValueError(
                f"Constraint groups {missing} not found in the training data of '{self.by}' "
                f"(available: {list(self._by_classes)})."
            )
        return [variables[:, self._by_int_map[g]] for g in groups]

    def _slope_constraints(self, v: cp.Expression, lo: float, hi: float, sign: int) -> list:
        """CVXPY constraints making ``sign * slope >= 0`` on ``[lo, hi]`` (override per spline)."""
        raise NotImplementedError(f"{type(self).__name__} has no slope constraints.")

    def _curvature_constraints(self, v: cp.Expression, lo: float, hi: float, sign: int) -> list:
        """CVXPY constraints making ``sign * curvature >= 0`` on ``[lo, hi]`` (override per spline)."""
        raise NotImplementedError(f"{type(self).__name__} has no curvature constraints.")

    def _breakpoints(self) -> np.ndarray:
        """Feature values where the spline may change slope abruptly (e.g. knots)."""
        return np.array([])

    def _bound_domain(self) -> Tuple[float, float]:
        """Default region checked by `bounded` when `start`/`end` are not given."""
        x_range = getattr(self, '_x_range', None)
        if x_range is None:
            raise ValueError(f"Cannot infer the domain of {self!r}: pass start and end to bounded().")
        return x_range

    def _bound_basis(self, start: Optional[float], end: Optional[float], n: int) -> np.ndarray:
        """Basis rows at which `bounded` checks the spline values."""
        if start is None or end is None:
            d_lo, d_hi = self._bound_domain()
            start = d_lo if start is None else start
            end = d_hi if end is None else end
        grid = np.linspace(start, end, n)
        knots = np.asarray(self._breakpoints(), dtype=float)
        grid = np.union1d(grid, knots[(knots >= start) & (knots <= end)])
        return self._build_basis(grid)

    def _check_constraint_conflicts(self) -> None:
        """
        Raise early on constraints that obviously contradict each other.

        Currently: a `passes_through` point outside a `bounded` range that covers it.

        Raises
        ------
        ValueError
            If a conflict is found.
        """
        anchors = [c for c in self._constraints if isinstance(c, Anchor)]
        bounds = [c for c in self._constraints if isinstance(c, Bound)]
        for a in anchors:
            for b in bounds:
                if a.groups is not None and b.groups is not None and not set(a.groups) & set(b.groups):
                    continue
                lo, hi = b.window
                for x, y in a.points:
                    inside = not isinstance(x, (int, float, np.number)) or lo <= x <= hi
                    if inside and ((b.lower is not None and y < b.lower) or (b.upper is not None and y > b.upper)):
                        raise ValueError(
                            f"Spline '{self.tag}': {a.describe(self.term)} conflicts with "
                            f"{b.describe(self.term)}."
                        )

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
            A 1D array of group identifiers (strings or integers).

        Returns
        -------
        np.ndarray
            A 2D binary numpy array of shape `(n_samples, n_classes)`.
        """
        if getattr(self, '_by_int_map', None) is not None:
            by_mapped = np.array([self._by_int_map.get(v, -1) for v in by])
        else:
            by_mapped = np.array(by).astype(int)
            
        n = len(by_mapped)
        out = np.zeros((n, len(self._by_classes)))
        mask = (by_mapped >= 0) & (by_mapped < len(self._by_classes))
        out[np.arange(n)[mask], by_mapped[mask]] = 1.0
        return out

    @staticmethod
    def _apply_mask(basis: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
        """Zero the basis rows outside `mask`.

        Rows are assigned rather than multiplied so that NaN features on
        masked-out rows do not propagate.
        """
        if mask is None:
            return basis
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (basis.shape[0],):
            raise ValueError(f"Mask of shape {mask.shape} does not match {basis.shape[0]} rows.")
        basis = np.array(basis, dtype=float, copy=True)
        basis[~mask] = 0.0
        return basis

    def __call__(self, x: np.ndarray, by: np.ndarray = None, mask: Optional[np.ndarray] = None) -> cp.Expression:
        """
        Evaluates the symbolic CVXPY spline expression for the given input `x`.

        Parameters
        ----------
        x : np.ndarray
            The 1D input feature array for the spline evaluation.
        by : np.ndarray, default=None
            The 1D integer-encoded grouping array, if the `by` argument is specified.
        mask : Optional[np.ndarray], default=None
            Boolean row mask; the spline is 0 on rows where it is False.

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

        basis = self._apply_mask(self._build_basis(x), mask)

        if by is None:
            return basis @ variables
        else:
            onehotby = self._build_one_hot_matrix(by=by)
            out = basis @ variables
            out = cp.multiply(out, onehotby)
            out = cp.sum(out, axis=1)
            return out


    def eval(self, x: np.ndarray, return_basis: bool = False, by: np.ndarray = None,
             mask: Optional[np.ndarray] = None) -> np.ndarray:
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
        mask : Optional[np.ndarray], default=None
            Boolean row mask; the spline is 0 on rows where it is False.

        Returns
        -------
        np.ndarray
            A 1D numpy array of shape `(n_samples,)` representing the predicted values.

        Raises
        ------
        ValueError
            If the spline has not been fitted and coefficients are not available.
        """
        coefficients = self.coefficients  # raises if not fitted
        basis = self._apply_mask(self._build_basis(x), mask)
        if return_basis:
            return basis

        if by is None:
            return basis @ coefficients
        else:
            onehotby = self._build_one_hot_matrix(by=by)
            out = basis @ coefficients
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
            # Keep the written order: `spline + model` puts the spline first.
            other.splines.insert(0, self)
            other._check_tags()
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
