import numpy as np
import polars as pl
from typing import Callable, Optional, Union
from ..optimizer import LpRegressor

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def _identity(x):
    return x

def logit(p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return np.log(p / (1 - p))

class Link(LpRegressor):
    """
    Wrap an additive spline model in a function: ``y = inv_link(s_1(x) + s_2(x) + ...)``.

    The wrapper is named after the function applied to the sum of the splines,
    e.g. ``Exp(model)`` means ``y = exp(model)``. Fitting applies `link` (the
    inverse of that function) to the target and fits the splines on it with
    least squares; predicting applies `inv_link` to the sum of the splines.

    Constraints act on the sum of the splines, before `inv_link`: for instance
    ``bounded(lower=0)`` inside ``Exp(...)`` means ``y >= 1``.
    """
    def __init__(
        self, 
        regressor: LpRegressor, 
        link: Optional[Callable] = None, 
        inv_link: Optional[Callable] = None
    ):
        """
        Initialize the Link wrapper.

        Parameters
        ----------
        regressor : LpRegressor
            The regressor instance to wrap.
        link : Callable, optional
            Applied to the target before fitting (e.g. ``np.log`` for ``Exp``).
            Defaults to identity.
        inv_link : Callable, optional
            Applied to the sum of the splines when predicting (e.g. ``np.exp`` for ``Exp``).
            Defaults to identity.
        """
        self.regressor = regressor
        self.splines = regressor.splines
        
        # Initialize parent attributes if they exist
        self.problem = getattr(regressor, 'problem', None)
        self._summary_data = getattr(regressor, '_summary_data', None)
        self._status = getattr(regressor, '_status', None)
        
        # Module-level identity (not a lambda) keeps the model picklable.
        self._link = link if link is not None else _identity
        self._inv_link = inv_link if inv_link is not None else _identity

    def link(self, x):
        """Apply the link function."""
        return self._link(x)

    def inv_link(self, x):
        """Apply the inverse link function."""
        return self._inv_link(x)

    def fit(self, X: pl.DataFrame, y: pl.Series, **kwargs) -> "Link":
        """
        Fit the model by transforming the target using the link function.
        """
        y_val = y.to_numpy()
        y_transformed = self.link(y_val)
        
        # LpRegressor.fit expects a pl.Series
        y_series = pl.Series(y.name, y_transformed)
        
        self.regressor.fit(X, y_series, **kwargs)
        
        # Sync state
        self.problem = self.regressor.problem
        self._summary_data = self.regressor._summary_data
        self._status = getattr(self.regressor, '_status', None)
        
        return self

    def predict(self, X: pl.DataFrame, return_components: bool = False) -> np.ndarray:
        """
        Predict by applying the inverse link function to the linear predictor.
        """
        res = self.regressor.predict(X, return_components=return_components)
        
        # Link will not be applied to individual components
        if return_components:
            return res
        
        return self.inv_link(res)

    def __getattr__(self, name):
        """Delegate attribute access to the wrapped regressor."""
        # Guard against infinite recursion while `regressor` is not yet set
        # (e.g. during unpickling/copying) and never delegate dunder lookups.
        if name == 'regressor' or (name.startswith('__') and name.endswith('__')):
            raise AttributeError(name)
        return getattr(self.regressor, name)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.regressor})"
