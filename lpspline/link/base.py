import numpy as np
import polars as pl
from typing import Callable, Optional, Union
from ..optimizer import LpRegressor

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def logit(p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return np.log(p / (1 - p))

class Link(LpRegressor):
    """
    A wrapper for LpRegressor that applies a link function to the target 
    during fitting and an inverse link function to the predictions.
    
    This allows for generalized additive models such as log-linear or logistic regression.
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
            The link function to apply to the target during fit (e.g., np.log).
            Defaults to identity if not provided.
        inv_link : Callable, optional
            The inverse link function to apply to predictions (e.g., np.exp).
            Defaults to identity if not provided.
        """
        self.regressor = regressor
        self.splines = regressor.splines
        
        # Initialize parent attributes if they exist
        self.problem = getattr(regressor, 'problem', None)
        self._summary_data = getattr(regressor, '_summary_data', None)
        self._status = getattr(regressor, '_status', None)
        
        self._link = link if link is not None else lambda x: x
        self._inv_link = inv_link if inv_link is not None else lambda x: x

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
        return getattr(self.regressor, name)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.regressor})"
