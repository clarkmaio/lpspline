"""Exp wrapper: ``y = exp(splines)``."""
import numpy as np

from ..optimizer import LpRegressor
from .base import Link


class Exp(Link):
    """
    Model ``y = exp(s_1(x) + s_2(x) + ...)``.

    The splines are fitted on ``log(y)``, so the target must be strictly positive.
    Predictions are ``exp`` of the sum of the splines.

    Example: ``Exp(bs("x", knots=5) + l("z"))``.

    Parameters
    ----------
    regressor : LpRegressor
        The additive spline model inside the exponential.
    """

    def __init__(self, regressor: LpRegressor):
        super().__init__(regressor, link=np.log, inv_link=np.exp)
