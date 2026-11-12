"""Log wrapper: ``y = log(splines)``."""
import numpy as np

from ..optimizer import LpRegressor
from .base import Link


class Log(Link):
    """
    Model ``y = log(s_1(x) + s_2(x) + ...)``.

    The splines are fitted on ``exp(y)``. Predictions are ``log`` of the sum of
    the splines (NaN where that sum is not positive).

    Example: ``Log(bs("x", knots=5) + l("z"))``.

    Parameters
    ----------
    regressor : LpRegressor
        The additive spline model inside the logarithm.
    """

    def __init__(self, regressor: LpRegressor):
        super().__init__(regressor, link=np.exp, inv_link=np.log)
