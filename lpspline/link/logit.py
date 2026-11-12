"""Sigmoid wrapper: ``y = sigmoid(splines)``."""
from ..optimizer import LpRegressor
from .base import Link, logit, sigmoid


class Sigmoid(Link):
    """
    Model ``y = sigmoid(s_1(x) + s_2(x) + ...)``.

    The splines are fitted on ``logit(y)``, so the target must lie in ``(0, 1)``.
    Predictions are ``sigmoid`` of the sum of the splines.

    Example: ``Sigmoid(bs("x", knots=5) + l("z"))``.

    Parameters
    ----------
    regressor : LpRegressor
        The additive spline model inside the sigmoid.
    """

    def __init__(self, regressor: LpRegressor):
        super().__init__(regressor, link=logit, inv_link=sigmoid)
