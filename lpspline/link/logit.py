from ..optimizer import LpRegressor
from .base import Link, logit, sigmoid

class Sigmoid(Link):
    """Sigmoid/Logit link function (y = sigmoid(Xb))."""
    def __init__(self, regressor: LpRegressor):
        super().__init__(regressor, link=logit, inv_link=sigmoid)
