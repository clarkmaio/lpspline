import numpy as np
from ..optimizer import LpRegressor
from .base import Link

class Log(Link):
    """Log link function (y = exp(Xb))."""
    def __init__(self, regressor: LpRegressor):
        super().__init__(regressor, link=np.log, inv_link=np.exp)
