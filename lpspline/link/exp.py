import numpy as np
from ..optimizer import LpRegressor
from .base import Link

class Exp(Link):
    """Exp link function (y = log(Xb))."""
    def __init__(self, regressor: LpRegressor):
        super().__init__(regressor, link=np.exp, inv_link=np.log)
