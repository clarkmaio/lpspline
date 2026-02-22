import cvxpy as cp
from .base import Penalty
from ..spline import Spline

class Ridge(Penalty):
    def __init__(self, alpha: float = 1.0):
        self._alpha = alpha

    @property
    def alpha(self) -> float:
        return self._alpha

    def build_penalty(self, s: Spline) -> list:
        """
        Creates a Ridge penalty: alpha * sum(variables**2).
        Returns a list of cvxpy expressions to be added to the objective.
        """
        penalties = []
        for var in s._variables:
            penalties.append(self.alpha * cp.sum_squares(var))
        return penalties



class Lasso(Penalty):
    def __init__(self, alpha: float = 1.0):
        self._alpha = alpha

    @property
    def alpha(self) -> float:
        return self._alpha

    def build_penalty(self, s: Spline) -> list:
        """
        Creates a Lasso penalty: alpha * sum(abs(variables)).
        Returns a list of cvxpy expressions to be added to the objective.
        """
        penalties = []
        for var in s._variables:
            penalties.append(self.alpha * cp.sum_abs(var))
        return penalties
