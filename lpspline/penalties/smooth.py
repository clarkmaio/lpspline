import cvxpy as cp
from .base import Penalty
from ..spline import Spline

class Ridge(Penalty):
    """
    L2 Ridge Regularization targeting coefficient smoothness.
    """
    def __init__(self, alpha: float = 1.0):
        """
        Initialize the Ridge penalty structure.

        Parameters
        ----------
        alpha : float, default=1.0
            The optimization severity weighting.
        """
        self._alpha = alpha

    @property
    def alpha(self) -> float:
        """
        Returns the scaling severity.

        Returns
        -------
        float
            Penalty tuning constant.
        """
        return self._alpha

    def build_penalty(self, s: Spline) -> list:
        """
        Creates a Ridge penalty: $alpha * \\sum v^2$.

        Parameters
        ----------
        s : Spline
            The targeted function modeling bounds.
        
        Returns
        -------
        list
            Returns a list of CVXPY expressions to be mathematically subtracted/added 
            into the objective solver metric.
        """
        penalties = []
        for var in s._variables:
            penalties.append(self.alpha * cp.sum_squares(var))
        return penalties



class Lasso(Penalty):
    """
    L1 Lasso Regularization targeting coefficient sparsity.
    """
    def __init__(self, alpha: float = 1.0):
        """
        Initialize the Lasso penalty structure.

        Parameters
        ----------
        alpha : float, default=1.0
            The optimization severity weighting scaling absolute value costs.
        """
        self._alpha = alpha

    @property
    def alpha(self) -> float:
        """
        Returns the scaling severity.

        Returns
        -------
        float
            Penalty tuning constant.
        """
        return self._alpha

    def build_penalty(self, s: Spline) -> list:
        """
        Creates a Lasso penalty sum sequence evaluating $alpha * \\sum |v|$.
        
        Parameters
        ----------
        s : Spline
            The targeted function modeling bounds.

        Returns
        -------
        list
            A list of CVXPY absolute weighting expressions applied to the core equation metric.
        """
        penalties = []
        for var in s._variables:
            penalties.append(self.alpha * cp.sum_abs(var))
        return penalties
