import abc
from typing import List
from ..spline import Spline
from cvxpy import Expression

class Penalty(abc.ABC):
    """
    Abstract base class defining the algorithmic penalty interface.
    """
    @abc.abstractmethod
    def build_penalty(self, s: Spline) -> List[Expression]:
        """
        Builds CVXPY objective cost penalty combinations given the current spline.

        Parameters
        ----------
        s : Spline
            The Spline instance determining optimization variables.

        Returns
        -------
        List[Expression]
            A list containing numeric cost formulations.
        """
        pass
