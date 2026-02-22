import abc
from typing import List
from ..spline import Spline
from cvxpy import Expression

class Penalty(abc.ABC):
    @abc.abstractmethod
    def build_penalty(self, s: Spline) -> List[Expression]:
        """
        Build cvxpy penalty components for the given spline.
        """
        pass
