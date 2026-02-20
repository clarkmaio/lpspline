import abc

class Constraint(abc.ABC):
    @abc.abstractmethod
    def build_constraint(self, s) -> list:
        """
        Build cvxpy constraints for the given spline.
        """
        pass
