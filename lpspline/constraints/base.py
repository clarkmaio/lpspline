import abc

class Constraint(abc.ABC):
    """
    Abstract base class defining the shape constraint interface.
    """
    @abc.abstractmethod
    def build_constraint(self, s) -> list:
        """
        Builds CVXPY constraints functionally mapping to the given spline structure.

        Parameters
        ----------
        s : Spline
            The initialized Spline instance to construct constraints upon.

        Returns
        -------
        list
            A list containing CVXPY constraint objects.
        """
        pass
