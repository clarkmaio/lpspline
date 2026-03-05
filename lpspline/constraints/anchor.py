import cvxpy as cp
import numpy as np
from typing import List
from .base import Constraint

class Anchor(Constraint):
    """
    Anchor constraint binding the curve to pass explicitly through specified coordinate points.
    """
    def __init__(self, *args):
        """
        Initialize the Anchor constraint.

        Parameters
        ----------
        *args : tuple
            A sequence of exactly (x, y) tuples defining coordinates the spline must intersect.

        Raises
        ------
        ValueError
            If arguments are empty or not formatted explicitly as structural (x, y) coordinates.
        """
        self.xy = args
            
        if len(self.xy) == 0:
            raise ValueError("xy must not be empty")

        for point in self.xy:
            if len(point) != 2:
                raise ValueError("Each point must be a tuple of (x, y)")
            

    def build_constraint(self, s) -> list:
        """
        Constructs CVXPY equality conditions restricting basis sums at anchoring points.

        Parameters
        ----------
        s : Spline
            The Spline component applying this positional constraint.

        Returns
        -------
        list
            A list specifying `basis @ variable == y` strict equalities.
        """
        constraints = []
        basis = s._build_basis(np.array([x for x, _ in self.xy]))
        variables = s._build_variables()
        M = len(s._by_classes) if getattr(s, 'by', None) is not None else 1
        
        for c in range(M):
            v_chunk = variables[:, c] if getattr(s, 'by', None) is not None else variables
            for i in range(len(self.xy)):
                constraints.append(basis[i] @ v_chunk == self.xy[i][1])
            
        return constraints
