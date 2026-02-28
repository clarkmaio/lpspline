import cvxpy as cp
import numpy as np
from typing import List
from .base import Constraint

class Anchor(Constraint):
    def __init__(self, *args):
        """
        xy: List of (x, y) tuples
        """
        self.xy = args
            
        if len(self.xy) == 0:
            raise ValueError("xy must not be empty")

        for point in self.xy:
            if len(point) != 2:
                raise ValueError("Each point must be a tuple of (x, y)")
            

    def build_constraint(self, s) -> list:
        constraints = []
        basis = s._build_basis(np.array([x for x, _ in self.xy]))
        variables = s._build_variables()[0]
        
        for i in range(len(self.xy)):
            constraints.append(basis[i] @ variables == self.xy[i][1])
            
        return constraints
