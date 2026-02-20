
import numpy as np
import cvxpy as cp
from typing import List
from .base import Spline

class BSpline(Spline):
    """
    B-Spline implementation using the Cox-de Boor recursion algorithm.
    """
    def __init__(self, term: str, knots: List[float], degree: int = 3):
        """
        Initialize the B-Spline.
        
        Args:
            term: The column name in the DataFrame.
            knots: List of knot positions.
            degree: The degree of the spline (default 3 for cubic).
        """
        super().__init__(term)
        self.knots = np.sort(np.array(knots))
        self.degree = degree
        self._variables = []

    def _build_basis(self, x: np.ndarray) -> np.ndarray:
        """
        Builds the B-Spline basis functions using Cox-de Boor recursion.
        
        Args:
            x: Input array of values.
            
        Returns:
            Basis matrix with shape (n_samples, n_basis_funcs).
        """
        t = self.knots
        k = self.degree
        x = np.array(x).flatten()
        n = len(x)
        m = len(t)
        
        num_basis = m - k - 1
        if num_basis <= 0:
             raise ValueError("Not enough knots for the given degree. Need len(knots) > degree + 1")

        b = self._initialize_basis(x, t, n, m)

        # Recursive step
        for p in range(1, k + 1):
            b = self._compute_next_degree_basis(b, x, t, p, n, m)
            
        return b

    def _initialize_basis(self, x: np.ndarray, t: np.ndarray, n: int, m: int) -> np.ndarray:
        """
        Initialize degree 0 basis functions.
        B_{i,0}(x) = 1 if t_i <= x < t_{i+1}, else 0.
        """
        b = np.zeros((n, m - 1))
        for i in range(m - 1):
            mask = (x >= t[i]) & (x < t[i+1])
            b[mask, i] = 1.0
        return b

    def _compute_next_degree_basis(self, b_prev: np.ndarray, x: np.ndarray, t: np.ndarray, p: int, n: int, m: int) -> np.ndarray:
        """
        Compute basis for degree p given basis for degree p-1 using Cox-de Boor formula.
        """
        b_new = np.zeros((n, m - p - 1))
        for i in range(m - p - 1):
            term1 = self._compute_term(x, t, b_prev[:, i], i, p, term_type=1)
            term2 = self._compute_term(x, t, b_prev[:, i+1], i, p, term_type=2)
            b_new[:, i] = term1 + term2
        return b_new

    def _compute_term(self, x: np.ndarray, t: np.ndarray, b_col: np.ndarray, i: int, p: int, term_type: int) -> np.ndarray:
        """
        Compute a single term of the recursion formula.
        Type 1: ((x - t_i) / (t_{i+p} - t_i)) * B_{i, p-1}
        Type 2: ((t_{i+p+1} - x) / (t_{i+p+1} - t_{i+1})) * B_{i+1, p-1}
        """
        if term_type == 1:
            numerator = x - t[i]
            denominator = t[i+p] - t[i]
        else:
            numerator = t[i+p+1] - x
            denominator = t[i+p+1] - t[i+1]
            
        if denominator == 0:
            return np.zeros_like(x)
        return (numerator / denominator) * b_col

    def _build_variables(self) -> List[cp.Variable]:
        """
        Create cvxpy variables for the spline coefficients.
        """
        if not self._variables:
            dim = len(self.knots) - self.degree - 1
            if dim <= 0:
                 raise ValueError(f"Invalid knots/degree config: len(knots)={len(self.knots)}, degree={self.degree}. Dim would be {dim}")
            self._variables = [cp.Variable(shape=(dim,), name=f"{self.term}_bspline")]
        return self._variables

    def __repr__(self):
        # Convert numpy array to list for cleaner repr if it's small, or specific format
        k_list = self.knots.tolist() if isinstance(self.knots, np.ndarray) else self.knots
        return f"BSpline(term='{self.term}', degree={self.degree}, knots={k_list})"
