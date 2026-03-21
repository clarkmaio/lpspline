
import numpy as np
import cvxpy as cp
from typing import List, Optional, Union
from .base import Spline

class BSpline(Spline):
    """
    B-Spline implementation using the Cox-de Boor recursion algorithm.
    """
    def __init__(self, term: str, knots: Union[int, np.ndarray], degree: int = 3, by: Optional[str] = None, tag: Optional[str] = 'bspline'):
        """
        Initialize the B-Spline.
        
        Parameters
        ----------
        term : str
            The column name in the DataFrame this spline models.
        knots : Union[int, np.ndarray]
            List or array of knot positions, or an integer specifying the number of knots to generate automatically.
        degree : int, default=3
            The degree of the spline (e.g., 3 for cubic splines).
        by : Optional[str], default=None
            The column name denoting group classes if interaction modeling is required.
        tag : Optional[str], default='bspline'
            A tag for identification.
        """
        super().__init__(term=term, tag=tag)
        self._knots = knots
        self._degree = degree
        self._by = by
        self._by_classes = None
        self._variables = []

    @property
    def degree(self) -> int:
        """
        Returns the degree of the spline.

        Returns
        -------
        int
            The polynomial degree.
        """
        return self._degree

    @property
    def by(self) -> Optional[str]:
        """
        Returns the grouping variable column name.

        Returns
        -------
        Optional[str]
            The name of the grouping column, or None.
        """
        return self._by

    @property
    def knots(self) -> Union[int, np.ndarray]:
        """
        Returns the knot definitions for this spline.

        Returns
        -------
        Union[int, np.ndarray]
            The array of knot locations or integer count.
        """
        return self._knots
        

    def init_spline(self, x: np.ndarray, by: np.ndarray = None):
        """
        Initializes the spline parameters based on input data.

        If `knots` was initially provided as an integer, it is correctly mapped to 
        linearly spaced values spanning the minimum and maximum of the input array `x`.

        Parameters
        ----------
        x : np.ndarray
            The 1D input array of training features.
        by : np.ndarray, default=None
            The array of grouping values.
        """
        super().init_spline(x, by)
        if isinstance(self._knots, int):
            self._knots = np.linspace(np.min(x), np.max(x), self._knots)
        else:
            self._knots = np.sort(self._knots)


    def _pad_knots(self, knots: np.ndarray, degree: int) -> np.ndarray:
        """
        Pad the knots so the final splines span the provided input knots.

        This ensures the outer boundary conditions for the B-Spline calculations
        are satisfied. 

        Parameters
        ----------
        knots : np.ndarray
            The interior knot sequence.
        degree : int
            The polynomial degree.

        Returns
        -------
        np.ndarray
            The padded knot sequence.
        """
        t_input = knots
        k = degree

        lowstep = t_input[1] - t_input[0]
        highstep = t_input[-1] - t_input[-2]

        for i in range(k):
            t_input = np.insert(t_input, 0, t_input[0] - lowstep)
            t_input = np.insert(t_input, -1, t_input[-1] + highstep)
            
        return t_input

    def _build_basis(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Builds the B-Spline basis functions using Cox-de Boor recursion.
        
        Parameters
        ----------
        x : np.ndarray
            Input array of feature values to evaluate.
        **kwargs : dict
            Additional arguments.
            
        Returns
        -------
        np.ndarray
            Basis matrix with shape `(n_samples, n_basis_funcs)`.
        
        Raises
        ------
        ValueError
            If there are not enough knots relative to the specified degree.
        """
        t = self._pad_knots(self.knots, self.degree) # knots of degree 0 base basis
        k = self.degree
        
        x = np.array(x).flatten()
        n = len(x)
        m = len(t)
        
        num_basis = m - k - 1
        if num_basis <= 0:
             raise ValueError(f"Not enough knots for the given degree. Need len(knots) > degree + 1")

        b = self._initialize_basis(x, t, n, m)

        # Recursive step
        for p in range(1, k + 1):
            b = self._compute_next_degree_basis(b_prev=b, x=x, t=t, p=p, n=n, m=m)
        base_basis = b
        return base_basis


    def _initialize_basis(self, x: np.ndarray, t: np.ndarray, n: int, m: int) -> np.ndarray:
        """
        Initialize degree 0 basis functions.

        $B_{i,0}(x) = 1$ if $t_i \\leq x < t_{i+1}$, else 0.

        Parameters
        ----------
        x : np.ndarray
            Evaluation sequence.
        t : np.ndarray
            Padded knots sequence.
        n : int
            Number of observations in `x`.
        m : int
            Number of padded knots in `t`.

        Returns
        -------
        np.ndarray
            The degree zero base basis array.
        """
        b = np.zeros((n, m - 1))
        for i in range(m - 1):
            mask = (x >= t[i]) & (x < t[i+1])
            b[mask, i] = 1.0
        return b

    def _compute_next_degree_basis(self, b_prev: np.ndarray, x: np.ndarray, t: np.ndarray, p: int, n: int, m: int) -> np.ndarray:
        """
        Compute basis for degree `p` given basis for degree `p-1` using the Cox-de Boor recursive formula.

        Parameters
        ----------
        b_prev : np.ndarray
            The basis calculation from the previous degree iteration.
        x : np.ndarray
            The input evaluation locations.
        t : np.ndarray
            The padded knot sequence.
        p : int
            The current degree iterator.
        n : int
            Length of the evaluation sequence.
        m : int
            Length of the knots sequence.

        Returns
        -------
        np.ndarray
            The updated basis array for current degree iteration.
        """
        b_new = np.zeros((n, m - p - 1))
        for i in range(m - p - 1):
            term1 = self._compute_term(x=x, t=t, b_col=b_prev[:, i], i=i, p=p, term_type=1)
            term2 = self._compute_term(x=x, t=t, b_col=b_prev[:, i+1], i=i, p=p, term_type=2)
            b_new[:, i] = term1 + term2
        return b_new

    def _compute_term(self, x: np.ndarray, t: np.ndarray, b_col: np.ndarray, i: int, p: int, term_type: int) -> np.ndarray:
        r"""
        Compute a single fractional component term of the Cox-de Boor recursion formula.

        Type 1: ((x - t_i) / (t_{i+p} - t_i)) * B_{i, p-1}
        Type 2: ((t_{i+p+1} - x) / (t_{i+p+1} - t_{i+1})) * B_{i+1, p-1}

        Parameters
        ----------
        x : np.ndarray
            Evaluation array.
        t : np.ndarray
            Padded knot sequence.
        b_col : np.ndarray
            Single column extracted from the `p-1` degree basis matrix.
        i : int
            Basis and knot iterator index.
        p : int
            Current degree iterator.
        term_type : int
            Identifies whether calculating fractional component 1 or 2 in Cox-de-Boor.

        Returns
        -------
        np.ndarray
            A 1D array representing the weighted basis evaluation for this fractional segment.
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

    def _build_variables(self) -> cp.Variable:
        """
        Create CVXPY variables representing the spline coefficients.

        Returns
        -------
        cp.Variable
            A CVXPY Variable of shape `(n_knots + degree - 1, len(by_classes) if by else 1)`.
        """
        if not self._variables:
            basedim = len(self.knots) + self.degree - 1
            if self.by is None:
                self._variables = cp.Variable(shape=(basedim,), name=f"{self.term}_bspline")
            else:
                self._variables = cp.Variable(shape=(basedim, len(self._by_classes)), name=f"{self.term}_bspline")
        return self._variables



    def __repr__(self):
        # Convert numpy array to list for cleaner repr if it's small, or specific format
        k_list = self.knots.tolist() if isinstance(self.knots, np.ndarray) else self.knots
        return f"BSpline(term='{self.term}', degree={self.degree}, knots={k_list}, by={self._by})"
