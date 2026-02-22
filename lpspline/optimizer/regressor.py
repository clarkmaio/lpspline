
import polars as pl
import cvxpy as cp
import numpy as np
from typing import List, Optional, Union, Dict, Any, Tuple
from ..spline import base as base_spline
from .summary import print_summary

class LpRegressor:
    """
    LpRegressor class to fit linear spline models using convex optimization.
    """

    def __init__(self, splines: Union["base_spline.Spline", List["base_spline.Spline"]]):
        """
        Initialize the LpRegressor.

        Args:
            splines: A single Spline object or a list of Spline objects.
        """
        if isinstance(splines, base_spline.Spline):
            self.splines = [splines]
        else:
            self.splines = splines
        
        self._check_tags()
        self.problem: Optional[cp.Problem] = None

    def _check_tags(self):
        """
        Check if all splines have unique tags.
        """
        tags = [spline.tag for spline in self.splines]
        if len(tags) != len(set(tags)):
            raise ValueError("All splines must have unique tags.")


    def get_spline(self, tag: str) -> base_spline.Spline:
        """
        Returns the spline with the given tag.
        """
        for spline in self.splines:
            if spline.tag == tag:
                return spline
        raise ValueError(f"Spline with tag '{tag}' not found.")
        

    def fit(self, X: pl.DataFrame, y: pl.Series) -> None:
        """
        Fit the additive model to the data.

        Minimizes the L2 norm of the difference between the sum of spline terms and the target y.
        ||Sum(Spline_i(X)) - y||_2

        Args:
            X: Input DataFrame containing the features.
            y: Target Series.
        
        Raises:
            ValueError: If no splines are provided or if required columns are missing.
        """
        self._validate_input(X)

        total_expression, summary_data = self._build_model_expression(X)
        
        self._solve_problem(total_expression, y)        
        print_summary(summary_data, self.problem)

    def predict(self, X: pl.DataFrame, return_components: bool = False) -> np.ndarray:
        """
        Predict target values for new data.

        Args:
            X: Input DataFrame.
            return_components: If True, returns a matrix where each column corresponds to a spline's output.

        Returns:
            Numpy array of predictions. 
            If return_components is True, shape (n_samples, n_splines).
            Else, shape (n_samples,).
        
        Raises:
            ValueError: If required columns are missing.
        """
        if return_components:
            return self._predict_components(X)
        return self._predict_total(X)

    def _predict_components(self, X: pl.DataFrame) -> np.ndarray:
        """Calculate predictions for each spline individually."""
        n_samples = len(X)
        components = np.zeros((n_samples, len(self.splines)))
        for i, spline in enumerate(self.splines):
            components[:, i] = self._evaluate_spline(spline, X)
        return components

    def _predict_total(self, X: pl.DataFrame) -> np.ndarray:
        """Calculate total predictions by summing all spline components."""
        n_samples = len(X)
        total_value = np.zeros(n_samples)
        for spline in self.splines:
            total_value += self._evaluate_spline(spline, X)
        return total_value

    def _evaluate_spline(self, spline: "base_spline.Spline", X: pl.DataFrame) -> np.ndarray:
        """Evaluate a single spline on the input data."""
        self._validate_term_in_dataframe(spline.term, X)
        x_data = X[spline.term].to_numpy()
        
        spline_expr = spline(x_data)
        spline_val = spline_expr.value
        
        if spline_val is None:
             print(f"Warning: Spline for term {spline.term} has no value. Using zeros.")
             return np.zeros(len(X))
             
        return spline_val

    def __add__(self, other: Union["base_spline.Spline", "LpRegressor"]) -> "LpRegressor":
        """
        Add a Spline or another LpRegressor to this LpRegressor.
        Allows syntax like: model = spline1 + spline2
        """
        if isinstance(other, base_spline.Spline):
            self.splines.append(other)
            return self
        elif isinstance(other, LpRegressor):
            self.splines.extend(other.splines)
            return self
        else:
             raise TypeError(f"Cannot add LpRegressor and {type(other)}")
    
    def __repr__(self):
        return f"LpRegressor(splines={self.splines})"

    def _validate_input(self, X: pl.DataFrame) -> None:
        """Ensure there are splines to fit."""
        if not self.splines:
             raise ValueError("No splines to fit.")

    def _validate_term_in_dataframe(self, term: str, X: pl.DataFrame) -> None:
        """Check if the term exists in the DataFrame columns."""
        if term not in X.columns:
                raise ValueError(f"Term {term} not found in input DataFrame columns: {X.columns}")

    def _build_model_expression(self, X: pl.DataFrame) -> Tuple[cp.Expression, List[Dict[str, Any]]]:
        """
        Construct the cvxpy expression for the model and collect summary info.

        Returns:
            Tuple containing the total cvxpy expression and a list of summary dictionaries.
        """
        total_expression = 0
        summary_data = []

        for spline in self.splines:
            self._validate_term_in_dataframe(spline.term, X)
            
            x_data = X[spline.term].to_numpy()
            spline_expr = spline(x_data)
            
            # Collect info for summary
            num_params = sum(v.size for v in spline._variables)
            constraint_names = [type(c).__name__ for c in spline.constraints]
            constraints_str = ", ".join(constraint_names) if constraint_names else "None"
            summary_data.append({
                "Spline Type": type(spline).__name__,
                "Term": spline.term,
                "Parameters": num_params,
                "Constraints": constraints_str
            })
            
            total_expression += spline_expr
            
        return total_expression, summary_data

    def _solve_problem(self, expression: cp.Expression, y: pl.Series) -> None:
        """
        Set up and solve the convex optimization problem.
        """
        y_np = y.to_numpy()
        objective = cp.Minimize(cp.sum_squares(expression - y_np))
        
        all_constraints = []
        for spline in self.splines:
            for c in spline.constraints:
                all_constraints.extend(c.build_constraint(spline))
        
        self.problem = cp.Problem(objective, all_constraints)
        self.problem.solve()

