
import polars as pl
import cvxpy as cp
import numpy as np
import pickle
import pathlib
import copy
from typing import List, Optional, Union, Dict, Any, Tuple
from ..spline import base as base_spline
from .summary import print_summary

class LpRegressor:
    """
    Algorithmic LpRegressor class fitting generalized additive spline models using convex optimization.
    """

    def __init__(self, splines: Union["base_spline.Spline", List["base_spline.Spline"]]):
        """
        Initialize the localized LpRegressor.

        Parameters
        ----------
        splines : Union[Spline, List[Spline]]
            A standalone Spline definition object or iterable combinations spanning the predictive framework.
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
        Returns the spline component isolated utilizing its corresponding explicit identifier.

        Parameters
        ----------
        tag : str
            The explicit string tag name applied to a spline.

        Returns
        -------
        Spline
            The specified matching component spline instance.
        
        Raises
        ------
        ValueError
            If the supplied specified tag isn't contained within the current configuration splines.
        """
        for spline in self.splines:
            if spline.tag == tag:
                return spline
        raise ValueError(f"Spline with tag '{tag}' not found.")
        

    def fit(self, X: pl.DataFrame, y: pl.Series) -> None:
        """
        Compute basis coefficients mapping combinations within additive splines.

        Minimizes the specified L2 norm determining the optimal linear composition evaluating structural distances.
        `||Sum(Spline_i(X)) - y||_2`

        Parameters
        ----------
        X : pl.DataFrame
            The independent predictive training feature frame subset containing all modeled keys.
        y : pl.Series
            Dependent observation labels associated mapping.

        Raises
        ------
        ValueError
            If no splines were initiated or structural dependencies are incorrectly verified.
        """
        self._validate_input(X)

        for spline in self.splines:

            if spline.by is not None:
                spline.init_spline(X[spline.term].to_numpy(), by=X[spline.by].to_numpy())
            else:
                spline.init_spline(X[spline.term].to_numpy())

        total_expression, summary_data = self._build_model_expression(X)
        
        self._solve_problem(total_expression, y)        
        print_summary(summary_data, self.problem)

    def predict(self, X: pl.DataFrame, return_components: bool = False) -> np.ndarray:
        """
        Predict target sequential observations for new domain instances evaluating trained coefficients.

        Parameters
        ----------
        X : pl.DataFrame
            Unseen independent predictors formatted natively identical to initial modeling.
        return_components : bool, default=False
            If True, calculates output sequentially isolated over all respective model components matrices.

        Returns
        -------
        np.ndarray
            Numpy array mapping evaluations across sequential instances.
            Returns shape `(n_samples, n_splines)` if returning explicitly defined components.
            Else, computes overall predicted structure `(n_samples, )`.
        
        Raises
        ------
        ValueError
            If structural dataframe column dependencies aren't accurately mirrored natively.
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
        
        if spline.by is not None:
            self._validate_term_in_dataframe(spline.by, X)

        x = X[spline.term].to_numpy()
        
        if spline.by is not None:
            by = X[spline.by].to_numpy()
            spline_val = spline.eval(x, by=by)
        else:
            spline_val = spline.eval(x)
        
        if spline_val is None:
             print(f"Warning: Spline for term {spline.term} has no value. Using zeros.")
             return np.zeros(len(X))
             
        return spline_val

    def __add__(self, other: Union["base_spline.Spline", "LpRegressor"]) -> "LpRegressor":
        """
        Appends Splines globally into compound configurations enabling straightforward summation chaining structures.
        Allows explicit configurations natively mimicking syntax constructs: `model = spline1 + spline2`.

        Parameters
        ----------
        other : Union[Spline, LpRegressor]
            Target instance to linearly concatenate into sequential memory configuration bindings.

        Returns
        -------
        LpRegressor
            The locally active appended LpRegressor combination.

        Raises
        ------
        TypeError
            If attempting explicit concatenation bypassing structurally approved model instances natively.
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
        Construct the global structural mathematical logic natively isolating expressions targeting individual component matrices.

        Returns
        -------
        Tuple[cp.Expression, List[Dict[str, Any]]]
            Combination defining explicit numerical constraints modeling logic equations, matched along natively sequential lists evaluating descriptive representations for final output presentation reporting.
        """
        total_expression = 0
        summary_data = []

        for spline in self.splines:
            self._validate_term_in_dataframe(spline.term, X)
            
            if spline.by is not None:
                x_data = X[spline.term].to_numpy()
                by_data = X[spline.by].to_numpy()
                spline_expr = spline(x_data, by=by_data)
            else:
                x_data = X[spline.term].to_numpy()
                spline_expr = spline(x_data)
            
            # Collect info for summary
            num_params = sum(v.size for v in spline._variables)
            constraint_names = [type(c).__name__ for c in spline.constraints]
            constraints_str = ", ".join(constraint_names) if constraint_names else "None"
            penalty_names = [type(p).__name__ for p in getattr(spline, 'penalties', [])]
            penalties_str = ", ".join(penalty_names) if penalty_names else "None"
            
            summary_data.append({
                "Spline Type": type(spline).__name__,
                "Term": spline.term,
                "Tag": spline.tag,
                "Parameters": num_params,
                "Constraints": constraints_str,
                "Penalties": penalties_str
            })
            
            total_expression += spline_expr
            
        return total_expression, summary_data

    def _solve_problem(self, expression: cp.Expression, y: pl.Series) -> None:
        """
        Set up and solve the convex optimization problem.
        """
        y_np = y.to_numpy()
        
        main_loss = cp.sum_squares(expression - y_np)
        penalty_loss = 0
        
        all_constraints = []
        for spline in self.splines:
            for c in spline.constraints:
                all_constraints.extend(c.build_constraint(spline))
            for p in getattr(spline, 'penalties', []):
                for p_expr in p.build_penalty(spline):
                    penalty_loss += p_expr
                    
        objective = cp.Minimize(main_loss + penalty_loss)
        
        self.problem = cp.Problem(objective, all_constraints)
        self.problem.solve()

    def save(self, path: Union[str, pathlib.Path]) -> None:
        """
        Save the model to a file.

        Parameters
        ----------
        path : Union[str, pathlib.Path]
            The path to the file where the model will be saved.
        """
        obj_copy = copy.copy(self)
        obj_copy.problem = None
        
        with open(path, 'wb') as f:
            pickle.dump(obj_copy, f)

    @staticmethod
    def load(path: Union[str, pathlib.Path]) -> "LpRegressor":
        """
        Load a model from a file.

        Parameters
        ----------
        path : Union[str, pathlib.Path]
            The path to the file from which the model will be loaded.

        Returns
        -------
        LpRegressor
            The loaded model instance.
        """
        with open(path, 'rb') as f:
            return pickle.load(f)

