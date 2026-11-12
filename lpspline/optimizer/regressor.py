
import polars as pl
import cvxpy as cp
import numpy as np
import pickle
import pathlib
import warnings
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
        self._summary_data = None
        self._status: Optional[str] = None

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
        

    def fit(self, X: pl.DataFrame, y: pl.Series, summary: bool = True,
            solver: Optional[str] = None, **solver_kwargs: Any) -> None:
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
        summary : bool, default=True
            Whether to print the model summary after fitting.
        solver : Optional[str], default=None
            CVXPY solver name. Defaults to CLARABEL when installed: the CVXPY default
            for these problems (OSQP) frequently stops at `user_limit` or
            `optimal_inaccurate`, leaving constraints only approximately satisfied.
        **solver_kwargs : Any
            Extra keyword arguments forwarded to `cvxpy.Problem.solve`.

        Raises
        ------
        ValueError
            If no splines were initiated or structural dependencies are incorrectly verified.
        """
        self._validate_input(X)

        fit_masks = []
        for spline in self.splines:
            self._validate_term_in_dataframe(spline.term, X)
            if spline.by is not None:
                self._validate_term_in_dataframe(spline.by, X)

            fit_mask = spline.fit_mask(X)
            if not fit_mask.any():
                raise ValueError(
                    f"Spline '{spline.tag}' has no rows to train on: "
                    "train_on/active_where exclude every row."
                )
            self._warn_active_outside_training(spline, X)
            fit_masks.append(fit_mask)

            # Knots, period, by groups and classes come from the training rows only.
            x = X[spline.term].to_numpy()[fit_mask]
            if spline.by is not None:
                spline.init_spline(x, by=X[spline.by].to_numpy()[fit_mask])
            else:
                spline.init_spline(x)
            spline._check_constraint_conflicts()

        total_expression, self._summary_data = self._build_model_expression(X, fit_masks)
        
        self._solve_problem(total_expression, y, solver=solver, **solver_kwargs)
        self._status = self.problem.status
        if self._status != cp.OPTIMAL:
            hint = (" The constraints may contradict each other or the data."
                    if self._status in (cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE) else "")
            warnings.warn(
                f"Optimization finished with status '{self._status}'; "
                f"coefficients may be inaccurate or missing.{hint}",
                RuntimeWarning,
                stacklevel=2,
            )
        if summary:
            self.summary()

    @staticmethod
    def _warn_active_outside_training(spline: "base_spline.Spline", X: pl.DataFrame) -> None:
        """Warn when `train_on` leaves rows that the spline will predict on at inference."""
        if spline.train_condition is None:
            return
        outside = spline.active_mask(X) & ~spline.train_mask(X)
        if outside.any():
            warnings.warn(
                f"Spline '{spline.tag}' is active on {int(outside.sum())} rows outside "
                f"train_on({spline.train_condition}): it is 0 there during fit but will "
                "contribute when predicting them. Add active_where(...) to switch it off "
                "on those rows if that is not intended.",
                UserWarning,
                stacklevel=3,
            )

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
        # Only `active_where` applies at inference; `train_on` is a fit-time setting.
        mask = spline.active_mask(X) if spline.active_condition is not None else None

        if spline.by is not None:
            by = X[spline.by].to_numpy()
            spline_val = spline.eval(x, by=by, mask=mask)
        else:
            spline_val = spline.eval(x, mask=mask)
        
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
        elif isinstance(other, LpRegressor):
            self.splines.extend(other.splines)
        else:
            raise TypeError(f"Cannot add LpRegressor and {type(other)}")
        self._check_tags()
        return self

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

    def _build_model_expression(self, X: pl.DataFrame,
                                fit_masks: Optional[List[np.ndarray]] = None
                                ) -> Tuple[cp.Expression, List[Dict[str, Any]]]:
        """
        Construct the global structural mathematical logic natively isolating expressions targeting individual component matrices.

        Parameters
        ----------
        X : pl.DataFrame
            Training data.
        fit_masks : Optional[List[np.ndarray]], default=None
            One boolean row mask per spline (`train_on` AND `active_where`);
            each spline is 0 outside its mask. None means all rows.

        Returns
        -------
        Tuple[cp.Expression, List[Dict[str, Any]]]
            Combination defining explicit numerical constraints modeling logic equations, matched along natively sequential lists evaluating descriptive representations for final output presentation reporting.
        """
        total_expression = 0
        summary_data = []

        if fit_masks is None:
            fit_masks = [None] * len(self.splines)

        for spline, mask in zip(self.splines, fit_masks):
            self._validate_term_in_dataframe(spline.term, X)
            x_data = X[spline.term].to_numpy()

            if spline.by is not None:
                by_data = X[spline.by].to_numpy()
                spline_expr = spline(x_data, by=by_data, mask=mask)
            else:
                spline_expr = spline(x_data, mask=mask)

            # Collect info for summary
            num_params = spline.variables.size
            n_rows = len(X) if mask is None else int(mask.sum())
            if n_rows < num_params:
                warnings.warn(
                    f"Spline '{spline.tag}' trains on {n_rows} rows but has {num_params} "
                    "parameters: its coefficients are weakly determined.",
                    UserWarning,
                    stacklevel=3,
                )
            constraint_names = [c.method for c in spline.constraints]
            constraints_str = ", ".join(constraint_names) if constraint_names else "None"
            penalty_names = [type(p).__name__ for p in getattr(spline, 'penalties', [])]
            penalties_str = ", ".join(penalty_names) if penalty_names else "None"
            
            summary_data.append({
                "Spline Type": type(spline).__name__,
                "Term": spline.term,
                "Tag": spline.tag,
                "Parameters": num_params,
                "Constraints": constraints_str,
                "Penalties": penalties_str,
                "Constraint details": [c.describe(spline.term) for c in spline.constraints],
                "Train on": None if spline.train_condition is None else str(spline.train_condition),
                "Active where": None if spline.active_condition is None else str(spline.active_condition),
                "Rows": n_rows,
            })
            
            total_expression += spline_expr
            
        return total_expression, summary_data

    def _solve_problem(self, expression: cp.Expression, y: pl.Series,
                       solver: Optional[str] = None, **solver_kwargs: Any) -> None:
        """
        Set up and solve the convex optimization problem.

        Parameters
        ----------
        expression : cp.Expression
            Sum of all spline expressions.
        y : pl.Series
            Target values.
        solver : Optional[str], default=None
            CVXPY solver name; CLARABEL is used when None and available.
        **solver_kwargs : Any
            Extra keyword arguments forwarded to `cvxpy.Problem.solve`.
        """
        y_np = np.asarray(y.to_numpy(), dtype=float)
        if len(y_np) != expression.shape[0]:
            raise ValueError(f"X has {expression.shape[0]} rows but y has {len(y_np)}.")
        
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
        if solver is None and cp.CLARABEL in cp.installed_solvers():
            solver = cp.CLARABEL
        self.problem.solve(solver=solver, **solver_kwargs)

    def save(self, path: Union[str, pathlib.Path]) -> None:
        """
        Save the model to a file.

        Parameters
        ----------
        path : Union[str, pathlib.Path]
            The path to the file where the model will be saved.
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

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


    def __getstate__(self) -> Dict[str, Any]:
        """Drop the cvxpy Problem when pickling: it is large and not needed to predict."""
        state = self.__dict__.copy()
        state['problem'] = None
        return state

    def summary(self) -> None:
        """
        Print the model summary.

        Raises
        ------
        ValueError
            If the model has not been fitted.
        """
        if self._summary_data is None:
            raise ValueError('Model has not been trained properly')
        print_summary(self._summary_data, self._status)

