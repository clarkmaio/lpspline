

import matplotlib.pyplot as plt 
import numpy as np
import pimpmyplot as pmp
import polars as pl
from .optimizer import LpRegressor
from .spline.factor import Factor


def _plot_partial_residuals(ax, spline, X, y, components, i, comp_vals, x_vals):
    y_arr = y.to_numpy() if isinstance(y, pl.Series) else np.asarray(y)
    other_components_sum = np.sum(components, axis=1) - comp_vals
    partial_resid = y_arr - other_components_sum
    
    if getattr(spline, '_by_classes', None) is not None:
        for j, c in enumerate(spline._by_classes):
            class_filter_idx = np.where(X[spline._by] == c)[0]
            if class_filter_idx.size > 0:
                ax.scatter(x_vals[class_filter_idx], partial_resid[class_filter_idx], 
                           color=f'C{j % 10}', alpha=0.5, s=10, marker='.')
    else:
        ax.scatter(x_vals, partial_resid, color='black', alpha=0.5, s=10, marker='.')

def _plot_spline(ax, spline, X, x_vals, comp_vals, color):
    idx = np.argsort(x_vals)
    if isinstance(spline, Factor):
        ax.scatter(x_vals, comp_vals, color=color, s=10, marker='o', label=spline.tag)
    elif spline._by_classes is None:
        ax.plot(x_vals[idx], comp_vals[idx], linestyle='-', color=color, linewidth=4, label=spline.tag)
    else:
        for j, c in enumerate(spline._by_classes):
            class_filter_idx = np.where(X[spline._by] == c)[0]
            if class_filter_idx.size > 0:
                idx = np.argsort(x_vals[class_filter_idx])
                ax.plot(x_vals[class_filter_idx][idx], 
                        comp_vals[class_filter_idx][idx], 
                        linestyle='-', color=f'C{j % 10}', linewidth=2, label=f'{spline.tag} {c}')

def plot_diagnostic(model: LpRegressor, X: pl.DataFrame, ncols: int = 4, y: pl.Series = None):
    """
    Generate a diagnostic plot rendering the individually learned spline components natively.

    Dynamically sizes subplot grids according to the mathematical component complexity 
    of the underlying optimized model.

    Parameters
    ----------
    model : LpRegressor
        An LPSpline regression model object which has completed the `fit()` cycle.
    X : pl.DataFrame
        A Polars DataFrame containing the predictive feature fields.
    ncols : int, default=4
        The maximum number of subplots generated per row.
    y : pl.Series, default=None
        Optional true response series. If provided, calculates and overlays partial residuals.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The rendered Matplotlib figure window.
    axes : numpy.ndarray
        Array containing the individual subplot Matplotlib Axes.
    """
    components = model.predict(X, return_components=True)

    n_splines = len(model.splines)
    n_cols = min(n_splines, ncols)
    n_rows = int(np.ceil(n_splines / ncols)) if n_splines > 0 else 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    
    if isinstance(axes, plt.Axes):
        axes = np.array([axes])
    axes = axes.flatten()

    for i, spline in enumerate(model.splines):
        feature = spline.term
        x_vals = X[feature].to_numpy()
        comp_vals = components[:, i]
        color = f'C{i % 10}'
        
        if y is not None:
            _plot_partial_residuals(
                ax=axes[i], 
                spline=spline, 
                X=X, 
                y=y, 
                components=components, 
                i=i, 
                comp_vals=comp_vals, 
                x_vals=x_vals
            )
        
        _plot_spline(
            ax=axes[i], 
            spline=spline, 
            X=X, 
            x_vals=x_vals, 
            comp_vals=comp_vals, 
            color=color
        )

        axes[i].set_xlabel(feature)
        pmp.remove_axis('top', 'right', ax=axes[i])
        pmp.bullet_grid(stepinch=.3, ax=axes[i], alpha=0.4)
        pmp.legend(ax=axes[i], loc='upper left', ncol=1)

    # Hide any unused subplots
    for j in range(n_splines, len(axes)):
        fig.delaxes(axes[j])
    
    fig.suptitle("Diagnostic", fontsize=20, y=1.02)    
    fig.tight_layout()
    
    return fig, axes