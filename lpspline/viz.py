

import matplotlib.pyplot as plt 
import numpy as np
import pimpmyplot as pmp
import polars as pl
from .optimizer import LpRegressor


def plot_components(model: LpRegressor, X: pl.DataFrame, ncols: int = 4, y: pl.Series = None):
    """
    Create a plot of the learned spline components.
    Create a plot with at most 4 columns. Modify subplot size according to number of model splines
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
        
        # Sort for plotting lines nicely
        idx = np.argsort(x_vals)
        color = f'C{i % 10}'
        
        if y is not None:
            y_arr = y.to_numpy() if isinstance(y, pl.Series) else np.asarray(y)
            other_components_sum = np.sum(components, axis=1) - comp_vals
            partial_resid = y_arr - other_components_sum
            axes[i].scatter(x_vals, partial_resid, color='black', alpha=0.1, s=10, marker='.')
        
        # If categorical style it differently
        if len(np.unique(x_vals)) < 10:
            axes[i].plot(x_vals[idx], comp_vals[idx], 'o', color=color, markersize=8, label=spline.tag)
        else:
            axes[i].plot(x_vals[idx], comp_vals[idx], '-', color=color, linewidth=2, label=spline.tag)
            
        axes[i].set_xlabel(feature)
        pmp.remove_axis('top', 'right', ax=axes[i])
        pmp.bullet_grid(stepinch=.3, ax=axes[i])
        pmp.legend(ax=axes[i], loc='upper left')

    # Hide any unused subplots
    for j in range(n_splines, len(axes)):
        fig.delaxes(axes[j])
    

    fig.suptitle("LPSpline Components", fontsize=20, y=1.02)    
    fig.tight_layout()
    
    return fig, axes