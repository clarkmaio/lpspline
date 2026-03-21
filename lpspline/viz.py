

import matplotlib.pyplot as plt 
import numpy as np
import pimpmyplot as pmp
import polars as pl
from typing import List, Optional, Any

from .optimizer import LpRegressor
from .spline.factor import Factor

import altair as alt


def _plot_partial_residuals(ax: plt.Axes, spline: object, X: pl.DataFrame, y: pl.Series, 
                            components: np.ndarray, i: int, comp_vals: np.ndarray, x_vals: np.ndarray):
    """
    Scatter plot partial residuals for a specific spline component.

    Calculates partial residuals as: target - (total_prediction - current_spline_contribution).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The target axes for plotting.
    spline : object
        The spline component instance.
    X : polars.DataFrame
        The feature data used for plotting.
    y : polars.Series
        The actual target values.
    components : np.ndarray
        The pre-calculated individual spline components.
    i : int
        The index of the current spline in `components`.
    comp_vals : np.ndarray
        The specific predictions for this spline component.
    x_vals : np.ndarray
        The feature values for the spline's primary term.
    """
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

def _plot_spline(ax: plt.Axes, spline: object, X: pl.DataFrame, x_vals: np.ndarray, 
                 comp_vals: np.ndarray, color: str):
    """
    Render a single spline's fitted curve or scatter points (for Factors).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The target axes for plotting.
    spline : object
        The spline component instance.
    X : polars.DataFrame
        The feature data used for plotting.
    x_vals : np.ndarray
        The feature values for the spline's primary term.
    comp_vals : np.ndarray
        The specific predictions for this spline component.
    color : str
        The color to use for the primary spline line/points.
    """
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
        x_vals_raw = X[feature].to_numpy()
        comp_vals = components[:, i]
        color = f'C{i % 10}'
        
        if isinstance(spline, Factor):
            x_vals = np.array([spline._int_map.get(v, -1) for v in x_vals_raw])
        else:
            x_vals = x_vals_raw
        
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
        if isinstance(spline, Factor):
            axes[i].set_xticks(range(len(spline._classes)))
            axes[i].set_xticklabels(spline._classes)
        pmp.bullet_grid(stepinch=.3, ax=axes[i], alpha=0.4)
        pmp.legend(ax=axes[i], loc='upper left', ncol=1)

    # Hide any unused subplots
    for j in range(n_splines, len(axes)):
        fig.delaxes(axes[j])
    
    fig.suptitle("Diagnostic", fontsize=20, y=1.02)    
    fig.tight_layout()
    
    return fig, axes


def _prepare_interactive_data(model: LpRegressor, X: pl.DataFrame, y: pl.Series, xcol: str, spline_indices: List[int]) -> pl.DataFrame:
    """
    Helper to prepare a concatenated Polars DataFrame suitable for Altair plotting.

    Includes target, model predictions, sample IDs, and individual spline component 
    values and partial residuals.

    Parameters
    ----------
    model : LpRegressor
        The fitted regression model.
    X : pl.DataFrame
        Input features.
    y : pl.Series
        Target values.
    xcol : str
        Primary x-axis column for the top row plot.
    spline_indices : List[int]
        Indices of splines from the model to prepare data for.

    Returns
    -------
    pl.DataFrame
        A "wide" DataFrame containing all columns needed for interactive layering.
    """
    y_pred_total = model.predict(X)
    components = model.predict(X, return_components=True)
    
    # We need a unique ID for linked selection
    df_plot = X.select(pl.col(xcol)).with_columns([
        pl.Series("target", y),
        pl.Series("model", y_pred_total),
        pl.Series("__id__", np.arange(len(X)))
    ])

    for i in spline_indices:
        spline = model.splines[i]
        comp_vals = components[:, i]
        # Partial residual: y - sum_{j!=i} s_j(X) = y - (y_pred_total - s_i(X))
        resid_j = y.to_numpy() - (y_pred_total - comp_vals)
        
        df_plot = df_plot.with_columns([
            pl.Series(f"x_{spline.tag}", X[spline.term]),
            pl.Series(f"val_{spline.tag}", comp_vals),
            pl.Series(f"resid_{spline.tag}", resid_j)
        ])
    return df_plot

def _create_top_chart(df_plot: pl.DataFrame, xcol: str, selector: Any, width: int, height: int) -> alt.LayerChart:
    """
    Create the top plot showing target vs model with highlighting.

    Parameters
    ----------
    df_plot : polars.DataFrame
        The prepared plotting data.
    xcol : str
        The feature name for the x-axis.
    selector : Any
        The shared selection parameter for linked highlighting.
    width : int
        Width of the plot.
    height : int
        Height of the plot.

    Returns
    -------
    alt.LayerChart
        A layered chart containing the scatter plot and highlighted selection.
    """
    base_top = alt.Chart(df_plot).transform_fold(
        ['target', 'model'],
        as_=['Measurement', 'Value']
    ).mark_circle(opacity=0.6).encode(
        x=alt.X(xcol, title=xcol),
        y=alt.Y('Value:Q', title=''),
        color=alt.Color('Measurement:N', scale=alt.Scale(domain=['target', 'model'], range=['black', 'lime'])),
        tooltip=['__id__', xcol, 'target', 'model']
    ).properties(width=width, height=height).add_params(selector)

    highlight = alt.Chart(df_plot).mark_circle(color='red', size=200).encode(
        x=alt.X(xcol),
        y=alt.Y('model:Q'),
        opacity=alt.condition(selector, alt.value(1), alt.value(0))
    )
    return base_top + highlight

def _create_spline_chart(df_plot: pl.DataFrame, spline: object, selector: Any, width: int, height: int) -> alt.LayerChart:
    """
    Create a single spline subplot with residuals and highlight.

    Parameters
    ----------
    df_plot : polars.DataFrame
        The prepared plotting data.
    spline : object
        The spline component instance.
    selector : Any
        The shared selection parameter for linked highlighting.
    width : int
        Width of the plot.
    height : int
        Height of the plot.

    Returns
    -------
    alt.LayerChart
        A layered chart containing residuals, fitted curve, and selection highlight.
    """
    tag = spline.tag
    term = spline.term
    
    # Base for this spline plot
    base = alt.Chart(df_plot).encode(
        x=alt.X(f"x_{tag}", title=term),
    ).properties(width=width, height=height, title=f"{tag}")

    # Layer 1: Partial residuals
    residuals = base.mark_point(color='black', opacity=0.3, size=10).encode(
        y=alt.Y(f"resid_{tag}", title='')
    )

    # Layer 2: Spline output
    if isinstance(spline, Factor):
        spline_line = base.mark_point(size=50).encode(y=f"val_{tag}")
    else:
        # Sort data for line mark if it's not a Factor
        spline_line = alt.Chart(df_plot.sort(f"x_{tag}")).mark_line(strokeWidth=3).encode(
            x=f"x_{tag}",
            y=f"val_{tag}"
        )

    # Layer 3: Red dot for selected point
    selected_point = base.mark_circle(color='red', size=200).encode(
        y=f"val_{tag}",
        opacity=alt.condition(selector, alt.value(1), alt.value(0))
    )

    return alt.layer(residuals, spline_line, selected_point).add_params(selector)

def plot_interactive(model: LpRegressor, X: pl.DataFrame, y: pl.Series, xcol: str, 
                     show_splines: List[str] = None, width: int = 400, height: int = 300):
    """
    Create an interactive Altair plot showing the model fit and component splines.

    Parameters
    ----------
    model : LpRegressor
        The fitted regression model.
    X : pl.DataFrame
        Input features.
    y : pl.Series
        Target values.
    xcol : str
        The feature name to use for the x-axis in the top plot.
    show_splines : List[str], default=None
        A list of spline tags to display in the bottom row. If None, all splines are shown.
    width : int, default=400
        The total width of the plot.
    height : int, default=300
        The total height of the plot.

    Returns
    -------
    alt.VConcatChart
        The interactive Altair chart.
    """
    # Selection of splines to show
    splines_to_show = []
    spline_indices = []
    for i, s in enumerate(model.splines):
        if show_splines is None or s.tag in show_splines:
            splines_to_show.append(s)
            spline_indices.append(i)

    # Data Preparation
    df_plot = _prepare_interactive_data(model, X, y, xcol, spline_indices)

    # Shared Selection
    selector = alt.selection_point(on='mouseover', nearest=True, fields=['__id__'], empty=False)

    # Layout dimensions
    spacing = 10
    n_bottom = len(splines_to_show)
    top_height, top_width = height * 0.5, width
    bottom_height = height * 0.5
    bottom_width = (width - (n_bottom - 1) * spacing) / max(1, n_bottom)

    # --- Build Plots ---
    top_row = _create_top_chart(df_plot, xcol, selector, top_width, top_height)
    
    spline_charts = [
        _create_spline_chart(df_plot, s, selector, bottom_width, bottom_height) 
        for s in splines_to_show
    ]
    bottom_row = alt.hconcat(*spline_charts, spacing=spacing)

    return alt.vconcat(top_row, bottom_row).configure_title(fontSize=16).configure_axis(grid=False)