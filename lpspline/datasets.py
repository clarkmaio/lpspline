import numpy as np
import polars as pl
from typing import Tuple



def load_by_dataset(samples: int = 1000, type: str = 'linear') -> Tuple[pl.DataFrame, pl.Series]:
    """
    Generate a synthetic dataset for demonstrating LPSpline features including group effects.
    
    Parameters
    ----------
    samples : int, default=1000
        Number of samples to generate.
    type : str, default='cubic'
        Type of structural relationship for `y`. Options include: 'linear', 'cubic', 'cyclic'.
        
    Returns
    -------
    X : pl.DataFrame
        A DataFrame containing the predictive feature 'x' and grouping structure 'by'.
    y : pl.Series
        The response Target series.
    """

    x = np.linspace(-10, 10, samples)
    by = np.random.randint(low=0, high=3, size=samples)

    X = pl.DataFrame({
        'x': x,
        'by': by
    })
    
    if type == 'linear':
        y = pl.Series(x * (1 + by) + np.random.normal(size=samples) * 2)
    elif type == 'cubic':
        y = pl.Series(x**3 * (1 + by) + np.random.normal(size=samples) * 10)
    elif type == 'cyclic':
        y = pl.Series(np.sin(x) * (1 + by) + np.random.normal(size=samples) * 0.5)
    else:
        raise ValueError("type must be one of: 'linear', 'cubic', 'cyclic'")
    
    return X, y

def load_demo_dataset(samples: int = 1000) -> Tuple[pl.DataFrame, pl.Series]:
    """
    Generate a diverse synthetic dataset for demonstrating multiple LPSpline component features.
    
    Parameters
    ----------
    samples : int, default=1000
        Number of samples to generate.
        
    Returns
    -------
    X : pl.DataFrame
        A DataFrame containing multiple features tracking various generative relationships.
    y_series : pl.Series
        The synthesized composite target variable.
    """
    x_linear = np.linspace(0, 10, samples)
    x_pwl = np.linspace(0, 10, samples)
    x_bs = np.linspace(0, 10, samples)
    x_cyc = np.linspace(0, 2*np.pi, samples)
    x_factor = np.random.randint(0, 3, samples)
    
    # True functions
    y_pwl = np.where(x_pwl < 5, 0, x_pwl - 5) # Hinge at 5
    y_bs = np.sin(x_bs) # Sine wave
    y_cyc = np.cos(x_cyc) # Cosine wave
    y_factor = np.array([0, 2, -1])[x_factor] # Categorical effects
    slopes = np.array([-1, 0.5, 2])
    y_linear = slopes[x_factor] * x_linear
    
    # Combined target with noise
    y = (
        y_linear + 
        y_pwl + 
        y_bs + 
        y_cyc + 
        y_factor + 
        np.random.normal(0, 0.2, samples)
    )
    
    X = pl.DataFrame({
        "xl": x_linear,
        "xpwl": x_pwl,
        "xbs": x_bs,
        "xcyc": x_cyc,
        "xfactor": x_factor
    })
    
    y_series = pl.Series("target", y)
    
    return X, y_series
