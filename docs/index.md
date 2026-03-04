# LPSpline Documentation

<div align="center">
  <img src="_static/logo_round.png" width="200px" alt="lpspline logo">
</div>

LPSpline is a Python package for building and optimizing **linear spline models** using an intuitive additive API. It provides a flexible way to model non-linear relationships using various spline types with full support for shape constraints and penalties.

[![PyPI version](https://badge.fury.io/py/lpspline.svg)](https://badge.fury.io/py/lpspline)
![Python Versions](https://shields.io/badge/python-3.10+-blue)
[![Documentation Status](https://img.shields.io/readthedocs/lpspline?logo=readthedocs)](https://lpspline.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/clarkmaio/lpspline/blob/main/LICENSE)


## Getting Started

LPSpline lets you compose complex additive models from simple building blocks:

```python
from lpspline import l, pwl, bs, cs, f

model = (
    + l(term='xl', by='xfactor')
    + pwl(term='xpwl', knots=3)
    + bs(term="xbs", knots=10, degree=2)
    + cs(term="xcyc", order=3)
    + f(term="xfactor")
)

model.fit(X, y)
```

See the [Quick Start](notebooks/quick_start.ipynb) for a complete walkthrough.


## Installation

```bash
pip install lpspline
```


## Features

- **Additive API** — compose models using `+` operator
- **Spline Types** — Linear, Piecewise Linear, B-Splines, Cyclic Splines, Factors
- **Shape Constraints** — Monotonicity, Convexity, Concavity, Anchoring
- **Penalties** — Ridge, Lasso
- **Polars support** — first-class DataFrame integration
- **CVXPY backend** — powered by convex optimization
