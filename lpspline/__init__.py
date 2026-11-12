"""LPSpline: constrained additive spline models fitted with convex optimization."""

__version__ = "1.0.0"

from .optimizer import LpRegressor

from .spline import Linear as l
from .spline import PiecewiseLinear as pwl
from .spline import BSpline as bs
from .spline import CyclicSpline as cs
from .spline import Factor as f
from .spline import Constant as c

from .link import Link, Log, Sigmoid, Exp
