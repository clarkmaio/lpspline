"""Wrappers applying a function to the sum of splines (e.g. ``Exp(model)``)."""
from .base import Link
from .exp import Exp
from .log import Log
from .logit import Sigmoid

__all__ = ["Link", "Exp", "Log", "Sigmoid"]
