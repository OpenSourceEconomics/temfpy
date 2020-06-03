"""Tests for optimization module.
"""
import numpy as np

from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats
from hypothesis import given

from temfpy.optimization import ackley
from temfpy.optimization import rastrigin


def get_strategies(name, x):
    if name == "ackley":
        valid_floats = floats(-10000, 10000, allow_nan=False, allow_infinity=False)
        x_strategy = arrays(np.float, len(x), elements=valid_floats)
        strategy = (x_strategy, valid_floats)
    elif name == "rastrigin":
        valid_floats = floats(-10000, 10000, allow_nan=False, allow_infinity=False)
        x_strategy = arrays(np.float, len(x), elements=valid_floats)
        strategy = (x_strategy, valid_floats)
    else:
        raise NotImplementedError

    return strategy


@given(*get_strategies("ackley", x))
def test_ackley(x, a, b, c):
    ackley(x, a, b, c)


@given(*get_strategies("rastrigin", x))
def test_rastrigin(x, a):
    rastrigin(x, a)