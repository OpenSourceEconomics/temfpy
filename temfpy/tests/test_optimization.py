"""Tests for optimization module."""
import numpy as np

from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats
from hypothesis.strategies import integers
from hypothesis import given

from temfpy.optimization import ackley
from temfpy.optimization import rastrigin


def get_strategies(name):
    if name == "ackley": 
        valid_floats = floats(-10000, 10000, allow_nan=False, allow_infinity=False)
        x_strategy = arrays(np.float, shape=integers(1, 10), elements=valid_floats)
        strategy = (x_strategy, valid_floats, valid_floats, valid_floats)
    elif name == "rastrigin":
        valid_floats = floats(-10000, 10000, allow_nan=False, allow_infinity=False)
        x_strategy = arrays(np.float, shape=integers(1, 10), elements=valid_floats)
        strategy = (x_strategy, valid_floats)
    else:
        raise NotImplementedError

    return strategy


@given(get_strategies("ackley"))
def test_ackley(x, a, b, c):
    ackley(x, a, b, c)


@given(get_strategies("rastrigin"))
def test_rastrigin(x, a):
    rastrigin(x, a)
