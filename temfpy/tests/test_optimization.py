"""Tests for optimization module."""
import numpy as np
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats
from hypothesis.strategies import integers

from temfpy.optimization import ackley
from temfpy.optimization import carlberg
from temfpy.optimization import rastrigin
from temfpy.optimization import rosenbrock


def get_strategies(name):
    if name == "ackley":
        valid_floats = floats(-10000, 10000, allow_nan=False, allow_infinity=False)
        x_strategy = arrays(np.float, shape=integers(1, 10), elements=valid_floats)
        strategy = (x_strategy, valid_floats, valid_floats, valid_floats)
    elif name == "carlberg":
        valid_floats_x = floats(-10000, 10000, allow_nan=False, allow_infinity=False)
        valid_floats_a = floats(-100, 100, allow_nan=False, allow_infinity=False)
        valid_floats_b = floats(0, 10, allow_nan=False, allow_infinity=False)
        dim = np.random.randint(1, 10)
        x_strategy = arrays(np.float, shape=dim, elements=valid_floats_x)
        a_strategy = arrays(np.float, shape=dim, elements=valid_floats_a)
        strategy = (x_strategy, a_strategy, valid_floats_b)
    elif name == "rastrigin":
        valid_floats = floats(-10000, 10000, allow_nan=False, allow_infinity=False)
        x_strategy = arrays(np.float, shape=integers(1, 10), elements=valid_floats)
        strategy = (x_strategy, valid_floats)
    elif name == "rosenbrock":
        valid_floats = floats(-10000, 10000, allow_nan=False, allow_infinity=False)
        x_strategy = arrays(np.float, shape=integers(2, 10), elements=valid_floats)
        strategy = x_strategy
    else:
        raise NotImplementedError

    return strategy


@given(*get_strategies("ackley"))
def test_ackley(x, a, b, c):
    ackley(x, a, b, c)


@given(*get_strategies("carlberg"))
def test_carlberg(x, a, b):
    carlberg(x, a, b)


@given(*get_strategies("rastrigin"))
def test_rastrigin(x, a):
    rastrigin(x, a)


@given(get_strategies("rosenbrock"))
def test_rosenbrock(x):
    rosenbrock(x)
