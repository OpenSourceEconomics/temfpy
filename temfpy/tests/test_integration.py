"""Tests for nonlinear equations module."""
import numpy as np
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats
from hypothesis.strategies import integers

from temfpy.integration import continuous
from temfpy.integration import corner_peak
from temfpy.integration import discontinuous
from temfpy.integration import gaussian_peak
from temfpy.integration import oscillatory
from temfpy.integration import product


def get_strategies(name):
    valid_floats = floats(0, 1, allow_nan=False, allow_infinity=False)
    in_strategy = arrays(np.float, 3, elements=valid_floats)
    if name == "continuous":
        strategy = (in_strategy, in_strategy, in_strategy)
    elif name == "corner-peak":
        strategy = (in_strategy, in_strategy)
    elif name == "gaussian_peak":
        strategy = (in_strategy, in_strategy, in_strategy)
    elif name == "discontinuous":
        u_strategy = arrays(np.float, 2, elements=valid_floats)
        strategy = (in_strategy, u_strategy, in_strategy)
    elif name == "oscillatory":
        strategy = (in_strategy, in_strategy, integers(1, 3))
    elif name == "product":
        strategy = (in_strategy, in_strategy, in_strategy)
    else:
        raise NotImplementedError

    return strategy


@given(*get_strategies("continuous"))
def test_continuous(x, u, a):
    continuous(x, u, a)


@given(*get_strategies("corner_peak"))
def test_corner_peak(x, a):
    corner_peak(x, a)


@given(*get_strategies("discontiuous"))
def test_discontinuous(x, u, a):
    discontinuous(x, u, a)


@given(*get_strategies("gaussian_peak"))
def test_gaussian_peak(x, u, a):
    gaussian_peak(x, u, a)


@given(*get_strategies("oscillatory"))
def test_oscillatory(x, a, b):
    oscillatory(x, a, b)


@given(*get_strategies("product"))
def test_product(x, u, a):
    product(x, u, a)
