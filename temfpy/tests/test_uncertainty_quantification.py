"""Tests for uncertainty quantification module.
"""
import numpy as np

from hypothesis.strategies import integers
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats
from hypothesis import given

from temfpy.uncertainty_quantification import borehole
from temfpy.uncertainty_quantification import ishigami
from temfpy.uncertainty_quantification import simple_linear_function
from temfpy.uncertainty_quantification import eoq_model


def get_strategies(name):
    if (name == "eoq_model") or (name == "ishigami"):
        valid_floats = floats(0.01, 10000, allow_nan=False, allow_infinity=False)
        x_strategy = arrays(np.float, 3, elements=valid_floats)
        strategy = (x_strategy, valid_floats)
    elif (name == "borehole"):
        valid_floats = floats(0.01, 10000, allow_nan=False, allow_infinity=False)
        x_strategy = arrays(np.float, 8, elements=valid_floats)
        strategy = (x_strategy, valid_floats)
    elif name == "simple_linear_function":
        strategy = arrays(np.float, integers(1, 100))
    else:
        raise NotImplementedError

    return strategy


@given(*get_strategies("borehole"))
def test_borehole(x):
    borehole(x)


@given(*get_strategies("ishigami"))
def test_ishigami(x, a, b):
    ishigami(x, a, b)


@given(*get_strategies("eoq_model"))
def test_eoq_model(x, r):
    eoq_model(x, r)


@given(get_strategies("simple_linear_function"))
def test_simple_linear_function(x):
    simple_linear_function(x)
