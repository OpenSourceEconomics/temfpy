"""Tests for nonlinear functions module."""
import numpy as np
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats
from hypothesis.strategies import integers

from temfpy.nonlinear_functions import exponential
from temfpy.nonlinear_functions import trig_exp
from temfpy.nonlinear_functions import broyden
from temfpy.unonlinear_functions import rosenbrock_ext
from temfpy.unonlinear_functions import chandrasekhar


def get_strategies(name):
    valid_floats = floats(0.01, 10000, allow_nan=False, allow_infinity=False)
    valid_floats_exp = floats(-100, 500, allow_nan=False, allow_infinity=False)
    if name == "exponential":
        strategy = arrays(np.float, 3, elements=valid_floats_exp)
    elif name == "trig_exp":
        strategy = arrays(np.float, 3, elements=valid_floats_exp)
    elif name == "broyden":
        strategy = arrays(np.float, 3, elements=valid_floats)
    elif name == "rosenbrock_ext":
        strategy = arrays(np.float, 3, elements=valid_floats)
    elif name == "chandrasekhar":
        x_strategy = arrays(np.float, integers(1, 100))
        y_strategy = arrays(np.float, integers(1, 100))
        strategy = (x_strategy, y_strategy, integers(1, 100))
    else:
        raise NotImplementedError

    return strategy


@given(*get_strategies("exponential"))
def test_exponential(x):
    exponential(x)


@given(*get_strategies("trig_exp"))
def test_trig_exp(x):
    trig_exp(x)


@given(get_strategies("broyden"))
def test_broyden(x):
    broyden(x)


@given(get_strategies("rosenbrock_ext"))
def test_rosenbrock_ext(x):
    rosenbrock_ext(x)


@given(get_strategies("chandrasekhar"))
def test_chandrasekhar(x, y, c):
    chandrasekhar(x, y, c)
