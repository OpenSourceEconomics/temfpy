"""Tests for nonlinear functions module."""
import numpy as np
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats
from hypothesis.strategies import integers

from temfpy.nonlinear_functions import exponential
from temfpy.nonlinear_functions import trig_exp
from temfpy.nonlinear_functions import broyden
from temfpy.nonlinear_functions import rosenbrock_ext
from temfpy.nonlinear_functions import troesch
from temfpy.nonlinear_functions import chandrasekhar


def get_strategies(name):
    valid_floats = floats(0.01, 10000, allow_nan=False, allow_infinity=False)
    valid_floats_exp = floats(-100, 500, allow_nan=False, allow_infinity=False)
    if name == "exponential":
        x_strategy = arrays(np.float, 3, elements=valid_floats_exp)
        strategy = (x_strategy, integers(1, 100), integers(1, 100))
    elif name == "trig_exp":
        x_strategy = arrays(np.float, 3, elements=valid_floats_exp)
        a_strategy = arrays(np.float, 9, elements=integers(1, 20))
        strategy = (x_strategy, a_strategy)
    elif name == "broyden":
        x_strategy = arrays(np.float, 3, elements=valid_floats)
        a_strategy = arrays(np.float, 4, elements=integers(1, 20))
        strategy = (x_strategy, integers(1, 100), integers(1, 100))
    elif name == "rosenbrock_ext":
        x_strategy = arrays(np.float, 3, elements=valid_floats)
        a_strategy = arrays(np.float, 2, elements=integers(1, 20))
        strategy = (x_strategy, integers(1, 100), integers(1, 100))
    elif name == "troesch":
        x_strategy = arrays(np.float, 3, elements=valid_floats)
        a_strategy = arrays(np.float, 1, elements=integers(1, 20))
        strategy = (x_strategy, integers(1, 100), integers(1, 100))
    elif name == "chandrasekhar":
        x_strategy = arrays(np.float, 3, elements=valid_floats)
        y_strategy = arrays(np.float, 3, elements=valid_floats)
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


@given(get_strategies("troesch"))
def test_troesch(x):
    troesch(x)


@given(get_strategies("rosenbrock_ext"))
def test_rosenbrock_ext(x):
    rosenbrock_ext(x)


@given(get_strategies("chandrasekhar"))
def test_chandrasekhar(x, y, c):
    chandrasekhar(x, y, c)
