"""Tests for nonlinear equations module."""
import numpy as np
import pytest
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats
from hypothesis.strategies import integers

from temfpy.nonlinear_equations import broyden
from temfpy.nonlinear_equations import chandrasekhar
from temfpy.nonlinear_equations import exponential
from temfpy.nonlinear_equations import rosenbrock_ext
from temfpy.nonlinear_equations import trig_exp
from temfpy.nonlinear_equations import troesch


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
        strategy = (x_strategy, a_strategy)
    elif name == "rosenbrock_ext":
        x_strategy = arrays(np.float, 4, elements=valid_floats)
        a_strategy = arrays(np.float, 2, elements=integers(1, 20))
        strategy = (x_strategy, a_strategy)
    elif name == "troesch":
        x_strategy = arrays(
            np.float,
            3,
            elements=floats(-1, 1, allow_nan=False, allow_infinity=False),
        )
        strategy = (x_strategy, integers(1, 100), integers(1, 100))
    elif name == "chandrasekhar":
        x_strategy = arrays(np.float, 3, elements=valid_floats)
        y_strategy = arrays(np.float, 3, elements=valid_floats)
        strategy = (x_strategy, y_strategy, integers(1, 100))
    else:
        raise NotImplementedError

    return strategy


@given(*get_strategies("exponential"))
def test_exponential(x, a, b):
    exponential(x, a, b, jac=True)


@given(*get_strategies("trig_exp"))
def test_trig_exp(x, a):
    trig_exp(x, a, jac=True)


@given(*get_strategies("broyden"))
def test_broyden(x, a):
    broyden(x, a, jac=True)


@given(*get_strategies("rosenbrock_ext"))
def test_rosenbrock_ext(x, a):
    rosenbrock_ext(x, a, jac=True)


@given(*get_strategies("troesch"))
def test_troesch(x, rho, a):
    troesch(x, rho, a, jac=True)


@given(*get_strategies("chandrasekhar"))
def test_chandrasekhar(x, y, c):
    chandrasekhar(x, y, c, jac=True)


def test_exponential_exit_zero_a():
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        exponential([3, 2, 3, 4], a=0, b=1, jac=False)
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == "a must be different from 0."
