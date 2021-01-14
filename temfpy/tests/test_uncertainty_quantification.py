"""Tests for uncertainty quantification module."""
import numpy as np
import pytest
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats
from hypothesis.strategies import integers

from temfpy.uncertainty_quantification import borehole
from temfpy.uncertainty_quantification import eoq_model
from temfpy.uncertainty_quantification import ishigami
from temfpy.uncertainty_quantification import simple_linear_function


def get_strategies(name):
    valid_floats = floats(0.01, 10000, allow_nan=False, allow_infinity=False)
    if name == "eoq_model":
        x_strategy = arrays(np.float, 3, elements=valid_floats)
        strategy = (x_strategy, valid_floats)
    elif name == "ishigami":
        x_strategy = arrays(np.float, 3, elements=valid_floats)
        strategy = (x_strategy, valid_floats, valid_floats)
    elif name == "borehole":
        strategy = arrays(np.float, 8, elements=valid_floats)
    elif name == "simple_linear_function":
        strategy = arrays(np.float, integers(1, 100))
    else:
        raise NotImplementedError

    return strategy


@given(*get_strategies("eoq_model"))
def test_eoq_model(x, r):
    eoq_model(x, r)


@given(*get_strategies("ishigami"))
def test_ishigami(x, a, b):
    ishigami(x, a, b)


@given(get_strategies("borehole"))
def test_borehole(x):
    borehole(x)


@given(get_strategies("simple_linear_function"))
def test_simple_linear_function(x):
    simple_linear_function(x)

 
def test_borehole_exit_zero():
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        borehole([1,2,3,4,0,6,7,8])
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == "x5, x7 and x8 must be different from 0."

    
def test_borehole_exit_negative_log():
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        borehole([1,2,3,-4,5,6,7,8])
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == "x4 divided by x5 must be greater than 0." 


def test_eoq_exit_negative_x():  
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        eoq_model([-1,2,3], r=0.1)
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == "m and s must be greater or equal to zero and c must be greater than 0." 

    
def test_eoq_exit_negative_r():  
    with pytest.raises(SystemExit) as pytest_wrapped_e:   
        eoq_model([1,2,3], r=-0.1)
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == "r must be greater than 0."
