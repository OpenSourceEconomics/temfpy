"""Tests for linear equations module."""
from hypothesis import given
from hypothesis.strategies import integers

from temfpy.linear_equations import get_ill_cond_lin_eq


def get_strategies(name):
    if name == "get_ill_cond_lin_eq":
        strategy = integers(1)
    else:
        raise NotImplementedError

    return strategy


@given(*get_strategies("get_ill_cond_lin_eq"))
def test_get_ill_cond_lin_eq(n):
    get_ill_cond_lin_eq(n)
