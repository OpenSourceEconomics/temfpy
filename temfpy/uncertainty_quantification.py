"""Test capabilities for uncertainty quantification.

This module contains a host of test models and functions often used in the uncertainty
quantification literate.

"""
import numpy as np
import math


def borehole(x):
    """The Borehole function models water flow through a borehole. Its simplicity and quick
    evaluation makes it a commonly used function for testing a wide variety of methods in
    computer experiments.
    """
    assert len(x) == 8

    r_w = x[0]
    r = x[1]
    T_u = x[2]
    H_u = x[3]
    T_l = x[4]
    H_l = x[5]
    L = x[6]
    K_w = x[7]

    a = 2 * math.pi * T_u * (H_u - H_l)
    b = np.log(r / r_w)
    c = (2 * L * T_u) / (b * r_w ** 2 * K_w)
    d = T_u / T_l

    rslt = a / (b * (1 + c + d))
    return rslt


def ishigami(x, a=7, b=0.1):
    """
    The Ishigami function of Ishigami & Homma (1990) is used as an example for uncertainty and
    sensitivity analysis methods, because it exhibits strong nonlinearity and nonmonotonicity.
    """
    assert len(x) == 3

    rslt = (1 + b * x[2] ** 4) * np.sin(x[0]) + a * np.sin(x[1]) ** 2
    return rslt


def eoq_harris(x, r=10):
    r"""Economic order quantity model.

    This function computes the optimal economic order quantity (EOQ) based on the model presented in
    [1]_. The EOQ minimizes the holding costs as well as ordering costs. The core parameters of
    the model are the units per months `x[0]`, the unit price of items in stock `x[1]`,
    and the setup costs of an order `x[2]`. The annual interest rate `r` is treated as an
    additional parameter.

    Parameters
    ----------
    x : array_like
        Core parameters of the model

    r : float, optional
        Annual interest rate

    Returns
    -------

    float
        Optimal order quantity

    Notes
    -----

    A historical perspective on the model is provided by [2]_. A brief description with the core
    equations is available in [3]_.

    References
    ----------

    .. [1] Harris, F. W. (1990). 
        How Many Parts to Make at Once.
        Operations Research, 38(6), 947–950.

    .. [2] Erlenkotter, D. (1990).
        Ford Whitman Harris and the Economic Order Quantity Model.
        Operations Research, 38(6), 937–946.

    .. [3] Economic order quantity.
        (2020, April 3).
        In Wikipedia.
        Retrieved from https://en.wikipedia.org/w/index.php?title=Economic_order_quantity&oldid=948881557

    Examples
    --------

    >>> x = [1, 2, 3]
    >>> y = eoq_harris(x, r=10)
    >>> np.testing.assert_almost_equal(y, 12.649110640673518)
    """

    m, s, c = x
    y = np.sqrt((24 * r * m * s) / c)

    return y


def simple_linear_function(x):
    """Simple linear function.

    Parameters
    ----------
    x : array_like
        Array of summands

    Examples
    --------

    >>> x = [1, 2, 3]
    >>> y = simple_linear_function(x)
    >>> np.testing.assert_almost_equal(y, 6)
    """
    return sum(x)
