"""Test capabilities for uncertainty quantification.

This module contains a host of test models and functions often used in the uncertainty
quantification literate.

"""
import numpy as np
import math


def borehole(x):
    r"""Borehole function.

    The Borehole function was developed by Harper and Gupta [H1983]_ to model steady state flow
    through a hypothetical borehole. It is widely used as a testing function for a variety of methods
    due to its simplicity and quick evaluation (e.g. [X2013]_).

    Parameters
    ----------

    x : array_like
        Core parameters of the model with dimension 8.

    Returns
    -------

    float
        Flow rate in :math:`m^3/yr`.

    Notes
    -----

    Harper and Gupta [H1983]_ used the function originally to compare the results of a sensitivity
    analysis to the results based on Latin hypercube sampling.   

    References
    ----------
    .. [H1983] Harper, W. V., and Gupta, S. K. (1983)
	Sensitivity/uncertainty analysis of a borehole scenario comparing Latin hypercube sampling 		and deterministc sensitivity approaches.
	Office of Nuclear Waste Isolation, Battelle Memorial Institute.

    .. [X2013] Xiong, S., and Qian, P. Z., and Wu, C. J. (2013).
	Sequential design and analysis of high-accuracy and low-accuracy computer codes.
	Technometrics, 55(1), 37-46.


    Examples
    --------

    >>> x = [1, 2, 3, 4, 5, 6, 7, 8]
    >>> y = borehole(x)
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


def ishigami(x, a=7, b=0.05):
    r"""Ishigami function.

    This function was specifically developed by Ishigami and Homma [I1990]_ as a test function used for 	  uncertainty analysis. It is characterized by its strong nonlinearity and nonmonotonicity.

    Parameters
    ----------

    x : array_like
        Core parameters of the model with dimension 3.

    float a, b
	The default values are 7 and 0.05 respectively, as used by Sobol' and Levitan [S1999]_.
        

    Returns
    -------

    float
        Output domain

    Notes
    -----

    Sobol' and Levitan [S1999]_ note that the Ishigami function has a strong dependance
    on :math:`x_3`.
    

    References
    ----------

    .. [I1990] Ishigami, T., and Homma, T. (1990).
        An importance quantification technique in uncertainty analysis for computer models.
        In: Uncertainty Modeling and Analysis, 1990. Proceedings., First International Symposium on 		(pp. 398-403).

    .. [S1999] Sobol', I. M., and Levitan, Y. L (1999).
	On the use of variance reducing multipliers in Monte Carlo computations of a global 		sensitivity index.
	Computer Physics Communications, 117(1), 52-61.

    Examples
    --------

    >>> x = [1, 2, 3]
    >>> y = ishigami(x)
    """
    assert len(x) == 3

    rslt = (1 + b * x[2] ** 4) * np.sin(x[0]) + a * np.sin(x[1]) ** 2
    return rslt


def eoq_model(x, r=0.1):
    r"""Economic order quantity model.

    This function computes the optimal economic order quantity (EOQ) based on the model presented in
    [H1990]_. The EOQ minimizes the holding costs as well as ordering costs. The core parameters of
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

    A historical perspective on the model is provided by [E1990]_. A brief description with the core
    equations is available in [W2020]_.

    References
    ----------

    .. [H1990] Harris, F. W. (1990).
        How many parts to make at once.
        Operations Research, 38(6), 947–950.

    .. [E1990] Erlenkotter, D. (1990).
        Ford Whitman Harris and the economic order quantity model.
        Operations Research, 38(6), 937–946.

    .. [W2020] Economic order quantity. (2020, April 3). In Wikipedia.
        Retrieved from
        `https://en.wikipedia.org/w/index.php\
        ?title=Economic_order_quantity&oldid=948881557 <https://en.wikipedia.org/w/index.php
        ?title=Economic_order_quantity&oldid=948881557>`_

    Examples
    --------

    >>> x = [1, 2, 3]
    >>> y = eoq_model(x, r=0.1)
    >>> np.testing.assert_almost_equal(y, 18.973665961010276)
    """

    m, c, s = x
    y = np.sqrt((24 * m * s) / (r * c))

    return y


def simple_linear_function(x):
    r"""Simple linear function.

    This function computes the sum of all elements of a given array.

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
