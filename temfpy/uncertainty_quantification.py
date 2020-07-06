"""Test capabilities for uncertainty quantification.

This module contains a host of test models and functions often used in the uncertainty
quantification literate.

"""
import numpy as np


def borehole(x):
    r"""Borehole function.

    .. math::
        f(x) = \frac{2 \pi x_1 (x_2 - x_3)}{\ln{\left(x_4/x_5\right)} \left(1 +
        \frac{2 x_1 x_6}{\ln{(x_4/x_5)}x_5^2 x_7} + \frac{x_1}{x_8}\right)}

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
    The Borehole function was developed by Harper and Gupta [H1983]_
    to model steady state flow through a hypothetical borehole.
    It is widely used as a testing function for a variety of methods
    due to its simplicity and quick evaluation (e.g. [X2013]_).
    Harper and Gupta used the function originally to compare the results of a sensitivity
    analysis to results based on Latin hypercube sampling.

    References
    ----------
    .. [H1983] Harper, W. V., and Gupta, S. K. (1983)
        Sensitivity/uncertainty analysis of a borehole scenario comparing Latin hypercube sampling
        and deterministc sensitivity approaches.
        Office of Nuclear Waste Isolation, Battelle Memorial Institute.

    .. [X2013] Xiong, S., and Qian, P. Z., and Wu, C. J. (2013).
        Sequential design and analysis of high-accuracy and low-accuracy computer codes.
        Technometrics, 55(1), 37-46.


    Examples
    --------
    >>> from temfpy.uncertainty_quantification import borehole
    >>> import numpy as np
    >>>
    >>> x = [1, 2, 3, 4, 5, 6, 7, 8]
    >>> y = borehole(x)
    >>> np.testing.assert_almost_equal(y, 34.43500403827335)
    """
    assert len(x) == 8

    a = 2 * np.pi * x[0] * (x[1] - x[2])
    b = np.log(x[3] / x[4])
    c = 2 * x[0] * x[5]
    d = b * x[4] ** 2 * x[6]
    e = x[0] / x[7]
    rslt = a / (b * (1 + c / d + e))
    return rslt


def ishigami(x, a=7, b=0.05):
    r"""Ishigami function.

    .. math::
        f(x) = \sin(x_1) + a \sin^2(x_2) + b x_3^4 \sin(x_1)

    Parameters
    ----------
    x : array_like
        Core parameters of the model with dimension 3.

    a : float, optional
        The default value is 7, as used by Sobol' and Levitan in [S1999]_.

    b : float, optional
        The default value is 0.05, as used by Sobol' and Levitan.

    Returns
    -------
    float
        Output domain

    Notes
    -----
    This function was specifically developed by Ishigami and Homma [I1990]_
    as a test function used for uncertainty analysis.
    It is characterized by its strong nonlinearity and nonmonotonicity.
    Sobol' and Levitan note that the Ishigami function has a strong dependence
    on :math:`x_2`.


    References
    ----------
    .. [I1990] Ishigami, T., and Homma, T. (1990).
        An importance quantification technique in uncertainty analysis for computer models.
        In: Uncertainty Modeling and Analysis, 1990. Proceedings.,
        First International Symposium on (pp. 398-403).

    .. [S1999] Sobol', I. M., and Levitan, Y. L (1999).
        On the use of variance reducing multipliers in Monte Carlo computations of a
        global sensitivity index.
        Computer Physics Communications, 117(1), 52-61.

    Examples
    --------
    >>> from temfpy.uncertainty_quantification import ishigami
    >>> import numpy as np
    >>>
    >>> x = [1, 2, 3]
    >>> y = ishigami(x)
    >>> np.testing.assert_almost_equal(y, 10.037181146302519)
    """
    assert len(x) == 3

    rslt = (1 + b * x[2] ** 4) * np.sin(x[0]) + a * np.sin(x[1]) ** 2
    return rslt


def eoq_model(x, r=0.1):
    r"""Economic order quantity model.

    .. math::
        y = \sqrt{\frac{24 x_0 x_2}{r x_1}}

    Parameters
    ----------
    x : array_like
        Core parameters of the model.

    r : float, optional
        Annual interest rate (default value is 0.1).

    Returns
    -------
    y : float
        Optimal order quantity.

    Notes
    -----
    This function computes the optimal economic order quantity (EOQ) based on the model presented in
    [H1990]_. The EOQ minimizes the holding costs as well as ordering costs. The core parameters of
    the model are the units per months :math:`x_0`, the unit price of items in stock :math:`x_1`,
    and the setup costs of an order :math:`x_2`. The annual interest rate `r` is treated as an
    additional parameter. A historical perspective on the model is provided by [E1990]_.
    A brief description with the core equations is available in [W2020]_. The figure below
    illustrates the core trade-off in the model. Holding :math:`x_1` and :math:`x_2` constant,
    an increase in :math:`x_0` results in a decrease in the setup cost per unit, but an increase
    in capital cost increases as the stock of inventory increase.

    .. figure:: ../../docs/_static/images/fig-eoq-tradeoff.png
       :align: center

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
    >>> from temfpy.uncertainty_quantification import eoq_model
    >>> import numpy as np
    >>>
    >>> x = [1, 2, 3]
    >>> y = eoq_model(x, r=0.1)
    >>> np.testing.assert_almost_equal(y, 18.973665961010276)
    """
    m, c, s = x
    y = np.sqrt((24 * m * s) / (r * c))

    return y


def simple_linear_function(x):
    r"""Uncomplicated linear function.

    This function computes the sum of all elements of a given array.

    Parameters
    ----------
    x : array_like
        Array of summands

    Examples
    --------
    >>> from temfpy.uncertainty_quantification import simple_linear_function
    >>> import numpy as np
    >>>
    >>> x = [1, 2, 3]
    >>> y = simple_linear_function(x)
    >>> np.testing.assert_almost_equal(y, 6)
    """
    return sum(x)
