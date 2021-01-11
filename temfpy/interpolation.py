"""Test capabilities for interpolation.
This module contains a host of models and functions often used for testing
 interpolation algorithms.
"""
import numpy as np


def runge(x):
    r"""Runge function.
    .. math::
        f(x) = \frac{1}{1 + 25x^2}
    Parameters
    ----------
    x : float
        Input number. It is usually evaluated on the interval
        :math:`x_i \in [-1, 1]`.

    Returns
    -------
    float
         Output domain
    Notes
    -----
    Runge found that interpolating this function with a Polynomial
    :math:`p_n(x)` of degree :math:`n` on an equidistant grid
    with grid points :math:`x_i = \frac{2i}{n}-1`, results in an interpolation
    function that oscilliates close to the interval boundaries :math:`-1` and
    :math:`1`.
    .. figure:: ../../docs/_static/images/fig-runge.png
       :align: center
    References
    ----------
    .. [R1901] Runge, K (1901).
       Über empirische Funktionen und die Interpolation
       zwischen äquidistanten Ordinaten.
       Zeitschrift für Mathematik und Physik.
    Examples
    --------
    >>> from temfpy.interpolation import runge
    >>> import numpy as np
    >>>
    >>> x = 0
    >>> y = runge(x)
    >>> np.testing.assert_almost_equal(y, 1)
    """

    x = np.atleast_1d(x)
    rslt = 1 / (1 + 25 * x ** 2)

    if len(x) == 1:
        rslt = float(rslt)

    return rslt

