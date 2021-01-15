"""Test capabilities for optimization.
This module contains a host of models and functions often used for testing optimization
algorithms.
"""
import sys

import numpy as np
import pandas as pd
from scipy.optimize import rosen


def ackley(x, a=20, b=0.2, c=2 * np.pi):
    r"""Ackley function.

    .. math::
        f(x) = -a \exp{\left(-b \sqrt{\frac{1}{p} \sum_{i=1}^p x_i^2}\right)}
        \exp{\left(\frac{1}{p} \sum_{i=1}^p \cos(c x_i)\right)} + a + \exp(1)

    Parameters
    ----------
    x : array_like
        Input domain with dimension :math:`p`.
        It is usually evaluated on the hypercube
        :math:`x_i \in [-32.768, 32.768]`, for all :math:`i = 1, \dots, p`.
    a : float, optional
        The default value is 20.
    b : float, optional
        The default value is 0.2.
    c : float, optional
        The default value is 2π.

    Returns
    -------
    float
         Output domain

    Notes
    -----
    This function was proposed by David Ackley in [A1987]_ and used in [B1996]_
    and [M2005]_. It is characterized by an almost flat outer region and a central hole
    or peak where modulations become more and more influential. The function has
    its global minimum :math:`f(x) = 0` at :math:`x = \begin{pmatrix}0 & \dots & 0
    \end{pmatrix}^T`.

    .. figure:: ../../docs/_static/images/fig-ackley.png
       :align: center

    References
    ----------
    .. [A1987] Ackley, D. H. (1987).
       A connectionist machine for genetic hillclimbing.
       Boston, MA: Kluwer Academic Publishers.
    .. [B1996] Back, T. (1996).
       Evolutionary algorithms in theory and practice:
       Evolution strategies, evolutionary programming, genetic algorithms.
       Oxford, UK: Oxford University Press.
    .. [M2005] Molga, M., and Smutnicki, C. (2005).
       Test functions for optimization needs.
       Retrieved June 2020, from
       http://www.zsd.ict.pwr.wroc.pl/files/docs/functions.pdf.

    Examples
    --------
    >>> from temfpy.optimization import ackley
    >>> import numpy as np
    >>>
    >>> x = [0, 0]
    >>> y = ackley(x)
    >>> np.testing.assert_almost_equal(y, 0)
    """
    rslt = (
        a + np.exp(1) - (a * (np.exp(-b * np.sqrt(1 / len(x) * np.sum(np.square(x))))))
    )
    rslt -= np.exp(1 / len(x) * np.sum(np.cos(np.multiply(c, x))))

    return rslt


def rastrigin(x, a=10):
    r"""Rastrigin function.

    .. math::
        f(x) = a p + \sum_{i=1}^p \left(x_i^2 - 10 \cos(2\pi x_i)\right)

    Parameters
    ----------
    x : array_like
        Input domain with dimension :math:`p`.
        It is usually evaluated on the hypercube
        :math:`x_i\in [-5.12, 5.12]`, for all :math:`i = 1, \dots, p`.
    a : float, optional
        The default value is 10.

    Returns
    -------
    float
         Output domain

    Notes
    -----
    The function was first proposed by Leonard Rastrigin in [R1974]_.
    It produces frequent local minima as it is highly multimodal.
    However, the location of the minima are regularly distributed.
    The function has its global minimum :math:`f(x) = 0` at
    :math:`x = \begin{pmatrix}0 & \dots & 0 \end{pmatrix}^T`.

    .. figure:: ../../docs/_static/images/fig-rastrigin.png
       :align: center

    References
    ----------
    .. [R1974] Rastrigin, L. A. (1974).
       Systems of extremal control.
       Moscow, Russia: Mir.

    Examples
    --------
    >>> from temfpy.optimization import rastrigin
    >>> import numpy as np
    >>>
    >>> x = [0, 0]
    >>> y = rastrigin(x)
    >>> np.testing.assert_almost_equal(y, 0)
    """
    rslt = a * len(x) + np.sum(
        np.multiply(x, x) - 10 * np.cos(2 * np.multiply(np.pi, x)),
    )

    return rslt


def rosenbrock(x):
    r"""Rosenbrock function.

    .. math::
        f(x) = \sum^{p-1}_{i = 1} \left[100(x_{i+1}-x_i^2)^2 + (1-x_i^2) \right]

    Parameters
    ----------
    x : array_like
        Input domain with dimension :math:`p > 1`.

    Returns
    -------
    float
         Output domain

    Notes
    -----
    The function was first proposed by Howard H. Rosenbrock in [R1960]_ and
    is often also referred to, due to its shape, as Rosenbrock's valley or
    Rosenbrock's Banana function.
    The function has its global minimum at
    :math:`x = \begin{pmatrix}1 & \dots & 1 \end{pmatrix}^T`

    .. figure:: ../../docs/_static/images/fig-rosenbrock.png
       :align: center

    References
    ----------
    .. [R1960] Rosenbrock, H. H. (1960).
       An Automatic Method for Finding the Greatest
       or Least Value of a Function.
       The Computer Journal, Volume 3, Issue 3, Pages 175-184

    Examples
    --------
    >>> from temfpy.optimization import rosenbrock
    >>> import numpy as np
    >>>
    >>> x = [1, 1]
    >>> y = rosenbrock(x)
    >>> np.testing.assert_almost_equal(y, 0)
    """

    if not isinstance(x, (list, tuple, pd.core.series.Series, np.ndarray)):
        sys.exit("The parameter x must be an array like object.")
    else:
        if len(x) < 2:
            sys.exit("The input array x must be at least of length 2.")

    rslt = rosen(x)

    return rslt


def carlberg(x, a, b):
    r"""Carlberg function.

    .. math::
        f(x) = \frac{1}{2}\sum_{i=1}^p a_i (x_i - 1)^2 + b \left[p -
        \sum_{i=1}^p \cos(2 \pi(x_i-1)) \right]

    Parameters
    ----------
    x : array_like
        Input vector with dimension :math:`p`.
    a : array_like
        Input vector with dimension :math:`p`.
    b : float
        Must not be smaller than zero.
        For more information see Notes.

    Returns
    -------
    float
         Output domain

    Notes
    -----
    If the values in :math:`a` are widely distributed the function is
    said to be ill-conditioned and it is hard to minimize in some
    directions for Hessian free numerical methods.
    If :math:`b=0` (see second graph below) the function is
    convex, smooth and has its minimum at
    :math:`x = \begin{pmatrix}1 & \dots & 1 \end{pmatrix}^T`. For
    :math:`b>0` the function is no longer convex and has many local
    minima (see first graph below), making it hard for
    local optimization methods to find the global minimum,
    which is still at
    :math:`x = \begin{pmatrix}1 & \dots & 1 \end{pmatrix}^T`.

    .. figure:: ../../docs/_static/images/fig-carlberg_noise.png
       :align: center
    .. figure:: ../../docs/_static/images/fig-carlberg_no_noise.png
       :align: center

    References
    ----------
    .. [C2019] Carlberg, K. (2019).
       Optimization in Python.
       Fundamentals of Data Science Summer Workshops, Stanford.

    Examples
    --------
    >>> from temfpy.optimization import carlberg
    >>> import numpy as np
    >>>
    >>> x = [1, 1]
    >>> a = [1, 1]
    >>> b = 1
    >>> y = carlberg(x,a,b)
    >>> np.testing.assert_almost_equal(y, 0)
    """

    if b < 0:
        sys.exit("Input parameter b must not be smaller than zero.")

    x, a = np.atleast_1d(x), np.atleast_1d(a)

    dimension = len(x)

    fval = 0
    fval += 0.5 * np.sum(a * (x - np.ones(dimension)) ** 2)
    fval += b * dimension
    fval -= b * np.sum(np.cos(2 * np.pi * (x - np.ones(dimension))))

    return fval
