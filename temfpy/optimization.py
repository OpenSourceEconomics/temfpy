"""Test capabilities for optimization.

This module contains a host of models and functions often used for testing optimization algorithms.

"""
import numpy as np
import math


def ackley(x, a=20, b=0.2, c=2 * math.pi):
    r"""Ackley function.

    This function was proposed by David Ackley in [A1987]_.
    Originally, it was formulated only for the two-dimensional case;
    it is characterized by an almost flat outer region and a central hole or peak
    where modulations become more and more influential. The function has
    its global minimum :math:`f(x) = 0` at :math:`x = (0, \dots, 0)`.

    Parameters
    ----------

    x : array_like
        Input domain with dimension d. It is usually evaluated on the hypercube
        :math:`x_i\in [-32.768, 32.768]`, for all :math:`i = 1, \dots, d`.

    float a, b, c
        Recommended variable values are 20, 0.2, 2π, respectively.

    Returns
    -------

    float
         Output domain

    Notes
    -----
    A brief description of the function is available in [B1996]_ and in [M2005]_.

    References
    ----------

    .. [A1987] Ackley, D. H. (1987).
       A connectionist machine for genetic hillclimbing.
       Boston, MA: Kluwer Academic Publishers.

    .. [B1996] Back, T. (1996).
       Evolutionary algorithms in theory and practice:
       evolution strategies, evolutionary programming, genetic algorithms.
       Oxford, UK: Oxford University Press.

    .. [M2005] Molga, M., and Smutnicki, C. (2005).
       Test functions for optimization needs.
       Retrieved June 2020, from http://www.zsd.ict.pwr.wroc.pl/files/docs/functions.pdf.

    Examples
    --------

    >>> x = [0, 0]
    >>> y = ackley(x)

    """
    n = len(x)

    rslt = (
        a + math.exp(1) - (a * (math.exp(-b * math.sqrt(1 / n * np.sum(np.square(x))))))
    )
    rslt -= math.exp(1 / n * np.sum(np.cos(np.multiply(c, x))))

    return rslt


def rastrigin(x, a=10):
    r"""Rastrigin function.

    The function was first proposed by Rastrigin as a 2-dimensional function in [R1974]_.
    It produces frequent local minima; thus, it is highly multimodal.
    However, the location of the minima are regularly distributed.
    The function has its global minimum :math:`f(x) = 0` at :math:`x = (0, \dots, 0)`.

    Parameters
    ----------
    a : float
        Recommended variable value is 10.

    x : array_like
        Input domain with dimension d. It is usually evaluated on the hypercube
        :math:`x_i\in [-5.12, 5.12]`, for all :math:`i = 1, \dots, d`.

    Returns
    -------

    float
         Output domain

    Notes
    -----
    A brief description of the function is available in [M2005]_.

    References
    ----------

    .. [R1974] Rastrigin, L. A. (1974).
       Systems of extremal control.
       Moscow, Russia: Mir.

    Examples
    --------

    >>> x = [0, 0]
    >>> y = rastrigin(x)
    """
    n = len(x)
    rslt = a * n
    for y in x:
        assert abs(y) < 5.12
        rslt += y ** 2 - a * np.cos(2 * math.pi * y)
    return rslt
