"""Linear equations.
We provide a variety of linear equations.
"""
import sys

import numpy as np


def get_ill_cond_lin_eq(n):

    r"""Create ill-conditioned system of linear equations given a 1-D solution array of ones.

    .. math:: Ax=b

    Parameters
    ----------

    n :  non-negative integer
         Dimension of linear equation.

    Returns
    -------

    a : array_like
        2-D ill-conditioned array.
    x : array_like
        1-D solution array of ones.
    b : array_like
        1-D data array.

    References
    ----------
    .. [V2009] Varadhan, R., and Gilbert, P. D. (2009). BB: An R package for solving a
               large system of nonlinear equations and for optimizing a high-dimensional
               nonlinear objective function. *Journal of Statistical Software*,
               32(1): 1–26.

    Examples
    --------
    >>> import numpy as np
    >>> from temfpy.linear_equations import get_ill_cond_lin_eq
    >>>
    >>> n = 5
    >>> a, x, b = get_ill_cond_lin_eq(n)
    """

    if n <= 0:
        sys.exit("n must be a non-negative, non-zero integer.")

    a = np.vander(1 + np.arange(n))
    x = np.ones(n)
    b = a @ x

    return a, x, b
