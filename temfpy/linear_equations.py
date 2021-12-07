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
    .. [W2021] *Vandermonde matrix*. (2021, December 1). In Wikipedia.
                Retrieved from
                https://en.wikipedia.org/wiki/Vandermonde_matrix.

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
