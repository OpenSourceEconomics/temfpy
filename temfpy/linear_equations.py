"""Linear equations.
We provide a variety of linear equations.
"""
import sys

import numpy as np


def get_ill_problem_1(n):

    r"""Get ill problem.

    .. math::
        x &\mapsto \begin{pmatrix} F_1(x) & F_2(x) & \dots & F_p(x) \end{pmatrix}^T \\
        F_1(x) &= e^{x_1} - b \\
        F_i(x) &= \frac{i}{a} (e^{x_i} +x_{i-1}) - b \\
        & \quad i = 2,3, \dots, p

    Parameters
    ----------
    x : array_like
        Input domain with dimension :math:`p`.
    a : float, optional
        The default value is 10.
    b : float, optional
        The default value is 1.
    jac : bool
          If True, an additional array containing the numerically derived Jacobian
          and the analytically derived Jacobian is returned.
          The default is False.

    Returns
    -------
    array_like
        Output domain
    array_like
        Only returned if :math:`jac = True`.
        Tuple containing the analytically derived Jacobian and the
        numerically derived Jacobian.

    References
    ----------
    .. [V2009] Varadhan, R., and Gilbert, P. D. (2009). BB: An R package for solving a
               large system of nonlinear equations and for optimizing a high-dimensional
               nonlinear objective function. *Journal of Statistical Software*,
               32(1): 1–26.

    Examples
    --------
    >>> import numpy as np
    >>> from temfpy.nonlinear_equations import exponential
    >>>
    >>> np.random.seed(123)
    >>> p = np.random.randint(1,20)
    >>> x = np.zeros(p)
    >>> np.allclose(exponential(x), np.zeros(p))
    True
    """

    a = np.vander(1 + np.arange(n))
    x = np.ones(n)
    b = a @ x

    return a, b, x
