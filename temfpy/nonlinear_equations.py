"""Nonlinear equations.
We provide a variety of non-linear equations used for testing
numerical optimization algorithms.
"""
import sys

import numdifftools as nd
import numpy as np
import pandas as pd
import functools as ft


def _check_if_number(a, name):
    r"""Function to check if object `a` is a number.

    Parameters
    ----------
    a : object
        Object for which it should be tested whether it is a number or not.
    name : str
        String including the name of the object that appears in the error notification.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import sys
    >>>
    >>> from temfpy.nonlinear_equations import _check_if_number
    >>> a = 10
    >>> _check_if_number(a, 'a')
    """
    if not (isinstance(a, (int, float)) and not isinstance(a, bool)):
        sys.exit(f"The parameter `{name}` must either be of type int or float.")


def _check_if_array(x, name, length=None, length_type="equal"):
    r"""Function to check if object `x` is an array and optionally if it has a specified length.

        Currently considered as arrays are list, numpy.array or pandas.Series type of objects.

    Parameters
    ----------
    x : object
        Object for which it should be tested whether it is an array or not.
    name : str
           String including the name of the object that
           appears in the error notification.
    length : int
             Desired length of array `x` for which the function
             does not yield an error notification.
    length_type : str
                  Indicates how the array length should be evaluated.
                  'equal' yields an error if the length of `x`
                  is not equal to the integer specified for `length`
                  and 'grtr_equ' yields an error if the length of `x`
                  is smaller.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import sys
    >>> from temfpy.nonlinear_equations import _check_if_array
    >>>
    >>> x = [10, 1, 4, 6]
    >>> _check_if_array(x, 'x')
    """
    if not isinstance(x, (list, pd.core.series.Series, np.ndarray)):
        sys.exit(
            f"The parameter `{name}` must either be a list, numpy.array or pandas.Series.",
        )
    if length is not None:
        if (length_type == "equal") and (len(x) != length):
            sys.exit(f"The array `{name}` must have length {length}.")
        if (length_type == "grtr_equ") and (len(x) < length):
            sys.exit(f"The array `{name}` must have at least length {length}.")


def _exponential_val(x, a=10, b=1):
    r"""Exponential function.

    .. math::
        x &\mapsto \begin{pmatrix} F_1(x) & F_2(x) & \dots & F_p(x) \end{pmatrix}^T \\
        F_1(x) &= e^{x_1} - b \\
        F_i(x) &= \frac{i}{a} (e^{x_i} +x_{i-1}) - b, i = 2,3, \dots, p

    Parameters
    ----------
    x : array_like
        Input domain with dimension :math:`p`.
    a : float, optional
        The default value is 10.
    b : float, optional
        The default value is 1.

    Returns
    -------
    array_like
         Output domain

    Examples
    --------
    >>> import numpy as np
    >>> import numdifftools as nd
    >>> from temfpy.nonlinear_equations import _exponential_val
    >>>
    >>> p = 10
    >>> np.random.seed(123)
    >>> x = np.random.normal(size = p)
    >>> value = _exponential_val(x)
    """
    _check_if_number(a, "a")
    _check_if_number(b, "b")
    _check_if_array(x, "x", length=1, length_type="grtr_equ")
    x = np.array(x)

    p = len(x)

    x_im1 = np.concatenate((0, np.delete(x, p - 1)), axis=None)
    rslt = (
        (np.exp(x) + x_im1 - b)
        * np.concatenate((a, np.array(range(2, p + 1))), axis=None)
        / a
    )

    return np.array(rslt)


def _exponential_jacobian(x, a=10):
    r"""Analytical and numerical jacobian of the exponential function.

    .. math::
       F_1(x) &= e^{x_1} - b \\
       F_i(x) &= \frac{i}{a} (e^{x_i} +x_{i-1}) - b, i = 2,3, \dots, p

    Parameters
    ----------
    x : array_like
        Input domain with dimension :math:`p`.
    a : float, optional
        The default value is 10.

    Returns
    -------
    numpy.array
        Analytically derived Jacobian
    numpy.array
        Numerically derived Jacobian

    Examples
    --------
    >>> import numpy as np
    >>> import numdifftools as nd
    >>> from temfpy.nonlinear_equations import _exponential_jacobian
    >>>
    >>> p = 10
    >>> np.random.seed(123)
    >>> x = np.random.normal(size = p)
    >>> analytical_jacobian, numerical_jacobian = _exponential_jacobian(x)
    >>> np.allclose(analytical_jacobian, numerical_jacobian)
    True
    """
    _check_if_number(a, "a")
    _check_if_array(x, "x", length=1, length_type="grtr_equ")
    x = np.array(x)

    p = len(x)
    x = np.array(x)

    diag_mat = np.diag(
        np.exp(np.array(x))
        * np.array(range(1, p + 1))
        / np.append([1], np.repeat(a, p - 1)),
    )
    off_diag_mat = np.diag(
        np.array(range(2, p + 1)) / np.array(np.repeat(a, p - 1)),
        k=-1,
    )
    jacobian = diag_mat + off_diag_mat

    j_numdiff = nd.Jacobian(_exponential_val)
    jacobian_numdiff = j_numdiff(np.array(x))

    return jacobian, jacobian_numdiff


def exponential(x, a=10, b=1):
    r"""Exponential function.

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

    Returns
    -------
    array_like
        Output domain
    array_like
        Tuple containing the analytically derived Jacobian and the
        numerically derived Jacobian

    References
    ----------
    .. [V2009] Varadhan, R., and Gilbert, P. D. (2009). BB: An R Package for Solving a Large
               System of Nonlinear Equations and for Optimizing a High-Dimensional Nonlinear
               Objective Function. *Journal of Statistical Software*, 32(1):1–26, 2009.

    Examples
    --------
    >>> import numpy as np
    >>> import numdifftools as nd
    >>> from temfpy.nonlinear_equations import exponential
    >>>
    >>> p = 10
    >>> np.random.seed(123)
    >>> x = np.random.normal(size = p)
    >>> value, jacobian = exponential(x)
    >>> analytical_jacobian = jacobian[0]
    >>> numerical_jacobian = jacobian[1]
    >>> np.allclose(analytical_jacobian, numerical_jacobian)
    True
    """

    return _exponential_val(x, a=a, b=b), _exponential_jacobian(x, a=a)


def _trig_exp_i(xi, a=None):
    r"""Trigonometrical exponential function. Used to build
    the function :func:`trig_exp_val`.

    .. math::
        F_i(x) = - x_{i-1}e^(x_{i-1} - x_i) + x_i(a_4+a_5x_i^2)
        + a_6x_{i+1} + \sin(x_i - x_{i+1})\sin(x_i + x_{i+1}) - a_7,
        i = 2,3, \dots, p-1

    Parameters
    ----------
    x_i : array_like
        Input domain with dimension :math:`3`.
    a : array_like, optional
        The default array is [3,2,5,4,3,2,8,4,3].

    Returns
    -------
    float
         Output domain

    Examples
    --------
    >>> import numpy as np
    >>> import numdifftools as nd
    >>> from temfpy.nonlinear_equations import _trig_exp_i
    >>>
    >>> p = 3
    >>> np.random.seed(123)
    >>> x = np.random.normal(size = p)
    >>> F_i = _trig_exp_i(x)
    """
    if a is None:
        a = [3, 2, 5, 4, 3, 2, 8, 4, 3]

    _check_if_array(a, "a", length=9)
    _check_if_array(xi, "xi", length=3)
    xi = np.array(xi)
    a = np.array(a)

    rslt = (
        -xi[0] * np.exp(xi[0] - xi[1])
        + xi[1] * (a[3] + a[4] * xi[1] ** 2)
        + a[5] * xi[2]
        + np.sin(xi[1] - xi[2]) * np.sin(xi[1] + xi[2])
        - a[6]
    )

    return rslt


def _trig_exp_val(x, a=None):
    r"""Trigonometrical exponential function.

    .. math::
        F_1(x) &= a_1x_1^3 + a_2x_2 - a_3 + \sin(x_1 - x_2)\sin(x1+x2) \\
        F_i(x) &= - x_{i-1}e^{x_{i-1} - x_i} + x_i(a_4+a_5x_i^2)
        + a_6x_{i+1} + \sin(x_i - x_{i+1})\sin(x_i + x_{i+1}) - a_7,
        i = 2,3, \dots, p-1 \\
        F_p(x) &= -x_{p-1}e^{x_{p-1}-x_p} + a_8x_p - a_9

    Parameters
    ----------
    x : array_like
        Input domain with dimension :math:`p > 1`.
    a : array_like, optional
        The default array is [3,2,5,4,3,2,8,4,3].

    Returns
    -------
    array_like
         Output domain

    Examples
    --------
    >>> import numpy as np
    >>> import numdifftools as nd
    >>> from temfpy.nonlinear_equations import _trig_exp_val
    >>>
    >>> p = 10
    >>> np.random.seed(123)
    >>> x = np.random.normal(size = p)
    >>> value = _trig_exp_val(x)
    """
    if a is None:
        a = [3, 2, 5, 4, 3, 2, 8, 4, 3]

    _check_if_array(a, "a", length=9)
    _check_if_array(x, "x", length=1, length_type="grtr_equ")
    x = np.array(x)
    a = np.array(a)

    p = len(x)
    rslt = [
        a[0] * x[0] ** 3
        + a[1] * x[1]
        - a[2]
        + np.sin(x[0] - x[1]) * np.sin(x[0] + x[1]),
    ]
    for i in range(2, p):
        rslt.append(_trig_exp_i(xi=x[(i - 2) : (i + 1)], a=a))

    rslt.append(-x[p - 2] * np.exp(x[p - 2] - x[p - 1]) + a[7] * x[p - 1] - a[8])

    return np.array(rslt)


def _trig_exp_jacobian(x, a=None):
    r"""Trigonometrical exponential function.

    .. math::
        x &\mapsto \begin{pmatrix} F_1(x) & F_2(x) & \dots & F_p \end{pmatrix}^T \\
        F_1(x) &= a_1x_1^3 + a_2x_2 - a_3 + \sin(x_1 - x_2)\sin(x1+x2) \\
        F_i(x) &= - x_{i-1}e^{x_{i-1} - x_i} + x_i(a_4+a_5x_i^2)
        + a_6x_{i+1} \\
        & \quad + \sin(x_i - x_{i+1})\sin(x_i + x_{i+1}) - a_7,
        i = 2,3, \dots, p-1 \\
        F_p(x) &= -x_{p-1}e^{x_{p-1}-x_p} + a_8x_p - a_9

    Parameters
    ----------
    x : array_like
        Input domain with dimension :math:`p`.
    a : array_like, optional
        The default array is [3,2,5,4,3,2,8,4,3].

    Returns
    -------
    array_like
         Output domain

    Examples
    --------
    >>> import numpy as np
    >>> import numdifftools as nd
    >>> from temfpy.nonlinear_equations import _trig_exp_jacobian
    >>>
    >>> p = 10
    >>> np.random.seed(123)
    >>> x = np.random.normal(size = p)
    >>> analytical_jacobian, numerical_jacobian = _trig_exp_jacobian(x)
    >>> np.allclose(analytical_jacobian, numerical_jacobian)
    True
    """
    if a is None:
        a = [3, 2, 5, 4, 3, 2, 8, 4, 3]

    _check_if_array(a, "a", length=9)
    _check_if_array(x, "x", length=1, length_type="grtr_equ")
    x = np.array(x)
    a = np.array(a)

    p = len(x)

    x_im1 = np.delete(x, [p - 1, p - 2])
    x_i = np.delete(x, [0, p - 1])
    x_ip1 = np.delete(x, [0, 1])
    diag_mat = np.diag(
        np.concatenate(
            (
                np.array(
                    3 * a[0] * x[0] ** 2
                    + np.sin(x[0] - x[1]) * np.cos(x[0] + x[1])
                    + np.cos(x[0] - x[1]) * np.sin(x[0] + x[1]),
                ),
                (
                    x_im1 * np.exp(x_im1 - x_i)
                    + np.repeat(a[3], p - 2)
                    + 3 * np.repeat(a[4], p - 2) * x_i ** 2
                    + np.sin(x_i - x_ip1) * np.cos(x_i + x_ip1)
                    + np.cos(x_i - x_ip1) * np.sin(x_i + x_ip1)
                ),
                np.array(x[p - 2] * np.exp(x[p - 2] - x[p - 1]) + a[7]),
            ),
            axis=None,
        ),
    )
    off_diag_p1_mat = np.diag(
        (
            np.concatenate((a[1], np.repeat(a[5], p - 2)), axis=None)
            + np.sin(np.delete(x, [p - 1]) - np.delete(x, [0]))
            * np.cos(np.delete(x, [p - 1]) + np.delete(x, [0]))
            - np.cos(np.delete(x, [p - 1]) - np.delete(x, [0]))
            * np.sin(np.delete(x, [p - 1]) + np.delete(x, [0]))
        ),
        k=1,
    )
    off_diag_m1_mat = np.diag(
        -(np.delete(x, [p - 1]) + np.repeat(1, p - 1))
        * np.exp(np.delete(x, [p - 1]) - np.delete(x, [0])),
        k=-1,
    )
    jacobian = off_diag_p1_mat + off_diag_m1_mat + diag_mat

    j_numdiff = nd.Jacobian(_trig_exp_val)
    jacobian_numdiff = j_numdiff(np.array(x))

    return jacobian, jacobian_numdiff


def trig_exp(x, a=None):
    r"""Trigonometrical exponential function.

    .. math::
        x &\mapsto \begin{pmatrix} F_1(x) & F_2(x) & \dots & F_p(x) \end{pmatrix}^T \\
        F_1(x) &= a_1x_1^3 + a_2x_2 - a_3 + \sin(x_1 - x_2)\sin(x_1+x_2) \\
        F_i(x) &= - x_{i-1}e^{x_{i-1} - x_i} + x_i(a_4+a_5x_i^2)
        + a_6x_{i+1} \\
        & \quad + \sin(x_i - x_{i+1})\sin(x_i + x_{i+1}) - a_7 \\
        & \quad i = 2,3, \dots, p-1 \\
        F_p(x) &= -x_{p-1}e^{x_{p-1}-x_p} + a_8x_p - a_9

    Parameters
    ----------
    x : array_like
        Input domain with dimension :math:`p > 1`.
    a : array_like, optional
        The default array is [3,2,5,4,3,2,8,4,3].

    Returns
    -------
    array_like
        Output domain
    array_like
        Tuple containing the analytically derived Jacobian and the numerically derived Jacobian.

    References
    ----------
    .. [V2009] Varadhan, R., and Gilbert, P. D. (2009). BB: An R Package for Solving a Large
               System of Nonlinear Equations and for Optimizing a High-Dimensional Nonlinear
               Objective Function. *Journal of Statistical Software*, 32(1):1–26, 2009.

    Examples
    --------
    >>> import numpy as np
    >>> import numdifftools as nd
    >>> from temfpy.nonlinear_equations import trig_exp
    >>>
    >>> p = 10
    >>> np.random.seed(123)
    >>> x = np.random.normal(size = p)
    >>> value, jacobian = trig_exp(x)
    >>> analytical_jacobian = jacobian[0]
    >>> numerical_jacobian = jacobian[1]
    >>> np.allclose(analytical_jacobian, numerical_jacobian)
    True
    """
    if a is None:
        a = [3, 2, 5, 4, 3, 2, 8, 4, 3]

    return _trig_exp_val(x, a=a), _trig_exp_jacobian(x, a=a)


def _broyden_val(x, a=None):
    r"""Broyden tridiagonal function.

    .. math::
        F_1(x) &= x_1(a_1 - a_2 x_1) -a_3 x_{2} + a_4 \\
        F_i(x) &= x_i(a_1 - a_2 x_i)-x_{i-1} -a_3 x_{i+1}
        + a_4, i = 2,3, \dots, p-1 \\
        F_p(x) &= x_p(a_1 - a_2 x_p)-x_{p-1} + a_4

    Parameters
    ----------
    x : array_like
        Input domain with dimension :math:`p > 1`.
    a : array_like, optional
        The default array is [3, 0.5, 2, 1]

    Returns
    -------
    float
         Output domain

    Examples
    --------
    >>> import numpy as np
    >>> import numdifftools as nd
    >>> from temfpy.nonlinear_equations import _broyden_val
    >>>
    >>> p = 10
    >>> np.random.seed(123)
    >>> x = - np.random.uniform(size = p)
    >>> value = _broyden_val(x)
    """
    if a is None:
        a = [3, 0.5, 2, 1]

    _check_if_array(a, "a", length=4)
    _check_if_array(x, "x", length=1, length_type="grtr_equ")
    x = np.array(x)
    a = np.array(a)

    p = len(x)

    x_ip1 = np.concatenate((np.delete(x, 0), 0), axis=None)
    x_im1 = np.concatenate((0, np.delete(x, p - 1)), axis=None)

    rslt = x * (a[0] - a[1] * x) - x_im1 - a[2] * x_ip1 + 1

    return np.array(rslt)


def _broyden_jacobian(x, a=None):
    r"""Broyden tridiagonal function.

    .. math::
        F_1(x) &= x_1(a_1 - a_2 x_1) -a_3 x_{2} + a_4 \\
        F_i(x) &= x_i(a_1 - a_2 x_i)-x_{i-1} -a_3 x_{i+1}
        + a_4, i = 2,3, \dots, p-1 \\
        F_p(x) &= x_p(a_1 - a_2 x_p)-x_{p-1} + a_4

    Parameters
    ----------
    x : array_like
        Input domain with dimension :math:`p > 1`.
    a : array_like, optional
        The default array is [3, 0.5, 2, 1]

    Returns
    -------
    numpy.array
        Analytically derived Jacobian
    numpy.array
        Numerically derived Jacobian

    Examples
    --------
    >>> import numpy as np
    >>> import numdifftools as nd
    >>> from temfpy.nonlinear_equations import _broyden_jacobian
    >>>
    >>> p = 10
    >>> np.random.seed(123)
    >>> x = - np.random.uniform(size = p)
    >>> analytical_jacobian, numerical_jacobian = _broyden_jacobian(x)
    >>> np.allclose(analytical_jacobian, numerical_jacobian)
    True
    """
    if a is None:
        a = [3, 0.5, 2, 1]

    _check_if_array(a, "a", length=4)
    _check_if_array(x, "x", length=1, length_type="grtr_equ")
    x = np.array(x)
    a = np.array(a)

    p = len(x)

    jacobian = (
        np.diag(np.repeat(a[0], p) - x * np.repeat(2 * a[1], p))
        + np.diag(np.repeat(-a[2], p - 1), k=1)
        + np.diag(np.repeat(-1, p - 1), k=-1)
    )

    j_numdiff = nd.Jacobian(_broyden_val)
    jacobian_numdiff = j_numdiff(np.array(x))

    return jacobian, jacobian_numdiff


def broyden(x, a=None):
    r"""Broyden tridiagonal function.

    .. math::
       x &\mapsto \begin{pmatrix} F_1(x) & F_2(x) & \dots & F_p(x) \end{pmatrix}^T \\
       F_1(x) &= x_1(a_1 - a_2 x_1) -a_3 x_{2} + a_4 \\
       F_i(x) &= x_i(a_1 - a_2 x_i)-x_{i-1} -a_3 x_{i+1} + a_4 \\
       & \quad i = 2,3, \dots, p-1 \\
       F_p(x) &= x_p(a_1 - a_2 x_p)-x_{p-1} + a_4

    Parameters
    ----------
    x : array_like
        Input domain with dimension :math:`p`.
    a : array_like, optional
        The default array is [3, 0.5, 2, 1].

    Returns
    -------
    array_like
        Output domain.
    array_like
        Tuple containing the analytically derived Jacobian and the numerically derived Jacobian.

    References
    ----------
    .. [V2009] Varadhan, R., and Gilbert, P. D. (2009). BB: An R Package for Solving a Large
               System of Nonlinear Equations and for Optimizing a High-Dimensional Nonlinear
               Objective Function. *Journal of Statistical Software*, 32(1):1–26, 2009.

    Examples
    --------
    >>> import numpy as np
    >>> import numdifftools as nd
    >>> from temfpy.nonlinear_equations import broyden
    >>>
    >>> p = 10
    >>> np.random.seed(123)
    >>> x = - np.random.uniform(size = p)
    >>> value, jacobian = broyden(x)
    >>> analytical_jacobian = jacobian[0]
    >>> numerical_jacobian = jacobian[1]
    >>> np.allclose(analytical_jacobian, numerical_jacobian)
    True
    """
    if a is None:
        a = [3, 0.5, 2, 1]

    return _broyden_val(x=x, a=a), _broyden_jacobian(x=x, a=a)


def _rosenbrock_ext_val(x, a=None):
    r"""Extended-Rosenbrock function.

    .. math::
        F_{2i-1}(x) &= a_1(x_{2i} - x_{2i-1}^2) \\
        F_{2i}(x) &= a_2 - x_{2i-1}, i = 1,2,3, \dots, \frac{p}{2}


    Parameters
    ----------
    x : array_like
        Input domain with dimension :math:`p`.
    a : array_like, optional
        The default array is [10,1]


    Returns
    -------
    float
         Output domain

    Examples
    --------
    >>> import numpy as np
    >>> import numdifftools as nd
    >>> from temfpy.nonlinear_equations import _rosenbrock_ext_val
    >>>
    >>> p = 10
    >>> np.random.seed(123)
    >>> x = - np.random.uniform(size = p)
    >>> value = _rosenbrock_ext_val(x)


    """
    if a is None:
        a = [10, 1]

    _check_if_array(a, "a", length=2)
    _check_if_array(x, "x", length=1, length_type="grtr_equ")
    x = np.array(x)
    a = np.array(a)

    p = len(x)

    xl = np.concatenate((np.delete(x, 0), 0), axis=None)
    xh = np.concatenate((np.delete(x, p - 1), 0), axis=None)
    f_odd = a[0] * (xl - xh ** 2) * np.resize((1, 0), p)
    f_even = np.delete(1 - np.concatenate((0, x), axis=None), p) * np.resize((0, 1), p)

    rslt = f_odd + f_even

    return np.array(rslt)


def _rosenbrock_ext_jacobian(x, a=None):
    r"""Extended-Rosenbrock function.

    .. math::
        F_{2i-1}(x) &= a_1(x_{2i} - x_{2i-1}^2) \\
        F_{2i}(x) &= a_2 - x_{2i-1}, i = 1,2,3, \dots, \frac{p}{2}

    Parameters
    ----------
    x : array_like
        Input domain with dimension :math:`p`.
    a : array_like, optional
        The default array is [10,1]


    Returns
    -------
    numpy.array
        Analytically derived Jacobian
    numpy.array
        Numerically derived Jacobian

    Examples
    --------
    >>> import numpy as np
    >>> import numdifftools as nd
    >>> from temfpy.nonlinear_equations import _rosenbrock_ext_jacobian
    >>>
    >>> p = 10
    >>> np.random.seed(123)
    >>> x = np.random.uniform(size = p)
    >>> analytical_jacobian, numerical_jacobian = _rosenbrock_ext_jacobian(x)
    >>> np.allclose(analytical_jacobian, numerical_jacobian)
    True
    """
    if a is None:
        a = [10, 1]

    _check_if_array(a, "a", length=2)
    _check_if_array(x, "x", length=1, length_type="grtr_equ")
    x = np.array(x)
    a = np.array(a)

    p = len(x)

    diag_mat = np.diag(np.repeat(-2, p) * np.repeat(a[0], p) * x * np.resize([1, 0], p))
    off_diag_p1_mat = np.diag(np.resize([a[0], 0], p - 1), k=1)
    off_diag_m1_mat = np.diag(np.resize([-1, 0], p - 1), k=-1)
    jacobian = diag_mat + off_diag_p1_mat + off_diag_m1_mat

    j_numdiff = nd.Jacobian(_rosenbrock_ext_val)
    jacobian_numdiff = j_numdiff(np.array(x))

    return jacobian, jacobian_numdiff


def rosenbrock_ext(x, a=None):
    r"""Extended-Rosenbrock function.

    .. math::
        x &\mapsto \begin{pmatrix} F_1(x) & F_2(x) & \dots & F_p(x) \end{pmatrix}^T \\
        F_{2i-1}(x) &= a_1(x_{2i} - x_{2i-1}^2) \\
        F_{2i}(x) &= a_2 - x_{2i-1}, \\
        & \quad i = 1,2,3, \dots, \frac{p}{2}

    Parameters
    ----------
    x : array_like
        Input domain with dimension :math:`p`.
    a : array_like, optional
        The default array is [10,1]

    Returns
    -------
    array_like
        Output domain
    array_like
        Tuple containing the analytically derived Jacobian and the
        numerically derived Jacobian

    References
    ----------
    .. [V2009] Varadhan, R., and Gilbert, P. D. (2009).
    BB: An R Package for Solving a Large System of Nonlinear Equations and for
    Optimizing a High-Dimensional Nonlinear Objective Function.

    Examples
    --------
    >>> import numpy as np
    >>> import numdifftools as nd
    >>> from temfpy.nonlinear_equations import rosenbrock_ext

    >>> p = 10
    >>> np.random.seed(123)
    >>>
    >>> x = np.random.uniform(size = p)
    >>> value, jacobian = rosenbrock_ext(x)
    >>> analytical_jacobian = jacobian[0]
    >>> numerical_jacobian = jacobian[1]
    >>> np.allclose(analytical_jacobian, numerical_jacobian)
    True
    """
    if a is None:
        a = [10, 1]

    return _rosenbrock_ext_val(x=x, a=a), _rosenbrock_ext_jacobian(x=x, a=a)


def _troesch_val(x, rho=10, a=2):
    r"""Troesch function.

    .. math::
        F_1(x) &= a_1x_1 + \rho h^2 \sinh(\rho x_1) - x_{2}, \\
        F_i(x) &= a_1x_i + \rho h^2 \sinh(\rho x_i) - x_{i-1} - x_{i+1}, i = 2,3, \dots, p-1\\
        F_p(x) &= a_1x_p + \rho h^2 \sinh(\rho x_p) - x_{p-1}

    Parameters
    ----------
    x : array_like
        Input domain with dimension :math:`p > 1`.
    rho : float, optional
        The default value is 10
    a : float, optional
        The default value is 2

    Returns
    -------
    array_like
        Output domain
    Notes
    -----
    :math:'h = \frac{1}{p+1}'

    Examples
    --------
    >>> import numpy as np
    >>> import numdifftools as nd
    >>> from temfpy.nonlinear_equations import _troesch_val
    >>>
    >>> p = 10
    >>> np.random.seed(123)
    >>> x = np.random.uniform(size = p)
    >>> val = _troesch_val(x)
    """
    _check_if_number(rho, "rho")
    _check_if_number(a, "a")
    _check_if_array(x, "x", length=1, length_type="grtr_equ")
    x = np.array(x)

    p = len(x)
    h = 1 / (p + 1)

    x_ip1 = np.concatenate((np.delete(x, 0), 1), axis=None)
    x_im1 = np.concatenate((0, np.delete(x, p - 1)), axis=None)

    rslt = a * x + rho * h ** 2 * np.sinh(rho * x) - x_ip1 - x_im1

    return np.array(rslt)


def _troesch_jacobian(x, rho=10, a=2):
    r"""Troesch function.

    .. math::
        x &\mapsto \begin{pmatrix} F_1(x) & F_2(x) & \dots & F_p(x) \end{pmatrix}^T \\
        h &= \frac{1}{p+1} \\
        F_1(x) &= ax_1 + \rho h^2 \sinh(\rho x_1) - x_{2}, \\
        F_i(x) &= ax_i + \rho h^2 \sinh(\rho x_i) - x_{i-1} - x_{i+1} \\
        & \quad i = 2,3, \dots, p-1 \\
        F_p(x) &= ax_p + \rho h^2 \sinh(\rho x_p) - x_{p-1

    Parameters
    ----------
    x : array_like
        Input domain with dimension :math:`p > 1`.
    rho : float, optional
        The default value is 10
    a : float, optional
        The default value is 2

    Returns
    -------
    numpy.array
        Analytically derived Jacobian
    numpy.array
        Numerically derived Jacobian

    Examples
    --------
    >>> import numpy as np
    >>> import numdifftools as nd
    >>> from temfpy.nonlinear_equations import _troesch_jacobian
    >>>
    >>> p = 10
    >>> np.random.seed(123)
    >>> x = np.random.uniform(size = p)
    >>> analytical_jacobian, numerical_jacobian = _troesch_jacobian(x)
    >>> np.allclose(analytical_jacobian, numerical_jacobian)
    True
    """
    _check_if_number(rho, "rho")
    _check_if_number(a, "a")
    _check_if_array(x, "x", length=1, length_type="grtr_equ")
    x = np.array(x)

    p = len(x)
    h = 1 / (p + 1)

    diag_mat = np.diag(
        np.repeat(a, p)
        + np.repeat(rho ** 2, p)
        * np.repeat(h ** 2, p)
        * np.cosh(np.repeat(rho, p) * x),
    )
    off_diag_p1_mat = np.diag(np.repeat(-1, p - 1), k=1)
    off_diag_m1_mat = np.diag(np.repeat(-1, p - 1), k=-1)
    jacobian = diag_mat + off_diag_p1_mat + off_diag_m1_mat

    j_numdiff = nd.Jacobian(_troesch_val)
    jacobian_numdiff = j_numdiff(np.array(x))

    return jacobian, jacobian_numdiff


def troesch(x, rho=10, a=2):
    r"""Troesch function.

    .. math::
        x &\mapsto \begin{pmatrix} F_1(x) & F_2(x) & \dots & F_p(x) \end{pmatrix}^T \\
        h &= \frac{1}{p+1} \\
        F_1(x) &= ax_1 + \rho h^2 \sinh(\rho x_1) - x_{2}, \\
        F_i(x) &= ax_i + \rho h^2 \sinh(\rho x_i) - x_{i-1} - x_{i+1} \\
        & \quad i = 2,3, \dots, p-1 \\
        F_p(x) &= ax_p + \rho h^2 \sinh(\rho x_p) - x_{p-1}

    Parameters
    ----------
    x : array_like
        Input domain with dimension :math:`p`.
    rho : float, optional
        The default value is 10
    a : float, optional
        The default value is 2

    Returns
    -------
    array_like
        Output domain
    array_like
        Tuple containing the analytically derived Jacobian and the
        numerically derived Jacobian

    References
    ----------
    .. [V2009] Varadhan, R., and Gilbert, P. D. (2009). BB: An R Package for Solving a Large
               System of Nonlinear Equations and for Optimizing a High-Dimensional Nonlinear
               Objective Function. *Journal of Statistical Software*, 32(1):1–26, 2009.

    Examples
    --------
    >>> import numpy as np
    >>> import numdifftools as nd
    >>> from temfpy.nonlinear_equations import troesch
    >>>
    >>> p = 10
    >>> np.random.seed(123)
    >>> x = np.random.uniform(size = p)
    >>> value, jacobian = troesch(x)
    >>> analytical_jacobian = jacobian[0]
    >>> numerical_jacobian = jacobian[1]
    >>> np.allclose(analytical_jacobian, numerical_jacobian)
    True
    """

    return _troesch_val(x=x, rho=rho, a=a), _troesch_jacobian(x=x, rho=rho, a=a)


def _chandrasekhar_val(x, y, c):
    r"""Discretized version of Chandrasekhar’s H-equation.

    .. math::
        x &\mapsto \begin{pmatrix} F_1(x) & F_2(x) & \dots & F_p(x) \end{pmatrix}^T \\
        F_i(x) &= x_i - \left(1 - \frac{c}{2p} \sum^p_{j=1}
        \frac{y_i x_j}{y_i + y_j} \right)^{-1} \\
        & \quad i = 1,2, \dots, p

    Parameters
    ----------
    x : array_like
        Input domain with dimension :math:`p`.
    y : array_like,
        Array of constants with dimension :math:`p`
    c : float
        Constant parameter

    Returns
    -------
    array_like
        Output domain
    array_like
        Tuple containing the analytically derived Jacobian and the
        numerically derived Jacobian

    References
    ----------
    .. [V2009] Varadhan, R., and Gilbert, P. D. (2009). BB: An R Package for Solving a Large
               System of Nonlinear Equations and for Optimizing a High-Dimensional Nonlinear
               Objective Function. *Journal of Statistical Software*, 32(1):1–26, 2009.

    Examples
    --------
    >>> import numpy as np
    >>> import numdifftools as nd
    >>> from temfpy.nonlinear_equations import chandrasekhar
    >>>
    >>> p = 10
    >>> np.random.seed(123)
    >>> x = np.random.uniform(size = p)
    >>> y = np.random.normal(size = p)
    >>> c = 2
    >>> value, jacobian = chandrasekhar(x,y, c)
    >>> analytical_jacobian = jacobian[0]
    >>> numerical_jacobian = jacobian[1]
    >>> np.allclose(analytical_jacobian, numerical_jacobian)
    True
    """
    p = len(x)
    x = np.array(x)
    y = np.array(y)

    term_sum = []
    for i in range(0, p):
        term_sum.append(np.sum(x / (y[i] + y)))

    rslt = x - 1 / (1 - c * y / (2 * p) * term_sum)

    return np.array(rslt)


def _chandrasekhar_jacobian(x, y, c):
    r"""Discretized version of Chandrasekhar’s H-equation.

    .. math::
        F_i(x) = x_i - \left(1 - \frac{c}{2p} \sum^p_{j=1}
        \frac{y_i x_j}{y_i + y_j} \right)^{-1}
        i = 1,2, \dots, p

    Parameters
    ----------
    x : array_like
        Input domain with dimension :math:`p`.
    y : array_like,
        Array of constants with dimension :math:'p'
    c : float
        Constant parameter

    Returns
    -------
    numpy.array
        Analytically derived Jacobian
    numpy.array
        Numerically derived Jacobian

    Examples
    --------
    >>> import numpy as np
    >>> import numdifftools as nd
    >>> from temfpy.nonlinear_equations import _chandrasekhar_jacobian
    >>>
    >>> p = 10
    >>> np.random.seed(123)
    >>> x = np.random.uniform(size = p)
    >>> y = np.random.normal(size = p)
    >>> c = 2
    >>> analytical_jacobian, numerical_jacobian = _chandrasekhar_jacobian(x,y, c)
    >>> np.allclose(analytical_jacobian, numerical_jacobian)
    True

    """

    _check_if_number(c, "c")

    _check_if_array(x, "x", length=1, length_type="grtr_equ")
    _check_if_array(y, "y")

    if len(x) != len(y):
        sys.exit("The arrays `x` and `y` must have the same length.")

    p = len(x)
    x = np.array(x)
    y = np.array(y)

    jacobian = np.zeros((p, p))
    for k in range(0, p):
        term_sum = []
        for i in range(0, p):
            term_sum.append(np.sum(x / (y[i] + y)))

        column_k = (
            -1 / (1 - c * y / (2 * p) * term_sum) ** 2 * (c * y / (2 * p) / (y + y[k]))
        )

        jacobian[:, k] = column_k
    jacobian = jacobian + np.identity(p)

    _chandrasekhar_help_num_ft = ft.partial(_chandrasekhar_val, y=y, c=c)

    j_numdiff = nd.Jacobian(_chandrasekhar_help_num_ft)
    jacobian_numdiff = j_numdiff(np.array(x))

    return jacobian, jacobian_numdiff


def chandrasekhar(x, y, c):
    r"""Discretized version of Chandrasekhar’s H-equation.

    .. math::
        x &\mapsto \begin{pmatrix} F_1(x) & F_2(x) & \dots & F_p(x) \end{pmatrix}^T \\
        F_i(x) &= x_i - \left(1 - \frac{c}{2p} \sum^p_{j=1}
        \frac{y_i x_j}{y_i + y_j} \right)^{-1} \\
        & \quad i = 1,2, \dots, p

    Parameters
    ----------
    x : array_like
        Input domain with dimension :math:`p`.
    y : array_like,
        Array of constants with dimension :math:`p`
    c : float
        Constant parameter

    Returns
    -------
    array_like
        Output domain
    array_like
        Tuple containing the analytically derived Jacobian and the
        numerically derived Jacobian

    References
    ----------
    .. [V2009] Varadhan, R., and Gilbert, P. D. (2009). BB: An R Package for Solving a Large
               System of Nonlinear Equations and for Optimizing a High-Dimensional Nonlinear
               Objective Function. *Journal of Statistical Software*, 32(1):1–26, 2009.

    Examples
    --------
    >>> import numpy as np
    >>> import numdifftools as nd
    >>> from temfpy.nonlinear_equations import chandrasekhar
    >>>
    >>> p = 10
    >>> np.random.seed(123)
    >>> x = np.random.uniform(size = p)
    >>> y = np.random.normal(size = p)
    >>> c = 2
    >>> value, jacobian = chandrasekhar(x,y, c)
    >>> analytical_jacobian = jacobian[0]
    >>> numerical_jacobian = jacobian[1]
    >>> np.allclose(analytical_jacobian, numerical_jacobian)
    True
    """
    _check_if_number(c, "c")

    _check_if_array(x, "x", length=1, length_type="grtr_equ")
    _check_if_array(y, "y")

    if len(x) != len(y):
        sys.exit("The arrays `x` and `y` must have the same length.")

    return _chandrasekhar_val(x=x, y=y, c=c), _chandrasekhar_jacobian(x=x, y=y, c=c)
