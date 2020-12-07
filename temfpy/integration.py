"""Integration.
We provide a variety of non-linear equations used for testing
numerical integration algorithms.
"""

import sys

import numpy as np


def _vector_interval(x, lower, upper):
    r"""Check if one value of a numeric vector is not
        in a specific interval.

    Parameters
    ----------
    x : array_like
        Array for which it should be checked if all values lie in the
        specific interval.
    lower : float
            Lower bound of the interval.
    upper : float
            Upper bound of the interval

    Returns
    -------
    output : bool
             True if one component of the vector is not in the specified
             interval.

    """
    boolean = (x < lower).any() or (x > upper).any()

    if boolean:
        sys.exit(
            f"Any component of the input vector {x} must be between {lower} and {upper}"
        )


def continuous(x, u, a):
    r"""Continuous Integrand Family.

    .. math::
        f(x)=\exp{\left(-\sum_{i=1}^d a_i \mid x_i -u_i \mid \right)}

    Parameters
    ----------
    x : array_like
        Input domain with dimension :math:`d` and :math:`x \in [0,1]^d`.
    u : array_like
        Location vector with dimension :math:`d` and :math:`u \in [0,1]^d`.
    a : array_like
        Weight vector with dimension :math:`d`.

    Returns
    -------
    float
         Output domain

    Notes
    -----
    This function was proposed by Alan Genz in [G1984]_. It can be
    integrated analytically quickly with high precision. Evaluated at two
    dimensions, the function has a flat surface and one peak
    close to the centre of the integration region. Large values
    in the location vector :math:`a` result in a sharp peak and a more
    difficult integration.

    References
    ----------
    .. [G1984] Genz, A. (1984). Testing multidimensional integration routines.
       In Proc. of international conference on Tools, methods and languages
       for scientific and engineering computation (pp. 81-94). Elsevier
       North-Holland.

    Examples
    --------
    >>> import numpy as np
    >>> from temfpy.integration import continuous
    >>>
    >>> d = np.random.randint(1,20)
    >>> x = np.repeat(0.5,d)
    >>> u = x
    >>> a = np.repeat(5,d)
    >>> np.allclose(continuous(x,u,a), 1)
    True
    """
    x = np.array(x)
    u = np.array(u)
    a = np.array(a)
    _vector_interval(x, 0, 1)
    _vector_interval(u, 0, 1)

    fval = np.exp(-np.sum(np.abs(x - u) * a))

    return fval


def corner_peak(x, a):
    r"""Corner Peak Integrand Family.

    .. math::
        f(x)=\left(1 + \sum_{i=1}^d a_i x_i \right)^{d+1}

    Parameters
    ----------
    x : array_like
        Input domain with dimension :math:`d` and :math:`x \in [0,1]^d`.
    a : array_like
        Weight vector with dimension :math:`d`.

    Returns
    -------
    float
         Output domain

    Notes
    -----
    This function was proposed by Alan Genz in [G1984]_. It can be
    integrated analytically quickly with high precision. Evaluated at two
    dimensions, the function has a flat surface and one sharp peak
    in one corner of the integration region. Large values
    in the location vector :math:`a` result in a more sharp peak and a more
    difficult integration.

    References
    ----------
    .. [G1984] Genz, A. (1984). Testing multidimensional integration routines.
       In Proc. of international conference on Tools, methods and languages
       for scientific and engineering computation (pp. 81-94). Elsevier
       North-Holland.

    Examples
    --------
    >>> import numpy as np
    >>> from temfpy.integration import corner_peak
    >>>
    >>> d = np.random.randint(1,20)
    >>> x = np.repeat(0,d)
    >>> a = np.repeat(5,d)
    >>> np.allclose(corner_peak(x,a), 1)
    True
    """
    x = np.array(x)
    a = np.array(a)
    _vector_interval(x, 0, 1)
    power = float(-len(x) - 1)

    fval = (1 + np.sum(a * x)) ** (power)

    return fval


def discontinuous(x, u, a):
    r"""Discontinuous Integrand Family.

    .. math::
          f(x) = \begin{cases}
                 0, & x_1 > u_1 \text{ or } x_2 > u_2 \\
                 \exp{\left(\sum_{i=1}^d a_i x_i \right)}, & \text{otherwise}
                 \end{cases}

    Parameters
    ----------
    x : array_like
        Input domain with dimension :math:`d` and :math:`x \in [0,1]^d`.
    u : array_like
        Location vector with dimension :math:`1`, if :math:`d = 1` and
        dimension :math:`2`, if :math:`d > 1` that determines in which
        area the function is equal to zero.
    a : array_like
        Weight vector with dimension :math:`d`.

    Returns
    -------
    float
         Output domain

    Notes
    -----
    This function was proposed by Alan Genz in [G1984]_. It can be
    integrated analytically quickly with high precision. Evaluated at two
    dimensions, the function has one peak close to the centre of the
    integration region and is flat at zero for values greater as specified
    in :math:`u`. Large values
    in the location vector :math:`a` result in a sharp peak and a more
    difficult integration.

    References
    ----------
    .. [G1984] Genz, A. (1984). Testing multidimensional integration routines.
       In Proc. of international conference on Tools, methods and languages
       for scientific and engineering computation (pp. 81-94). Elsevier
       North-Holland.

    Examples
    --------
    >>> import numpy as np
    >>> from temfpy.integration import discontinuous
    >>>
    >>> d = np.random.randint(1,20)
    >>> x = np.repeat(0,d)
    >>> u = [0.5,0.5]
    >>> a = np.repeat(5,d)
    >>> np.allclose(discontinuous(x,u,a), 1)
    True
    """
    x = np.array(x)
    u = np.array(u)
    a = np.array(a)
    _vector_interval(x, 0, 1)

    if len(x) > 1:
        x_if = [x[0], x[1]]
        u_if = [u[0], u[1]]
    else:
        x_if = x
        u_if = u

    if (x_if <= u_if).all():
        fval = np.exp(np.sum(a * x))
    else:
        fval = 0

    return fval


def gaussian_peak(x, u, a):
    r"""Gaussian peak Integrand Family.

    .. math::
        f(x)=\exp{\left(-\sum_{i=1}^d a_i^2 (x_i -u_i)^2 \right)}

    Parameters
    ----------
    x : array_like
        Input domain with dimension :math:`d` and :math:`x \in [0,1]^d`.
    u : array_like
        Location vector with dimension :math:`d` and :math:`u \in [0,1]^d`.
    a : array_like
        Weight vector with dimension :math:`d`.

    Returns
    -------
    float
         Output domain

    Notes
    -----
    This function was proposed by Alan Genz in [G1984]_. It can be
    integrated analytically quickly with high precision. Evaluated at two
    dimensions, the function has a Gaussian peak
    close to the centre of the integration region. Large values
    in the location vector :math:`a` result in a sharp peak and a more
    difficult integration.

    References
    ----------
    .. [G1984] Genz, A. (1984). Testing multidimensional integration routines.
       In Proc. of international conference on Tools, methods and languages
       for scientific and engineering computation (pp. 81-94). Elsevier
       North-Holland.

    Examples
    --------
    >>> import numpy as np
    >>> from temfpy.integration import gaussian_peak
    >>>
    >>> d = np.random.randint(1,20)
    >>> x = np.repeat(0.5,d)
    >>> u = x
    >>> a = np.repeat(5,d)
    >>> np.allclose(gaussian_peak(x,u,a), 1)
    True
    """
    x = np.array(x)
    u = np.array(u)
    a = np.array(a)
    _vector_interval(x, 0, 1)
    _vector_interval(u, 0, 1)

    fval = np.exp(-np.sum(np.abs(x - u) ** 2 * a ** 2))

    return fval


def oscillatory(x, a, b):
    r"""Oscillatory Integrand Family.

    .. math::
        f(x)= \cos\left(2 \pi b + \sum_{i=1}^d a_i x_i \right)

    Parameters
    ----------
    x : array_like
        Input domain with dimension :math:`d` and :math:`x \in [0,1]^d`.
    a : array_like
        Weight vector with dimension :math:`d`.

    Returns
    -------
    float
         Output domain

    Notes
    -----
    This function was proposed by Alan Genz in [G1984]_. It can be
    integrated analytically quickly with high precision.Large values
    in the location vector :math:`a` result in a higher frequency of
    oscillations and a more difficult integration.

    References
    ----------
    .. [G1984] Genz, A. (1984). Testing multidimensional integration routines.
       In Proc. of international conference on Tools, methods and languages
       for scientific and engineering computation (pp. 81-94). Elsevier
       North-Holland.

    Examples
    --------
    >>> import numpy as np
    >>> from temfpy.integration import oscillatory
    >>>
    >>> d = np.random.randint(1,20)
    >>> x = np.repeat(np.pi/4,d)
    >>> a = np.repeat(-6/d,d)
    >>> b = 1
    >>> np.allclose(oscillatory(x,a,b), 0)
    True
    """
    x = np.array(x)
    a = np.array(a)
    _vector_interval(x, 0, 1)

    fval = np.cos(2 * np.pi * b + np.sum(a * x))

    return fval


def product(x, u, a):
    r"""Product Peak Integrand Family.

    .. math::
        f(x)=\prod \frac{1}{\sum_{i=1}^d a_i^{-2} (x_i -u_i)^2}

    Parameters
    ----------
    x : array_like
        Input domain with dimension :math:`d` and :math:`x \in [0,1]^d`.
    u : array_like
        Location vector with dimension :math:`d` and :math:`u \in [0,1]^d`.
    a : array_like
        Weight vector with dimension :math:`d`.

    Returns
    -------
    float
         Output domain

    Notes
    -----
    This function was proposed by Alan Genz in [G1984]_. It can be
    integrated analytically quickly with high precision. Evaluated at two
    dimensions, the function has a peak
    in the centre of the integration region. Large values
    in the location vector :math:`a` result in a larger peak and a more
    difficult integration.

    References
    ----------
    .. [G1984] Genz, A. (1984). Testing multidimensional integration routines.
       In Proc. of international conference on Tools, methods and languages
       for scientific and engineering computation (pp. 81-94). Elsevier
       North-Holland.

    Examples
    --------
    >>> import numpy as np
    >>> from temfpy.integration import product
    >>>
    >>> d = np.random.randint(1,20)
    >>> x = np.repeat(0.5,d)
    >>> u = x
    >>> a = np.repeat(1,d)
    >>> np.allclose(product(x,u,a), 1)
    True
    """
    x = np.array(x)
    u = np.array(u)
    a = np.array(a)
    _vector_interval(x, 0, 1)
    _vector_interval(u, 0, 1)

    fval = np.prod((a ** (float(-2)) + (x - u) ** 2) ** (float(-1)))

    return fval
