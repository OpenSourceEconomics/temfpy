import numpy as np
import math


def borehole(x):
    """The Borehole function models water flow through a borehole. Its simplicity and quick
    evaluation makes it a commonly used function for testing a wide variety of methods in
    computer experiments.
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


def ishigami(x, a=7, b=0.1):
    """
    The Ishigami function of Ishigami & Homma (1990) is used as an example for uncertainty and
    sensitivity analysis methods, because it exhibits strong nonlinearity and nonmonotonicity.
    """
    assert len(x) == 3

    rslt = (1 + b * x[2] ** 4) * np.sin(x[0]) + a * np.sin(x[1]) ** 2
    return rslt
