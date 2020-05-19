import numpy as np
import math


"""
The Borehole function models water flow through a borehole. Its simplicity and quick evaluation makes it a commonly used function for testing a wide variety of methods in computer experiments.
"""


def borehole(x):
    r_w = np.random.normal(0.1, 0.0161812)
    r = np.random.lognormal(7.71, 1.0056)
    T_u = np.random.uniform(63070, 115600)
    H_u = np.random.uniform(990, 1110)
    T_l = np.random.uniform(63.1, 116)
    H_l = np.random.uniform(700, 820)
    L = np.random.uniform(1120, 1680)
    K_w = np.random.uniform(9855, 12045)

    a = 2 * math.pi * T_u * (H_u - H_l)
    b = np.log(r / r_w)
    c = (2 * L * T_u) / (b * r_w ** 2 * K_w)
    d = T_u / T_l

    rslt = a / (b * (1 + c + d))
    return rslt


"""
The Ishigami function of Ishigami & Homma (1990) is used as an example for uncertainty and sensitivity analysis methods, because it exhibits strong nonlinearity and nonmonotonicity.
"""


def ishigami(x, y, z, a=7, b=0.1):
    rslt = (1 + b * z ** 4) * np.sin(x) + a * np.sin(y) ** 2
    return rslt
