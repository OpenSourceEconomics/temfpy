import numpy as np
import math


def ackley(x, a=20, b=0.2, c=2 * math.pi):
    n = len(x)
    rslt = (
        a
        + math.exp(1)
        - (a * (math.exp(-b * math.sqrt(1 / n * np.sum(x ** 2)))))
        - math.exp(1 / n * np.sum(np.cos(c * x)))
    )
    return rslt


def rastrigin(x, a=10):
    n = len(x)
    rslt = a * n
    for y in x:
        assert abs(y) < 5.12
        rslt += y ** 2 - a * np.cos(2 * math.pi * y)
    return rslt
