import numpy as np
import math


def ackley(x, a=20, b=0.2, c=2 * math.pi):
    n = len(x)
    function = (
        a
        + math.exp(1)
        - (a * (math.exp(-b * math.sqrt(1 / n * np.sum(np.square(x))))))
        - math.exp(1 / n * np.sum(np.cos(c * x)))
    )
    return function


def rastrigin(y, a=10):
    n = len(y)
    rslt = a * n
    for x in y:
        if abs(x) > 5.12:
            return "At least one of the items in the list you provided is not within [-5.12, 5.12]."
        rslt += np.square(x) - a * np.cos(2 * math.pi * x)
    return rslt
