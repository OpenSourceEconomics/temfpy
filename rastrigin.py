import numpy as np
import math

def rastrigin(y, a=10):
    n = len(y)
    rslt = a * n
    for x in y:
        if abs(x) > 5.12:
            return "At least one of the items in the list you provided is not within [-5.12, 5.12]."
        rslt += np.square(x) - a * np.cos(2*math.pi * x)
    return rslt
