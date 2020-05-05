import numpy as np
import math

def ackley(x, a = 20, b = 0.2, c = 2*math.pi):
    n = len(x)
    function = a + math.exp(1) - (a * (math.exp(-b * math.sqrt(1/n * np.sum(np.square(x)))))) - math.exp(1/n * np.sum(np.cos(c*x)))
    return function