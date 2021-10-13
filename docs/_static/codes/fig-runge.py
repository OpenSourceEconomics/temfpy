"""Figure of the Runge function in 3D.

x is evaluated on [-1, 1]
y is the result of applying the Runge function on each element in x

"""
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial as Poly

from temfpy.interpolation import runge


def get_interpolator_runge_baseline(func, degree):
    xnodes = np.linspace(-1, 1, degree)
    poly = Poly.fit(xnodes, func(xnodes), degree)
    return poly


xvalues = np.linspace(-1, 1, 10000)

interpolant1 = get_interpolator_runge_baseline(runge, 5)
yfit1 = interpolant1(xvalues)

interpolant2 = get_interpolator_runge_baseline(runge, 10)
yfit2 = interpolant2(xvalues)

fig, ax = plt.subplots()
ax.plot(xvalues, runge(xvalues), label="Runge Function")
ax.plot(xvalues, yfit1, label="Approx. Degree 5")
ax.plot(xvalues, yfit2, label="Approx. Degree 10")
ax.set_xlabel("$x$")
ax.set_ylabel("$f(x)$")
ax.set_title("Runge Function and Polynomial Approximations over Uniform Grid")
ax.legend()

fig.savefig("fig-runge")
