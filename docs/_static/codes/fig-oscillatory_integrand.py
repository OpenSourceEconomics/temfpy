"""Figure of the Oscillatory function in 3D.

x1 is evaluated on [0, 1]
x2 is evaluated on [0, 1]
y is the result of applying the Continuous Integrand function on
each combination of x1 and x2 and u1,u2 = 5

"""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from temfpy.integration import oscillatory

x1 = np.linspace(0, 1)
x2 = np.linspace(0, 1)

xvalues = []
array = []
for i in range(0, len(x1)):
    for j in range(0, len(x2)):
        array = [x1[i], x2[j]]
        xvalues.append(array)

yvalues = np.linspace(0, 0, len(xvalues))
for i, j in zip(xvalues, range(0, len(xvalues))):
    yvalues[j] = oscillatory(i, [5,5], 0.5)

xvalues = np.array(xvalues)
xvalues1 = xvalues[:, 0]
xvalues2 = xvalues[:, 1]

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(xvalues1, xvalues2, yvalues, c=yvalues, cmap="viridis", linewidth=0.05)
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$f(x_1, x_2)$")
fig.savefig("fig-oscillatory_integrand")
