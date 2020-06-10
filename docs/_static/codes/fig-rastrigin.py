"""Figure of the Rastrigin function in 3D.

x1 is evaluated on [-5.12, 5.12]
x2 is evaluated on [-5.12, 5.12]
y is the result of applying the Rastrigin function on each combination of x1 and x2

"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from temfpy.optimization import rastrigin

x1 = np.linspace(-5.12, 5.12, 500)
x2 = np.linspace(-5.12, 5.12, 500)

xvalues = []
array = []
for i in range(0, len(x1)):
    for j in range(0, len(x2)):
        array = [x1[i], x2[j]]
        xvalues.append(array)

yvalues = np.linspace(0, 0, len(xvalues))
for i, j in zip(xvalues, range(0, len(xvalues))):
    yvalues[j] = rastrigin(i)

xvalues = np.array(xvalues)
xvalues1 = xvalues[:, 0]
xvalues2 = xvalues[:, 1]

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(xvalues1, xvalues2, yvalues, c=yvalues, cmap="viridis", linewidth=0.05)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("f(x1, x2)")
ax.set_title("Rastrigin function")
fig.savefig("fig-rastrigin-500")
