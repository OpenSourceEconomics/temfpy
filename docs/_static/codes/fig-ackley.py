import numpy as np
import matplotlib.pyplot as plt
from temfpy.optimization import ackley

x1 = np.linspace(-32.768, 32.768, 500)
x2 = np.linspace(-32.768, 32.768, 500)

xvalues = []
array = []
for i in range(0, len(x1)):
    for j in range(0, len(x2)):
        array = [x1[i], x2[j]]
        xvalues.append(array)

yvalues = np.linspace(0, 0, len(xvalues))
for i, j in zip(xvalues, range(0, len(xvalues))):
    yvalues[j] = ackley(i)

xvalues = np.array(xvalues)
xvalues1 = xvalues[:, 0]
xvalues2 = xvalues[:, 1]

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter(xvalues1, xvalues2, yvalues, c=yvalues, cmap="viridis", linewidth=0.05)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("f(x1, x2)")
ax.set_title("Ackley function")
fig.savefig("docs/_static/images/fig-ackley-500")
