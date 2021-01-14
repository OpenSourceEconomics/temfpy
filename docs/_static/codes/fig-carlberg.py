"""Figure of the carlberg function with noise in 3D.

x1 is evaluated on [-2, 5]
x2 is evaluated on [-2, 5]
y is the result of applying the carlberg function on each combination of x1 and x2

"""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from temfpy.optimization import carlberg


def plot_carlberg(x1, x2, a, b, title, save_name):
    r"""plot the carlberg function.

    Parameters
    ----------
    x1 : array_like
        Input vector with points at which the function should be
        plotted at the first dimension.
    x2 : array_like
        Input vector with points at which the function should be
        plotted at the second dimension.
    a : array_like
        Input vector with dimension :math:`2` passed to carlberg.
    b : integer
        Scaling factor of the sinusoidal terms passed to carlberg.
    title : string
            Title of the graph.
    save_name : string
                Name under which the graph will be saved.

    Returns
    -------
    graph
         Returns graph in png format with name 'save_name.png'.
    """

    xvalues = []
    array = []
    for i in range(0, len(x1)):
        for j in range(0, len(x2)):
            array = [x1[i], x2[j]]
            xvalues.append(array)

    yvalues = np.linspace(0, 0, len(xvalues))
    for i, j in zip(xvalues, range(0, len(xvalues))):
        yvalues[j] = carlberg(i, a, b)

    xvalues = np.array(xvalues)
    xvalues1 = xvalues[:, 0]
    xvalues2 = xvalues[:, 1]

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(xvalues1, xvalues2, yvalues, c=yvalues, cmap="viridis", linewidth=0.05)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$f(x_1, x_2)$")
    fig.suptitle(title)
    fig.savefig(save_name)


x1 = np.linspace(-2, 5, 500)
x2 = np.linspace(-2, 5, 500)
a = (1, 1)

plot_carlberg(
    x1=x1, x2=x2, a=a, b=0, title="Without Noise", save_name="fig-carlberg_no_noise",
)
plot_carlberg(
    x1=x1, x2=x2, a=a, b=1, title="Noise Included", save_name="fig-carlberg_noise",
)
