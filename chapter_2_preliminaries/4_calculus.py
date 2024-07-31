import numpy as np
import matplotlib.pyplot as plt
import torch


def set_figsize(figsize=(3.5, 2.5)) -> None:
    plt.rcParams["figure.figsize"] = figsize


def set_up_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend) -> None:
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)

    axes.set_xlim(xlim)
    axes.set_ylim(ylim)

    axes.set_xscale(xscale)
    axes.set_yscale(yscale)

    if legend:
        axes.legend(legend)

    axes.grid()


def plot(
    X,
    Y=None,
    xlabel=None,
    ylabel=None,
    legend=None,
    xlim=None,
    ylim=None,
    xscale="linear",
    yscale="linear",
    formats=("-", "m--", "g-.", "r:"),
    figsize=(3.5, 2.5),
    axes=None,
) -> None:
    if legend is None:
        legend = []

    set_figsize(figsize)
    # If axes is not None, False, or an empty list, then assign the value of axes to itself.
    # If the axes do not exist, then call plt.gca() which fetches the current axes.
    axes = axes if axes else plt.gca()

    # Return True if `X` (tensor or list) has 1 axis
    def has_one_axis(X):
        return (
            hasattr(X, "ndim")
            and X.ndim == 1
            or isinstance(X, list)
            and not hasattr(X[0], "__len__")
        )

    if has_one_axis(X):
        X = [X]

    if Y is None:
        # Assigns [[]] * len(X) to X, and the original value of X to Y.
        # If X = [1, 2, 3], then X = [[], [], []] and Y = [1, 2, 3]
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]

    if len(X) != len(Y):
        X = X * len(Y)

    # Clears the current axes of titles, labels, etc.
    axes.cla()

    for x, y, format in zip(X, Y, formats):
        if len(x):
            axes.plot(x, y, format)
        else:
            axes.plot(y, format)

    set_up_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)


def f(x):
    return 3 * x**2 - 4 * x


def main():
    x = np.arange(0, 3, 0.1)
    plot(
        X=x,
        Y=[f(x), 2 * x - 3],
        xlabel="x",
        ylabel="f(x)",
        legend=["f(x)", "Tangent line (x=1)"],
    )

    plt.savefig("image/plot.png")


if __name__ == "__main__":
    main()
