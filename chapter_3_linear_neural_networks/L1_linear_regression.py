import math
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from util.plot import plot


def vectorization_speedups() -> None:
    # Slow solution: manually adding each element from a and b
    n = 10000
    a = torch.ones(n)
    b = torch.ones(n)
    c = torch.zeros(n)

    t = time.time()

    for i in range(n):
        c[i] = a[i] + b[i]

    print(f"{time.time() - t:.5f} sec")

    # Faster solution: use the overloaded + operator to compute the elementwise sum
    t = time.time()
    d = a + b

    print(f"{time.time() - t:.5f} sec")


def normal_visualization() -> None:
    def normal(x, mu, sigma) -> np.ndarray:
        p = 1 / math.sqrt(2 * math.pi * sigma**2)
        return p * np.exp(-0.5 * (x - mu) ** 2 / sigma**2)

    x = np.arange(-7, 7, 0.01)
    params = [(1, 3), (3, 2), (0, 1)]

    plot(
        X=x,
        Y=[normal(x, mu, sigma) for mu, sigma in params],
        xlabel="x",
        ylabel="p(x)",
        figsize=(4.5, 2.5),
        legend=[f"mean {mu}, std {sigma}" for mu, sigma in params],
    )

    plt.savefig("image/linear_regression_plot.png")


if __name__ == "__main__":
    vectorization_speedups()
    normal_visualization()
