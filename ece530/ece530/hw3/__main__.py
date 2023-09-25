from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

DATA = np.array(
    [
        (1.00, 1.10),
        (1.50, 1.62),
        (2.00, 1.98),
        (2.50, 2.37),
        (3.00, 3.23),
        (3.50, 3.69),
        (4.00, 3.97),
    ]
).T


def scipy_regression(
    x: np.ndarray, y: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    r = linregress(x, y)
    print(f"Intercept: {r.intercept}")
    f = lambda x: r.slope * x + r.intercept

    return f


def my_regression(x: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    N = y.size
    d = (np.sum(x**2) * np.sum(y) - np.sum(x) * np.sum(x * y)) / (
        N * np.sum(x**2) - np.sum(x) ** 2
    )
    c = (np.sum(x) * np.sum(y) - N * np.sum(x * y)) / (
        np.sum(x) ** 2 - N * np.sum(x**2)
    )

    return lambda x: c * x + d


if __name__ == "__main__":
    x = DATA[0]
    y = DATA[1]
    f_hat = scipy_regression(x, y)
    f_mine = my_regression(x, y)
    x_base = np.linspace(0, 5, 100)
    y_hat_scipy = f_hat(x_base)
    y_hat_mine = f_mine(x_base)
    plt.figure()
    plt.title("Linear Regressions")
    plt.scatter(x, y, label="Raw Data", c="red")
    plt.plot(x_base, y_hat_scipy, label="SciPy Regression")
    plt.plot(
        x_base, y_hat_mine, label="Exlicitly calculated Regression", linestyle="dashed"
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()

    plt.show()
