from typing import Union
from decimal import Decimal
import time

import numpy as np
import matplotlib.pyplot as plt

from .. import numerical_methods as nm

h = 0.5


def f(x: np.ndarray) -> np.ndarray:
    x1, x2 = x[0], x[1]
    fx1 = 4 * x2**2 + 4 * x2 + 52 * x1 - 19
    fx2 = 169 * x1**2 + 3 * x2**2 + 111 * x1 - 10 * x2 - 10

    return np.array([fx1, fx2])


def Jf(x: np.ndarray) -> np.ndarray:
    x1, x2 = x[0], x[1]
    ret = np.array([[52.0, 8.0 * x2 + 4.0], [338.0 * x1 + 111.0, 6.0 * x2 - 10.0]])

    return ret


def symbolic_newton(x):
    return x - (np.linalg.pinv(Jf(x)) @ f(x))


def forward_difference_newton(x):
    J = np.zeros((2, 2))
    J[:, 0] = (f(x + np.array([h, 0])) - f(x)) / h
    J[:, 1] = (f(x + np.array([0, h])) - f(x)) / h

    return x - (np.linalg.pinv(J) @ f(x))


def center_difference_newton(x):
    J = np.zeros((2, 2))
    J[:, 0] = (f(x + np.array([h, 0])) - f(x - np.array([h, 0]))) / (2 * h)
    J[:, 1] = (f(x + np.array([0, h])) - f(x - np.array([0, h]))) / (2 * h)

    return x - (np.linalg.pinv(J) @ f(x))


class BroydensMethod:
    def __init__(self, J0):
        self.J = J0

    def __call__(self, x):
        fx = f(x)
        dx = -np.linalg.solve(self.J, fx)
        x_i = x + dx
        dfx = f(x_i) - fx
        self.J += np.outer((dfx - (self.J @ dx)), dx.T) / (
            np.linalg.norm(dx, ord=2) ** 2
        )

        return x_i


def solve_and_plot(
    desc: str,
    iterator,
    x0: np.ndarray,
    fig: plt.Figure,
    ax: plt.Axes,
    epsilon: float = 1e-10,
    iter: int = 100,
) -> None:
    _ = fig
    x = x0
    epsilons = []
    t0 = time.time()
    tf = np.inf
    for i in range(iter):
        try:
            x = iterator(x)
            eps = np.linalg.norm(f(x), ord=2)
            epsilons.append(eps)
            if eps < epsilon:
                tf = time.time() - t0
                print(f"Converged at iteration {i+1}")
                break
        except Exception as e:
            print(f"Exception occurred at iterate {i}")
            raise e
    else:
        print(f"Failed to converge in {iter} iterations!")

    ax.semilogy(epsilons)
    ax.set_xlabel("Iteration Number")
    ax.set_ylabel("||f(x)||_2")
    ax.set_title(
        f"{desc}\nx0: {x0}, Min Error: {Decimal(min(epsilons)):.3E}, Convergence Time: {tf:.1E} seconds"
    )


if __name__ == "__main__":
    x0s = [
        np.array([5.0, 0.0]),
        np.array([5.1, -0.1]),
        np.array([0.0, 5.0]),
        np.array([100, -500]),
    ]

    for x0 in x0s:
        fig, axes = plt.subplots(2, 2)
        fig.tight_layout()
        solve_and_plot("Symbolic Newton-Raphson", symbolic_newton, x0, fig, axes[0, 0])

        solve_and_plot(
            "Forward-Difference Newton-Raphson",
            forward_difference_newton,
            x0,
            fig,
            axes[0, 1],
        )

        solve_and_plot(
            "Center-Difference Newton-Raphson",
            center_difference_newton,
            x0,
            fig,
            axes[1, 0],
        )

        J0 = np.zeros((2, 2))
        J0[:, 0] = (f(x0 + np.array([h, 0])) - f(x0 - np.array([h, 0]))) / (2 * h)
        J0[:, 1] = (f(x0 + np.array([0, h])) - f(x0 - np.array([0, h]))) / (2 * h)
        print(f"Broyden's Method J0:\n{J0}")
        solve_and_plot(
            "Broydens Method, initialized with center-difference",
            BroydensMethod(J0),
            x0,
            fig,
            axes[1, 1],
        )

    plt.show()
