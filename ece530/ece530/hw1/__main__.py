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
        print("dx:", dx)
        print("dfx:", dfx)
        self.J += np.outer((dfx - (self.J @ dx)), dx.T) / (
            np.linalg.norm(dx, ord=2) ** 2
        )

        return x_i


def solve_and_plot(
    desc: str,
    iterator,
    x0: np.ndarray,
    epsilon: float = 1e-10,
    iter: int = 100,
) -> None:
    x = x0
    epsilons = []
    t0 = time.time()
    tf = -1
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

    plt.figure()
    plt.semilogy(epsilons)
    plt.xlabel("Iteration Number")
    plt.ylabel("||f(x)||_2")
    plt.title(
        f"{desc}\nMin Error: {Decimal(min(epsilons)):.3E}\nConvergence Time: {tf:.1E} seconds"
    )


if __name__ == "__main__":
    x0 = np.array([5.0, 0])

    solve_and_plot("Symbolic Newton-Raphson", symbolic_newton, x0)

    solve_and_plot("Forward-Difference Newton-Raphson", forward_difference_newton, x0)

    solve_and_plot("Center-Difference Newton-Raphson", center_difference_newton, x0)

    solve_and_plot("Broydens Method, J0=I", BroydensMethod(J0=Jf(x0)), x0)

    plt.show()
