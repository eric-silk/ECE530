from typing import Union
from decimal import Decimal

import numpy as np
import matplotlib.pyplot as plt

from .. import numerical_methods as nm


def f(x: np.ndarray) -> np.ndarray:
    x1, x2 = x[0], x[1]
    fx1 = 4 * x2**2 + 4 * x2 + 52 * x1 - 19
    fx2 = 169 * x1**2 + 3 * x2**2 + 111 * x1 - 10 * x2 - 10

    return np.array([fx1, fx2])


def Jf(x: np.ndarray) -> np.ndarray:
    x1, x2 = x[0], x[1]
    ret = np.array([[52, 8 * x2 + 4], [338 * x1 + 111, 6 * x2 - 10]])

    return ret


def solve_and_plot(
    desc: str,
    iterator: Union[nm.NumericalIterator, nm.BroydensMethod, nm.BroydensInvMethod],
    x0: np.ndarray,
    epsilon: float = 1e-10,
    iter: int = 100,
) -> None:
    x = x0
    epsilons = []
    for i in range(iter):
        x = iterator(x)
        eps = np.linalg.norm(iterator.f(x), ord=2)
        epsilons.append(eps)
        if eps < epsilon:
            print(f"Converged at iteration {i+1}")
            break
    else:
        print(f"Failed to converge in {iter} iterations!")

    plt.figure()
    plt.semilogy(epsilons)
    plt.xlabel("Iteration Number")
    plt.ylabel("||f(x)||_2")
    plt.title(f"{desc}\nMin Error: {Decimal(min(epsilons)):.3E}")


if __name__ == "__main__":
    x0 = np.array([5, 0])

    iterator1 = nm.NewtonRaphson(f, Jf)
    solve_and_plot("Symbolic Newton-Raphson", iterator1, x0)

    iterator2 = nm.NewtonRaphson(f, nm.ForwardDifferenceDerivative(f, 0.5))
    solve_and_plot("Forward-Difference Newton-Raphson", iterator2, x0)

    # plt.show()
