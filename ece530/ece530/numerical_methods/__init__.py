from abc import ABC, abstractmethod
from typing import Any, Callable
import warnings

import numpy as np


class NumericalIterator(ABC):
    """
    A generic interface for a solver.

    args:
        - f: The function itself
        - J: The Jacobian object
        - J_inv: Whether the Jacobian object is already inverted (i.e. J^{-1})
    """

    def __init__(
        self,
        f: Callable[[np.ndarray], np.ndarray],
        J: Callable[[np.ndarray], np.ndarray],
        J_inv: bool = False,
    ) -> None:
        self.f = f
        self.J = J
        self.J_inv = J_inv

    def __call__(self, x_i: np.ndarray) -> np.ndarray:
        return self.step(x_i)

    @abstractmethod
    def step(self, x_i: np.ndarray) -> np.ndarray:
        """
        Given an iterate x_i, take a step using the solver
        """
        pass


class NewtonRaphson(NumericalIterator):
    """
    Implements the Newton-Raphson root-finding algorithm's iteration. Note that Misters Newton and Raphson provide no warranty,
    express or implied, as to the validity of the roots found.
    """

    def _step(self, x_i: np.ndarray) -> np.ndarray:
        return x_i - (np.linalg.pinv(self.J(x_i)) @ self.f(x_i))

    def _step_inv(self, x_i: np.ndarray) -> np.ndarray:
        return x_i - (self.J(x_i) @ self.f(x_i))

    def step(self, x_i: np.ndarray) -> np.ndarray:
        if self.J_inv:
            return self._step_inv(x_i)
        return self._step(x_i)


class NumericalDerivative(ABC):
    """
    A generic interface for a numerical differentiation scheme.
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.derivative(x)

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        pass


class FiniteDifferenceDerivative(NumericalDerivative):
    def __init__(self, f: Callable[[np.ndarray], np.ndarray], h: float) -> None:
        self.f = f
        if h == 0.0:
            raise ValueError(f"Finite difference h ({h}) must not be 0!")
        if not h > 0.0:
            warnings.warn(
                f"Provided difference magnitude h ({h}) was non-positive; taking absolute value..."
            )
        self.h = np.abs(h)


class ForwardDifferenceDerivative(FiniteDifferenceDerivative):
    """
    Calculates the forward difference approximation for the Jacobian of f()
    Assumes x is a vector
    """

    def derivative(self, x: np.ndarray) -> np.ndarray:
        J = np.zeros((x.size, x.size))
        fx = self.f(x)
        print("x:", x)
        for row, _ in enumerate(J):
            tmp_x = x
            print("tmp_x:", tmp_x)
            tmp_x[row] += self.h

            J[:, row] = (self.f(tmp_x) - fx) / self.h

        print(J)
        return J


class BackwardDifferenceDerivative(FiniteDifferenceDerivative):
    """
    Calculates the backward difference approximation for the Jacobian of f()
    """

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return (self.f(x) - self.f(x - self.h)) / self.h


class CenterDifferenceDerivative(FiniteDifferenceDerivative):
    """
    Calculates the center difference derivative approximation for the Jacobian of f()
    """

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return (self.f(x + self.h) - self.f(x - self.h)) / (2 * self.h)


class BroydensMethod:
    def __init__(self, f: Callable[[np.ndarray], np.ndarray], J0: np.ndarray) -> None:
        self.f = f
        self.J = J0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.step(x)

    def step(self, x: np.ndarray) -> np.ndarray:
        fx = self.f(x)
        dx = -np.linalg.solve(self.J, fx)
        # Next iterate
        x_i = x + dx
        # Slightly less than perfectly efficient because we're having to evaluate f(x) more times than needed. But whatevs.
        dfx = self.f(x_i) - fx
        # Update our estimate of the Jacobian
        self.J += ((dfx - self.J @ dx) @ dx.T) / (np.linalg.norm(dx, ord=2) ** 2)

        return x_i


class BroydensInvMethod:
    def __init__(
        self, f: Callable[[np.ndarray], np.ndarray], Jinv0: np.ndarray
    ) -> None:
        self.f = f
        self.Jinv = Jinv0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.step(x)

    def step(self, x: np.ndarray) -> np.ndarray:
        fx = self.f(x)
        dx = -(self.Jinv @ fx)
        x_i = x + dx
        dfx = self.f(x_i)
        Jin_dfx = self.Jinv @ dfx

        self.Jinv += (dfx - Jin_dfx) / (dx.T @ self.Jinv @ dfx) @ dx.T @ self.Jinv
