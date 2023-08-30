from abc import ABC, abstractmethod
from typing import Any, Callable
import warnings

import numpy as np


class NumericalSolver(ABC):
    """
    A generic interface for a solver. Accepts a function "f" (mapping R^n->R) and its Jacobian "J" (mapping R^n->R^n)
    and provides methods to step given some input x
    """

    def __init__(
        self,
        f: Callable[[np.ndarray], np.ndarray],
        J: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        self.f = f
        self.J = J

    def __call__(self, x_i: np.ndarray) -> np.ndarray:
        return self.step(x_i)

    @abstractmethod
    def step(self, x_i: np.ndarray) -> np.ndarray:
        """
        Given an iterate x_i, take a step using the solver
        """
        pass


class NewtonRaphson(NumericalSolver):
    """
    Implements the Newton-Raphson root-finding algorithm. Note that Misters Newton and Raphson provide no warranty,
    express or implied, as to the validity of the roots found.
    """

    def step(self, x_i: np.ndarray) -> np.ndarray:
        return x_i - self.f(x_i) / self.J(x_i)


class NumericalDerivative(ABC):
    """
    A generic interface for a numerical differentiation scheme.
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.derivative(x)

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        pass


class FiniteDifferenceDerivative(ABC):
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
    """

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return (self.f(x + self.h) - self.f(x)) / self.h


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
