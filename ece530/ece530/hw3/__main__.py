from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress, chi2

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


def problem_one() -> None:
    x = DATA[0]
    y = DATA[1]
    f_hat = scipy_regression(x, y)
    f_mine = my_regression(x, y)
    x_base = np.linspace(0, 5, 100)
    y_hat_scipy = f_hat(x_base)
    y_hat_mine = f_mine(x_base)
    plt.figure(figsize=(8, 8))
    plt.title("Linear Regressions")
    plt.scatter(x, y, label="Raw Data", c="red")
    plt.plot(x_base, y_hat_scipy, label="SciPy Regression")
    plt.plot(
        x_base, y_hat_mine, label="Exlicitly calculated Regression", linestyle="dashed"
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()

    plt.savefig("./hw3_regression.png")


def generate_e(mean: np.ndarray, covariance_matrix: np.ndarray) -> np.ndarray:
    return np.random.multivariate_normal(mean, cov=covariance_matrix)


class NoisySystem:
    def __init__(
        self, A: np.ndarray, x: np.ndarray, mean: np.ndarray, R: np.ndarray
    ) -> None:
        self.A = A
        self.x = x
        self.mean = mean
        self.R = R

    def __call__(self) -> np.ndarray:
        return self.A @ self.x + generate_e(self.mean, self.R)

    def __repr__(self) -> str:
        return f"A:\n{self.A}\nx:\n{self.x}\nMean:\n{self.mean}\nR:\n{self.R}"


class LinearEstimator:
    def __init__(self, A: np.ndarray, R: np.ndarray) -> None:
        # I know, I know, directly calling an inverse is bad...
        self.R_inv = np.linalg.inv(R)
        self.A = A
        self.da_matrix = np.linalg.inv(A.T @ self.R_inv @ A) @ A.T @ self.R_inv

    def __call__(self, y: np.ndarray) -> np.ndarray:
        return self.da_matrix @ y

    def estimate_error(self, y: np.ndarray, x_hat: np.ndarray) -> float:
        tmp = y - (self.A @ x_hat)
        return tmp.T @ self.R_inv @ tmp


def problem_2_experiment(system: NoisySystem, estimator: LinearEstimator) -> float:
    y = system()
    x_hat = estimator(y)
    return estimator.estimate_error(y, x_hat)


def problem_two() -> None:
    mean = np.zeros(4)
    R = np.diag([0.1, 0.2, 0.3, 0.4])

    A = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    x_true = np.array([1, 1])
    system = NoisySystem(A, x_true, mean, R)
    estimator = LinearEstimator(A, R)

    first_y = system()
    print(f"First y:\n{first_y}")
    x_hat = estimator(first_y)
    print(f"x hat:\n{x_hat}")
    J = estimator.estimate_error(first_y, x_hat)
    print(f"J: {J}, Conforms to error model (with 95% confidence): {J <= 5.9915}")

    EXPERIMENT_COUNT = 10000
    rslts = np.zeros(EXPERIMENT_COUNT)
    for i in range(EXPERIMENT_COUNT):
        rslts[i] = problem_2_experiment(system, estimator)

    times_conformed = np.count_nonzero(rslts <= 5.9915)
    error_rate = (times_conformed / EXPERIMENT_COUNT) * 100

    df = 2
    x = np.linspace(rslts.min(), rslts.max(), 10000)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.hist(rslts, bins=100, density=True)
    ax.set_title(
        f"Estimated Error of 10k Measurements and their MLE Parameter Estimation\nConformance Rate: {error_rate}%"
    )
    ax.set_xlabel("Absolute Error Magnitude, Arbitrary Units")
    ax.set_ylabel("Probability Density")
    ax.plot(x, chi2(df).pdf(x), label="Chi Squared (df=2) PDF")
    ax.legend()
    plt.savefig("./problem2_hist.png")


if __name__ == "__main__":
    problem_one()
    problem_two()
